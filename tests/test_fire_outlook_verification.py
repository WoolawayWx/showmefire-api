"""Tests for services/fire_outlook/verification.py: real worker runs on a tiny grid + a temp observations DB."""
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer

from services.fire_outlook import assemble, moisture, rules, verification, windows, worker
from tests.test_fire_outlook_driver import GRID, member
from tests.test_fire_outlook_worker import CYCLE, GRID_DEF, cfg


def thresholds():
    from core.fire_danger import RULE_SPEC
    return RULE_SPEC["thresholds"]


def center(r, c):
    lon, lat = Transformer.from_crs(GRID_DEF["crs"], 4326, always_xy=True).transform(
        GRID_DEF["west"] + (c + 0.5) * 3000, GRID_DEF["north"] - (r + 0.5) * 3000)
    return lat, lon


STATIONS = [{"station_id": "A", "lat": center(0, 0)[0], "lon": center(0, 0)[1]},
            {"station_id": "B", "lat": center(1, 2)[0], "lon": center(1, 2)[1]},
            {"station_id": "C", "lat": center(2, 3)[0], "lon": center(2, 3)[1]},
            {"station_id": "OUT", "lat": 10.0, "lon": 10.0}]          # outside the grid: dropped


def adapters(stations=STATIONS, n=4):
    # members differ in temperature so member distributions are not degenerate
    members = [member(temp=18.0 + i, rh=40.0, wind=3.0, rain=0.2) for i in range(n)]
    state = moisture.initial_state(np.full(GRID, 20.0), np.full(GRID, 40.0))
    return worker.Adapters(
        inputs_ready=lambda c: (True, {"available_utc": CYCLE + timedelta(hours=5)}), members=lambda c: iter(members),
        n_members=n, conditioning=lambda c: (state, {"level": "conditioned"}),
        rules=lambda: rules.rule_functions(gdd_accum=None), grid=GRID_DEF,
        source_cycles=lambda c: {"gefs": "x"}, stations=(lambda: list(stations)) if stations is not None else None)


@pytest.fixture
def run_root(tmp_path):
    out = worker.run_cycle(CYCLE, adapters(), cfg(tmp_path, publish=False), now_fn=lambda: CYCLE + timedelta(hours=6))
    assert out["ok"], out
    return tmp_path / "outlook"


def make_db(path, rows):
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE observations (id INTEGER PRIMARY KEY AUTOINCREMENT, station_id TEXT, observation_date TEXT,
        fuel_moisture_percentage REAL, temp_c REAL, rel_humidity REAL, wind_speed_ms REAL, precip_accum_1h_mm REAL,
        latitude REAL, longitude REAL, UNIQUE(station_id, observation_date))""")
    con.executemany("INSERT INTO observations (station_id, observation_date, fuel_moisture_percentage, temp_c, rel_humidity,"
                    " wind_speed_ms, precip_accum_1h_mm, latitude, longitude) VALUES (?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    return path


def obs_rows(station, hours, fm=8.0, temp=20.0, rh=40.0, wind=3.0, precip=0.0, fmt="%Y-%m-%dT%H:%M:%SZ", minute=0):
    lat, lon = next((s["lat"], s["lon"]) for s in STATIONS if s["station_id"] == station)
    return [(station, (CYCLE + timedelta(hours=h, minutes=minute)).strftime(fmt), fm, temp, rh, wind, precip, lat, lon)
            for h in hours]


def records(run_root):
    return verification.read_table(run_root / "verification" / "records" / "20261029T00Z")


# --- forecast side ------------------------------------------------------------------------------------------

def test_forecast_rows_written_with_run_and_columns_match_training_contract(run_root):
    df = verification.read_table(run_root / "verification" / "forecast" / "20261029T00Z")
    assert set(df["station_id"]) == {"A", "B", "C"}                   # 'OUT' dropped
    assert len(df) == 3 * 28 and df["n_members"].eq(4).all() and (df["window_idx"].between(0, 27)).all()
    for col in ("agreement_elevated", "agreement_critical", "fm10_p10", "fm10_p50", "fm10_p90", "t2m_c_mean", "t2m_c_p10",
                "t2m_c_p90", "rh_pct_mean", "wind_ms_mean", "precip_mm_p90"):
        assert col in df, col
    assert (df["agreement_critical"] <= df["agreement_elevated"] + 1e-9).all()
    assert (df["fm10_p10"] <= df["fm10_p50"]).all() and (df["fm10_p50"] <= df["fm10_p90"]).all()
    w0 = df[(df.station_id == "A") & (df.window_idx == 0)].iloc[0]
    assert w0["t2m_c_p10"] < w0["t2m_c_mean"] < w0["t2m_c_p90"]       # members span 18..21 degC
    assert w0["wind_ms_mean"] == pytest.approx(3.0, abs=1e-3)         # knots -> m/s, wind_factor undone
    assert w0["precip_mm_mean"] == pytest.approx(6 * 0.2, abs=1e-3)
    assert w0["conditioning_level"] == "conditioned"
    assert json.loads((run_root / "status.json").read_text())["verification_export"]["stations"] == 3


def test_forecast_side_matches_grid_and_does_not_change_the_public_run(tmp_path, run_root):
    df = verification.read_table(run_root / "verification" / "forecast" / "20261029T00Z")
    pct = assemble.read_agreement_percent(run_root / "runs" / "20261029T00Z", "elevated")
    a = df[(df.station_id == "B") & (df.window_idx == 5)].iloc[0]
    assert a["agreement_elevated"] * 100 == pytest.approx(pct[5, 1, 2], abs=0.05)
    # the same run without stations produces identical public grids
    other = worker.run_cycle(CYCLE, adapters(stations=None), cfg(tmp_path / "x", publish=False), now_fn=lambda: CYCLE)
    assert other["ok"]
    a_dir = run_root / "runs" / "20261029T00Z"
    b_dir = tmp_path / "x" / "outlook" / "runs" / "20261029T00Z"
    for name in ("agreement_elevated.npz", "agreement_critical.npz", "fm10_min.npz", "weather.npz"):
        with np.load(a_dir / name) as za, np.load(b_dir / name) as zb:
            for k in za.files:
                np.testing.assert_array_equal(za[k], zb[k])


def test_failing_station_provider_is_not_fatal(tmp_path):
    def boom():
        raise sqlite3.OperationalError("unable to open database file")
    ad = adapters()
    ad.stations = boom
    out = worker.run_cycle(CYCLE, ad, cfg(tmp_path, publish=False), now_fn=lambda: CYCLE)
    assert out["ok"]
    st = json.loads((tmp_path / "outlook" / "status.json").read_text())
    assert "OperationalError" in st["verification_export"]["error"] and st["verification_export"]["stations"] == 0
    assert not (tmp_path / "outlook" / "verification").exists()


# --- observation side -------------------------------------------------------------------------------------

def _hot_obs():
    th = thresholds()
    wind_ms = (th["critical_wind"] + 5) / verification.KTS_PER_MS
    return dict(fm=th["elevated_fm"] - 2, rh=th["critical_rh"] - 3, wind=wind_ms)


def _at(hours):
    return CYCLE + timedelta(hours=hours)


def test_records_join_observations_for_matured_windows_only(tmp_path, run_root):
    hot = _hot_obs()
    rows = (obs_rows("A", range(0, 12), fm=hot["fm"], rh=hot["rh"], wind=hot["wind"], temp=25.0)   # fire weather
            + obs_rows("B", range(0, 12), fm=20.0, rh=80.0, wind=1.0, temp=12.0, precip=0.5, fmt="%Y-%m-%d %H:%M:%S")  # calm, wet
            + obs_rows("C", [0, 1], fm=9.0))                                                      # only 2 hours: insufficient
    db = make_db(tmp_path / "obs.db", rows)
    path = verification.build_records(run_root, "20261029T00Z", verification.SqliteObservations(db), _at(12 + 6))
    df = records(run_root)
    assert path is not None and sorted(df["window_idx"].unique()) == [0, 1]            # windows 0,1 ended 6 h ago; 2 did not
    a0 = df[(df.station_id == "A") & (df.window_idx == 0)].iloc[0]
    assert a0["fm10_obs"] == pytest.approx(hot["fm"]) and a0["t2m_c_obs"] == 25.0
    assert bool(a0["obs_elevated"]) and bool(a0["obs_critical"]) and bool(a0["qc_ok"])
    assert a0["obs_concurrent_hours"] == 6 and a0["wind_ms_obs"] == pytest.approx(hot["wind"])
    b1 = df[(df.station_id == "B") & (df.window_idx == 1)].iloc[0]
    assert not bool(b1["obs_elevated"]) and not bool(b1["obs_critical"]) and bool(b1["qc_ok"])
    assert b1["precip_mm_obs"] == pytest.approx(3.0)                                    # 6 h x 0.5 mm
    c0 = df[(df.station_id == "C") & (df.window_idx == 0)].iloc[0]
    assert pd.isna(c0["obs_elevated"]) and pd.isna(c0["fm10_obs"]) and not bool(c0["qc_ok"])  # NaN, never zero-filled


def test_window_not_verified_before_latency(tmp_path, run_root):
    obs = verification.SqliteObservations(make_db(tmp_path / "obs.db", obs_rows("A", range(0, 12), **_hot_obs())))
    assert verification.build_records(run_root, "20261029T00Z", obs, _at(11) + timedelta(minutes=59)) is None
    verification.build_records(run_root, "20261029T00Z", obs, _at(12) + timedelta(minutes=1))
    assert sorted(records(run_root)["window_idx"].unique()) == [0]


def test_persistence_is_causal(tmp_path, run_root):
    rows = (obs_rows("A", [-1], fm=11.0, temp=9.0) + obs_rows("A", [0], fm=10.0, temp=10.0)   # at/before issue
            + obs_rows("A", [1], fm=2.0, temp=30.0)                                           # AFTER issue: must not leak in
            + obs_rows("A", range(2, 8), fm=7.0))
    verification.build_records(run_root, "20261029T00Z",
                               verification.SqliteObservations(make_db(tmp_path / "obs.db", rows)), _at(12))
    a = records(run_root).query("station_id == 'A'").iloc[0]
    assert a["fm10_init_obs"] == 10.0 and a["t2m_c_persist"] == 10.0
    # a station with only post-issue observations has no persistence value
    verification.build_records(run_root, "20261029T00Z", verification.SqliteObservations(
        make_db(tmp_path / "obs2.db", obs_rows("B", range(1, 8)))), _at(12))
    assert pd.isna(records(run_root).query("station_id == 'B'").iloc[0]["fm10_init_obs"])


def test_idempotent_grows_as_windows_mature_and_missing_forecast_file_is_skipped(tmp_path, run_root):
    obs = verification.SqliteObservations(make_db(tmp_path / "obs.db", obs_rows("A", range(0, 60), **_hot_obs())))
    verification.build_records(run_root, "20261029T00Z", obs, _at(12))
    first = records(run_root)
    verification.build_records(run_root, "20261029T00Z", obs, _at(12))
    assert records(run_root).equals(first)
    verification.build_records(run_root, "20261029T00Z", obs, _at(36))
    assert records(run_root)["window_idx"].max() == 4          # window w is verified at (w + 1) * 6 + 6 h latency <= 36 h
    assert verification.build_records(run_root, "20990101T00Z", obs, _at(36)) is None


def test_build_all_skips_old_runs_and_isolates_errors(tmp_path, run_root):
    obs = verification.SqliteObservations(make_db(tmp_path / "obs.db", obs_rows("A", range(0, 12), **_hot_obs())))
    assert "20261029T00Z" in verification.build_all(run_root, obs, now=_at(24))
    assert verification.build_all(run_root, obs, now=_at(24 * 20)) == {}               # older than max_age_days

    class Broken:
        def hourly(self, *a, **k):
            raise RuntimeError("db locked")
    out = verification.build_all(run_root, Broken(), now=_at(24))
    assert out["20261029T00Z"].startswith("error") and "db locked" in out["20261029T00Z"]


def test_stations_query_uses_recent_fm10_reporters(tmp_path):
    rows = obs_rows("A", [0], fm=9.0) + [("D", CYCLE.strftime("%Y-%m-%dT%H:%M:%SZ"), None, 20.0, 40.0, 2.0, 0.0, 38.0, -92.0)]
    obs = verification.SqliteObservations(make_db(tmp_path / "o.db", rows))
    assert [s["station_id"] for s in obs.stations(CYCLE - timedelta(days=1))] == ["A"]  # D never reported fuel moisture
    assert obs.stations(CYCLE + timedelta(days=5)) == []


def test_write_table_falls_back_to_csv_without_pyarrow(tmp_path, monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "pyarrow":
            raise ImportError()
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    path = verification.write_table(pd.DataFrame({"a": [1, 2]}), tmp_path / "t")
    monkeypatch.undo()
    assert path.suffix == ".csv" and verification.read_table(tmp_path / "t")["a"].tolist() == [1, 2]
