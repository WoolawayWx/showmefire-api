"""Tests for services/fire_outlook/conditioning.py (injected RTMA replay; real RAWS anchor)."""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from services.fire_outlook import conditioning, driver

CYCLE = datetime(2026, 10, 29, 12, tzinfo=timezone.utc)
LAT, LON = np.meshgrid(np.linspace(37.0, 39.0, 4), np.linspace(-94.0, -91.0, 5), indexing="ij")


def fake_condition(lat, lon, analysis_hour=None, raws_payload=None):
    assert raws_payload is None, "anchoring must be done causally by the wrapper, not inside the replay"
    assert analysis_hour == CYCLE
    ones = np.ones(lat.shape)
    return {"analysis_hour": "2026-10-29T12:00:00", "fm1_pct": 8 * ones, "fm10_pct": 10 * ones,
            "fm100_pct": 14 * ones, "conditioning_hours_available": 168, "conditioning_hours_missing": 2}


def station(stid, lat, lon, value, minutes_before_cycle, state="MO"):
    stamp = (CYCLE - timedelta(minutes=minutes_before_cycle)).isoformat().replace("+00:00", "Z")
    return {"stid": stid, "state": state, "latitude": lat, "longitude": lon,
            "observations": {"fuel_moisture": {"value": value, "time": stamp}}}


def payload(*stations):
    return {"stations": list(stations)}


def test_causal_filter_rejects_future_stale_untimed_and_out_of_state():
    good = station("A", 38.0, -92.0, 7.0, 30)
    future = station("B", 38.1, -92.1, 7.0, -10)
    stale = station("C", 38.2, -92.2, 7.0, 400)
    other = station("D", 38.3, -92.3, 7.0, 30, state="KS")
    untimed = station("E", 38.4, -92.4, 7.0, 30)
    untimed["observations"]["fuel_moisture"].pop("time")
    out_of_range = station("F", 38.5, -92.5, 80.0, 30)
    pts = conditioning.causal_raws_points(payload(good, future, stale, other, untimed, out_of_range), CYCLE)
    assert [p["stid"] for p in pts] == ["A"]


def test_old_cycle_still_uses_observations_valid_then():
    """Age is measured against the cycle, not now(): a backfill keeps its own-era observations."""
    old = datetime(2025, 3, 1, 0, tzinfo=timezone.utc)
    stamp = (old - timedelta(minutes=20)).isoformat().replace("+00:00", "Z")
    st = {"stid": "A", "state": "MO", "latitude": 38.0, "longitude": -92.0,
          "observations": {"fuel_moisture": {"value": 9.0, "time": stamp}}}
    assert len(conditioning.causal_raws_points(payload(st), old)) == 1


def test_anchored_when_enough_causal_stations():
    stations = [station(f"S{i}", 37.5 + i * 0.4, -93.5 + i * 0.5, 6.0, 20) for i in range(4)]
    state, q = conditioning.condition_initial_state(CYCLE, LAT, LON, payload(*stations), condition=fake_condition)
    assert q["level"] == conditioning.LEVEL_ANCHORED and q["raws_station_count"] == 4
    assert state["fm10"].mean() < 10.0                      # pulled toward the drier observations
    assert (state["fm1"] == 8).all() and (state["fm100"] == 14).all()  # only fm10 is anchored
    assert q["newest_raws_age_minutes"] == pytest.approx(20)


def test_unanchored_when_too_few_stations_and_future_obs_ignored():
    stations = [station("A", 38.0, -92.0, 5.0, 20), station("B", 38.2, -92.2, 5.0, 20),
                station("C", 38.4, -92.4, 5.0, -30)]  # third is in the future
    state, q = conditioning.condition_initial_state(CYCLE, LAT, LON, payload(*stations), condition=fake_condition)
    assert q["level"] == conditioning.LEVEL_CONDITIONED and q["raws_station_count"] == 2
    assert (state["fm10"] == 10).all()


def test_insufficient_history_is_strict_by_default():
    def short(*a, **k):
        raise RuntimeError("insufficient RTMA history: 40/120 hours")
    with pytest.raises(conditioning.ConditioningUnavailable):
        conditioning.condition_initial_state(CYCLE, LAT, LON, None, condition=short)


def test_emc_seed_only_when_opted_in_and_labelled():
    def short(*a, **k):
        raise RuntimeError("insufficient RTMA history: 40/120 hours")
    fb = (np.full(LAT.shape, 20.0), np.full(LAT.shape, 40.0))
    state, q = conditioning.condition_initial_state(CYCLE, LAT, LON, None, condition=short, fallback_weather=fb)
    assert q["level"] == conditioning.LEVEL_EMC_SEED and "insufficient" in q["reason"]
    assert set(state) == {"fm1", "fm10", "fm100"}


def test_state_feeds_the_driver():
    state, _ = conditioning.condition_initial_state(CYCLE, LAT, LON, None, condition=fake_condition)
    assert all(v.shape == LAT.shape for v in state.values())


def test_public_grid_lat_lon():
    lat, lon = conditioning.public_lat_lon()
    assert lat.shape == (264, 267) and lon.shape == (264, 267)
    # buffered Missouri domain: ~34.75-41.9 N
    assert 34.0 < lat.min() and lat.max() < 43.0 and -98.0 < lon.min() and lon.max() < -86.0
    assert lat[0, 0] > lat[-1, 0]  # row 0 is the north edge


# --- RAWS anchor from the observations table (causal) ---------------------------------------------------------

def _obs_db(tmp_path, rows):
    import sqlite3
    con = sqlite3.connect(tmp_path / "obs.db")
    con.execute("CREATE TABLE observations (id INTEGER PRIMARY KEY AUTOINCREMENT, station_id TEXT, observation_date TEXT, "
                "fuel_moisture_percentage REAL, temp_c REAL, rel_humidity REAL, wind_speed_ms REAL, precip_accum_1h_mm REAL, "
                "latitude REAL, longitude REAL, UNIQUE(station_id, observation_date))")
    con.executemany("INSERT INTO observations (station_id, observation_date, fuel_moisture_percentage, latitude, longitude) "
                    "VALUES (?,?,?,?,?)", rows)
    con.commit()
    con.close()
    return tmp_path / "obs.db"


def test_observations_table_payload_is_causal_and_anchors_without_a_state_filter(tmp_path):
    from services.fire_outlook import verification
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    t = lambda minutes: (CYCLE + timedelta(minutes=minutes)).strftime(fmt)  # noqa: E731
    rows = []
    for i in range(4):                                             # four good stations, newest reading 20 min before the cycle
        rows += [(f"S{i}", t(-200), 12.0, 37.5 + i * 0.4, -93.5 + i * 0.5), (f"S{i}", t(-20), 6.0, 37.5 + i * 0.4, -93.5 + i * 0.5)]
    rows += [("FUT", t(+30), 3.0, 38.0, -92.0),                    # after the cycle: must never be used
             ("OLD", t(-600), 3.0, 38.1, -92.1),                   # older than 3 h
             ("NOFM", t(-10), None, 38.2, -92.2)]                  # no fuel moisture
    payload = verification.SqliteObservations(_obs_db(tmp_path, rows)).fuel_moisture_payload(CYCLE)
    assert sorted(s["stid"] for s in payload["stations"]) == ["S0", "S1", "S2", "S3"]
    assert all(s["observations"]["fuel_moisture"]["value"] == 6.0 for s in payload["stations"])   # newest <= cycle, not the 200-min one
    state, q = conditioning.condition_initial_state(CYCLE, LAT, LON, payload, condition=fake_condition, raws_states=None)
    assert q["level"] == conditioning.LEVEL_ANCHORED and q["raws_station_count"] == 4
    # with the default MO-only filter the same stations (state "") are not used: that is why the provider passes raws_states=None
    _, q2 = conditioning.condition_initial_state(CYCLE, LAT, LON, payload, condition=fake_condition)
    assert q2["level"] == conditioning.LEVEL_CONDITIONED
