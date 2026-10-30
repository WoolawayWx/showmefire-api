"""Tests for services/fire_outlook/nbm.py and its integration into the verification rows / worker (no network)."""
import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from services.fire_outlook import nbm, verification, windows, worker
from tests.test_fire_outlook_verification import STATIONS, adapters
from tests.test_fire_outlook_worker import CYCLE, cfg

STN = [{"station_id": "A", "lat": 38.0, "lon": -92.0}, {"station_id": "B", "lat": 39.0, "lon": -93.0}]


def fake_fetch_factory(calls=None, missing=()):
    """T rises 0.1 K/h from 280 K, RH falls 0.1 %/h from 80, wind = 2 + 0.01*lead, 6 h precip = 1 mm per 6-hour window."""
    def fetch(session, cycle, lead, stations, cells, tmp_dir):
        if calls is not None:
            calls.append(lead)
        if lead in missing:
            raise nbm.NBMUnavailable(f"NBM f{lead:03d} not published")
        S = len(stations)
        out = {"t2m": np.full(S, 280.0 + 0.1 * lead), "rh": np.full(S, 80.0 - 0.1 * lead), "wind": np.full(S, 2.0 + 0.01 * lead)}
        if lead % windows.WINDOW_HOURS == 0:
            out["precip6"] = np.full(S, 1.0)
        return out, (np.zeros(S, int), np.zeros(S, int))
    return fetch


def test_url_and_cadence_match_what_noaa_publishes():
    assert nbm.grib_url(CYCLE, 51).endswith("/blend.20261029/00/core/blend.t00z.core.f051.co.grib2")
    leads = nbm.LEADS
    assert leads[:48] == tuple(range(1, 49)) and leads[48:51] == (51, 54, 57) and leads[-1] == 168 and len(leads) == 88
    assert 49 not in leads and 50 not in leads and 52 not in leads              # not published (probed)
    assert sum(1 for l in leads if l % 6 == 0) == windows.N_WINDOWS            # one 6 h accumulation per window


def test_nearest_cells_and_distance_guard():
    lat2d, lon2d = np.meshgrid(np.linspace(36, 41, 201), np.linspace(-97, -88, 361), indexing="ij")
    rows, cols, km = nbm.nearest_cells(lat2d, lon2d, STN)
    assert abs(lat2d[rows[0], cols[0]] - 38.0) < 0.03 and abs(lon2d[rows[1], cols[1]] + 93.0) < 0.03 and (km < 3).all()
    with pytest.raises(nbm.NBMUnavailable):
        nbm.nearest_cells(lat2d, lon2d, [{"station_id": "X", "lat": 10.0, "lon": 10.0}])


def test_station_series_interpolates_after_f048_and_uses_six_hour_precip(tmp_path):
    calls = []
    s = nbm.station_series(CYCLE, STN, tmp_path, workers=2, fetch=fake_fetch_factory(calls))
    assert sorted(calls) == sorted(nbm.LEADS)                                   # only published leads are requested
    k = np.arange(1, 169)
    np.testing.assert_allclose(s["t_c"][:, 0], 280.0 + 0.1 * k - 273.15, atol=1e-9)   # exact for a linear field, incl. f049/f050
    np.testing.assert_allclose(s["wind_ms"][:, 1], 2.0 + 0.01 * k, atol=1e-9)
    assert s["t_c"].shape == (168, 2) and s["precip6"].shape == (28, 2) and (s["precip6"] == 1.0).all()


def test_cache_hit_skips_network_and_station_change_refetches(tmp_path):
    calls = []
    nbm.station_series(CYCLE, STN, tmp_path, workers=2, fetch=fake_fetch_factory(calls))
    n = len(calls)
    nbm.station_series(CYCLE, STN, tmp_path, workers=2, fetch=fake_fetch_factory(calls))
    assert len(calls) == n
    nbm.station_series(CYCLE, STN[:1], tmp_path, workers=2, fetch=fake_fetch_factory(calls))
    assert len(calls) == 2 * n


def test_any_missing_lead_means_no_baseline_and_no_cache(tmp_path):
    out = nbm.reference_for(CYCLE, STN, tmp_path, workers=2, fetch=fake_fetch_factory(missing={96}))
    assert out is None and not list(tmp_path.glob("nbm_*"))                     # never a partial baseline


def test_window_stats_definitions():
    k = np.arange(1, 169)
    series = {"t_c": np.sin(k / 7.0)[:, None] * np.ones((1, 2)), "rh": (50 + k % 13)[:, None] * np.ones((1, 2)) * 1.0,
              "wind_ms": (k % 5)[:, None] * np.ones((1, 2)) * 1.0, "precip6": np.arange(28)[:, None] * np.ones((1, 2))}
    w = nbm.window_stats(series)
    assert all(v.shape == (28, 2) for v in w.values())
    assert w["t2m_c"][0, 0] == pytest.approx(np.sin(k[:6] / 7.0).max()) and w["rh_pct"][1, 0] == (50 + k[6:12] % 13).min()
    assert w["wind_ms"][2, 0] == (k[12:18] % 5).max() and w["precip_mm"][9, 0] == 9


# --- integration ---------------------------------------------------------------------------------------------------

def _reference(n_stations):
    return {v: np.tile(np.linspace(1, 2, 28)[:, None] * (i + 1), (1, n_stations))
            for i, v in enumerate(("t2m_c", "rh_pct", "wind_ms", "precip_mm"))}


def run_with(tmp_path, reference):
    ad = adapters()
    ad.reference = reference
    out = worker.run_cycle(CYCLE, ad, cfg(tmp_path, publish=False), now_fn=lambda: CYCLE + timedelta(hours=6))
    assert out["ok"], out
    root = tmp_path / "outlook"
    return root, verification.read_table(root / "verification" / "forecast" / "20261029T00Z"), \
        json.loads((root / "status.json").read_text())["verification_export"]


def test_nbm_columns_flow_into_forecast_rows_and_training_schema(tmp_path):
    root, df, status = run_with(tmp_path, lambda cycle, stations: _reference(len(stations)))
    for v in ("t2m_c", "rh_pct", "wind_ms", "precip_mm"):
        assert f"{v}_nbm" in df and df[f"{v}_nbm"].notna().all()
    a = df[(df.station_id == "A") & (df.window_idx == 27)].iloc[0]
    assert a["t2m_c_nbm"] == pytest.approx(2.0) and a["precip_mm_nbm"] == pytest.approx(8.0)
    assert status["nbm"] is True and "nbm_error" not in status


def test_missing_or_failing_nbm_is_not_fatal_and_leaves_no_columns(tmp_path):
    _, df, status = run_with(tmp_path / "none", lambda cycle, stations: None)
    assert not [c for c in df.columns if c.endswith("_nbm")] and status["nbm"] is False and "unavailable" in status["nbm_error"]

    def boom(cycle, stations):
        raise ConnectionError("noaa-nbm-grib2-pds unreachable")
    _, df2, status2 = run_with(tmp_path / "boom", boom)
    assert not [c for c in df2.columns if c.endswith("_nbm")] and "ConnectionError" in status2["nbm_error"]
    assert len(df2) == 3 * 28                                                  # the rest of the export is intact


def test_without_a_reference_provider_nothing_changes(tmp_path):
    _, df, status = run_with(tmp_path, None)
    assert not [c for c in df.columns if c.endswith("_nbm")] and status["nbm"] is False and "nbm_error" not in status
