"""Tests for spread-rate moisture conditioning and RAWS anchoring."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import xarray as xr

from services.spread_rate_moisture import (
    MIN_RAW_STATIONS,
    apply_raws_10hr_correction,
    condition_moisture,
    extract_raws_fuel_points,
    nelson_emc,
)


def test_nelson_emc_clips_to_expected_range():
    emc = nelson_emc(np.array([20.0]), np.array([40.0]))
    assert 1.0 <= emc[0] <= 40.0


def test_extract_raws_fuel_points_filters_state_and_freshness():
    now = datetime.now(timezone.utc)
    payload = {
        "stations": [
            {
                "stid": "MO1",
                "state": "MO",
                "latitude": 38.0,
                "longitude": -92.0,
                "observations": {
                    "fuel_moisture": {
                        "value": 12.0,
                        "time": now.isoformat().replace("+00:00", "Z"),
                    }
                },
            },
            {
                "stid": "KS1",
                "state": "KS",
                "latitude": 39.0,
                "longitude": -95.0,
                "observations": {"fuel_moisture": {"value": 12.0}},
            },
            {
                "stid": "STALE",
                "state": "MO",
                "latitude": 37.0,
                "longitude": -91.0,
                "observations": {
                    "fuel_moisture": {
                        "value": 12.0,
                        "time": (now - timedelta(hours=5)).isoformat().replace("+00:00", "Z"),
                    }
                },
            },
        ]
    }
    points = extract_raws_fuel_points(payload)
    assert len(points) == 1
    assert points[0]["stid"] == "MO1"


def test_apply_raws_10hr_correction_requires_minimum_stations():
    lat = np.array([[38.0, 39.0], [37.0, 38.5]])
    lon = np.array([[-92.0, -91.0], [-93.0, -92.5]])
    fm10 = np.full(lat.shape, 10.0)
    stations = [
        {"lat": 38.0, "lon": -92.0, "fm10": 14.0},
        {"lat": 39.0, "lon": -91.0, "fm10": 13.0},
    ]
    with pytest.raises(ValueError, match=str(MIN_RAW_STATIONS)):
        apply_raws_10hr_correction(fm10, lat, lon, stations)


def test_apply_raws_10hr_correction_adjusts_field():
    lat = np.array([[38.0, 39.0], [37.0, 38.5]])
    lon = np.array([[-92.0, -91.0], [-93.0, -92.5]])
    fm10 = np.full(lat.shape, 10.0)
    stations = [
        {"lat": 38.0, "lon": -92.0, "fm10": 16.0},
        {"lat": 39.0, "lon": -91.0, "fm10": 15.0},
        {"lat": 37.0, "lon": -93.0, "fm10": 14.0},
    ]
    corrected, meta = apply_raws_10hr_correction(fm10, lat, lon, stations)
    assert corrected.shape == fm10.shape
    assert meta["station_count"] == 3
    assert np.nanmean(corrected) > np.nanmean(fm10)


def _write_rtma_hour(path, hour: datetime, lat: np.ndarray, lon: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = 20.0 + np.zeros(lat.shape, dtype=np.float32)
    rh = 45.0 + np.zeros(lat.shape, dtype=np.float32)
    ds = xr.Dataset(
        {
            "t2m": (("y", "x"), temp),
            "r2": (("y", "x"), rh),
            "u10": (("y", "x"), np.full(lat.shape, 3.0, dtype=np.float32)),
            "v10": (("y", "x"), np.full(lat.shape, 1.0, dtype=np.float32)),
            "apcp": (("y", "x"), np.zeros(lat.shape, dtype=np.float32)),
            "latitude": (("y", "x"), lat.astype(np.float32)),
            "longitude": (("y", "x"), lon.astype(np.float32)),
        }
    )
    ds.to_netcdf(path, engine="netcdf4")


def test_condition_moisture_warms_up_with_insufficient_history(tmp_path, monkeypatch):
    lat = np.array([[38.0, 39.0], [37.0, 38.5]], dtype=float)
    lon = np.array([[-92.0, -91.0], [-93.0, -92.5]], dtype=float)
    cache_dir = tmp_path / "cache" / "rtma"
    end = datetime(2026, 8, 30, 14, tzinfo=timezone.utc).replace(tzinfo=None)
    for offset in range(24):
        stamp = end - timedelta(hours=offset)
        _write_rtma_hour(cache_dir / f"rtma_{stamp:%Y%m%d_%H}z.nc", stamp, lat, lon)

    monkeypatch.setattr("services.spread_rate_moisture._root", lambda: tmp_path)
    with pytest.raises(RuntimeError, match="insufficient RTMA history"):
        condition_moisture(lat, lon, analysis_hour=end)


def test_condition_moisture_returns_fields_when_history_is_sufficient(tmp_path, monkeypatch):
    lat = np.array([[38.0, 39.0], [37.0, 38.5]], dtype=float)
    lon = np.array([[-92.0, -91.0], [-93.0, -92.5]], dtype=float)
    cache_dir = tmp_path / "cache" / "rtma"
    end = datetime(2026, 8, 30, 14, tzinfo=timezone.utc).replace(tzinfo=None)
    for offset in range(5 * 24):
        stamp = end - timedelta(hours=offset)
        _write_rtma_hour(cache_dir / f"rtma_{stamp:%Y%m%d_%H}z.nc", stamp, lat, lon)

    monkeypatch.setattr("services.spread_rate_moisture._root", lambda: tmp_path)
    result = condition_moisture(lat, lon, analysis_hour=end)
    assert result["fm10_pct"].shape == lat.shape
    assert "u_ms" in result and "v_ms" in result
