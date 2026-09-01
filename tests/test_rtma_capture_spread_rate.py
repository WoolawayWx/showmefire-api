"""RTMA cache contract checks for spread-rate inputs."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import xarray as xr

from services.rtma_capture import (
    REQUIRED_RTMA_VARS,
    ensure_analysis_hour_cached,
    is_analysis_hour_cached,
    spread_rate_poll_minutes,
)


def test_complete_cache_must_include_precipitation(tmp_path):
    cache_dir = tmp_path / "rtma"
    cache_dir.mkdir(parents=True)
    run_dt = datetime(2026, 8, 30, 14)
    path = cache_dir / f"rtma_{run_dt:%Y%m%d_%H}z.nc"
    lat = np.array([[38.0]], dtype=float)
    lon = np.array([[-92.0]], dtype=float)
    ds = xr.Dataset(
        {
            "t2m": (("y", "x"), np.full(lat.shape, 295.0, dtype=np.float32)),
            "r2": (("y", "x"), np.full(lat.shape, 45.0, dtype=np.float32)),
            "u10": (("y", "x"), np.full(lat.shape, 3.0, dtype=np.float32)),
            "v10": (("y", "x"), np.full(lat.shape, 1.0, dtype=np.float32)),
            "latitude": (("y", "x"), lat.astype(np.float32)),
            "longitude": (("y", "x"), lon.astype(np.float32)),
        }
    )
    ds.to_netcdf(path, engine="netcdf4")

    with xr.open_dataset(path) as cached:
        assert not REQUIRED_RTMA_VARS.issubset(cached.data_vars)


def test_precipitation_enabled_cache_is_complete(tmp_path):
    cache_dir = tmp_path / "rtma"
    cache_dir.mkdir(parents=True)
    run_dt = datetime(2026, 8, 30, 14)
    path = cache_dir / f"rtma_{run_dt:%Y%m%d_%H}z.nc"
    lat = np.array([[38.0, 39.0]], dtype=float)
    lon = np.array([[-92.0, -91.0]], dtype=float)
    ds = xr.Dataset(
        {
            "t2m": (("y", "x"), np.full(lat.shape, 295.0, dtype=np.float32)),
            "r2": (("y", "x"), np.full(lat.shape, 45.0, dtype=np.float32)),
            "u10": (("y", "x"), np.full(lat.shape, 3.0, dtype=np.float32)),
            "v10": (("y", "x"), np.full(lat.shape, 1.0, dtype=np.float32)),
            "apcp": (("y", "x"), np.full(lat.shape, 0.5, dtype=np.float32)),
            "latitude": (("y", "x"), lat.astype(np.float32)),
            "longitude": (("y", "x"), lon.astype(np.float32)),
        }
    )
    ds.to_netcdf(path, engine="netcdf4")

    assert is_analysis_hour_cached(run_dt, cache_dir=cache_dir)


def test_ensure_analysis_hour_cached_uses_existing_file(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache" / "rtma"
    run_dt = datetime(2026, 8, 30, 14)

    lat = np.array([[38.0]], dtype=float)
    lon = np.array([[-92.0]], dtype=float)
    path = cache_dir / f"rtma_{run_dt:%Y%m%d_%H}z.nc"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {
            "t2m": (("y", "x"), np.full(lat.shape, 295.0, dtype=np.float32)),
            "r2": (("y", "x"), np.full(lat.shape, 45.0, dtype=np.float32)),
            "u10": (("y", "x"), np.full(lat.shape, 3.0, dtype=np.float32)),
            "v10": (("y", "x"), np.full(lat.shape, 1.0, dtype=np.float32)),
            "apcp": (("y", "x"), np.zeros(lat.shape, dtype=np.float32)),
            "latitude": (("y", "x"), lat.astype(np.float32)),
            "longitude": (("y", "x"), lon.astype(np.float32)),
        }
    )
    ds.to_netcdf(path, engine="netcdf4")
    monkeypatch.setattr("services.rtma_capture._root", lambda: tmp_path)

    with patch("services.rtma_capture.fetch_rtma") as fetch_mock:
        result = ensure_analysis_hour_cached(run_dt, cache_dir=cache_dir)
        fetch_mock.assert_not_called()

    assert result["cached"] is True
    assert result["fetched"] is False


def test_spread_rate_poll_minutes_defaults_to_15(monkeypatch):
    monkeypatch.delenv("SPREAD_RATE_POLL_MINUTES", raising=False)
    assert spread_rate_poll_minutes() == 15


def test_spread_rate_poll_minutes_clamps_invalid_values(monkeypatch):
    monkeypatch.setenv("SPREAD_RATE_POLL_MINUTES", "3")
    assert spread_rate_poll_minutes() == 5
    monkeypatch.setenv("SPREAD_RATE_POLL_MINUTES", "120")
    assert spread_rate_poll_minutes() == 60
