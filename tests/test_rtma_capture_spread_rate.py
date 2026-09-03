"""RTMA cache contract checks for spread-rate inputs."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyproj
import xarray as xr

from services.rtma_capture import (
    REQUIRED_RTMA_VARS,
    _align_precipitation,
    ensure_analysis_hour_cached,
    is_analysis_hour_cached,
    spread_rate_poll_minutes,
)


def _grid_dataset(x_values, y_values, variable):
    crs = pyproj.CRS.from_epsg(5070)
    xx, yy = np.meshgrid(x_values, y_values)
    longitude, latitude = pyproj.Transformer.from_crs(
        crs, "EPSG:4326", always_xy=True
    ).transform(xx, yy)
    projection = xr.DataArray(0.0, attrs=crs.to_cf())
    return xr.Dataset(
        {
            variable: (("y", "x"), np.ones(xx.shape, dtype=np.float32)),
            "gribfile_projection": projection,
        },
        coords={
            "latitude": (("y", "x"), latitude),
            "longitude": (("y", "x"), longitude),
        },
    )


def test_precipitation_aligns_with_grib_projection_fallback():
    source = _grid_dataset(np.arange(0, 5000, 1000), np.arange(0, 4000, 1000), "apcp")
    target = _grid_dataset(np.arange(0, 4000, 1000), np.arange(0, 3000, 1000), "t2m")
    grib_attrs = {
        "GRIB_gridType": "lambert",
        "GRIB_LoVInDegrees": 264.0,
        "GRIB_LaDInDegrees": 23.0,
        "GRIB_Latin1InDegrees": 29.5,
        "GRIB_Latin2InDegrees": 45.5,
    }
    source_data = source["apcp"].copy(deep=False)
    source_data.attrs.update(grib_attrs)
    source_data.attrs.pop("_projection_attrs", None)
    target = target.drop_vars("gribfile_projection")
    target["t2m"].attrs.update(grib_attrs)

    aligned = _align_precipitation(source_data, target)

    assert aligned.sizes == target.sizes
    assert np.isfinite(aligned.values).all()


def test_precipitation_aligns_to_analysis_grid():
    source = _grid_dataset(np.arange(0, 5000, 1000), np.arange(0, 4000, 1000), "apcp")
    source_data = source["apcp"].copy(deep=False)
    source_data.attrs["_projection_attrs"] = pyproj.CRS.from_epsg(5070).to_cf()
    target = _grid_dataset(np.arange(0, 4000, 1000), np.arange(0, 3000, 1000), "t2m")

    aligned = _align_precipitation(source_data, target)

    assert aligned.sizes == target.sizes
    assert "_projection_attrs" not in aligned.attrs


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
