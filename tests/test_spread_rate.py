"""Tests for Rothermel spread-rate classification and publication."""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from scripts.build_synthetic_fire_behavior_bundle import build as build_synthetic_bundle
from services import spread_rate
from services.spread_rate import (
    ROS_CLASS_BOUNDS,
    ROS_CLASS_LABELS,
    classify_ros_ch_per_h,
    compute_spread_rate_grid,
    generate_spread_rate,
    m_per_min_to_ch_per_h,
    run_spread_rate_pipeline,
    spread_direction_degrees,
    wind_from_degrees,
)


def test_m_per_min_to_ch_per_h_conversion():
    assert np.isclose(m_per_min_to_ch_per_h(1.0), 60.0 / 20.1168)


@pytest.mark.parametrize(
    "rate,label",
    [
        (0.0, "Very Low"),
        (1.9, "Very Low"),
        (2.0, "Low"),
        (4.9, "Low"),
        (5.0, "Moderate"),
        (19.9, "Moderate"),
        (20.0, "High"),
        (49.9, "High"),
        (50.0, "Very High"),
        (149.9, "Very High"),
        (150.0, "Extreme"),
        (500.0, "Extreme"),
    ],
)
def test_classify_ros_boundaries(rate, label):
    classes = classify_ros_ch_per_h(np.array([rate], dtype=float))
    assert ROS_CLASS_LABELS[classes[0]] == label


def test_wind_from_degrees_matches_meteorological_convention():
    # Wind toward the east => from 270 degrees
    direction = wind_from_degrees(np.array([5.0]), np.array([0.0]))[0]
    assert np.isclose(direction, 270.0, atol=1.0)


def test_spread_direction_from_unit_vector():
    assert np.isclose(spread_direction_degrees((0.0, 1.0, 0.0)), 0.0)
    assert np.isclose(spread_direction_degrees((1.0, 0.0, 0.0)), 90.0)


def test_compute_spread_rate_grid_on_synthetic_bundle(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "static"
    bundle_path = build_synthetic_bundle(bundle_dir, version="test-v1")
    manifest = json.loads(bundle_path.with_suffix(".json").read_text())

    with xr.open_dataset(bundle_path) as ds:
        static = {
            "bundle_version": manifest["bundle_version"],
            "lat": np.asarray(ds.latitude.values, dtype=float),
            "lon": np.asarray(ds.longitude.values, dtype=float),
            "slope_deg": np.asarray(ds.slope_degrees.values, dtype=float),
            "aspect_sin": np.asarray(ds.aspect_sin.values, dtype=float),
            "aspect_cos": np.asarray(ds.aspect_cos.values, dtype=float),
            "canopy_cover_pct": np.asarray(ds.canopy_cover_pct.values, dtype=float),
            "canopy_height_m": np.asarray(ds.canopy_height_m.values, dtype=float),
            "fuel_model_code": np.rint(np.asarray(ds.fuel_model_fbfm40.values)).astype(np.int32),
            "valid_mask": np.asarray(ds.static_valid_mask.values) > 0.5,
        }

    shape = static["lat"].shape
    moisture = {
        "fm1_pct": np.full(shape, 8.0),
        "fm10_pct": np.full(shape, 10.0),
        "fm100_pct": np.full(shape, 12.0),
        "live_herbaceous_pct": np.full(shape, 80.0),
        "live_woody_pct": np.full(shape, 100.0),
        "wind_ms": np.full(shape, 5.0),
        "u_ms": np.full(shape, 5.0),
        "v_ms": np.zeros(shape),
    }
    grids = compute_spread_rate_grid(static, moisture)
    finite = grids["ros_ch_per_h"][np.isfinite(grids["ros_ch_per_h"])]
    assert finite.size > 0
    assert np.all(finite >= 0.0)
    assert np.all(grids["class"][np.isfinite(grids["ros_ch_per_h"])] < len(ROS_CLASS_LABELS))


def _write_rtma_cache(cache_dir: Path, end: datetime, lat: np.ndarray, lon: np.ndarray, hours: int):
    for offset in range(hours):
        stamp = end - timedelta(hours=offset)
        path = cache_dir / f"rtma_{stamp:%Y%m%d_%H}z.nc"
        path.parent.mkdir(parents=True, exist_ok=True)
        ds = xr.Dataset(
            {
                "t2m": (("y", "x"), np.full(lat.shape, 25.0, dtype=np.float32)),
                "r2": (("y", "x"), np.full(lat.shape, 40.0, dtype=np.float32)),
                "u10": (("y", "x"), np.full(lat.shape, 4.0, dtype=np.float32)),
                "v10": (("y", "x"), np.full(lat.shape, 1.0, dtype=np.float32)),
                "apcp": (("y", "x"), np.zeros(lat.shape, dtype=np.float32)),
                "latitude": (("y", "x"), lat.astype(np.float32)),
                "longitude": (("y", "x"), lon.astype(np.float32)),
            }
        )
        ds.to_netcdf(path, engine="netcdf4")


def test_generate_spread_rate_publishes_artifacts(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "static"
    bundle_path = build_synthetic_bundle(bundle_dir, version="test-v1")
    monkeypatch.setenv("FIRE_BEHAVIOR_STATIC_BUNDLE", str(bundle_path))
    monkeypatch.setattr(spread_rate, "BETA_ROOT", tmp_path / "testbed")
    monkeypatch.setattr(spread_rate, "SPREAD_RATE_DIR", tmp_path / "testbed" / "spread_rate")
    monkeypatch.setattr(spread_rate, "SPREAD_RATE_GIS_DIR", tmp_path / "testbed" / "gis" / "spread_rate")
    monkeypatch.setattr(spread_rate, "SPREAD_RATE_IMAGE_DIR", tmp_path / "testbed" / "images")
    monkeypatch.setattr(spread_rate, "STATUS_PATH", tmp_path / "testbed" / "spread_rate" / "status.json")
    monkeypatch.setattr(spread_rate, "TIF_PATH", tmp_path / "testbed" / "gis" / "spread_rate" / "spread_rate_latest.tif")
    monkeypatch.setattr(spread_rate, "PNG_PATH", tmp_path / "testbed" / "images" / "spread_rate_latest.png")

    with xr.open_dataset(bundle_path) as ds:
        lat = np.asarray(ds.latitude.values, dtype=float)
        lon = np.asarray(ds.longitude.values, dtype=float)
    end = datetime(2026, 8, 30, 14, tzinfo=timezone.utc).replace(tzinfo=None)
    cache_dir = tmp_path / "cache" / "rtma"
    _write_rtma_cache(cache_dir, end, lat, lon, hours=5 * 24)
    monkeypatch.setattr("services.spread_rate_moisture._root", lambda: tmp_path)
    monkeypatch.setattr("services.rtma_capture._root", lambda: tmp_path)

    result = generate_spread_rate(analysis_hour=end)
    assert result["status"] == "ready"
    assert spread_rate.TIF_PATH.is_file()
    assert spread_rate.PNG_PATH.is_file()
    assert spread_rate.STATUS_PATH.is_file()


def test_generate_spread_rate_reports_warming_without_history(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "static"
    bundle_path = build_synthetic_bundle(bundle_dir, version="test-v1")
    monkeypatch.setenv("FIRE_BEHAVIOR_STATIC_BUNDLE", str(bundle_path))
    monkeypatch.setattr(spread_rate, "BETA_ROOT", tmp_path / "testbed")
    monkeypatch.setattr(spread_rate, "STATUS_PATH", tmp_path / "testbed" / "spread_rate" / "status.json")
    monkeypatch.setattr("services.spread_rate_moisture._root", lambda: tmp_path)
    monkeypatch.setattr("services.spread_rate.is_analysis_hour_cached", lambda *_args, **_kwargs: True)
    result = generate_spread_rate(allow_warming=True)
    assert result["status"] == "warming"


def test_generate_spread_rate_waits_for_uncached_rtma(tmp_path, monkeypatch):
    bundle_dir = tmp_path / "static"
    bundle_path = build_synthetic_bundle(bundle_dir, version="test-v1")
    monkeypatch.setenv("FIRE_BEHAVIOR_STATIC_BUNDLE", str(bundle_path))
    monkeypatch.setattr(spread_rate, "BETA_ROOT", tmp_path / "testbed")
    monkeypatch.setattr(spread_rate, "STATUS_PATH", tmp_path / "testbed" / "spread_rate" / "status.json")
    end = datetime(2026, 8, 30, 14, tzinfo=timezone.utc).replace(tzinfo=None)
    monkeypatch.setattr("services.spread_rate.is_analysis_hour_cached", lambda *_args, **_kwargs: False)
    result = generate_spread_rate(analysis_hour=end, allow_warming=False)
    assert result["status"] == "waiting_for_rtma"


def test_run_spread_rate_pipeline_fetches_before_generate(tmp_path, monkeypatch):
    calls = []

    def fake_ensure():
        calls.append("ensure")
        return {"analysis_hour": "2026-08-30T14:00:00Z", "cached": True, "fetched": False}

    def fake_generate(*_args, **_kwargs):
        calls.append("generate")
        return {"status": "ready"}

    monkeypatch.setattr("services.spread_rate.ensure_latest_analysis_cached", fake_ensure)
    monkeypatch.setattr("services.spread_rate.generate_spread_rate", fake_generate)
    result = run_spread_rate_pipeline()
    assert calls == ["ensure", "generate"]
    assert result["status"] == "ready"
