from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import geopandas as gpd
import numpy as np
import pytest
import rasterio

from services import gis_publisher
from forecast.export_fire_danger_gis import export_geotiff


def _source_grid():
    lon, lat = np.meshgrid(np.linspace(-96.0, -89.0, 12), np.linspace(35.5, 41.0, 10))
    values = np.arange(lon.size, dtype=float).reshape(lon.shape)
    return values, lon, lat


def test_canonical_grid_is_epsg_32615_and_three_kilometers():
    grid = gis_publisher.canonical_grid()
    assert grid["crs"] == "EPSG:32615"
    assert grid["resolution"] == 3000
    assert grid["width"] > 100
    assert grid["height"] > 100


def test_categorical_regrid_does_not_invent_classes():
    values, lon, lat = _source_grid()
    values = (values % 5).astype(float)
    projected = gis_publisher.regrid_lonlat(values, lon, lat, categorical=True)
    assert set(np.unique(projected[np.isfinite(projected)])) <= {0, 1, 2, 3, 4}


def test_accumulated_regrid_preserves_domain_mean():
    values, lon, lat = _source_grid()
    values = np.maximum(values, 0)
    projected = gis_publisher.regrid_lonlat(values, lon, lat, accumulated=True)
    assert np.nanmean(projected) == pytest.approx(np.nanmean(values), rel=0.001)


def test_publish_raster_writes_atomic_catalog_index_and_native_crs(tmp_path):
    values, lon, lat = _source_grid()
    valid = datetime(2026, 9, 3, 16, tzinfo=timezone.utc)
    path = gis_publisher.publish_raster(
        "forecast_temperature", values, lon, lat, units="degree_Fahrenheit",
        run_time=valid - timedelta(hours=4), valid_time=valid, root=tmp_path,
    )
    with rasterio.open(path) as dataset:
        assert dataset.crs.to_epsg() == 32615
        assert dataset.transform.a == 3000
        assert dataset.tags()["VALID_TIME"] == "2026-09-03T16:00:00Z"
    catalog = json.loads((tmp_path / "catalog.json").read_text())
    assert catalog["native_crs"] == "EPSG:32615"
    assert catalog["products"]["forecast_temperature"]["latest"] == "latest/forecast_temperature.tif"
    assert (tmp_path / "raster_catalog.gpkg").is_file()
    index = gpd.read_file(tmp_path / "raster_catalog.gpkg", layer="rasters")
    assert index.iloc[0]["product"] == "forecast_temperature"


def test_publish_vectors_writes_geojson_4326_and_geopackage_32615(tmp_path):
    features = [{
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [-92.3, 38.5]},
        "properties": {"stid": "TEST"},
    }]
    paths = gis_publisher.publish_vectors("weather_stations", features, root=tmp_path)
    payload = json.loads(paths["geojson"].read_text())
    assert payload["features"][0]["geometry"]["coordinates"] == [-92.3, 38.5]
    frame = gpd.read_file(paths["geopackage"], layer="weather_stations")
    assert frame.crs.to_epsg() == 32615


def test_publish_vectors_supports_an_empty_live_feed(tmp_path):
    paths = gis_publisher.publish_vectors("fire_detections", [], root=tmp_path)
    payload = json.loads(paths["geojson"].read_text())
    assert payload["features"] == []
    assert paths["geopackage"].is_file()


def test_cleanup_retention_keeps_latest(tmp_path):
    old = datetime.now(timezone.utc) - timedelta(days=40)
    values, lon, lat = _source_grid()
    path = gis_publisher.publish_raster(
        "realtime_rh", values, lon, lat, units="percent", observation_time=old, root=tmp_path,
    )
    removed = gis_publisher.cleanup_retention(root=tmp_path)
    # The only item is the active latest item and must not be removed even if old.
    assert removed == []
    assert path.exists()


def test_staged_batch_is_invisible_until_commit(tmp_path):
    values, lon, lat = _source_grid()
    staging = gis_publisher.create_staging_root(root=tmp_path)
    gis_publisher.publish_raster(
        "forecast_rh", values, lon, lat, units="percent",
        valid_time=datetime(2026, 9, 3, 16, tzinfo=timezone.utc),
        root=staging, rebuild_index=False,
    )
    assert not (tmp_path / "catalog.json").exists()
    assert not (tmp_path / "latest" / "forecast_rh.tif").exists()
    gis_publisher.commit_staging_root(staging, root=tmp_path)
    assert (tmp_path / "catalog.json").is_file()
    assert (tmp_path / "latest" / "forecast_rh.tif").is_file()


def test_legacy_peak_export_uses_canonical_projected_grid(tmp_path):
    values, lon, lat = _source_grid()
    values = values % 5
    output = tmp_path / "peak_fire_danger.tif"
    assert export_geotiff(values, lon, lat, output)
    with rasterio.open(output) as dataset:
        assert dataset.crs.to_epsg() == 32615
        assert dataset.transform.a == 3000
