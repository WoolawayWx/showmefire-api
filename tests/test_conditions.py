import numpy as np
import pytest
import rasterio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from rasterio.transform import from_origin

import routers.conditions as conditions


@pytest.fixture
def client(tmp_path, monkeypatch):
    latest = tmp_path / "latest"
    latest.mkdir()
    # 10x10 grid in EPSG:4326 covering Missouri-ish; value = column index.
    transform = from_origin(-96.0, 41.0, 0.7, 0.7)
    for product in conditions.PRODUCTS:
        data = np.tile(np.arange(10, dtype="float32"), (10, 1))
        data[0, 0] = -9999.0
        with rasterio.open(
            latest / f"realtime_{'rh' if product.key == 'rh' else product.key}.tif", "w",
            driver="GTiff", height=10, width=10, count=1, dtype="float32",
            crs="EPSG:4326", transform=transform, nodata=-9999.0,
        ) as dst:
            dst.write(data, 1)
            dst.update_tags(OBSERVATION_TIME="2026-10-09T12:00:00Z")
    monkeypatch.setattr(conditions, "GIS_DIR", tmp_path)
    conditions._cache.clear()
    app = FastAPI()
    app.include_router(conditions.router)
    return TestClient(app)


def test_point_returns_every_product(client):
    body = client.get("/api/conditions/at", params={"lat": 38.6, "lon": -92.5}).json()
    assert set(body["values"]) == {p.key for p in conditions.PRODUCTS}
    assert body["observed_at"] == "2026-10-09T12:00:00Z"
    # lon -92.5 -> column floor((-92.5 + 96) / 0.7) = 5
    assert body["values"]["temperature"] == 5


def test_nodata_cell_is_null(client):
    body = client.get("/api/conditions/at", params={"lat": 40.65, "lon": -95.75}).json()
    assert body["values"]["temperature"] is None


def test_outside_missouri_is_rejected(client):
    assert client.get("/api/conditions/at", params={"lat": 45.0, "lon": -92.5}).status_code == 422


def test_layers_report_style_and_freshness(client):
    layers = {layer["id"]: layer for layer in client.get("/api/conditions/layers").json()["layers"]}
    assert layers["rh"]["colormap"] == "rdylbu" and layers["rh"]["available"] is True
    assert layers["fire_danger"]["categories"][0] == "Low"


STATION_FILLS = ("station_precip_1h", "station_precip_24h", "station_et_current",
                 "station_et_24h", "station_et_max", "station_et_min")


def test_station_precip_and_et_fills_are_listed_with_precise_legends(client):
    layers = {layer["id"]: layer for layer in client.get("/api/conditions/layers").json()["layers"]}
    for key in STATION_FILLS:
        assert layers[key]["group"] == "station_fill" and layers[key]["colormap"] == "rgba"
        assert layers[key]["available"] is False  # no GeoTIFF in this fixture yet
    # ET rates are hundredths of an inch per hour: a one-decimal legend would read 0.0 to 0.0.
    assert layers["station_et_current"]["legend"][0]["value"] == 0
    assert layers["station_et_current"]["legend"][-1]["value"] == 0.04
    assert layers["station_precip_1h"]["legend"][-1]["value"] == 1.0
    assert layers["station_precip_24h"]["units"] == "in"
    assert layers["station_et_current"]["units"] == "in/hr"


def test_station_fill_reports_freshness_from_the_geotiff(client, tmp_path):
    (tmp_path / "realtime").mkdir(exist_ok=True)
    with rasterio.open(tmp_path / "realtime/station_et_24h.tif", "w", driver="GTiff", height=2, width=2, count=4,
                       dtype="uint8", crs="EPSG:4326", transform=from_origin(-96.0, 41.0, 1.0, 1.0)) as dst:
        dst.write(np.full((4, 2, 2), 200, dtype="uint8"))
        dst.update_tags(CREATED="2026-10-14T21:00:00+00:00")
    layers = {layer["id"]: layer for layer in client.get("/api/conditions/layers").json()["layers"]}
    assert layers["station_et_24h"]["available"] is True
    assert layers["station_et_24h"]["observed_at"] == "2026-10-14T21:00:00+00:00"
    assert layers["station_et_max"]["available"] is False
