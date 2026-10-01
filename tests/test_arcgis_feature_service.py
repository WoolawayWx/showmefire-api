from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from shapely.geometry import box

from routers import arcgis_feature_service as svc

BASE = "/arcgis/rest/services/BurnBans/FeatureServer"
LIVE = {"bans": ["29019"]}      # what the "database" currently says


def _county(fips, name, x0, y0):
    return fips, name, box(x0, y0, x0 + 1, y0 + 1)


COUNTIES = [_county("29019", "Boone", -93, 38), _county("29051", "Cole", -92, 38), _county("29053", "Cooper", -93, 39)]


def _fake_burn_bans():
    out = []
    for fips, name, geom in COUNTIES:
        active = fips in LIVE["bans"]
        out.append({
            "OBJECTID": int(fips), "county_fips": fips, "county_name": name,
            "status": "active" if active else "inactive", "has_burn_ban": 1 if active else 0,
            "effective_at": svc._epoch_ms("2026-10-01T05:00:00Z") if active else None,
            "expires_at": None, "published_at": None, "updated_at": None, "proof_url": None, "_geometry": geom,
        })
    return out


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setattr(svc.LAYERS["BurnBans"], "load", _fake_burn_bans)
    monkeypatch.setattr(svc.LAYERS["BurnBans"], "ttl", 0.0)
    LIVE["bans"] = ["29019"]
    app = FastAPI()
    app.include_router(svc.router)
    return TestClient(app)


def q(client, **params):
    response = client.get(f"{BASE}/0/query", params={"f": "json", **params})
    assert response.status_code == 200
    return response.json()


def test_server_info_and_service_listing(client):
    assert client.get("/arcgis/rest/info").json()["currentVersion"] == svc.VERSION
    assert client.get("/arcgis/rest/services").json()["services"][0]["type"] == "FeatureServer"


def test_service_and_layer_metadata(client):
    service = client.get(BASE, params={"f": "json"}).json()
    assert service["capabilities"] == "Query" and service["hasStaticData"] is False
    assert service["layers"][0]["geometryType"] == "esriGeometryPolygon"
    layer = client.get(f"{BASE}/0", params={"f": "json"}).json()
    assert layer["type"] == "Feature Layer" and layer["objectIdField"] == "OBJECTID"
    assert layer["supportsStatistics"] and layer["supportsPagination"]
    assert {f["name"] for f in layer["fields"]} >= {"OBJECTID", "county_name", "status", "expires_at"}
    assert layer["drawingInfo"]["renderer"]["field1"] == "status"
    assert client.get(f"{BASE}/layers").json()["layers"][0]["id"] == 0


def test_unknown_service_or_layer_is_an_esri_style_error(client):
    body = client.get("/arcgis/rest/services/Nope/FeatureServer").json()
    assert body["error"]["code"] == 404
    assert client.get(f"{BASE}/7").json()["error"]["code"] == 404


def test_query_all_returns_esri_polygons(client):
    result = q(client, where="1=1", outFields="*")
    assert len(result["features"]) == 3 and result["exceededTransferLimit"] is False
    feature = result["features"][0]
    ring = feature["geometry"]["rings"][0]
    assert ring[0] == ring[-1] and len(ring) == 5
    # Esri wants exterior rings clockwise: signed area negative
    area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(ring, ring[1:]))
    assert area < 0
    assert result["spatialReference"]["wkid"] == 4326


def test_data_is_live_not_static(client):
    assert q(client, where="status='active'", returnCountOnly="true")["count"] == 1
    LIVE["bans"] = ["29019", "29051"]          # a ban is confirmed
    assert q(client, where="status='active'", returnCountOnly="true")["count"] == 2
    LIVE["bans"] = []                          # all lifted
    assert q(client, where="status='active'", returnCountOnly="true")["count"] == 0


def test_where_fields_ids_order_and_paging(client):
    names = q(client, where="county_name LIKE 'Co%'", outFields="county_name", returnGeometry="false",
              orderByFields="county_name DESC")["features"]
    assert [f["attributes"]["county_name"] for f in names] == ["Cooper", "Cole"]
    assert "geometry" not in names[0]
    assert q(client, objectIds="29051", returnIdsOnly="true")["objectIds"] == [29051]
    page = q(client, where="1=1", resultRecordCount="2", resultOffset="0", outFields="county_name")
    assert len(page["features"]) == 2 and page["exceededTransferLimit"] is True
    assert len(q(client, where="1=1", resultRecordCount="2", resultOffset="2")["features"]) == 1


def test_statistics_for_dashboard_indicators(client):
    stats = [{"statisticType": "count", "onStatisticField": "OBJECTID", "outStatisticFieldName": "n"}]
    total = q(client, outStatistics=json.dumps(stats), where="status='active'")
    assert total["features"][0]["attributes"] == {"n": 1}
    grouped = q(client, outStatistics=json.dumps(stats), groupByFieldsForStatistics="status")
    assert {f["attributes"]["status"]: f["attributes"]["n"] for f in grouped["features"]} == {"active": 1, "inactive": 2}
    assert [f["name"] for f in grouped["fields"]] == ["status", "n"]


def test_distinct_values_for_selectors(client):
    values = q(client, returnDistinctValues="true", outFields="status", returnGeometry="false")
    assert sorted(f["attributes"]["status"] for f in values["features"]) == ["active", "inactive"]


def test_extent_filter_in_web_mercator_and_wgs84(client):
    # Boone's box in Web Mercator (EPSG:102100) and in lon/lat
    from routers.arcgis_feature_service import _reproject
    wm = _reproject(box(-92.9, 38.1, -92.8, 38.2), 4326, 3857).bounds
    envelope = json.dumps({"xmin": wm[0], "ymin": wm[1], "xmax": wm[2], "ymax": wm[3], "spatialReference": {"wkid": 102100}})
    hit = q(client, geometry=envelope, geometryType="esriGeometryEnvelope", spatialRel="esriSpatialRelIntersects",
            inSR="102100", outFields="county_name", returnGeometry="false")
    assert [f["attributes"]["county_name"] for f in hit["features"]] == ["Boone"]
    hit = q(client, geometry="-93.5,38.5,-92.5,39.5", inSR="4326", outFields="county_name", returnGeometry="false")
    assert {f["attributes"]["county_name"] for f in hit["features"]} == {"Boone", "Cooper"}   # Cole lies east of -92.5


def test_out_sr_web_mercator_coordinates(client):
    result = q(client, where="county_name='Boone'", outSR="102100")
    x, y = result["features"][0]["geometry"]["rings"][0][0]
    assert abs(x) > 1e6 and abs(y) > 1e6                       # metres, not degrees
    assert result["spatialReference"] == {"wkid": 102100, "latestWkid": 3857}


def test_geojson_format(client):
    body = client.get(f"{BASE}/0/query", params={"f": "geojson", "where": "status='active'"}).json()
    assert body["type"] == "FeatureCollection" and len(body["features"]) == 1
    assert body["features"][0]["geometry"]["type"] == "Polygon"
    assert body["features"][0]["properties"]["county_name"] == "Boone"


def test_date_literals_and_post_form(client):
    assert q(client, where="effective_at >= timestamp '2026-09-30 00:00:00'", returnCountOnly="true")["count"] == 1
    assert q(client, where="effective_at >= timestamp '2026-10-02 00:00:00'", returnCountOnly="true")["count"] == 0
    response = client.post(f"{BASE}/0/query", data={"f": "json", "where": "status='inactive'", "returnCountOnly": "true"})
    assert response.json()["count"] == 2


@pytest.mark.parametrize("evil", [
    "1=1; DROP TABLE f",
    "1=1) UNION SELECT name FROM sqlite_master --",
    "load_extension('x')=1",
    "(SELECT count(*) FROM sqlite_master) > 0",
    "randomblob(100000000) IS NOT NULL",
    "x" * 3000,
    "county_name = ",
])
def test_hostile_where_clauses_fail_safely(client, evil):
    body = client.get(f"{BASE}/0/query", params={"f": "json", "where": evil}).json()
    assert "error" in body and "features" not in body


def test_endless_query_is_aborted(client):
    body = client.get(f"{BASE}/0/query", params={
        "f": "json",
        "where": "(WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT count(*) FROM c) > 0"}).json()
    assert "error" in body


def test_invalid_parameters_are_errors_not_crashes(client):
    assert "error" in q(client, outFields="nope")
    assert "error" in q(client, orderByFields="nope; DROP")
    assert "error" in q(client, outStatistics="not json")
    assert "error" in q(client, geometry="{bad")


def test_cors_and_no_cache_headers(client):
    response = client.get(f"{BASE}/0/query", params={"f": "json", "where": "1=1"}, headers={"Origin": "https://x.maps.arcgis.com"})
    assert response.headers["access-control-allow-origin"] == "*"
    assert response.headers["cache-control"] == "no-store"


def test_smooth_polygon_service_reads_the_published_file(tmp_path, monkeypatch):
    payload = {"type": "FeatureCollection", "features": [
        {"type": "Feature", "properties": {"danger_level": lvl, "label": lbl, "color": "#000000",
                                           "model_run": "2026-10-01T12:00:00Z", "resolution_m": 500, "area_km2": 1.0},
         "geometry": box(-93 + lvl, 38, -92 + lvl, 39).__geo_interface__}
        for lvl, lbl in [(0, "Low"), (1, "Moderate")]]}
    (tmp_path / svc._SMOOTH_FILE).write_text(json.dumps(payload))
    monkeypatch.setattr("core.config.GIS_DIR", tmp_path)
    monkeypatch.setattr(svc, "_smooth_cache", (-1.0, []))
    app = FastAPI(); app.include_router(svc.router)
    c = TestClient(app)
    url = "/arcgis/rest/services/PeakFireDangerSmooth/FeatureServer/0"
    assert c.get(url, params={"f": "json"}).json()["drawingInfo"]["renderer"]["field1"] == "danger_level"
    result = c.get(f"{url}/query", params={"f": "json", "where": "danger_level >= 1", "outFields": "label"}).json()
    assert [f["attributes"]["label"] for f in result["features"]] == ["Moderate"]
    # replacing the file (next forecast run) is picked up with no restart
    payload["features"].pop()
    import os, time
    path = tmp_path / svc._SMOOTH_FILE
    path.write_text(json.dumps(payload)); os.utime(path, (time.time() + 5, time.time() + 5))
    assert c.get(f"{url}/query", params={"f": "json", "returnCountOnly": "true"}).json()["count"] == 1


def test_missing_smooth_file_gives_an_empty_valid_layer(tmp_path, monkeypatch):
    monkeypatch.setattr("core.config.GIS_DIR", tmp_path)
    monkeypatch.setattr(svc, "_smooth_cache", (-1.0, []))
    app = FastAPI(); app.include_router(svc.router)
    c = TestClient(app)
    url = "/arcgis/rest/services/PeakFireDangerSmooth/FeatureServer"
    assert c.get(f"{url}/0", params={"f": "json"}).json()["type"] == "Feature Layer"
    assert c.get(f"{url}/0/query", params={"f": "json", "where": "1=1"}).json()["features"] == []
