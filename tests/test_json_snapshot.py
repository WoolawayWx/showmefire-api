"""The station feeds are encoded once per update and served as the same bytes FastAPI produced before."""
from datetime import datetime, timezone

import pytest
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from core import json_snapshot


def feed(stations, updated="2026-10-20T16:00:00+00:00", error=None):
    return {"stations": stations, "last_updated": updated, "error": error}


@pytest.fixture(autouse=True)
def fresh_cache(monkeypatch):
    monkeypatch.setattr(json_snapshot, "_snapshots", {})


def test_body_is_byte_identical_to_fastapi_encoding():
    # Datetimes (jsonable_encoder) and non-ASCII names (ensure_ascii=False) are where a hand-rolled json.dumps would differ.
    data = feed([{"stid": "KCOU", "name": "Columbia – Regional", "t": 61.5, "n": 3,
                  "at": datetime(2026, 10, 20, 16, tzinfo=timezone.utc), "gust": None}])
    body, _ = json_snapshot.snapshot("t", data)
    assert body == JSONResponse(jsonable_encoder(data)).body


def test_reused_until_the_feed_changes(monkeypatch):
    calls = []
    real = json_snapshot.jsonable_encoder
    monkeypatch.setattr(json_snapshot, "jsonable_encoder", lambda value: calls.append(1) or real(value))
    data = feed([{"stid": "A"}])
    first = json_snapshot.snapshot("t", data)
    assert json_snapshot.snapshot("t", data) == first and len(calls) == 1

    # A fetch replaces the list (fetch_synoptic_data, fetch_buffer_stations, ... all build a new one).
    data["stations"] = [{"stid": "B"}]
    second = json_snapshot.snapshot("t", data)
    assert len(calls) == 2 and second[0] != first[0] and second[1] != first[1]

    # A failed fetch keeps the old stations but sets `error`: that must show up too.
    data["error"] = "timeout"
    assert b"timeout" in json_snapshot.snapshot("t", data)[0] and len(calls) == 3


def test_feeds_are_cached_separately():
    a, b = feed([{"stid": "A"}]), feed([{"stid": "B"}])
    assert json_snapshot.snapshot("a", a)[0] != json_snapshot.snapshot("b", b)[0]
    assert b'"A"' in json_snapshot.snapshot("a", a)[0]


def test_route_serves_etag_and_304():
    from fastapi import FastAPI, Request

    app = FastAPI()
    data = feed([{"stid": "KCOU", "temp": 61.5}])

    @app.get("/stations")
    def stations(request: Request):
        return json_snapshot.snapshot_response("stations", data, request)

    client = TestClient(app)
    first = client.get("/stations")
    assert first.status_code == 200 and first.json() == data
    assert first.headers["content-type"] == "application/json"
    assert first.headers["cache-control"] == "private, max-age=60"
    etag = first.headers["etag"]

    assert client.get("/stations", headers={"If-None-Match": etag}).status_code == 304
    assert client.get("/stations", headers={"If-None-Match": f"W/{etag}"}).status_code == 304
    assert client.get("/stations", headers={"If-None-Match": '"other"'}).status_code == 200

    data["stations"] = [{"stid": "KCOU", "temp": 62.0}]
    changed = client.get("/stations", headers={"If-None-Match": etag})
    assert changed.status_code == 200 and changed.headers["etag"] != etag


def test_main_station_routes_use_the_snapshot(monkeypatch):
    import main
    from services import station_et, station_precip, synoptic, timeseries

    monkeypatch.setattr(synoptic, "station_data", feed([{"stid": "KCOU"}]))
    monkeypatch.setattr(timeseries, "timeseriesdata", feed([{"stid": "KJEF"}]))
    monkeypatch.setattr(synoptic, "buffer_station_data", feed([{"stid": "KMCI"}]))
    monkeypatch.setattr(station_precip, "precip_data", feed([{"stid": "KSTL"}]))
    monkeypatch.setattr(station_et, "et_data", feed([{"stid": "TS123"}]))
    client = TestClient(main.app)
    for path, stid in (("/stations", "KCOU"), ("/stations/timeseries", "KJEF"), ("/stations/buffer", "KMCI"),
                       ("/stations/precip", "KSTL"), ("/stations/et", "TS123")):
        response = client.get(path)
        assert response.status_code == 200 and response.json()["stations"][0]["stid"] == stid
        assert response.headers["etag"]
