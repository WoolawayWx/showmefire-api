import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

import routers.verification as v
from routers import api_index


def _client():
    app = FastAPI()
    app.include_router(api_index.router)
    app.include_router(v.router)
    return TestClient(app)


def setup_function():
    for limiter in (v.verification_limiter, v.station_limiter, api_index.index_limiter):
        limiter.reset()


def test_routes_json_lists_public_routes_with_limits():
    body = _client().get("/routes").json()
    paths = {r["path"] for r in body["routes"]}
    assert "/verification/rtma/{date}" in paths
    assert not any(p.startswith("/api/admin") for p in paths)
    assert all(r["rate_limit_per_minute"] >= 1 for r in body["routes"])


def test_routes_page_is_html():
    response = _client().get("/routes/page")
    assert response.status_code == 200
    assert "/verification/rtma/history" in response.text


def test_station_route_marked_key_only_when_gate_on():
    with patch.dict(os.environ, {"VERIFICATION_STATION_DATA_REQUIRES_KEY": "true"}):
        routes = {r["path"]: r for r in _client().get("/routes").json()["routes"]}
    assert routes["/verification/report/{date}/comparisons"]["access"] == "api key or admin"
    assert routes["/verification/history"]["access"] == "public"


def _write_report(tmp, date="2026-09-04"):
    entry = {
        "date": date, "generated_at": "2026-09-05T04:30:00Z", "record_count": 5,
        "metrics": {"Temperature (C)": {"mae": 1.0, "bias": 0.5, "count": 5}},
        "rtma_metrics": {
            "available": True,
            "station_point": {
                "record_count": 5, "stations_count": 2,
                "forecast_vs_rtma": {"Temperature (C)": {"mae": 2.0, "bias": 1.5, "count": 5}},
                "station_vs_rtma": {"Temperature (C)": {"mae": 1.0, "bias": 0.0, "count": 5}},
            },
            "gridded": None,
        },
    }
    day = Path(tmp) / date
    day.mkdir(parents=True)
    (day / "validation_summary.json").write_text(json.dumps(entry))
    return entry


def test_rtma_routes_read_stored_metrics():
    with tempfile.TemporaryDirectory() as tmp:
        _write_report(tmp)
        with patch.object(v, "REPORTS_DIR", tmp), patch.object(v, "HISTORY_FILE", Path(tmp) / "none.json"), \
                patch.object(v, "_report_covers_closed_window", return_value=True):
            client = _client()
            history = client.get("/verification/rtma/history").json()["days"]
            report = client.get("/verification/rtma/2026-09-04")
            missing = client.get("/verification/rtma/2026-01-01")
            plain = client.get("/verification/history").json()["dates"][0]
    assert history[0]["forecast_vs_rtma"]["Temperature (C)"]["bias"] == 1.5
    assert report.status_code == 200
    assert report.json()["rtma_metrics"]["station_point"]["stations_count"] == 2
    assert missing.status_code == 404
    assert plain["rtma_temp_bias"] == 1.5


def test_verification_routes_are_rate_limited():
    with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"RATE_LIMIT_VERIFICATION_PER_MIN": "2"}), \
            patch.object(v, "REPORTS_DIR", tmp), patch.object(v, "HISTORY_FILE", Path(tmp) / "none.json"):
        client = _client()
        codes = [client.get("/verification/history").status_code for _ in range(3)]
    assert codes == [200, 200, 429]
