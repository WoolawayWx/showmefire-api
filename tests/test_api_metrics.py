import json

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from core import metrics as metrics_module
from core.metrics import (LATENCY_EDGES_MS, MAX_ROUTE_KEYS, MetricsMiddleware, ApiMetrics, percentile, route_label,
                          _new_row, _add)
from core.security import require_admin


class Clock:
    def __init__(self, now=1_800_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


def test_counts_status_classes_latency_and_bytes(clock):
    collector = ApiMetrics(clock)
    for status, ms in ((200, 10), (200, 20), (304, 5), (404, 8), (500, 400)):
        collector.record("GET", "/things", status, ms, nbytes=100)
    overview = collector.snapshot(60)["overview"]
    assert overview["requests"] == 5 and overview["ok"] == 3
    assert overview["client_errors"] == 1 and overview["server_errors"] == 1
    assert overview["bytes"] == 500 and overview["max_ms"] == 400 and overview["avg_ms"] == pytest.approx(88.6)
    assert overview["server_error_rate"] == 0.2 and overview["client_error_rate"] == 0.2
    assert overview["requests_per_second"] == pytest.approx(5 / 3600, abs=1e-3)


def test_percentiles_come_from_the_histogram():
    row = _new_row()
    for _ in range(90):
        _add(row, 200, 8, 0)           # the 5-10 ms bucket
    for _ in range(10):
        _add(row, 200, 800, 0)         # the 500-1000 ms bucket
    assert 5 < percentile(row, 0.50) <= 10
    assert 500 < percentile(row, 0.95) <= 800 and percentile(row, 0.99) <= 800
    assert percentile(_new_row(), 0.5) is None
    slow = _new_row()
    _add(slow, 200, 30000, 0)                                                   # beyond the last edge
    assert LATENCY_EDGES_MS[-1] < percentile(slow, 0.99) <= 30000


def test_series_points_follow_the_window(clock):
    collector = ApiMetrics(clock)
    for minute in range(10):
        collector.record("GET", "/a", 200, 10)
        clock.now += 60
    short = collector.snapshot(15)["series"]
    assert short["step_minutes"] == 1 and sum(p["requests"] for p in short["points"]) == 10
    day = collector.snapshot(1440)["series"]
    assert day["step_minutes"] == 15 and sum(p["requests"] for p in day["points"]) == 10
    assert all(day["points"][i]["time"] < day["points"][i + 1]["time"] for i in range(len(day["points"]) - 1))
    assert collector.snapshot(5)["overview"]["requests"] <= 6                    # only the last five minutes


def test_routes_are_listed_by_template_busiest_first(clock):
    collector = ApiMetrics(clock)
    for _ in range(3):
        collector.record("GET", "/runs/{cycle}", 200, 10)
    collector.record("POST", "/login", 401, 40)
    routes = collector.snapshot(60)["routes"]
    assert [(r["method"], r["route"], r["requests"]) for r in routes] == [("GET", "/runs/{cycle}", 3), ("POST", "/login", 1)]
    assert routes[1]["client_errors"] == 1 and routes[0]["p95_ms"] is not None


def test_only_the_last_24_hours_are_kept(clock):
    collector = ApiMetrics(clock)
    collector.record("GET", "/old", 200, 10)
    clock.now += 25 * 3600
    collector.record("GET", "/new", 200, 10)                                    # recording prunes
    snapshot = collector.snapshot(1440)
    assert [r["route"] for r in snapshot["routes"]] == ["/new"] and snapshot["overview"]["requests"] == 1


def test_the_number_of_distinct_routes_is_bounded(clock):
    collector = ApiMetrics(clock)
    for index in range(MAX_ROUTE_KEYS + 25):
        collector.record("GET", f"/r{index}", 200, 1)
    routes = collector.snapshot(60, max_routes=10_000)["routes"]
    assert len(routes) == MAX_ROUTE_KEYS + 1 and any(r["route"] == "(other)" or r["method"] == "" for r in routes)
    assert sum(r["requests"] for r in routes) == MAX_ROUTE_KEYS + 25


def test_snapshot_survives_a_save_and_reload_and_drops_stale_rows(clock, tmp_path):
    path = tmp_path / "m.json"
    first = ApiMetrics(clock)
    first.record("GET", "/a", 200, 12, nbytes=7)
    first.save(path)
    second = ApiMetrics(clock)
    assert second.load(path) == 1
    assert second.snapshot(60)["overview"]["requests"] == 1 and second.snapshot(60)["overview"]["bytes"] == 7
    clock.now += 30 * 3600                                                       # the saved minute is now stale
    assert ApiMetrics(clock).load(path) == 0


def test_a_missing_or_damaged_snapshot_is_ignored(clock, tmp_path):
    collector = ApiMetrics(clock)
    assert collector.load(tmp_path / "nope.json") == 0
    (tmp_path / "bad.json").write_text("{not json")
    assert collector.load(tmp_path / "bad.json") == 0
    (tmp_path / "odd.json").write_text(json.dumps({"minutes": {"x": [1], "5": "no", "9": [1, 2]}, "routes": []}))
    assert collector.load(tmp_path / "odd.json") == 0


# --- the middleware, against a real app ------------------------------------------------------------------------------

def make_app(collector):
    app = FastAPI()

    @app.get("/items/{item_id}")
    def item(item_id: int):
        return {"id": item_id, "pad": "x" * 100}

    @app.get("/boom")
    def boom():
        raise RuntimeError("kaboom")

    @app.get("/teapot")
    def teapot():
        raise HTTPException(status_code=418, detail="short and stout")

    @app.get("/stream")
    def stream():
        return StreamingResponse(iter([b"a" * 10, b"b" * 20, b"c" * 30]), media_type="text/plain")

    @app.get("/admin/metrics/api")
    def own():
        return {}

    app.add_middleware(MetricsMiddleware, collector=collector)
    return app


def test_middleware_records_the_route_template_not_the_url(clock):
    collector = ApiMetrics(clock)
    client = TestClient(make_app(collector))
    for item_id in (1, 2, 3):
        assert client.get(f"/items/{item_id}").status_code == 200
    routes = collector.snapshot(60)["routes"]
    assert [(r["method"], r["route"], r["requests"]) for r in routes] == [("GET", "/items/{item_id}", 3)]
    assert routes[0]["bytes"] > 300


def test_middleware_counts_errors_unmatched_paths_and_streamed_bytes(clock):
    collector = ApiMetrics(clock)
    client = TestClient(make_app(collector), raise_server_exceptions=False)
    assert client.get("/boom").status_code == 500
    assert client.get("/teapot").status_code == 418
    assert client.get("/no/such/path").status_code == 404
    assert client.get("/stream").content == b"a" * 10 + b"b" * 20 + b"c" * 30
    by_route = {r["route"]: r for r in collector.snapshot(60)["routes"]}
    assert by_route["/boom"]["server_errors"] == 1
    assert by_route["/teapot"]["client_errors"] == 1
    assert by_route["(static/unmatched)"]["client_errors"] == 1
    assert by_route["/stream"]["bytes"] == 60
    assert collector.snapshot(60)["overview"]["inflight"] == 0


def test_middleware_does_not_measure_the_metrics_page_itself(clock):
    collector = ApiMetrics(clock)
    client = TestClient(make_app(collector))
    client.get("/admin/metrics/api")
    assert collector.snapshot(60)["overview"]["requests"] == 0


def test_route_label_for_a_scope_without_a_route():
    assert route_label({}) == "(static/unmatched)"


def test_a_failure_while_recording_never_breaks_the_request(clock, monkeypatch):
    collector = ApiMetrics(clock)
    monkeypatch.setattr(collector, "record", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk")))
    assert TestClient(make_app(collector)).get("/items/1").status_code == 200


# --- the admin endpoint ----------------------------------------------------------------------------------------------

def admin_client(monkeypatch, collector, admin=True):
    from routers import admin_metrics

    monkeypatch.setattr(admin_metrics, "metrics", collector)
    app = FastAPI()
    app.include_router(admin_metrics.router)
    if admin:
        app.dependency_overrides[require_admin] = lambda: "admin@example.org"
    return TestClient(app)


def test_admin_endpoint_returns_the_snapshot(monkeypatch, clock):
    collector = ApiMetrics(clock)
    collector.record("GET", "/a", 200, 15)
    body = admin_client(monkeypatch, collector).get("/admin/metrics/api?window=30").json()
    assert body["overview"]["requests"] == 1 and body["overview"]["window_minutes"] == 30
    assert body["routes"][0]["route"] == "/a" and body["series"]["step_minutes"] == 1


def test_admin_endpoint_validates_the_window_and_requires_admin(monkeypatch, clock):
    client = admin_client(monkeypatch, ApiMetrics(clock))
    assert client.get("/admin/metrics/api?window=1").status_code == 422
    assert client.get("/admin/metrics/api?window=5000").status_code == 422
    assert admin_client(monkeypatch, ApiMetrics(clock), admin=False).get("/admin/metrics/api").status_code == 401


def test_persistence_starts_reloads_and_saves_on_stop(clock, tmp_path):
    collector = ApiMetrics(clock)
    path = tmp_path / "saved.json"
    collector.record("GET", "/before", 200, 5)
    collector.save(path)
    restarted = ApiMetrics(clock)
    assert metrics_module.start_persistence(interval_seconds=3600, collector=restarted, path=path) == 1
    restarted.record("GET", "/after", 200, 5)
    metrics_module.stop_persistence(collector=restarted, path=path)
    again = ApiMetrics(clock)
    again.load(path)
    assert {r["route"] for r in again.snapshot(60)["routes"]} == {"/before", "/after"}
