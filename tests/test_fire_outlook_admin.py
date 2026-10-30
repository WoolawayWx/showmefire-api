"""The staff-only pilot: public routes closed by default, the admin copy behind an admin session, and what the admin status shows."""
import json
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.security import create_access_token
from routers import fire_outlook, fire_outlook_admin
from tests.test_fire_outlook_router import _write, _center, CYCLE, RID

TOKEN = None


def token():
    return create_access_token({"sub": "admin@example.com"})


def client(tmp_path, monkeypatch, public=False):
    monkeypatch.setenv("SMF_FIRE_OUTLOOK_DIR", str(tmp_path))
    if public:
        monkeypatch.setenv("SMF_FIRE_OUTLOOK_PUBLIC", "true")
    else:
        monkeypatch.delenv("SMF_FIRE_OUTLOOK_PUBLIC", raising=False)
    for limiter in (fire_outlook.status_limiter, fire_outlook.runs_limiter, fire_outlook.manifest_limiter, fire_outlook.point_limiter,
                    fire_outlook.file_limiter, fire_outlook_admin.admin_limiter):
        limiter.reset()
    app = FastAPI()
    app.include_router(fire_outlook.router)
    app.include_router(fire_outlook_admin.router)
    return TestClient(app)


PUBLIC_PATHS = ["/api/fire-outlook/status", "/api/fire-outlook/runs", f"/api/fire-outlook/runs/{RID}/manifest",
                f"/api/fire-outlook/runs/{RID}/point?lat=38&lon=-92", f"/api/fire-outlook/runs/{RID}/files/contours/elevated/w00.geojson"]


def test_public_routes_are_closed_until_the_switch_is_on(tmp_path, monkeypatch):
    _write(tmp_path)
    c = client(tmp_path, monkeypatch)
    for path in PUBLIC_PATHS:
        r = c.get(path)
        assert r.status_code == 404 and "not public yet" in r.json()["detail"], path
    on = client(tmp_path, monkeypatch, public=True)
    assert on.get("/api/fire-outlook/runs").status_code == 200
    assert on.get(f"/api/fire-outlook/runs/{RID}/files/contours/elevated/w00.geojson").status_code == 200


@pytest.mark.parametrize("value,expected", [("true", True), ("1", True), ("yes", True), ("false", False), ("", False), ("no", False)])
def test_public_switch_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("SMF_FIRE_OUTLOOK_PUBLIC", value)
    assert fire_outlook.public_enabled() is expected
    monkeypatch.delenv("SMF_FIRE_OUTLOOK_PUBLIC")
    assert fire_outlook.public_enabled() is False


ADMIN_PATHS = ["/api/admin/fire-outlook/status", "/api/admin/fire-outlook/runs", f"/api/admin/fire-outlook/runs/{RID}/manifest",
               f"/api/admin/fire-outlook/runs/{RID}/point?lat=38&lon=-92", f"/api/admin/fire-outlook/runs/{RID}/files/contours/elevated/w00.geojson"]


def test_admin_routes_need_an_admin_session(tmp_path, monkeypatch):
    _write(tmp_path)
    c = client(tmp_path, monkeypatch)
    for path in ADMIN_PATHS:
        assert c.get(path).status_code == 401, path
        assert c.get(path, params={"token": "not-a-real-token"}).status_code == 401, path


def test_admin_sees_the_outlook_while_the_public_routes_stay_closed(tmp_path, monkeypatch):
    run, counts = _write(tmp_path)
    c = client(tmp_path, monkeypatch)                                       # public switch OFF
    t = {"token": token()}
    runs = c.get("/api/admin/fire-outlook/runs", params=t).json()
    assert runs["latest"] == RID and runs["runs"][0]["member_count"] == 31
    manifest = c.get(f"/api/admin/fire-outlook/runs/latest/manifest", params=t).json()
    assert manifest["run_id"] == RID and len(manifest["windows"]) == 28 and manifest["serving"]["stale_after_hours"] == 24
    p = c.get(f"/api/admin/fire-outlook/runs/{RID}/point", params={**t, **dict(zip(("lat", "lon"), _center(5, 6).values()))}).json()
    assert p["in_domain"] and p["windows"][3]["elevated_member_count"] == int(counts[3, 5, 6])
    f = c.get(f"/api/admin/fire-outlook/runs/{RID}/files/contours/elevated/w00.geojson", params=t)
    assert f.status_code == 200 and f.headers["cache-control"].startswith("private")        # never in a shared cache
    assert c.get(f"/api/admin/fire-outlook/runs/{RID}/files/weather.npz", params=t).status_code == 404   # same whitelist
    assert c.get("/api/fire-outlook/runs").status_code == 404                                # and the public copy is still closed
    assert c.get(f"/api/admin/fire-outlook/runs/{RID}/point", params={**t, "lat": 95, "lon": 0}).status_code == 422


def test_admin_status_shows_what_the_worker_is_doing_including_errors(tmp_path, monkeypatch):
    _write(tmp_path)
    (tmp_path / "status.json").write_text(json.dumps({
        "state": "failed", "stage": "publishing", "cycle": "2026-10-29T00:00:00Z", "runtime_s": 612.0, "peak_rss_mb": 2432.5,
        "errors": [{"stage": "publishing", "type": "ConnectionError", "message": "R2 unreachable at /app/data/x", "at": "2026-10-29T05:40:00Z"}],
        "source_coverage": {"members_expected": 31, "members_ready": 31},
        "verification_export": {"stations": 61, "file": "20261029T00Z.parquet", "nbm": False, "nbm_error": "NBM baseline unavailable"}}))
    (tmp_path / "published.json").write_text(json.dumps({"run_ids": [RID]}))
    (tmp_path / "verification" / "forecast").mkdir(parents=True)
    (tmp_path / "verification" / "forecast" / f"{RID}.csv").write_text("a\n1\n")
    (tmp_path / "verification" / "records").mkdir()
    (tmp_path / "verification" / "records" / f"{RID}.csv").write_text("a\n1\n")
    for k, v in (("SMF_FIRE_OUTLOOK_ENABLED", "true"), ("SMF_FIRE_OUTLOOK_PUBLISH", "false")):
        monkeypatch.setenv(k, v)
    c = client(tmp_path, monkeypatch)
    s = c.get("/api/admin/fire-outlook/status", params={"token": token()}).json()
    assert s["mode"] == "admin_preview"
    assert s["settings"] == {"worker_enabled": True, "public": False, "publish_to_cdn": False, "threads": 1, "memory_limit_gb": 8.0,
                             "public_keep_days": 7, "cache_keep_days": 30}
    assert s["worker"]["errors"][0]["message"].startswith("R2 unreachable")            # the full text staff need
    assert s["worker"]["verification_export"]["stations"] == 61 and s["worker"]["verification_export"]["nbm"] is False
    latest = s["latest_run"]
    assert latest["run_id"] == RID and latest["member_count"] == 31 and latest["windows"] == 28 and latest["files"] > 50
    assert latest["conditioning"]["level"] == "conditioned" and "layers" in latest and latest["bytes"] > 0
    assert s["runs_on_disk"] == [RID] and s["uploaded_to_cdn"] == [RID]
    assert s["verification"]["records"]["files"] == 1 and s["verification"]["records"]["newest"] == f"{RID}.csv"
    assert s["verification"]["forecast_side"]["files"] == 1 and s["storage"]["runs"]["bytes"] > 0
    assert s["schedule"]["latest_cycle_utc"].endswith("Z") and "every 30 minutes" in s["schedule"]["checks"]
    # the public status hides that error text
    pub = client(tmp_path, monkeypatch, public=True).get("/api/fire-outlook/status").json()
    assert "errors" not in pub and "R2 unreachable" not in json.dumps(pub)


def test_admin_status_on_a_fresh_install(tmp_path, monkeypatch):
    monkeypatch.delenv("SMF_FIRE_OUTLOOK_ENABLED", raising=False)
    c = client(tmp_path / "empty", monkeypatch)
    s = c.get("/api/admin/fire-outlook/status", params={"token": token()}).json()
    assert s["worker"] == {"state": "never_run"} and s["latest_run"] is None and s["runs_on_disk"] == []
    assert s["settings"]["worker_enabled"] is False and s["verification"]["records"] == {"files": 0, "bytes": 0, "newest": None}
    assert c.get("/api/admin/fire-outlook/runs", params={"token": token()}).json() == {"dataset": "fire-outlook", "latest": None, "runs": []}


def test_the_admin_status_schedule_matches_the_worker_cycle_rule():
    from services.fire_outlook import worker
    now = datetime(2026, 10, 29, 17, 20, tzinfo=timezone.utc)
    s = fire_outlook_admin.build_admin_status(__import__("pathlib").Path("/nonexistent-fire-outlook"), now)
    assert s["schedule"]["latest_cycle_utc"] == "2026-10-29T12:00:00Z" and s["schedule"]["inputs_expected_utc"] == "2026-10-29T17:00:00Z"
    assert worker.latest_cycle(now).hour == 12


def test_main_registers_both_routers_and_admin_is_not_aliased():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / "main.py").read_text(encoding="utf-8")
    assert "app.include_router(fire_outlook_admin.router)" in source
    from core import api_aliases
    assert api_aliases._excluded("/api/admin/fire-outlook/status")          # session middleware matches /api/admin/*: never an alias
