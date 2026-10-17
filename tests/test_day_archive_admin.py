import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.security import require_admin
from routers import day_archive_admin as router_module
from services import day_archive as da
from tests.test_day_archive import DAY, NOW, env  # noqa: F401 - fixture


def _client(s3, admin=True):
    app = FastAPI()
    app.include_router(router_module.router)
    if admin:
        app.dependency_overrides[require_admin] = lambda: "admin@example.org"
    return TestClient(app)


def test_routes_have_no_api_prefix_and_require_admin(env):
    root, s3 = env
    assert router_module.router.prefix == "/admin/day-archive"
    client = _client(s3, admin=False)
    for method, path in [("get", "/admin/day-archive"), ("get", f"/admin/day-archive/{DAY}"),
                         ("get", f"/admin/day-archive/{DAY}/links"),
                         ("post", f"/admin/day-archive/{DAY}/mark-downloaded"), ("post", "/admin/day-archive/run")]:
        assert getattr(client, method)(path).status_code == 401, path


def test_list_manifest_links_and_mark_downloaded_read_from_r2(env):
    root, s3 = env
    da.build_day(s3, DAY, now=NOW)
    da.save_ledger(s3, {DAY: {"created": NOW.isoformat(), "expires": "2099-01-01T00:00:00+00:00",
                              "complete": True, "bytes": 10, "pending": []}})
    with patch.object(router_module, "_s3", return_value=s3):
        client = _client(s3)
        listing = client.get("/admin/day-archive").json()
        assert listing["days"][0]["day"] == DAY and listing["days"][0]["downloaded"] is False
        assert client.get(f"/admin/day-archive/{DAY}").json()["day"] == DAY
        links = client.get(f"/admin/day-archive/{DAY}/links?expires_in=600").json()["files"]
        assert all(link["url"].startswith("https://r2.example/day-archive/") for link in links)
        assert client.get("/admin/day-archive/2000-01-01").status_code == 404
        assert client.get("/admin/day-archive/not-a-date").status_code == 422
        marked = client.post(f"/admin/day-archive/{DAY}/mark-downloaded", json={"files": ["forecasts.zip"]})
        assert marked.status_code == 200
    receipt = s3.json(f"day-archive/receipts/{DAY}.json")
    assert receipt["downloaded_by"] == "admin@example.org" and receipt["verified"] is True


def test_mark_downloaded_unknown_day_is_404(env):
    root, s3 = env
    with patch.object(router_module, "_s3", return_value=s3):
        assert _client(s3).post("/admin/day-archive/2000-01-01/mark-downloaded").status_code == 404
