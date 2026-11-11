"""The QR code tool: UTM tagging, SVG/PNG output, tracked redirects, scan stats and the admin endpoints."""
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.security import require_admin
from routers import qr as public_module
from routers import qr_admin as admin_module
from services.qr import store

PHONE = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148 Safari/604.1"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PUBLIC_WEB_URL", "https://example.org/")
    store.reset_for_tests()
    public_module.scan_limiter.reset()
    app = FastAPI()
    app.include_router(public_module.router)
    app.include_router(admin_module.router)
    app.dependency_overrides[require_admin] = lambda: "admin-user"
    return SimpleNamespace(client=TestClient(app, follow_redirects=False))


def make(env, **extra):
    body = {"name": "Flyer", "destination": "https://showmefire.org/tools/burn-check", **extra}
    created = env.client.post("/admin/qr-codes", json=body)
    assert created.status_code == 201, created.text
    return created.json()["code"]


def test_utm_is_added_but_never_overrides():
    url = "https://a.org/p?x=1&utm_source=keep"
    out = store.with_utm(url, {"utm_source": "flyer", "utm_medium": "print", "utm_campaign": ""})
    assert "utm_source=keep" in out and "utm_source=flyer" not in out
    assert "utm_medium=print" in out and "utm_campaign" not in out and "x=1" in out


def test_tracked_code_encodes_short_address_and_untracked_encodes_destination(env):
    tracked = make(env, utm_source="flyer", utm_medium="print")
    assert tracked["encoded_text"] == f"https://example.org/q/{tracked['slug']}"
    assert "utm_source=flyer" in tracked["target_url"]
    static = make(env, name="Static", tracked=False, utm_source="flyer")
    assert static["encoded_text"] == static["target_url"] and "utm_source=flyer" in static["encoded_text"]


def test_text_codes_are_never_tracked(env):
    code = make(env, name="Wifi note", kind="text", destination="hello there")
    assert code["tracked"] is False and code["encoded_text"] == "hello there"


def test_validation(env):
    for body in ({"destination": "ftp://x.org"}, {"destination": "not a url"}, {"fg": "red"}, {"error_level": "Z"}, {"border": 40}, {"slug": "Bad Slug!"}):
        response = env.client.post("/admin/qr-codes", json={"name": "x", "destination": "https://a.org", **body})
        assert response.status_code == 422, body
    make(env, slug="taken")
    assert env.client.post("/admin/qr-codes", json={"name": "y", "destination": "https://a.org", "slug": "taken"}).status_code == 422


def test_svg_and_png_downloads(env):
    code = make(env, fg="#112233", bg="#ffeeff")
    svg = env.client.get(f"/admin/qr-codes/{code['id']}/image?download=true")
    assert svg.status_code == 200 and svg.headers["content-type"].startswith("image/svg+xml")
    assert 'fill="#112233"' in svg.text and 'fill="#ffeeff"' in svg.text and "attachment" in svg.headers["content-disposition"]
    png = env.client.get(f"/admin/qr-codes/{code['id']}/image?format=png")
    assert png.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_preview_does_not_save(env):
    response = env.client.post("/admin/qr-codes/preview", json={"destination": "https://a.org", "utm_source": "s"})
    assert response.status_code == 200 and response.json()["svg"].startswith("<svg")
    assert env.client.get("/admin/qr-codes").json()["codes"] == []


def test_scan_redirects_records_and_ignores_bots(env):
    code = make(env, utm_source="flyer")
    first = env.client.get(f"/q/{code['slug']}", headers={"user-agent": PHONE, "referer": "https://facebook.com/post/1"})
    assert first.status_code == 302 and first.headers["location"].endswith("utm_source=flyer")
    env.client.get(f"/q/{code['slug']}", headers={"user-agent": "Googlebot/2.1"})
    stats = env.client.get(f"/admin/qr-codes/{code['id']}/stats").json()
    assert stats["total"] == 1 and stats["bots_ignored"] == 1 and stats["devices"] == {"ios": 1}
    assert stats["referrers"] == [{"host": "facebook.com", "count": 1}] and sum(d["count"] for d in stats["per_day"]) == 1
    csv_text = env.client.get(f"/admin/qr-codes/{code['id']}/scans.csv").text.splitlines()
    assert len(csv_text) == 2 and "ios" in csv_text[1]
    assert env.client.get("/admin/qr-codes").json()["codes"][0]["scan_count"] == 1


def test_resolve_endpoint_uses_forwarded_visitor_details(env):
    code = make(env)
    response = env.client.get(f"/qr/go/{code['slug']}", headers={"x-scan-user-agent": PHONE})
    assert response.json() == {"url": code["target_url"]}
    assert env.client.get(f"/admin/qr-codes/{code['id']}/stats").json()["devices"] == {"ios": 1}


def test_destination_can_change_after_printing_and_inactive_codes_stop(env):
    code = make(env)
    changed = env.client.put(f"/admin/qr-codes/{code['id']}", json={"destination": "https://other.org/x", "utm_campaign": "spring"})
    assert changed.json()["code"]["encoded_text"] == code["encoded_text"]
    assert env.client.get(f"/q/{code['slug']}").headers["location"] == "https://other.org/x?utm_campaign=spring"
    env.client.put(f"/admin/qr-codes/{code['id']}", json={"active": False})
    assert env.client.get(f"/q/{code['slug']}").status_code == 404
    assert env.client.get("/q/nope").status_code == 404


def test_untracked_code_has_no_redirect_and_delete_removes_scans(env):
    static = make(env, tracked=False)
    assert env.client.get(f"/q/{static['slug']}").status_code == 404
    code = make(env, name="Other")
    env.client.get(f"/q/{code['slug']}", headers={"user-agent": PHONE})
    assert env.client.delete(f"/admin/qr-codes/{code['id']}").status_code == 200
    assert env.client.get(f"/admin/qr-codes/{code['id']}/stats").status_code == 404


def test_too_much_data_is_a_clean_error(env):
    response = env.client.post("/admin/qr-codes", json={"name": "big", "kind": "text", "destination": "x" * 1500, "error_level": "H"})
    assert response.status_code == 422
