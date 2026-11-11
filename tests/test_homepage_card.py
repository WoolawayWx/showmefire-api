from fastapi import FastAPI
from fastapi.testclient import TestClient

from core import database, security
from core.admin_auth import users
from routers import homepage_card, mobile


def test_admin_card_controls_both_homepages(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    database.init_database()
    app = FastAPI()
    app.include_router(homepage_card.router)
    app.include_router(mobile.router)
    client = TestClient(app)
    admin_url = "/api/admin/homepage-card"
    public_url = "/api/homepage-card"

    card = client.get(public_url).json()["card"]
    assert card["url"] == "https://showmefire.org/forms/research-survey"
    assert client.get("/api/mobile/content").json()["homepageCard"] == card
    assert client.get(admin_url).status_code == 401
    assert client.put(admin_url, json={**card, "enabled": False}).status_code == 401

    account = {"email": "editor@example.org", "role": "staff", "disabled_at": None, "expires_at": None}
    monkeypatch.setattr(security, "verify_token", lambda token: account["email"])
    monkeypatch.setattr(users, "get_by_email", lambda email: account)
    assert client.put(admin_url, json={**card, "enabled": False}).status_code == 403

    account["role"] = "admin"
    for url in ("javascript:alert(1)", "http://example.org", "https://user@example.org"):
        response = client.put(admin_url, json={**card, "enabled": True, "url": url})
        assert response.status_code == 422
    assert client.get(public_url).json()["card"] == card

    edited = {**card, "title": "Community survey", "url": "https://example.org/survey"}
    assert client.put(admin_url, json={**edited, "enabled": True}).status_code == 200
    assert client.get(public_url).json()["card"] == edited
    assert client.get("/api/mobile/content").json()["homepageCard"] == edited

    assert client.put(admin_url, json={**edited, "enabled": False}).status_code == 200
    database.init_database()
    assert client.get(admin_url).json() == {**edited, "enabled": False}
    assert client.get(public_url).json() == {"card": None}
    assert client.get("/api/mobile/content").json()["homepageCard"] is None
