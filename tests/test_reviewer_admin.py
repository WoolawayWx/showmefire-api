"""Owner tools for reviewer accounts: create (password shown once), reset, extend, and the sandbox reset button."""
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core import sandbox, sandbox_data, security
from core.admin_auth import lockdown, permissions, sessions, users
from core.admin_session import AdminCookieMiddleware
from routers import admin_auth, admin_mobile, admin_reviewers

OWNER = "owner@example.org"
PASSWORD = "correct horse battery 9"


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("STAFF_API_ENABLED", "true")
    monkeypatch.setenv("SANDBOX_ENABLED", "true")
    monkeypatch.setattr(security, "ADMIN_EMAIL", OWNER)
    monkeypatch.setattr(security, "ADMIN_PASSWORD_HASH", security.hash_password(PASSWORD))
    monkeypatch.setattr(security, "AUTH_COOKIE_SECURE", False)
    sessions._forget()
    application = FastAPI()
    application.add_middleware(lockdown.ReviewerLockdownMiddleware)
    application.add_middleware(AdminCookieMiddleware)
    application.include_router(admin_auth.router)
    application.include_router(admin_reviewers.router)
    application.include_router(admin_mobile.router)
    return application


@pytest.fixture
def owner(app):
    client = TestClient(app)
    assert client.post("/api/admin/login", json={"email": OWNER, "password": PASSWORD}).json()["success"] is True
    return client


def bearer(role):
    users.ensure_seeded()
    user = users.create_user(f"{role}@example.org", role.title(), role, PASSWORD, must_change=False)
    session_id, _ = sessions.create(user["id"], "127.0.0.1", "pytest", "mobile")
    return {"Authorization": f"Bearer {permissions.access_token_for(users.get_by_id(user['id']), session_id, client='mobile')}"}


def test_only_owners_may_use_them(app, owner):
    client = TestClient(app)
    assert client.get("/api/admin/reviewers").status_code == 401
    for role in ("admin", "staff", "partner"):
        headers = bearer(role)
        assert client.get("/api/admin/reviewers", headers=headers).status_code == 403
        assert client.post("/api/admin/reviewers", json={"label": "Apple review"}, headers=headers).status_code == 403
        assert client.post("/api/admin/sandbox/reset", headers=headers).status_code == 403


def test_creating_a_reviewer_returns_the_password_once_and_it_works_in_the_app(app, owner):
    created = owner.post("/api/admin/reviewers", json={"label": "Apple review, 1.3.0", "note": "App Store submission", "days": 45}).json()
    assert created["email"].startswith("apple-review-1-3-0-") and created["email"].endswith("@reviewer.showmefire.org")
    reviewer = created["reviewer"]
    assert reviewer["role"] == "reviewer" and reviewer["sandbox"] is True and reviewer["note"] == "App Store submission"
    assert abs(reviewer["expires_at"] - (time.time() + 45 * 86400)) < 60
    assert created["password"] not in owner.get("/api/admin/reviewers").text  # never shown again
    app_client = TestClient(app)
    signed_in = app_client.post("/api/admin/mobile/login", json={"email": created["email"], "password": created["password"]}).json()
    assert signed_in["success"] is True and signed_in["sandbox"] is True
    web = TestClient(app).post("/api/admin/login", json={"email": created["email"], "password": created["password"]})
    assert web.status_code == 401  # reviewers cannot use the website


def test_the_list_shows_reviewers_and_the_sandbox_state(app, owner):
    owner.post("/api/admin/reviewers", json={"label": "Google review"})
    listing = owner.get("/api/admin/reviewers").json()
    assert [r["display_name"] for r in listing["reviewers"]] == ["Google review"]
    assert listing["sandbox"]["enabled"] is True and listing["sandbox"]["auto_reset_after_minutes"] == 30
    assert listing["reviewers"][0]["expired"] is False


def test_label_and_days_are_checked(owner):
    assert owner.post("/api/admin/reviewers", json={"label": "ab"}).status_code == 422
    assert owner.post("/api/admin/reviewers", json={"label": "Apple review", "days": 0}).status_code == 422
    assert owner.post("/api/admin/reviewers", json={"label": "Apple review", "days": 9999}).status_code == 422


def test_resetting_a_password_ends_their_sessions(app, owner):
    created = owner.post("/api/admin/reviewers", json={"label": "Apple review"}).json()
    tokens = TestClient(app).post("/api/admin/mobile/login", json={"email": created["email"], "password": created["password"]}).json()
    reset = owner.post(f"/api/admin/reviewers/{created['reviewer']['id']}/reset-password").json()
    assert reset["password"] != created["password"]
    sessions._forget()
    assert TestClient(app).post("/api/admin/mobile/refresh", json={"refresh_token": tokens["refresh_token"]}).status_code == 401
    assert TestClient(app).post("/api/admin/mobile/login", json={"email": created["email"], "password": created["password"]}).status_code == 401
    assert TestClient(app).post("/api/admin/mobile/login", json={"email": created["email"], "password": reset["password"]}).json()["success"] is True


def test_extending_an_expired_account_brings_it_back(app, owner):
    created = owner.post("/api/admin/reviewers", json={"label": "Apple review"}).json()
    admin_id = created["reviewer"]["id"]
    users.set_expiry(admin_id, time.time() - 10)
    assert TestClient(app).post("/api/admin/mobile/login", json={"email": created["email"], "password": created["password"]}).status_code == 401
    assert owner.get("/api/admin/reviewers").json()["reviewers"][0]["expired"] is True
    extended = owner.post(f"/api/admin/reviewers/{admin_id}/extend", json={"days": 10}).json()["reviewer"]
    assert extended["expired"] is False
    assert TestClient(app).post("/api/admin/mobile/login", json={"email": created["email"], "password": created["password"]}).json()["success"] is True


def test_only_reviewer_accounts_can_be_reset_or_extended_here(app, owner):
    users.ensure_seeded()
    staff = users.create_user("staff@example.org", "Staff", "staff", PASSWORD, must_change=False)
    assert owner.post(f"/api/admin/reviewers/{staff['id']}/reset-password").status_code == 404
    assert owner.post(f"/api/admin/reviewers/{staff['id']}/extend", json={}).status_code == 404
    assert owner.post("/api/admin/reviewers/9999/extend", json={}).status_code == 404


def test_disabling_a_reviewer_uses_the_normal_account_route_and_signs_them_out(app, owner):
    created = owner.post("/api/admin/reviewers", json={"label": "Apple review"}).json()
    tokens = TestClient(app).post("/api/admin/mobile/login", json={"email": created["email"], "password": created["password"]}).json()
    assert owner.patch(f"/api/admin/users/{created['reviewer']['id']}", json={"disabled": True}).status_code == 200
    sessions._forget()
    assert TestClient(app).post("/api/admin/mobile/refresh", json={"refresh_token": tokens["refresh_token"]}).status_code == 401


def test_the_reset_button_resets_the_sandbox_and_is_refused_when_it_is_off(app, owner, monkeypatch):
    sandbox_data.build_template()
    sandbox_data.reset_sandbox()
    import sqlite3

    with sqlite3.connect(sandbox.sandbox_db_path()) as conn:
        conn.execute("UPDATE fire_events SET status = 'approved'")
    assert owner.post("/api/admin/sandbox/reset").json() == {"success": True}
    with sqlite3.connect(sandbox.sandbox_db_path()) as conn:
        assert conn.execute("SELECT COUNT(*) FROM fire_events WHERE status = 'pending' AND source = 'user_submission'").fetchone()[0] == 3
    monkeypatch.setenv("SANDBOX_ENABLED", "false")
    assert owner.post("/api/admin/sandbox/reset").status_code == 409


def test_a_reviewer_cannot_reach_these_routes(app, owner):
    created = owner.post("/api/admin/reviewers", json={"label": "Apple review"}).json()
    tokens = TestClient(app).post("/api/admin/mobile/login", json={"email": created["email"], "password": created["password"]}).json()
    headers = {"Authorization": f"Bearer {tokens['access_token']}"}
    client = TestClient(app)
    assert client.get("/api/admin/reviewers", headers=headers).status_code == 403
    assert client.post("/api/admin/sandbox/reset", headers=headers).status_code == 403
