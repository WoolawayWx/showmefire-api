"""The website's view of the moderation log, and the permission list the site uses to hide actions a role would be refused."""
import sqlite3
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core import database, security
from core.admin_auth import lockdown, permissions, sessions, users
from core.admin_session import AdminCookieMiddleware
from routers import admin_audit, admin_auth

PASSWORD = "correct horse battery 9"
OWNER = "owner@example.org"


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(security, "ADMIN_EMAIL", OWNER)
    monkeypatch.setattr(security, "ADMIN_PASSWORD_HASH", security.hash_password(PASSWORD))
    monkeypatch.setattr(security, "AUTH_COOKIE_SECURE", False)
    sessions._forget()
    database.init_database()
    application = FastAPI()
    application.add_middleware(lockdown.ReviewerLockdownMiddleware)
    application.add_middleware(AdminCookieMiddleware)
    application.include_router(admin_auth.router)
    application.include_router(admin_audit.router)
    return application


def signed_in(app, role="owner"):
    client = TestClient(app)
    if role == "owner":
        email = OWNER
    else:
        users.ensure_seeded()
        email = f"{role}@example.org"
        users.create_user(email, role.title(), role, PASSWORD, must_change=False,
                          **({"expires_at": time.time() + 3600} if role == "reviewer" else {}))
    response = client.post("/api/admin/login", json={"email": email, "password": PASSWORD})
    assert response.status_code == 200, response.text
    return client


def test_verify_and_me_tell_the_site_what_each_role_may_do(app):
    for role in ("owner", "admin", "staff", "partner"):
        client = signed_in(app, role)
        expected = sorted(permissions.permissions_for(role))
        assert client.post("/api/admin/verify").json()["permissions"] == expected
        assert client.get("/api/admin/me").json()["permissions"] == expected
    assert "reports.moderate" not in sorted(permissions.permissions_for("partner"))
    assert TestClient(app).post("/api/admin/verify").json() == {"valid": False}


def test_the_log_is_filterable_pageable_and_needs_a_signed_in_account(app):
    owner = signed_in(app)
    with sqlite3.connect(database.get_db_path()) as conn:
        for index in range(5):
            conn.execute("INSERT INTO fire_events (source, status, latitude, longitude, occurred_at) VALUES ('user_submission', 'pending', 37, -92, '2026-10-01T12:00:00Z')")
    for event_id in range(1, 6):
        database.set_fire_event_status(event_id, "approved" if event_id % 2 else "rejected", actor="a@example.org", reason=f"reason {event_id}")
    everything = owner.get("/api/admin/moderation-log?limit=3").json()
    assert len(everything["entries"]) == 3 and everything["next_before_id"] == everything["entries"][-1]["id"]
    older = owner.get(f"/api/admin/moderation-log?limit=3&before_id={everything['next_before_id']}").json()
    assert len(older["entries"]) == 2 and older["next_before_id"] is None
    one = owner.get("/api/admin/moderation-log?subject_type=fire_event&subject_id=2").json()["entries"]
    assert [e["reason"] for e in one] == ["reason 2"]
    assert "burn_ban" in owner.get("/api/admin/moderation-log").json()["subject_types"]
    assert owner.get("/api/admin/moderation-log?subject_type=nope").status_code == 400
    assert TestClient(app).get("/api/admin/moderation-log").status_code == 401


def test_every_role_that_can_sign_in_to_the_site_may_read_the_log_but_reviewers_cannot_reach_it(app):
    for role in ("admin", "staff", "partner"):
        assert signed_in(app, role).get("/api/admin/moderation-log").status_code == 200
    users.ensure_seeded()
    reviewer = users.create_user("rev@example.org", "Rev", "reviewer", PASSWORD, must_change=False, expires_at=time.time() + 3600)
    session_id, _ = sessions.create(reviewer["id"], "127.0.0.1", "pytest", "mobile")
    headers = {"Authorization": f"Bearer {permissions.access_token_for(users.get_by_id(reviewer['id']), session_id, client='mobile')}"}
    assert TestClient(app).get("/api/admin/moderation-log", headers=headers).status_code == 403  # locked out of everything but the staff API
