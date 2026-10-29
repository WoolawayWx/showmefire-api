"""The reviewer sandbox: reviewers only ever touch synthetic data, nothing they do leaves the server, and it fails closed."""
import re
import sqlite3
import time

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from core import database, sandbox, sandbox_data, security
from core.admin_auth import lockdown, permissions, sessions, users
from core.admin_session import AdminCookieMiddleware
from routers import admin_auth

PASSWORD = "correct horse battery 9"


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("sandbox-data")
    patch = pytest.MonkeyPatch()
    patch.setenv("DATA_DIR", str(path))
    patch.setenv("SANDBOX_ENABLED", "true")
    patch.setattr(security, "ADMIN_EMAIL", "owner@example.org")
    patch.setattr(security, "ADMIN_PASSWORD_HASH", security.hash_password(PASSWORD))
    database.init_database()  # the real database, with one real record the sandbox must never touch
    with sqlite3.connect(database.real_db_path()) as conn:
        conn.execute("INSERT INTO fire_events (source, status, latitude, longitude, occurred_at, description, reporter_name) "
                     "VALUES ('user_submission', 'pending', 38.5, -92.5, '2026-10-01T12:00:00Z', 'REAL report', 'Real Person')")
    sandbox_data.build_template()
    yield path
    patch.undo()


@pytest.fixture
def app(data_dir, monkeypatch):
    monkeypatch.setenv("SANDBOX_ENABLED", "true")
    sessions._forget()
    sandbox_data.reset_sandbox()
    application = FastAPI()
    application.add_middleware(lockdown.ReviewerLockdownMiddleware)
    application.add_middleware(AdminCookieMiddleware)
    application.include_router(admin_auth.router)

    @application.get("/api/staff/v1/where")
    def where(email: str = Depends(permissions.require_permission("staff.access"))):
        with sqlite3.connect(database.get_db_path()) as conn:
            rows = conn.execute("SELECT id, status, description FROM fire_events WHERE source = 'user_submission' ORDER BY id").fetchall()
        return {"sandbox": sandbox.is_sandbox(), "email": email, "reports": rows}

    @application.post("/api/staff/v1/approve/{event_id}")
    def approve(event_id: int, email: str = Depends(permissions.require_permission("reports.moderate"))):
        return database.set_fire_event_status(event_id, "approved", actor=email, reason="demo")

    @application.get("/api/admin/website-only")
    def website_only(email: str = Depends(security.require_admin)):
        return {"email": email}

    return application


def login_headers(role, email):
    users.ensure_seeded()
    extra = {"expires_at": time.time() + 3600} if role == "reviewer" else {}
    user = users.get_by_email(email) or users.create_user(email, role.title(), role, PASSWORD, must_change=False, **extra)
    user = users.get_by_id(user["id"])
    session_id, _ = sessions.create(user["id"], "127.0.0.1", "pytest", "mobile")
    return {"Authorization": f"Bearer {permissions.access_token_for(user, session_id, client='mobile')}"}


def real_status():
    with sqlite3.connect(database.real_db_path()) as conn:
        return conn.execute("SELECT status FROM fire_events WHERE description = 'REAL report'").fetchone()[0]


# ---------------------------------------------------------------------------------------------- isolation

def test_a_reviewers_staff_requests_run_against_the_sandbox_not_the_real_database(app):
    client = TestClient(app)
    seen = client.get("/api/staff/v1/where", headers=login_headers("reviewer", "apple@example.org")).json()
    assert seen["sandbox"] is True and len(seen["reports"]) == 5
    assert all("REAL report" not in row[2] for row in seen["reports"])
    real = client.get("/api/staff/v1/where", headers=login_headers("staff", "staff@example.org")).json()
    assert real["sandbox"] is False and [r[2] for r in real["reports"]] == ["REAL report"]


def test_what_a_reviewer_approves_never_reaches_real_data(app):
    client = TestClient(app)
    headers = login_headers("reviewer", "apple@example.org")
    pending = [r for r in client.get("/api/staff/v1/where", headers=headers).json()["reports"] if r[1] == "pending"]
    assert client.post(f"/api/staff/v1/approve/{pending[0][0]}", headers=headers).status_code == 200
    after = client.get("/api/staff/v1/where", headers=headers).json()["reports"]
    assert [r[1] for r in after if r[0] == pending[0][0]] == ["approved"]
    assert real_status() == "pending"
    with sqlite3.connect(database.real_db_path()) as conn:  # and the audit row went to the sandbox's log
        assert conn.execute("SELECT COUNT(*) FROM moderation_audit").fetchone()[0] == 0
    with sqlite3.connect(sandbox.sandbox_db_path()) as conn:
        assert conn.execute("SELECT client FROM moderation_audit ORDER BY id DESC LIMIT 1").fetchone()[0] == "sandbox"


def test_a_reviewer_cannot_reach_real_admin_routes(app):
    headers = login_headers("reviewer", "apple@example.org")
    client = TestClient(app)
    assert client.get("/api/admin/website-only", headers=headers).status_code == 403
    assert client.get("/api/admin/me", headers=headers).status_code == 200


# ---------------------------------------------------------------------------------------------- fail closed

def test_the_sandbox_switch_turns_reviewer_access_off(app, monkeypatch):
    headers = login_headers("reviewer", "apple@example.org")
    monkeypatch.setenv("SANDBOX_ENABLED", "false")
    response = TestClient(app).get("/api/staff/v1/where", headers=headers)
    assert response.status_code == 503


def test_a_sandbox_that_cannot_be_made_never_falls_back_to_real_data(app, monkeypatch):
    headers = login_headers("reviewer", "apple@example.org")

    def broken():
        raise FileNotFoundError("gone")

    monkeypatch.setattr(sandbox_data, "ensure_sandbox", broken)
    response = TestClient(app).get("/api/staff/v1/where", headers=headers)
    assert response.status_code == 503 and "REAL" not in response.text


# ---------------------------------------------------------------------------------------------- nothing leaves the server

def test_nothing_a_reviewer_does_can_send_anything_out(data_dir, monkeypatch):
    import httpx

    from ai.clef import ClefClient
    from ai.cloudflare import CloudflareAIClient, CloudflareAIError
    from services import discord_notifier, mobile_push

    def no_network(*args, **kwargs):
        raise AssertionError("a request left the server from the sandbox")

    monkeypatch.setattr(httpx, "Client", no_network)
    monkeypatch.setattr("threading.Thread", no_network)  # the Discord sender would start a thread
    with sandbox.sandbox_scope(sandbox.sandbox_db_path()):
        assert discord_notifier.notify_staff_alert(alert_type="fire_report", title="t", force=True) is False
        assert discord_notifier._send_event({"event_type": "staff_alert"}) is False
        assert mobile_push.send_mobile_event(event_type="forecast", event_key="k", title="t", body="b", url="/") == 0
        assert mobile_push._send_batch([{"to": "ExponentPushToken[x]"}]) == []
        with pytest.raises(CloudflareAIError, match="sandbox"):
            CloudflareAIClient(api_key="k", account_id="a").generate_text("hello")
        with pytest.raises(CloudflareAIError, match="sandbox"):
            ClefClient(api_key="k", account_id="a").run("s", {"q": {"type": "noul", "instructions": "?"}})


# ---------------------------------------------------------------------------------------------- the seed data

def test_the_demo_data_is_synthetic_and_inside_missouri(data_dir):
    with sqlite3.connect(sandbox.sandbox_template_path()) as conn:
        conn.row_factory = sqlite3.Row
        text = " ".join(str(value) for table in ("fire_events", "burn_ban_submissions", "fire_incident_feedback", "moderation_audit")
                        for row in conn.execute(f"SELECT * FROM {table}") for value in tuple(row))
        emails = set(re.findall(r"[\w.+-]+@[\w.-]+\.\w+", text))
        assert emails and all(email.endswith("@example.invalid") for email in emails), emails
        for lat, lon in conn.execute("SELECT latitude, longitude FROM fire_events"):
            assert 36.0 <= lat <= 40.7 and -95.8 <= lon <= -89.1
        assert conn.execute("SELECT COUNT(*) FROM fire_events WHERE source = 'user_submission' AND status = 'pending'").fetchone()[0] == 3
        assert conn.execute("SELECT COUNT(*) FROM burn_ban_submissions WHERE status = 'pending'").fetchone()[0] == 3
        assert conn.execute("SELECT COUNT(*) FROM moderation_audit").fetchone()[0] >= 3  # history, written by the real functions
        assert conn.execute("SELECT COUNT(*) FROM clef_assessments").fetchone()[0] == 5
    photos = list(sandbox.sandbox_media_dir().glob("*.png"))
    assert len(photos) == 4 and all(p.read_bytes().startswith(b"\x89PNG") for p in photos)


# ---------------------------------------------------------------------------------------------- reset

def test_reset_puts_everything_back(app):
    client = TestClient(app)
    headers = login_headers("reviewer", "apple@example.org")
    first = client.get("/api/staff/v1/where", headers=headers).json()["reports"][0][0]
    client.post(f"/api/staff/v1/approve/{first}", headers=headers)
    sandbox_data.reset_sandbox()
    assert [r[1] for r in client.get("/api/staff/v1/where", headers=headers).json()["reports"] if r[0] == first] == ["pending"]


def test_an_idle_modified_sandbox_resets_but_not_while_someone_is_using_it(app):
    client = TestClient(app)
    headers = login_headers("reviewer", "apple@example.org")
    first = client.get("/api/staff/v1/where", headers=headers).json()["reports"][0][0]
    assert sandbox_data.reset_if_idle() is False  # nothing changed yet
    client.post(f"/api/staff/v1/approve/{first}", headers=headers)  # a change, just now
    assert sandbox_data.reset_if_idle() is False  # a reviewer is mid-review
    sandbox_data._marker("last_activity").write_text(str(time.time() - sandbox_data.IDLE_RESET_SECONDS - 5))
    assert sandbox_data.reset_if_idle() is True
    assert sandbox_data.reset_if_idle() is False  # and it stays reset
    assert [r[1] for r in client.get("/api/staff/v1/where", headers=headers).json()["reports"] if r[0] == first] == ["pending"]
