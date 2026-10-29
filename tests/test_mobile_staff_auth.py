"""Mobile staff sign-in: same accounts and rules as the website, tokens in the body, rotating refresh, reviewer exceptions."""
import time

import pyotp
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from core import security
from core.admin_auth import events, lockdown, mfa, permissions, sessions, throttle, users
from core.admin_session import AdminCookieMiddleware
from routers import admin_mobile

PASSWORD = "correct horse battery 9"
OWNER = "owner@example.org"


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("STAFF_API_ENABLED", "true")
    monkeypatch.setenv("SANDBOX_ENABLED", "true")
    monkeypatch.delenv("ADMIN_MFA_ENFORCE", raising=False)
    monkeypatch.setattr(security, "ADMIN_EMAIL", OWNER)
    monkeypatch.setattr(security, "ADMIN_PASSWORD_HASH", security.hash_password(PASSWORD))
    sessions._forget()
    application = FastAPI()
    application.add_middleware(lockdown.ReviewerLockdownMiddleware)
    application.add_middleware(AdminCookieMiddleware)
    application.include_router(admin_mobile.router)

    @application.get("/api/staff/v1/ping")
    def ping(email: str = Depends(permissions.require_permission("staff.access"))):
        return {"email": email}

    return application


@pytest.fixture
def client(app):
    return TestClient(app)


def make(role="staff", email=None, must_change=False, **extra):
    users.ensure_seeded()
    if role == "reviewer":
        extra.setdefault("expires_at", time.time() + 3600)
    return users.create_user(email or f"{role}@example.org", role.title(), role, PASSWORD, must_change=must_change, **extra)


def login(client, email, password=PASSWORD):
    return client.post("/api/admin/mobile/login", json={"email": email, "password": password})


def authed(tokens):
    return {"Authorization": f"Bearer {tokens['access_token']}"}


# ------------------------------------------------------------------------------------------------- switch + basic login

def test_it_does_not_exist_until_switched_on(client, monkeypatch):
    monkeypatch.setenv("STAFF_API_ENABLED", "false")
    assert login(client, OWNER).status_code == 404
    assert client.post("/api/admin/mobile/refresh", json={"refresh_token": "x"}).status_code == 404


def test_signing_in_returns_tokens_that_work_as_a_bearer_header(client):
    make()
    body = login(client, "staff@example.org").json()
    assert body["success"] is True and body["expires_in"] == 15 * 60 and body["sandbox"] is False
    assert body["user"]["role"] == "staff" and "reports.moderate" in body["user"]["permissions"]
    assert "set-cookie" not in login(client, "staff@example.org").headers
    assert client.get("/api/staff/v1/ping", headers=authed(body)).json() == {"email": "staff@example.org"}
    assert client.get("/api/staff/v1/ping").status_code == 401
    assert any(s["method"] == "mobile" for s in sessions.list_for(users.get_by_email("staff@example.org")["id"]))  # visible in the web session list


def test_a_partner_gets_view_permissions_only(client):
    make("partner")
    perms = login(client, "partner@example.org").json()["user"]["permissions"]
    assert "reports.view" in perms and "reports.moderate" not in perms and "burnbans.moderate" not in perms


def test_wrong_credentials_get_the_same_answer_and_lock_out_like_the_website(client):
    make()
    for _ in range(3):
        assert login(client, "staff@example.org", "wrong password 1").json() == {"detail": "Invalid credentials"}
        assert login(client, "nobody@example.org", "wrong password 1").json() == {"detail": "Invalid credentials"}
    for _ in range(throttle.limits()["pw"]):
        login(client, "staff@example.org", "wrong password 1")
    assert login(client, "staff@example.org").status_code == 429  # even the right password, until the window passes
    assert "login_fail" in [e["kind"] for e in events.recent(users.get_by_email("staff@example.org")["id"])]


def test_a_disabled_account_cannot_sign_in(client):
    user = make()
    users.set_disabled(user["id"], True)
    assert login(client, "staff@example.org").status_code == 401


def test_a_temporary_password_must_be_replaced_on_the_website_first(client):
    make(email="new@example.org", must_change=True)
    body = login(client, "new@example.org").json()
    assert body["success"] is False and body["must_change_password"] is True and "access_token" not in body


# ------------------------------------------------------------------------------------------------- second step

def enrol(user):
    secret = mfa.totp_begin(user["id"], user["email"])["secret"]
    totp = pyotp.TOTP(secret)
    assert mfa.totp_confirm(user["id"], totp.now())
    return totp


def test_an_account_with_an_authenticator_app_needs_its_code(client):
    user = make()
    totp = enrol(user)
    first = login(client, "staff@example.org").json()
    assert first["mfa_required"] is True and "access_token" not in first
    wrong = client.post("/api/admin/mobile/login/totp", json={"mfa_token": first["mfa_token"], "code": "000000"})
    assert wrong.status_code == 401
    # an authenticator code cannot be used twice, so move to the next 30-second step like the website tests do
    code = totp.at(time.time() + 30)
    mfa_now = mfa._now
    try:
        mfa._now = lambda: time.time() + 30
        done = client.post("/api/admin/mobile/login/totp", json={"mfa_token": first["mfa_token"], "code": code}).json()
    finally:
        mfa._now = mfa_now
    assert done["success"] is True and client.get("/api/staff/v1/ping", headers=authed(done)).status_code == 200
    assert any(s["method"] == "mobile+totp" for s in sessions.list_for(user["id"]))


def test_a_recovery_code_signs_in_once(client):
    user = make()
    enrol(user)
    code = mfa.recovery_regenerate(user["id"])[0]
    first = login(client, "staff@example.org").json()
    ok = client.post("/api/admin/mobile/login/recovery", json={"mfa_token": first["mfa_token"], "code": code}).json()
    assert ok["success"] is True and ok["recovery_remaining"] == 9
    again = login(client, "staff@example.org").json()
    assert client.post("/api/admin/mobile/login/recovery", json={"mfa_token": again["mfa_token"], "code": code}).status_code == 401


def test_the_second_step_token_is_not_a_session(client):
    user = make()
    enrol(user)
    pending = login(client, "staff@example.org").json()["mfa_token"]
    assert client.get("/api/staff/v1/ping", headers={"Authorization": f"Bearer {pending}"}).status_code == 401


def test_when_two_step_is_enforced_an_account_without_it_is_sent_to_the_website(client, monkeypatch):
    make()
    monkeypatch.setattr(mfa, "enforcement_active", lambda: True)
    body = login(client, "staff@example.org").json()
    assert body["mfa_enrollment_required"] is True and "access_token" not in body


# ------------------------------------------------------------------------------------------------- refresh + logout

def test_each_refresh_replaces_the_refresh_token_and_a_replay_ends_the_session(client, monkeypatch):
    make()
    first = login(client, "staff@example.org").json()
    second = client.post("/api/admin/mobile/refresh", json={"refresh_token": first["refresh_token"]}).json()
    assert second["success"] and second["refresh_token"] != first["refresh_token"]
    assert client.get("/api/staff/v1/ping", headers=authed(second)).status_code == 200
    # inside the grace period a repeat is a lost response, not an attack
    assert client.post("/api/admin/mobile/refresh", json={"refresh_token": first["refresh_token"]}).status_code == 200
    monkeypatch.setattr(sessions, "ROTATION_GRACE_SECONDS", -1)
    assert client.post("/api/admin/mobile/refresh", json={"refresh_token": first["refresh_token"]}).status_code == 401  # replay
    sessions._forget()
    assert client.get("/api/staff/v1/ping", headers=authed(second)).status_code == 401  # the session is gone for everyone
    assert client.post("/api/admin/mobile/refresh", json={"refresh_token": second["refresh_token"]}).status_code == 401


def test_logging_out_ends_the_session_at_once(client):
    make()
    tokens = login(client, "staff@example.org").json()
    assert client.post("/api/admin/mobile/logout", headers=authed(tokens)).json() == {"success": True}
    sessions._forget()
    assert client.get("/api/staff/v1/ping", headers=authed(tokens)).status_code == 401
    assert client.post("/api/admin/mobile/refresh", json={"refresh_token": tokens["refresh_token"]}).status_code == 401


def test_a_phone_whose_access_token_expired_can_still_sign_out_with_the_refresh_token(client):
    make()
    tokens = login(client, "staff@example.org").json()
    assert client.post("/api/admin/mobile/logout", json={"refresh_token": tokens["refresh_token"]}).json() == {"success": True}
    assert client.post("/api/admin/mobile/refresh", json={"refresh_token": tokens["refresh_token"]}).status_code == 401
    assert client.post("/api/admin/mobile/logout").json() == {"success": True}  # and signing out never errors


def test_an_account_disabled_after_sign_in_cannot_refresh(client):
    user = make()
    tokens = login(client, "staff@example.org").json()
    users.set_disabled(user["id"], True)
    assert client.post("/api/admin/mobile/refresh", json={"refresh_token": tokens["refresh_token"]}).status_code == 401


# ------------------------------------------------------------------------------------------------- reviewers

def test_a_reviewer_signs_in_without_a_second_step_even_when_it_is_enforced(client, monkeypatch):
    make("reviewer", email="apple@example.org")
    monkeypatch.setattr(mfa, "enforcement_active", lambda: True)
    body = login(client, "apple@example.org").json()
    assert body["success"] is True and body["sandbox"] is True and body["user"]["role"] == "reviewer"
    assert "push.receive" not in body["user"]["permissions"]


def test_a_reviewer_cannot_sign_in_when_the_sandbox_is_off_or_after_expiry(client, monkeypatch):
    make("reviewer", email="apple@example.org")
    make("reviewer", email="old@example.org", expires_at=time.time() - 5)
    assert login(client, "old@example.org").status_code == 401
    monkeypatch.setenv("SANDBOX_ENABLED", "false")
    assert login(client, "apple@example.org").status_code == 503


def test_other_accounts_are_never_second_step_exempt(client):
    user = make()
    enrol(user)
    assert login(client, "staff@example.org").json().get("mfa_required") is True
