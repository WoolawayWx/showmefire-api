"""Passkey sign-in for the mobile app (and the matching tightening of the website's passkey sign-in).

The WebAuthn library's verification is stubbed, as in tests/test_admin_auth_flow.py; what is tested here is everything around it: the
account rules, the session it opens, and that the server's allowed origins (including Android's) reach the library."""
import importlib.util
import time
from pathlib import Path
from types import SimpleNamespace

import pyotp
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from webauthn.helpers import bytes_to_base64url

from core import security
from core.admin_auth import events, lockdown, mfa, permissions, sessions, users
from core.admin_auth import db as auth_db
from core.admin_session import AdminCookieMiddleware
from routers import admin_auth, admin_mobile

PASSWORD = "correct horse battery 9"
OWNER = "owner@example.org"
CREDENTIAL_ID = b"credential-one"


@pytest.fixture(autouse=True)
def fresh_limiter():
    admin_auth.passkey_limiter.reset()  # one in-memory budget per IP; do not let tests share it
    yield
    admin_auth.passkey_limiter.reset()


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("STAFF_API_ENABLED", "true")
    monkeypatch.setenv("SANDBOX_ENABLED", "true")
    monkeypatch.setenv("WEBAUTHN_RP_ID", "example.org")
    monkeypatch.setenv("WEBAUTHN_ORIGINS", "https://example.org")
    monkeypatch.delenv("ADMIN_MFA_ENFORCE", raising=False)
    monkeypatch.setattr(security, "ADMIN_EMAIL", OWNER)
    monkeypatch.setattr(security, "ADMIN_PASSWORD_HASH", security.hash_password(PASSWORD))
    monkeypatch.setattr(security, "AUTH_COOKIE_SECURE", False)
    sessions._forget()
    application = FastAPI()
    application.add_middleware(lockdown.ReviewerLockdownMiddleware)
    application.add_middleware(AdminCookieMiddleware)
    application.include_router(admin_mobile.router)
    application.include_router(admin_auth.router)

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


def add_passkey(user, credential_id=CREDENTIAL_ID):
    with auth_db.tx() as conn:
        conn.execute(
            "INSERT INTO admin_passkeys (admin_id, credential_id, public_key, sign_count, name, transports, created_at) VALUES (?, ?, ?, 0, 'Phone', '[]', ?)",
            (user["id"], credential_id, b"public-key", time.time()))


def credential(credential_id=CREDENTIAL_ID):
    return {"id": bytes_to_base64url(credential_id), "rawId": bytes_to_base64url(credential_id), "type": "public-key",
            "response": {"clientDataJSON": "x", "authenticatorData": "x", "signature": "x", "userHandle": "x"}}


def accept(monkeypatch):
    seen = []

    def verified(**kwargs):
        seen.append(kwargs)
        return SimpleNamespace(new_sign_count=1)

    monkeypatch.setattr("webauthn.verify_authentication_response", verified)
    return seen


def passkey_login(client, monkeypatch=None, cred=None):
    options = client.post("/api/admin/mobile/login/passkey/options").json()
    return options, client.post("/api/admin/mobile/login/passkey/verify", json={"challenge_id": options["challenge_id"], "credential": cred or credential()})


# ------------------------------------------------------------------------------------------------- signing in

def test_a_passkey_signs_in_with_no_password_and_returns_tokens_the_api_accepts(client, monkeypatch):
    user = make()
    add_passkey(user)
    accept(monkeypatch)
    options, response = passkey_login(client)
    assert options["options"]["rpId"] == "example.org" and options["options"]["userVerification"] == "required"
    assert options["options"].get("allowCredentials") in (None, [])  # discoverable: no email needed first
    body = response.json()
    assert response.status_code == 200 and body["success"] is True and body["expires_in"] == 15 * 60 and body["sandbox"] is False
    assert body["user"]["email"] == "staff@example.org" and "reports.moderate" in body["user"]["permissions"]
    assert client.get("/api/staff/v1/ping", headers={"Authorization": f"Bearer {body['access_token']}"}).json() == {"email": "staff@example.org"}
    assert any(s["method"] == "mobile+passkey" for s in sessions.list_for(user["id"]))
    assert "set-cookie" not in response.headers
    assert sessions.rotate(body["refresh_token"]) is not None  # a normal rotating mobile session


def test_a_passkey_replaces_the_second_step_even_for_an_account_that_has_one(client, monkeypatch):
    user = make()
    add_passkey(user)
    secret = mfa.totp_begin(user["id"], user["email"])["secret"]
    assert mfa.totp_confirm(user["id"], pyotp.TOTP(secret).now())
    accept(monkeypatch)
    _, response = passkey_login(client)
    assert response.json()["success"] is True and "mfa_required" not in response.json()


def test_the_allowed_origins_including_androids_are_what_the_library_checks_against(client, monkeypatch):
    monkeypatch.setenv("WEBAUTHN_ORIGINS", "https://example.org,android:apk-key-hash:qrvM3e7_ABEiM0RVZneImaq7zN3u_wARIjNEVWZ3iJk")
    user = make()
    add_passkey(user)
    seen = accept(monkeypatch)
    assert passkey_login(client)[1].status_code == 200
    assert seen[0]["expected_rp_id"] == "example.org"
    assert "android:apk-key-hash:qrvM3e7_ABEiM0RVZneImaq7zN3u_wARIjNEVWZ3iJk" in seen[0]["expected_origin"]
    assert seen[0]["require_user_verification"] is True  # a bare "present" touch is not enough


# ------------------------------------------------------------------------------------------------- refusals

def test_a_challenge_can_only_be_used_once(client, monkeypatch):
    add_passkey(make())
    accept(monkeypatch)
    options, first = passkey_login(client)
    assert first.status_code == 200
    again = client.post("/api/admin/mobile/login/passkey/verify", json={"challenge_id": options["challenge_id"], "credential": credential()})
    assert again.status_code == 401


def test_a_rejected_signature_and_an_unknown_passkey_sign_nobody_in(client, monkeypatch):
    user = make()
    add_passkey(user)

    def reject(**kwargs):
        raise ValueError("bad signature")

    monkeypatch.setattr("webauthn.verify_authentication_response", reject)
    assert passkey_login(client)[1].status_code == 401
    accept(monkeypatch)
    assert passkey_login(client, cred=credential(b"some-other-passkey"))[1].status_code == 401  # never enrolled here
    assert not any(s["method"] == "mobile+passkey" for s in sessions.list_for(user["id"]))


def test_failures_are_recorded_with_the_account_when_it_is_known(client, monkeypatch):
    user = make()
    add_passkey(user)
    users.set_disabled(user["id"], True)
    accept(monkeypatch)
    assert passkey_login(client)[1].status_code == 401
    assert "passkey_fail" in [e["kind"] for e in events.recent(user["id"])]


def test_a_disabled_account_cannot_use_its_passkey(client, monkeypatch):
    user = make()
    add_passkey(user)
    users.set_disabled(user["id"], True)
    accept(monkeypatch)
    assert passkey_login(client)[1].status_code == 401


def test_an_expired_reviewer_cannot_use_a_passkey(client, monkeypatch):
    user = make("reviewer", expires_at=time.time() - 5)
    add_passkey(user)
    accept(monkeypatch)
    assert passkey_login(client)[1].status_code == 401


def test_reviewers_follow_the_sandbox_switch(client, monkeypatch):
    user = make("reviewer")
    add_passkey(user)
    accept(monkeypatch)
    monkeypatch.setenv("SANDBOX_ENABLED", "false")
    assert passkey_login(client)[1].status_code == 503
    monkeypatch.setenv("SANDBOX_ENABLED", "true")
    _, response = passkey_login(client)
    assert response.json()["sandbox"] is True  # allowed in the app (never on the website: see below)


def test_a_temporary_password_still_means_the_website_first(client, monkeypatch):
    add_passkey(make(must_change=True))
    accept(monkeypatch)
    body = passkey_login(client)[1].json()
    assert body["success"] is False and body["must_change_password"] is True and "access_token" not in body


def test_it_does_not_exist_until_the_staff_api_is_switched_on(client, monkeypatch):
    monkeypatch.setenv("STAFF_API_ENABLED", "false")
    assert client.post("/api/admin/mobile/login/passkey/options").status_code == 404
    assert client.post("/api/admin/mobile/login/passkey/verify", json={"challenge_id": "x", "credential": {}}).status_code == 404


def test_it_is_rate_limited_per_client(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_ADMIN_PASSKEY_PER_MIN", "3")
    statuses = [client.post("/api/admin/mobile/login/passkey/options").status_code for _ in range(5)]
    assert statuses[:3] == [200, 200, 200] and statuses[3:] == [429, 429]


# ------------------------------------------------------------------------------------------------- the website's passkey path

def test_the_website_refuses_passkey_sessions_to_reviewers_and_expired_accounts(client, monkeypatch):
    add_passkey(make("reviewer"))
    accept(monkeypatch)
    options = client.post("/api/admin/login/passkey/options").json()
    assert client.post("/api/admin/login/passkey/verify", json={"challenge_id": options["challenge_id"], "credential": credential()}).status_code == 401

    other = make("reviewer", email="old@example.org", expires_at=time.time() - 5)
    add_passkey(other, b"credential-two")
    options = client.post("/api/admin/login/passkey/options").json()
    assert client.post("/api/admin/login/passkey/verify", json={"challenge_id": options["challenge_id"], "credential": credential(b"credential-two")}).status_code == 401


def test_the_website_still_signs_staff_in_with_a_passkey(client, monkeypatch):
    add_passkey(make())
    accept(monkeypatch)
    options = client.post("/api/admin/login/passkey/options").json()
    done = client.post("/api/admin/login/passkey/verify", json={"challenge_id": options["challenge_id"], "credential": credential()})
    assert done.status_code == 200 and client.get("/api/admin/me").status_code == 200


# ------------------------------------------------------------------------------------------------- configuration

def test_android_origins_in_the_server_setting_are_kept_and_replace_the_default_web_list(monkeypatch):
    monkeypatch.setenv("WEBAUTHN_RP_ID", "showmefire.org")
    monkeypatch.delenv("WEBAUTHN_ORIGINS", raising=False)
    rp, defaults = mfa.webauthn_settings()
    assert rp == "showmefire.org" and "https://dev.showmefire.org" in defaults
    monkeypatch.setenv("WEBAUTHN_ORIGINS", "https://showmefire.org,android:apk-key-hash:abc_DEF-123")
    _, listed = mfa.webauthn_settings()
    assert listed == ["https://showmefire.org", "android:apk-key-hash:abc_DEF-123"]  # explicit list REPLACES the defaults


def test_the_origin_helper_turns_a_fingerprint_into_the_android_origin():
    path = Path(__file__).resolve().parents[1] / "scripts" / "android_origin.py"
    spec = importlib.util.spec_from_file_location("android_origin", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fingerprint = "AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99:AA:BB:CC:DD:EE:FF:00:11:22:33:44:55:66:77:88:99"
    assert module.fingerprint_to_origin(fingerprint) == "android:apk-key-hash:qrvM3e7_ABEiM0RVZneImaq7zN3u_wARIjNEVWZ3iJk"
    assert module.fingerprint_to_origin(fingerprint.lower().replace(":", " ")) == module.fingerprint_to_origin(fingerprint)  # tolerant of format
    for bad in ("AA:BB", "not hex at all", ""):
        with pytest.raises(ValueError):
            module.fingerprint_to_origin(bad)
