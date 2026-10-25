"""Admin sign-in end to end: password, throttling, sessions, authenticator codes, recovery codes, passkeys, enforcement and owner tools.

Builds a small app around the auth router and the real cookie middleware (importing `main` needs packages this environment lacks)."""
import time
from types import SimpleNamespace

import pyotp
import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from webauthn.helpers import bytes_to_base64url

from core import security
from core.admin_auth import access, mfa, sessions, throttle, users
from core.admin_session import AdminCookieMiddleware
from core.security import require_admin
from routers import admin_auth, admin_confirm

_offset = [0]


def now():
    """The clock the authenticator-code check uses: real time plus however far a test has advanced it."""
    return time.time() + _offset[0]


def advance(seconds):
    _offset[0] += seconds


@pytest.fixture(autouse=True)
def clock(monkeypatch):
    _offset[0] = 0
    monkeypatch.setattr(mfa, "_now", now)


EMAIL = "owner@example.org"
PASSWORD = "correct horse battery 9"


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("WEBAUTHN_RP_ID", "localhost")
    monkeypatch.setenv("WEBAUTHN_ORIGINS", "http://localhost:3000")
    monkeypatch.delenv("ADMIN_MFA_ENFORCE", raising=False)
    monkeypatch.delenv("ADMIN_ALLOW_LEGACY_TOKENS", raising=False)
    monkeypatch.setattr(security, "ADMIN_EMAIL", EMAIL)
    monkeypatch.setattr(security, "ADMIN_PASSWORD_HASH", security.hash_password(PASSWORD))
    monkeypatch.setattr(security, "AUTH_COOKIE_SECURE", False)
    sessions._forget()
    application = FastAPI()
    application.add_middleware(AdminCookieMiddleware)
    application.include_router(admin_auth.router)
    application.include_router(admin_confirm.router)

    @application.get("/protected")
    def protected(email: str = Depends(require_admin)):
        return {"email": email}

    return application


@pytest.fixture
def client(app):
    return TestClient(app)


def sign_in(client, email=EMAIL, password=PASSWORD):
    return client.post("/api/admin/login", json={"email": email, "password": password})


def confirm_token(email=EMAIL):
    return security.create_confirm_token(email, "mfa_manage")


def enrol_totp(client, email=EMAIL):
    """Set up an authenticator app on a signed-in client; returns (TOTP generator, response of the confirm call)."""
    setup = client.post("/api/admin/mfa/totp/setup", json={"confirm_token": confirm_token(email)}).json()
    totp = pyotp.TOTP(setup["secret"])
    confirmed = client.post("/api/admin/mfa/totp/confirm", json={"confirm_token": confirm_token(email), "code": totp.at(now())})
    return totp, confirmed


# --------------------------------------------------------------------------- migration and password login

def test_existing_credentials_become_the_owner_account_and_still_work(client):
    response = sign_in(client)
    assert response.status_code == 200 and response.json()["success"] is True
    assert client.get("/protected").json() == {"email": EMAIL}
    probe = client.post("/api/admin/verify").json()
    assert probe["valid"] is True and probe["role"] == "owner" and probe["email"] == EMAIL
    assert users.get_by_email(EMAIL.upper())["role"] == "owner"


def test_wrong_password_and_unknown_email_get_the_same_answer_and_both_spend_a_verification(client, monkeypatch):
    calls = []
    real = security.verify_password
    monkeypatch.setattr(security, "verify_password", lambda plain, hashed: calls.append(1) or real(plain, hashed))
    wrong = sign_in(client, password="nope")
    unknown = sign_in(client, email="nobody@example.org", password="nope")
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()
    assert "set-cookie" not in wrong.headers
    assert len(calls) == 2


def test_repeated_failures_lock_the_password_path_but_not_a_passkey_and_clear_themselves(client, monkeypatch):
    monkeypatch.setenv("ADMIN_LOGIN_MAX_PW", "3")
    for _ in range(3):
        assert sign_in(client, password="nope").status_code == 401
    locked = sign_in(client)  # even the right password waits out the lock
    assert locked.status_code == 429 and int(locked.headers["Retry-After"]) > 0
    assert client.post("/api/admin/login/passkey/options").status_code == 200
    throttle.clear("pw", EMAIL)
    assert sign_in(client).status_code == 200


def test_failures_age_out_of_the_window(client, monkeypatch):
    monkeypatch.setenv("ADMIN_LOGIN_MAX_PW", "2")
    for _ in range(2):
        sign_in(client, password="nope")
    assert sign_in(client).status_code == 429
    monkeypatch.setattr(throttle, "WINDOW_SECONDS", 0)
    assert sign_in(client).status_code == 200


# --------------------------------------------------------------------------- sessions

def test_renewal_from_the_refresh_cookie_and_logout_revokes_it(client):
    sign_in(client)
    refresh = client.cookies.get("admin_refresh")
    client.cookies.delete("admin_access")
    renewed = client.get("/protected")
    assert renewed.status_code == 200 and "admin_access=" in renewed.headers["set-cookie"]
    logged_out = client.post("/api/admin/logout")
    assert "admin_access=" in logged_out.headers["set-cookie"] and "admin_refresh=" in logged_out.headers["set-cookie"]
    client.cookies.set("admin_refresh", refresh)
    client.cookies.delete("admin_access")
    assert client.get("/protected").status_code == 401


def test_revoking_a_session_stops_it_immediately_and_logout_all_ends_every_device(client, app):
    sign_in(client)
    other = TestClient(app)
    sign_in(other)
    listing = client.get("/api/admin/sessions").json()["sessions"]
    assert len(listing) == 2
    other_id = next(s["id"] for s in listing if not s["current"]) if any(not s["current"] for s in listing) else listing[0]["id"]
    assert client.delete(f"/api/admin/sessions/{other_id}").status_code == 200
    assert client.delete("/api/admin/sessions/does-not-exist").status_code == 404
    assert client.post("/api/admin/logout-all").status_code == 200
    assert client.get("/protected").status_code == 401 and other.get("/protected").status_code == 401


def test_sessions_without_an_id_are_refused_once_legacy_tokens_are_turned_off(client, monkeypatch):
    token = security.create_access_token({"sub": EMAIL})
    assert client.get("/protected", params={"token": token}).status_code == 200
    monkeypatch.setenv("ADMIN_ALLOW_LEGACY_TOKENS", "false")
    assert client.get("/protected", params={"token": token}).status_code == 401


def test_a_disabled_account_loses_its_sessions_and_cannot_sign_in(client, app):
    sign_in(client)
    created = client.post("/api/admin/users", json={"email": "second@example.org"}).json()
    second = TestClient(app)
    sign_in(second, email="second@example.org", password=created["temporary_password"])
    assert second.get("/protected").status_code == 200
    assert client.patch(f"/api/admin/users/{created['user']['id']}", json={"disabled": True}).status_code == 200
    assert second.get("/protected").status_code == 401
    assert sign_in(TestClient(app), email="second@example.org", password=created["temporary_password"]).status_code == 401
    assert client.patch(f"/api/admin/users/{users.get_by_email(EMAIL)['id']}", json={"disabled": True}).status_code == 422  # not yourself


def test_password_change_needs_the_current_password_and_ends_other_sessions(client, app):
    sign_in(client)
    other = TestClient(app)
    sign_in(other)
    assert client.post("/api/admin/password", json={"current_password": "wrong", "new_password": "a brand new pass 77"}).status_code == 401
    assert client.post("/api/admin/password", json={"current_password": PASSWORD, "new_password": "short"}).status_code == 422
    assert client.post("/api/admin/password", json={"current_password": PASSWORD, "new_password": "a brand new pass 77"}).json()["success"]
    assert other.get("/protected").status_code == 401
    assert sign_in(TestClient(app), password="a brand new pass 77").status_code == 200


def test_changing_the_email_keeps_the_account(client):
    sign_in(client)
    assert client.patch("/api/admin/profile", json={"email": "me@example.org", "display_name": "Me"}).status_code == 401
    changed = client.patch("/api/admin/profile", json={"email": "me@example.org", "display_name": "Me", "current_password": PASSWORD})
    assert changed.status_code == 200 and changed.json()["email"] == "me@example.org"
    assert sign_in(TestClient(client.app), email="me@example.org").status_code == 200


# --------------------------------------------------------------------------- authenticator app and recovery codes

def test_enrolling_an_authenticator_app_adds_a_second_step_and_issues_recovery_codes(client, app):
    sign_in(client)
    assert client.post("/api/admin/mfa/totp/setup", json={}).status_code == 403  # needs a fresh password confirmation
    totp, confirmed = enrol_totp(client)
    codes = confirmed.json()["recovery_codes"]
    assert confirmed.status_code == 200 and len(codes) == 10

    fresh = TestClient(app)
    step_one = sign_in(fresh).json()
    assert step_one["mfa_required"] is True and "totp" in step_one["methods"] and "admin_access" not in fresh.cookies
    token = step_one["mfa_token"]
    assert fresh.post("/api/admin/login/totp", json={"mfa_token": token, "code": "000000"}).status_code == 401
    advance(30)  # enrolling used this 30-second step, and a code can only be used once, so wait for the next one
    good = totp.at(now())
    assert fresh.post("/api/admin/login/totp", json={"mfa_token": token, "code": good}).status_code == 200
    assert fresh.get("/protected").status_code == 200
    # the same code is not accepted twice
    again = TestClient(app)
    token = sign_in(again).json()["mfa_token"]
    assert again.post("/api/admin/login/totp", json={"mfa_token": token, "code": good}).status_code == 401


def test_a_recovery_code_signs_in_once(client, app):
    sign_in(client)
    _, confirmed = enrol_totp(client)
    code = confirmed.json()["recovery_codes"][0]
    fresh = TestClient(app)
    token = sign_in(fresh).json()["mfa_token"]
    assert fresh.post("/api/admin/login/recovery", json={"mfa_token": token, "code": code.upper()}).json()["recovery_remaining"] == 9
    again = TestClient(app)
    token = sign_in(again).json()["mfa_token"]
    assert again.post("/api/admin/login/recovery", json={"mfa_token": token, "code": code}).status_code == 401


def test_the_second_step_token_cannot_be_used_as_a_session(client):
    sign_in(client)
    enrol_totp(client)
    token = sign_in(TestClient(client.app)).json()["mfa_token"]
    assert client.get("/protected", params={"token": token}).status_code == 401


def test_enforcement_waits_until_the_owner_has_a_factor_and_recovery_codes(client, app, monkeypatch):
    monkeypatch.setenv("ADMIN_MFA_ENFORCE", "true")
    assert mfa.enforcement_active() is False
    assert sign_in(client).json()["success"] is True  # switching it on early cannot lock the owner out
    enrol_totp(client)
    assert mfa.enforcement_active() is True


def test_with_enforcement_a_new_admin_must_enrol_before_getting_a_session(client, app, monkeypatch):
    sign_in(client)
    enrol_totp(client)
    monkeypatch.setenv("ADMIN_MFA_ENFORCE", "true")
    created = client.post("/api/admin/users", json={"email": "new@example.org"}).json()
    newcomer = TestClient(app)
    step = sign_in(newcomer, email="new@example.org", password=created["temporary_password"]).json()
    assert step["mfa_enrollment_required"] is True and "admin_access" not in newcomer.cookies
    headers = {"x-admin-enroll-token": step["enroll_token"]}
    setup = newcomer.post("/api/admin/mfa/totp/setup", json={}, headers=headers).json()
    totp = pyotp.TOTP(setup["secret"])
    done = newcomer.post("/api/admin/mfa/totp/confirm", json={"code": totp.at(now())}, headers=headers).json()
    assert len(done["recovery_codes"]) == 10
    assert newcomer.post("/api/admin/mfa/totp/setup", json={}).status_code in (401, 403)  # the enrol token is not a session


def test_with_enforcement_the_password_alone_no_longer_confirms_an_action(client, monkeypatch):
    sign_in(client)
    totp, _ = enrol_totp(client)
    monkeypatch.setenv("ADMIN_MFA_ENFORCE", "true")
    body = {"password": PASSWORD, "action": "run_forecast_v1"}
    assert client.post("/api/admin/verify-password", json=body).status_code == 401
    advance(30)
    ok = client.post("/api/admin/verify-password", json={**body, "code": totp.at(now())})
    assert ok.status_code == 200 and security.verify_confirm_token(ok.json()["confirm_token"], "run_forecast_v1") == EMAIL


def test_the_last_sign_in_method_cannot_be_removed_while_enforced(client, monkeypatch):
    sign_in(client)
    enrol_totp(client)
    monkeypatch.setenv("ADMIN_MFA_ENFORCE", "true")
    assert client.post("/api/admin/mfa/totp/remove", json={"confirm_token": confirm_token()}).status_code == 409


# --------------------------------------------------------------------------- passkeys (the WebAuthn library calls are mocked)

def test_passkey_registration_and_sign_in_with_single_use_challenges(client, app, monkeypatch):
    sign_in(client)
    credential_id = b"credential-one"
    registered = SimpleNamespace(credential_id=credential_id, credential_public_key=b"public-key", sign_count=0)
    monkeypatch.setattr("webauthn.verify_registration_response", lambda **kwargs: registered)

    options = client.post("/api/admin/mfa/passkeys/options", json={"confirm_token": confirm_token()}).json()
    assert options["options"]["rp"]["id"] == "localhost" and options["options"]["authenticatorSelection"]["userVerification"] == "required"
    credential = {"id": bytes_to_base64url(credential_id), "rawId": bytes_to_base64url(credential_id), "response": {"transports": ["internal"]}}
    saved = client.post("/api/admin/mfa/passkeys/verify", json={"confirm_token": confirm_token(), "challenge_id": options["challenge_id"],
                                                                  "credential": credential, "name": "Laptop"})
    assert saved.status_code == 200 and saved.json()["passkeys"][0]["name"] == "Laptop" and len(saved.json()["recovery_codes"]) == 10
    replay = client.post("/api/admin/mfa/passkeys/verify", json={"confirm_token": confirm_token(), "challenge_id": options["challenge_id"],
                                                                   "credential": credential, "name": "Laptop"})
    assert replay.status_code == 400

    monkeypatch.setattr("webauthn.verify_authentication_response", lambda **kwargs: SimpleNamespace(new_sign_count=1))
    fresh = TestClient(app)
    login = fresh.post("/api/admin/login/passkey/options").json()
    verified = fresh.post("/api/admin/login/passkey/verify", json={"challenge_id": login["challenge_id"], "credential": credential})
    assert verified.status_code == 200 and fresh.get("/protected").status_code == 200
    reused = fresh.post("/api/admin/login/passkey/verify", json={"challenge_id": login["challenge_id"], "credential": credential})
    assert reused.status_code == 401

    # confirming a sensitive action with the passkey mints the same token the password flow does
    ask = fresh.post("/api/admin/confirm/passkey/options").json()
    done = fresh.post("/api/admin/confirm/passkey/verify", json={"challenge_id": ask["challenge_id"], "credential": credential, "action": "run_forecast_v1"})
    assert security.verify_confirm_token(done.json()["confirm_token"], "run_forecast_v1") == EMAIL


def test_a_failed_passkey_check_signs_nobody_in(client, monkeypatch):
    def reject(**kwargs):
        raise ValueError("bad signature")

    sign_in(client)
    monkeypatch.setattr("webauthn.verify_registration_response", lambda **k: SimpleNamespace(credential_id=b"c1", credential_public_key=b"k", sign_count=0))
    options = client.post("/api/admin/mfa/passkeys/options", json={"confirm_token": confirm_token()}).json()
    credential = {"id": bytes_to_base64url(b"c1"), "rawId": bytes_to_base64url(b"c1"), "response": {}}
    client.post("/api/admin/mfa/passkeys/verify", json={"confirm_token": confirm_token(), "challenge_id": options["challenge_id"], "credential": credential})
    monkeypatch.setattr("webauthn.verify_authentication_response", reject)
    fresh = TestClient(client.app)
    login = fresh.post("/api/admin/login/passkey/options").json()
    assert fresh.post("/api/admin/login/passkey/verify", json={"challenge_id": login["challenge_id"], "credential": credential}).status_code == 401


def test_one_passkey_domain_covers_production_preview_and_dev(monkeypatch):
    monkeypatch.setenv("WEBAUTHN_RP_ID", "showmefire.org")
    monkeypatch.delenv("WEBAUTHN_ORIGINS", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "production")
    rp_id, origins = mfa.webauthn_settings()
    assert rp_id == "showmefire.org"
    assert {"https://showmefire.org", "https://preview.showmefire.org", "https://dev.showmefire.org"} <= set(origins)
    monkeypatch.delenv("WEBAUTHN_RP_ID")
    with pytest.raises(Exception) as caught:
        mfa.webauthn_settings()
    assert getattr(caught.value, "status_code", None) == 503


# --------------------------------------------------------------------------- owner tools

def test_only_owners_manage_accounts_and_the_last_owner_is_protected(client, app):
    sign_in(client)
    owner_id = users.get_by_email(EMAIL)["id"]
    created = client.post("/api/admin/users", json={"email": "helper@example.org", "display_name": "Helper"}).json()
    assert created["user"]["must_change_password"] is True
    assert client.post("/api/admin/users", json={"email": "helper@example.org"}).status_code == 422

    helper = TestClient(app)
    assert sign_in(helper, email="helper@example.org", password=created["temporary_password"]).json()["must_change_password"] is True
    assert helper.get("/api/admin/users").status_code == 403

    assert client.patch(f"/api/admin/users/{owner_id}", json={"role": "admin"}).status_code == 422
    assert client.patch(f"/api/admin/users/{owner_id}", json={"disabled": True}).status_code == 422


def test_resetting_an_account_clears_its_factors_and_sessions(client, app):
    sign_in(client)
    created = client.post("/api/admin/users", json={"email": "helper@example.org"}).json()
    helper = TestClient(app)
    sign_in(helper, email="helper@example.org", password=created["temporary_password"])
    helper_id = created["user"]["id"]
    mfa.recovery_regenerate(helper_id)
    reset = client.post(f"/api/admin/users/{helper_id}/reset").json()
    assert helper.get("/protected").status_code == 401
    assert mfa.recovery_remaining(helper_id) == 0
    assert sign_in(TestClient(app), email="helper@example.org", password=reset["temporary_password"]).status_code == 200


def test_the_command_line_tool_resets_factors_and_sessions(client, app, monkeypatch, capsys):
    from scripts import admin_user

    sign_in(client)
    enrol_totp(client)
    assert sign_in(TestClient(app)).json().get("mfa_required") is True
    admin_user.main(["reset-mfa", EMAIL])
    admin_user.main(["revoke-sessions", EMAIL])
    assert client.get("/protected").status_code == 401
    assert sign_in(TestClient(app)).json()["success"] is True
    admin_user.main(["list"])
    assert EMAIL in capsys.readouterr().out


# --------------------------------------------------------------------------- roles

def test_every_role_signs_in_the_same_way_and_reaches_the_admin_area(client, app):
    sign_in(client)
    roles = client.get("/api/admin/roles").json()["roles"]
    assert [r["id"] for r in roles] == ["owner", "admin", "staff", "partner"]
    for role in ("admin", "staff", "partner"):
        created = client.post("/api/admin/users", json={"email": f"{role}@example.org", "role": role}).json()
        member = TestClient(app)
        signed = sign_in(member, email=f"{role}@example.org", password=created["temporary_password"]).json()
        assert signed["success"] is True and signed["role"] == role
        assert member.get("/protected").status_code == 200  # no role-specific access yet
        assert member.post("/api/admin/verify").json()["role"] == role
        assert member.get("/api/admin/me").json()["role"] == role
        assert member.get("/api/admin/users").status_code == 403  # only owners manage accounts


def test_unknown_roles_are_refused_and_roles_can_be_changed(client):
    sign_in(client)
    assert client.post("/api/admin/users", json={"email": "x@example.org", "role": "superuser"}).status_code == 422
    created = client.post("/api/admin/users", json={"email": "p@example.org", "role": "partner"}).json()["user"]
    assert client.patch(f"/api/admin/users/{created['id']}", json={"role": "staff"}).json()["role"] == "staff"
    assert client.patch(f"/api/admin/users/{created['id']}", json={"role": "nope"}).status_code == 422


def test_require_role_is_ready_for_endpoints_that_need_it(client, app):
    @app.get("/staff-only", dependencies=[Depends(access.require_role("owner", "staff"))])
    def staff_only():
        return {"ok": True}

    sign_in(client)
    partner = client.post("/api/admin/users", json={"email": "p@example.org", "role": "partner"}).json()
    staff = client.post("/api/admin/users", json={"email": "s@example.org", "role": "staff"}).json()
    as_partner, as_staff = TestClient(app), TestClient(app)
    sign_in(as_partner, email="p@example.org", password=partner["temporary_password"])
    sign_in(as_staff, email="s@example.org", password=staff["temporary_password"])
    assert client.get("/staff-only").status_code == 200
    assert as_staff.get("/staff-only").status_code == 200
    assert as_partner.get("/staff-only").status_code == 403
    assert TestClient(app).get("/staff-only").status_code == 401
    with pytest.raises(ValueError):
        access.require_role("owner", "janitor")


def test_the_last_owner_cannot_be_demoted_to_a_new_role(client):
    sign_in(client)
    owner_id = users.get_by_email(EMAIL)["id"]
    assert client.patch(f"/api/admin/users/{owner_id}", json={"role": "staff"}).status_code == 422
