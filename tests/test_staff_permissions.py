"""Staff permissions, Bearer tokens, the reviewer lockdown and mobile refresh rotation.

Builds a small app around the real middlewares and auth router (importing `main` needs packages this environment lacks)."""
import time

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from core import security
from core.admin_auth import events, lockdown, permissions, sessions, users
from core.admin_session import AdminCookieMiddleware
from routers import admin_auth

OWNER = "owner@example.org"
PASSWORD = "correct horse battery 9"


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.delenv("ADMIN_MFA_ENFORCE", raising=False)
    monkeypatch.delenv("ADMIN_ALLOW_LEGACY_TOKENS", raising=False)
    monkeypatch.setattr(security, "ADMIN_EMAIL", OWNER)
    monkeypatch.setattr(security, "ADMIN_PASSWORD_HASH", security.hash_password(PASSWORD))
    monkeypatch.setattr(security, "AUTH_COOKIE_SECURE", False)
    sessions._forget()
    application = FastAPI()
    application.add_middleware(lockdown.ReviewerLockdownMiddleware)
    application.add_middleware(AdminCookieMiddleware)
    application.include_router(admin_auth.router)

    @application.get("/api/admin/website-only")
    def website_only(email: str = Depends(security.require_admin)):
        return {"email": email}

    @application.get("/api/admin/queue", dependencies=[Depends(permissions.require_permission("reports.view"))])
    def queue():
        return {"ok": True}

    @application.post("/api/admin/act")
    def act(email: str = Depends(permissions.require_permission("reports.moderate"))):
        return {"email": email}

    @application.get("/api/staff/v1/ping")
    def ping(email: str = Depends(permissions.require_permission("staff.access"))):
        return {"email": email}

    @application.get("/api/public")
    def public():
        return {"public": True}

    return application


def make_user(role, email=None, **extra):
    users.ensure_seeded()  # the owner first, so "the last active owner" rules see it
    email = email or f"{role}@example.org"
    expires = extra.pop("expires_at", time.time() + 3600 if role == "reviewer" else None)
    user = users.create_user(email, role.title(), role, PASSWORD, must_change=False, expires_at=expires, **extra)
    return users.get_by_id(user["id"])


def bearer(user, client="mobile"):
    session_id, refresh = sessions.create(user["id"], "127.0.0.1", "pytest", "mobile")
    token = permissions.access_token_for(user, session_id, client=client)
    return {"Authorization": f"Bearer {token}"}, session_id, refresh


# ------------------------------------------------------------------------------------------------- permission table

def test_the_permission_table_matches_the_decisions():
    p = permissions.permissions_for
    assert "accounts.manage" in p("owner") and "accounts.manage" not in p("admin")
    for role in ("owner", "admin", "staff", "reviewer"):
        assert {"reports.moderate", "burnbans.moderate"} <= p(role)
    assert not ({"reports.moderate", "burnbans.moderate"} & p("partner"))
    assert {"reports.view", "burnbans.view", "audit.view", "push.receive"} <= p("partner")
    assert "push.receive" not in p("reviewer")
    assert set(permissions.roles_with("push.receive")) == {"owner", "admin", "staff", "partner"}
    assert p("nobody") == frozenset()


def test_an_unknown_permission_name_is_refused_when_the_route_is_defined():
    with pytest.raises(ValueError):
        permissions.require_permission("reports.moderat")
    with pytest.raises(ValueError):
        permissions.requires("nope")


# ------------------------------------------------------------------------------------------------- bearer + checks

def test_a_bearer_token_works_like_the_cookie_and_other_bearer_values_are_ignored(app):
    headers, _, _ = bearer(make_user("staff"))
    client = TestClient(app)
    assert client.get("/api/admin/website-only", headers=headers).json() == {"email": "staff@example.org"}
    assert client.get("/api/admin/website-only").status_code == 401
    assert client.get("/api/admin/website-only", headers={"Authorization": "Bearer fld_not_a_jwt"}).status_code == 401
    assert client.get("/api/public", headers={"Authorization": "Bearer fld_not_a_jwt"}).status_code == 200
    # Bearer requests never set cookies
    assert "set-cookie" not in client.get("/api/admin/website-only", headers=headers).headers


def test_partners_can_look_but_not_act_and_everyone_else_can(app):
    client = TestClient(app)
    for role, can_act in (("owner", True), ("admin", True), ("staff", True), ("partner", False)):
        user = users.get_by_email(OWNER) if role == "owner" else make_user(role)
        headers, _, _ = bearer(user)
        assert client.get("/api/admin/queue", headers=headers).status_code == 200
        assert client.post("/api/admin/act", headers=headers).status_code == (200 if can_act else 403)
    assert client.post("/api/admin/act").status_code == 401


def test_a_disabled_account_is_refused_straight_away(app):
    user = make_user("staff")
    headers, _, _ = bearer(user)
    client = TestClient(app)
    assert client.post("/api/admin/act", headers=headers).status_code == 200
    users.set_disabled(user["id"], True)
    sessions._forget()
    assert client.post("/api/admin/act", headers=headers).status_code == 401


def test_a_role_change_ends_sessions_so_the_new_permissions_apply(app):
    owner = TestClient(app)
    assert owner.post("/api/admin/login", json={"email": OWNER, "password": PASSWORD}).json()["success"] is True
    created = owner.post("/api/admin/users", json={"email": "s@example.org", "role": "staff"}).json()
    staff = users.get_by_email("s@example.org")
    headers, session_id, _ = bearer(staff)
    client = TestClient(app)
    assert client.post("/api/admin/act", headers=headers).status_code == 200
    assert owner.patch(f"/api/admin/users/{created['user']['id']}", json={"role": "partner"}).status_code == 200
    sessions._forget()
    assert client.post("/api/admin/act", headers=headers).status_code == 401  # signed out; signs in again as a partner
    assert any(e["kind"] == "role_changed" for e in events.recent(users.get_by_email(OWNER)["id"]))


# ------------------------------------------------------------------------------------------------- reviewers

def test_a_reviewer_reaches_only_the_staff_api(app, monkeypatch):
    monkeypatch.setenv("SANDBOX_ENABLED", "true")  # staff API requests of a reviewer run in the sandbox (core/sandbox_data.py)
    headers, _, _ = bearer(make_user("reviewer"))
    client = TestClient(app)
    assert client.get("/api/staff/v1/ping", headers=headers).status_code == 200
    assert client.get("/api/admin/me", headers=headers).status_code == 200
    for path in ("/api/admin/website-only", "/api/admin/queue", "/api/admin/users", "/api/admin/sessions", "/api/admin/events"):
        assert client.get(path, headers=headers).status_code == 403, path
    assert client.post("/api/admin/act", headers=headers).status_code == 403
    assert client.get("/api/public", headers=headers).status_code == 403  # the app never sends the staff token to public routes


def test_other_roles_are_not_affected_by_the_lockdown(app):
    headers, _, _ = bearer(make_user("staff"))
    client = TestClient(app)
    assert client.get("/api/admin/website-only", headers=headers).status_code == 200
    assert client.get("/api/public", headers=headers).status_code == 200


def test_reviewers_sign_in_only_through_the_app_and_only_until_they_expire(app):
    make_user("reviewer", email="apple@example.org")
    assert users.authenticate("apple@example.org", PASSWORD) is None  # the website answers as for a wrong password
    assert users.authenticate("apple@example.org", PASSWORD, client="mobile")["role"] == "reviewer"
    assert TestClient(app).post("/api/admin/login", json={"email": "apple@example.org", "password": PASSWORD}).status_code == 401
    expired = make_user("reviewer", email="old@example.org", expires_at=time.time() - 5)
    assert users.authenticate("old@example.org", PASSWORD, client="mobile") is None
    headers, _, refresh = bearer(expired)
    assert TestClient(app).get("/api/staff/v1/ping", headers=headers).status_code == 401  # a session of an expired account is dead
    assert sessions.renew(refresh) is None


def test_reviewer_accounts_have_their_own_rules(app):
    with pytest.raises(users.AccountError):
        users.create_user("r@example.org", "R", "reviewer", PASSWORD)  # needs an expiry
    with pytest.raises(users.AccountError):
        users.create_user("s@example.org", "S", "staff", PASSWORD, expires_at=time.time() + 60)  # only reviewers expire
    staff = make_user("staff")
    with pytest.raises(users.AccountError):
        users.set_role(staff["id"], "reviewer")
    reviewer = make_user("reviewer")
    with pytest.raises(users.AccountError):
        users.set_role(reviewer["id"], "admin")
    later = time.time() + 86400
    users.set_expiry(reviewer["id"], later)
    assert users.get_by_id(reviewer["id"])["expires_at"] == later
    with pytest.raises(users.AccountError):
        users.set_expiry(staff["id"], later)


def test_the_normal_create_form_cannot_make_reviewers_and_the_role_list_hides_them(app):
    owner = TestClient(app)
    owner.post("/api/admin/login", json={"email": OWNER, "password": PASSWORD})
    assert owner.post("/api/admin/users", json={"email": "x@example.org", "role": "reviewer"}).status_code == 422
    assert "reviewer" not in [r["id"] for r in owner.get("/api/admin/roles").json()["roles"]]
    assert "reviewer" in [r["id"] for r in owner.get("/api/admin/roles?all=true").json()["roles"]]


# ------------------------------------------------------------------------------------------------- refresh rotation

def test_each_mobile_refresh_replaces_the_token(app):
    user = make_user("staff")
    _, session_id, refresh = bearer(user)
    email, sid, second = sessions.rotate(refresh)
    assert (email, sid) == (user["email"], session_id) and second != refresh
    _, _, third = sessions.rotate(second)
    assert third not in (refresh, second)


def test_a_lost_response_can_be_retried_within_the_grace_period(app):
    _, session_id, refresh = bearer(make_user("staff"))
    sessions.rotate(refresh)  # the app never received this answer
    again = sessions.rotate(refresh)  # so it asks again with the old token
    assert again is not None and again[1] == session_id
    assert sessions.is_active(session_id)


def test_replaying_an_old_token_after_the_grace_period_ends_the_session(app, monkeypatch):
    user = make_user("staff")
    _, session_id, refresh = bearer(user)
    _, _, current = sessions.rotate(refresh)
    monkeypatch.setattr(sessions, "ROTATION_GRACE_SECONDS", -1)
    assert sessions.rotate(refresh) is None  # a replay of the replaced token
    sessions._forget()
    assert not sessions.is_active(session_id)
    assert sessions.rotate(current) is None  # the thief's win is the owner's logout: nothing works any more
    assert any(e["kind"] == "refresh_token_reused" for e in events.recent(user["id"]))


def test_rotation_refuses_revoked_disabled_and_unknown_tokens(app):
    user = make_user("staff")
    _, session_id, refresh = bearer(user)
    assert sessions.rotate("not-a-token") is None and sessions.rotate("") is None
    sessions.revoke(session_id)
    assert sessions.rotate(refresh) is None
    _, other, token = bearer(user)
    users.set_disabled(user["id"], True)
    assert sessions.rotate(token) is None


# ------------------------------------------------------------------------------------------------- route report

def test_routes_without_a_permission_are_listed(app):
    missing = permissions.unguarded_admin_routes(app)
    assert ("GET", "/api/admin/website-only") in missing
    assert ("GET", "/api/admin/queue") not in missing  # declared through a dependency in the decorator
    assert ("POST", "/api/admin/act") not in missing

    @app.get("/api/admin/inline")
    @permissions.requires("burnbans.moderate")
    def inline(token: str = None):
        return permissions.authorize(token, "burnbans.moderate")

    assert ("GET", "/api/admin/inline") not in permissions.unguarded_admin_routes(app)
