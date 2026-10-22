"""Development-site access codes: custom codes, redeeming them, the signed tokens, revoking codes and ending sessions."""
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core import security
from core.security import require_admin
from routers import dev_access as module
from services.dev_access import store, tokens

SECRET = "s" * 40


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DEV_ACCESS_SECRET", SECRET)
    store.reset_for_tests()
    module.redeem_limiter.reset()
    module.reset_failures()
    app = FastAPI()
    app.include_router(module.router)
    app.include_router(module.admin_router)
    app.dependency_overrides[require_admin] = lambda: "admin-user"
    return SimpleNamespace(client=TestClient(app))


def make(env, code="Acme Fire 2026", **extra):
    response = env.client.post("/admin/dev-access-codes", json={"code": code, "label": "Acme FPD", **extra})
    assert response.status_code == 201, response.text
    return response.json()["code"]


def redeem(env, code="Acme Fire 2026"):
    return env.client.post("/v1/dev-access/redeem", json={"code": code}, headers={"user-agent": "pytest-browser"})


def validate(env, token):
    return env.client.post("/v1/dev-access/validate", json={"token": token}).json()


def test_a_custom_code_opens_a_session_and_is_matched_ignoring_case_and_spaces(env):
    code = make(env)
    assert code["code_hint"] == "Ac…26" and "code_hash" not in code
    response = redeem(env, "  acme   FIRE 2026 ")
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "Acme FPD"
    assert validate(env, body["token"])["valid"] is True
    sessions = env.client.get("/admin/dev-access-codes/sessions", params={"code_id": code["id"]}).json()["sessions"]
    assert len(sessions) == 1 and sessions[0]["user_agent"] == "pytest-browser" and sessions[0]["seen_count"] == 1
    listed = env.client.get("/admin/dev-access-codes").json()
    assert listed["codes"][0]["use_count"] == 1 and listed["codes"][0]["active_sessions"] == 1


def test_duplicate_and_bad_codes_are_refused(env):
    make(env)
    assert env.client.post("/admin/dev-access-codes", json={"code": "ACME FIRE 2026", "label": "x"}).status_code == 409
    assert env.client.post("/admin/dev-access-codes", json={"code": "abc", "label": "x"}).status_code == 422
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    assert env.client.post("/admin/dev-access-codes", json={"code": "later", "label": "x", "expires_at": past}).status_code == 422


def test_wrong_codes_fail_and_repeated_guessing_is_locked_out(env):
    make(env)
    for _ in range(module.MAX_FAILURES):
        assert redeem(env, "nope nope").status_code == 401
        module.redeem_limiter.reset()
    assert redeem(env).status_code == 429             # even the right code waits out the lockout


def test_revoking_a_code_ends_its_sessions_and_stops_new_uses(env):
    code = make(env)
    token = redeem(env).json()["token"]
    assert env.client.post(f"/admin/dev-access-codes/{code['id']}/revoke", json={}).status_code == 200
    assert validate(env, token) == {"valid": False, "reason": "ended"}
    assert redeem(env).status_code == 401
    assert env.client.post(f"/admin/dev-access-codes/{code['id']}/revoke", json={}).status_code == 404


def test_revoking_without_ending_sessions_still_closes_them(env):
    code = make(env)
    token = redeem(env).json()["token"]
    env.client.post(f"/admin/dev-access-codes/{code['id']}/revoke", json={"end_sessions": False})
    assert validate(env, token)["valid"] is False     # a session never outlives its code
    session = env.client.get("/admin/dev-access-codes/sessions").json()["sessions"][0]
    assert session["state"] == "code_revoked"


def test_one_session_can_be_ended_and_others_keep_working(env):
    make(env)
    first, second = redeem(env).json()["token"], redeem(env).json()["token"]
    sid = tokens.read(first)["sid"]
    assert env.client.post(f"/admin/dev-access-codes/sessions/{sid}/end").status_code == 200
    assert validate(env, first)["valid"] is False
    assert validate(env, second)["valid"] is True


def test_end_all_sessions_keeps_the_code_usable(env):
    code = make(env)
    token = redeem(env).json()["token"]
    assert env.client.post(f"/admin/dev-access-codes/{code['id']}/end-sessions").json()["ended"] == 1
    assert validate(env, token)["valid"] is False
    assert redeem(env).status_code == 200


def test_max_uses_stops_new_sessions_but_not_existing_ones(env):
    make(env, max_uses=1)
    token = redeem(env).json()["token"]
    assert redeem(env).status_code == 401
    assert validate(env, token)["valid"] is True
    assert env.client.get("/admin/dev-access-codes").json()["codes"][0]["state"] == "used_up"


def test_sessions_end_with_the_code_expiry(env):
    soon = datetime.now(timezone.utc) + timedelta(days=2)
    make(env, expires_at=soon.isoformat(), session_days=30)
    body = redeem(env).json()
    assert abs(tokens.read(body["token"])["exp"] - soon.timestamp()) < 2


def test_editing_a_code_can_clear_limits(env):
    code = make(env, max_uses=1)
    redeem(env)
    changed = env.client.patch(f"/admin/dev-access-codes/{code['id']}", json={"label": "Acme", "clear_max_uses": True}).json()["code"]
    assert changed["label"] == "Acme" and changed["max_uses"] is None and changed["active"]
    assert redeem(env).status_code == 200


def test_admin_can_sign_in_at_the_gate_without_a_code(env, monkeypatch):
    monkeypatch.setattr(security, "ADMIN_EMAIL", "boss@example.org")
    monkeypatch.setattr(security, "ADMIN_PASSWORD_HASH", security.hash_password("correct horse 1"))
    bad = env.client.post("/v1/dev-access/admin", json={"email": "boss@example.org", "password": "wrong"})
    assert bad.status_code == 401
    good = env.client.post("/v1/dev-access/admin", json={"email": "Boss@Example.org", "password": "correct horse 1"})
    assert good.status_code == 200 and validate(env, good.json()["token"])["valid"] is True
    session = env.client.get("/admin/dev-access-codes/sessions", params={"kind": "admin"}).json()["sessions"][0]
    assert session["admin_email"] == "boss@example.org"


def test_tokens_are_tamper_proof_and_expire(env):
    make(env)
    token = redeem(env).json()["token"]
    version, body, signature = token.split(".")
    assert tokens.read(f"{version}.{body}.{signature[:-2]}AA") is None
    assert tokens.read(token, now=time.time() + 400 * 86400) is None
    assert validate(env, "v1.junk.junk") == {"valid": False, "reason": "invalid"}


def test_logout_ends_the_session(env):
    make(env)
    token = redeem(env).json()["token"]
    env.client.post("/v1/dev-access/logout", json={"token": token})
    assert validate(env, token)["valid"] is False


def test_nothing_is_issued_without_a_secret(env, monkeypatch):
    make(env)
    monkeypatch.setenv("DEV_ACCESS_SECRET", "")
    assert redeem(env).status_code == 503
