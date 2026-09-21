"""Tests for the password re-confirmation step used to gate manual
forecast runs and model activate/rollback (see routers/admin_confirm.py).

Builds a minimal FastAPI app around just the routers under test instead of
importing `main` - main.py pulls in routers/tiles.py, which needs
`rio_tiler`, a dependency missing from this environment (pre-existing,
unrelated to this feature; see test_admin_auth.py's own `import main`
which hits the same wall).
"""
from datetime import timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core import security
from core.security import create_access_token, create_confirm_token, verify_confirm_token
from routers import admin_confirm, forecast_admin, forecast_admin_09z, forecast_v1_admin, model_admin


# --- unit tests: the token primitive itself ---------------------------------

def test_confirm_token_round_trips_for_its_own_action():
    token = create_confirm_token("admin@example.com", "run_forecast_09z")
    assert verify_confirm_token(token, "run_forecast_09z") == "admin@example.com"


def test_confirm_token_rejected_for_a_different_action():
    token = create_confirm_token("admin@example.com", "run_forecast_09z")
    assert verify_confirm_token(token, "activate_model:fire_weather_index") is None


def test_confirm_token_rejected_when_expired():
    token = create_access_token(
        {"sub": "admin@example.com", "type": "admin_confirm", "action": "run_forecast_09z"},
        timedelta(minutes=-1),
    )
    assert verify_confirm_token(token, "run_forecast_09z") is None


def test_missing_or_garbage_token_rejected():
    assert verify_confirm_token(None, "run_forecast_09z") is None
    assert verify_confirm_token("not-a-jwt", "run_forecast_09z") is None


# --- endpoint tests -----------------------------------------------------

@pytest.fixture
def app(monkeypatch):
    monkeypatch.setattr(security, "ADMIN_PASSWORD_HASH", "irrelevant-hash")
    monkeypatch.setattr(admin_confirm, "verify_password", lambda plain, hashed: plain == "correct-password")

    fastapi_app = FastAPI()
    fastapi_app.include_router(admin_confirm.router)
    fastapi_app.include_router(forecast_admin.router)
    fastapi_app.include_router(forecast_admin_09z.router)
    fastapi_app.include_router(forecast_v1_admin.router)
    fastapi_app.include_router(model_admin.router)
    return fastapi_app


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def admin_token():
    return create_access_token({"sub": "admin@example.com"})


def test_verify_password_rejects_wrong_password(client, admin_token):
    response = client.post(
        "/api/admin/verify-password",
        params={"token": admin_token},
        json={"password": "wrong", "action": "run_forecast_09z"},
    )
    assert response.status_code == 401


def test_verify_password_accepts_correct_password_and_mints_action_scoped_token(client, admin_token):
    response = client.post(
        "/api/admin/verify-password",
        params={"token": admin_token},
        json={"password": "correct-password", "action": "run_forecast_09z"},
    )
    assert response.status_code == 200
    confirm_token = response.json()["confirm_token"]
    assert verify_confirm_token(confirm_token, "run_forecast_09z") == "admin@example.com"
    # Minted for one action, can't be replayed against another endpoint's guard.
    assert verify_confirm_token(confirm_token, "run_forecast_testbed") is None


def test_verify_password_requires_an_existing_admin_session(client):
    response = client.post(
        "/api/admin/verify-password",
        json={"password": "correct-password", "action": "run_forecast_09z"},
    )
    assert response.status_code == 401


@pytest.mark.parametrize("path,action", [
    ("/api/admin/testbed/forecast/run", "run_forecast_testbed"),
    ("/api/admin/forecast-09z/run", "run_forecast_09z"),
    ("/api/admin/forecast-v1/run", "run_forecast_v1"),
])
def test_run_endpoint_rejects_missing_confirm_token(client, admin_token, path, action):
    response = client.post(path, params={"token": admin_token}, json={"confirm_token": ""})
    assert response.status_code == 401


@pytest.mark.parametrize("path,action", [
    ("/api/admin/testbed/forecast/run", "run_forecast_testbed"),
    ("/api/admin/forecast-09z/run", "run_forecast_09z"),
    ("/api/admin/forecast-v1/run", "run_forecast_v1"),
])
def test_run_endpoint_rejects_a_confirm_token_minted_for_a_different_action(client, admin_token, path, action):
    wrong_token = create_confirm_token("admin@example.com", "some_other_action")
    response = client.post(path, params={"token": admin_token}, json={"confirm_token": wrong_token})
    assert response.status_code == 401


def test_09z_run_endpoint_accepts_a_valid_confirm_token(client, admin_token, monkeypatch):
    monkeypatch.setattr(forecast_admin_09z, "trigger_09z_forecast", lambda email: {"status": "queued"})
    confirm_token = create_confirm_token("admin@example.com", "run_forecast_09z")
    response = client.post(
        "/api/admin/forecast-09z/run",
        params={"token": admin_token},
        json={"confirm_token": confirm_token},
    )
    assert response.status_code == 202
    assert response.json() == {"status": "queued"}


@pytest.mark.parametrize("path,make_body", [
    ("/api/admin/models/fire_weather_index/activate", lambda t: {"version": "0.0.1-beta.1", "confirm_token": t}),
    ("/api/admin/models/fire_weather_index/rollback", lambda t: {"confirm_token": t}),
])
def test_model_action_rejects_wrong_action_confirm_token(client, admin_token, path, make_body):
    wrong_token = create_confirm_token("admin@example.com", "run_forecast_09z")
    response = client.post(path, params={"token": admin_token}, json=make_body(wrong_token))
    assert response.status_code == 401


def test_model_activate_accepts_a_valid_confirm_token_and_proceeds_past_the_guard(client, admin_token):
    # Family isn't registered/importable in this minimal app, so a valid
    # confirm token should get past the password guard and hit the
    # "unknown model family" branch instead of the 401 guard.
    confirm_token = create_confirm_token("admin@example.com", "activate_model:not_a_real_family")
    response = client.post(
        "/api/admin/models/not_a_real_family/activate",
        params={"token": admin_token},
        json={"version": "1.0.0", "confirm_token": confirm_token},
    )
    assert response.status_code == 404
    assert "Unknown model family" in response.json()["detail"]
