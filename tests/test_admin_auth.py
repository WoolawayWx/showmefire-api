import pytest
from fastapi.testclient import TestClient

import main


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "ADMIN_EMAIL", "admin@example.com")
    monkeypatch.setattr(main, "ADMIN_PASSWORD_HASH", "hash")
    monkeypatch.setattr(main, "verify_password", lambda password, hashed: password == "correct")
    monkeypatch.setattr(main, "init_database", lambda: None)
    monkeypatch.setattr(main, "run_initial_fetches", lambda: _noop())
    with TestClient(main.app) as test_client:
        yield test_client


async def _noop():
    return None


def test_login_sets_httponly_access_and_refresh_cookies(client):
    response = client.post(
        "/api/admin/login",
        json={"email": "admin@example.com", "password": "correct"},
    )

    assert response.status_code == 200
    assert "admin_access=" in response.headers["set-cookie"]
    assert "admin_refresh=" in response.headers["set-cookie"]
    assert "HttpOnly" in response.headers["set-cookie"]

    verified = client.post("/api/admin/verify")
    assert verified.status_code == 200
    assert verified.json()["email"] == "admin@example.com"


def test_refresh_rotates_access_cookie_and_logout_clears_both(client):
    client.post(
        "/api/admin/login",
        json={"email": "admin@example.com", "password": "correct"},
    )

    refreshed = client.post("/api/admin/refresh")
    assert refreshed.status_code == 200
    assert "admin_access=" in refreshed.headers["set-cookie"]

    logged_out = client.post("/api/admin/logout")
    assert logged_out.status_code == 200
    assert "admin_access=" in logged_out.headers["set-cookie"]
    assert "admin_refresh=" in logged_out.headers["set-cookie"]
    assert client.post("/api/admin/verify").status_code == 401


def test_invalid_credentials_are_rejected_without_auth_cookies(client):
    response = client.post(
        "/api/admin/login",
        json={"email": "admin@example.com", "password": "wrong"},
    )

    assert response.status_code == 401
    assert "set-cookie" not in response.headers
