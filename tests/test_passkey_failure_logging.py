"""A passkey the server rejects is explained in the server log (not to the browser), so a wrong site address can be found."""
import logging

import pytest
from fastapi import HTTPException

from core import security
from core.admin_auth import mfa, sessions, users

PASSWORD = "correct horse battery 9"


@pytest.fixture
def user(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("WEBAUTHN_RP_ID", "example.org")
    monkeypatch.setenv("WEBAUTHN_ORIGINS", "https://example.org")
    monkeypatch.setattr(security, "ADMIN_EMAIL", "owner@example.org")
    monkeypatch.setattr(security, "ADMIN_PASSWORD_HASH", security.hash_password(PASSWORD))
    sessions._forget()
    users.ensure_seeded()
    return users.get_by_email("owner@example.org")


def test_a_rejected_registration_tells_the_browser_little_and_the_log_why(user, caplog):
    options = mfa.passkey_register_options(user)
    challenge_id = options["challenge_id"] if "challenge_id" in options else next(iter(options.values()))
    with caplog.at_level(logging.WARNING, logger="core.admin_auth.mfa"):
        with pytest.raises(HTTPException) as raised:
            mfa.passkey_register_verify(user, challenge_id, {"id": "x", "rawId": "x", "type": "public-key", "response": {}}, "Test")
    assert raised.value.status_code == 400 and raised.value.detail == "That passkey could not be verified."
    logged = " ".join(record.getMessage() for record in caplog.records)
    assert "was not accepted" in logged and "rp_id=example.org" in logged and "https://example.org" in logged
