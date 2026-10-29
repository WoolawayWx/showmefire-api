"""Staff push: who is eligible is decided at send time, payloads stay small, and Discord and staff push cannot break each other."""
import sqlite3
import time

import pytest

from core import database, sandbox, security
from core.admin_auth import db as auth_db
from core.admin_auth import permissions, sessions, users
from services import discord_notifier, staff_push

PASSWORD = "correct horse battery 9"
TOKEN_A = "ExponentPushToken[aaaaaaaaaaaaaaaaaaaaaa]"
TOKEN_B = "ExponentPushToken[bbbbbbbbbbbbbbbbbbbbbb]"
TOKEN_C = "ExponentPushToken[cccccccccccccccccccccc]"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("STAFF_PUSH_ENABLED", "true")
    monkeypatch.setattr(security, "ADMIN_EMAIL", "owner@example.org")
    monkeypatch.setattr(security, "ADMIN_PASSWORD_HASH", security.hash_password(PASSWORD))
    sessions._forget()
    staff_push._ready.clear()
    users.ensure_seeded()


def signed_in(role="staff", email=None, token=TOKEN_A, install="install-1", expires_at=None):
    user = users.create_user(email or f"{role}@example.org", role.title(), role, PASSWORD, must_change=False,
                             expires_at=expires_at or (time.time() + 3600 if role == "reviewer" else None))
    session_id, _ = sessions.create(user["id"], "127.0.0.1", "pytest", "mobile")
    assert staff_push.register_device(user["id"], session_id, install, token, "ios", "1.3.0")
    return user, session_id


def eligible_tokens():
    return sorted(device["expo_token"] for device in staff_push.eligible_devices())


# ------------------------------------------------------------------------------------------------- registration

def test_only_real_expo_tokens_register_and_one_phone_belongs_to_one_account(env):
    user, session_id = signed_in()
    assert not staff_push.register_device(user["id"], session_id, "i2", "not-a-token")
    assert not staff_push.register_device(user["id"], session_id, "", TOKEN_B)
    other = users.create_user("admin@example.org", "A", "admin", PASSWORD, must_change=False)
    other_session, _ = sessions.create(other["id"], "127.0.0.1", "pytest", "mobile")
    staff_push.register_device(other["id"], other_session, "install-1", TOKEN_A, "ios", "1.3.0")  # same phone, new account
    with auth_db.tx() as conn:
        rows = conn.execute("SELECT admin_id FROM staff_push_devices").fetchall()
    assert [row[0] for row in rows] == [other["id"]]


# ------------------------------------------------------------------------------------------------- access ends the push

def test_a_signed_in_staff_device_is_eligible(env):
    signed_in()
    assert eligible_tokens() == [TOKEN_A]


def test_signing_out_stops_pushes_and_removes_the_device(env):
    _, session_id = signed_in()
    sessions.revoke(session_id)
    assert eligible_tokens() == []
    with auth_db.tx() as conn:
        assert conn.execute("SELECT COUNT(*) FROM staff_push_devices").fetchone()[0] == 0


def test_signing_out_everywhere_or_by_refresh_token_stops_pushes(env):
    user, _ = signed_in()
    sessions.revoke_all(user["id"])
    assert eligible_tokens() == []
    user2, session2 = signed_in(email="two@example.org", token=TOKEN_B, install="install-2")
    with auth_db.tx() as conn:
        conn.execute("UPDATE admin_sessions SET revoked_at = NULL")
    _, refresh = sessions.create(user2["id"], "127.0.0.1", "pytest", "mobile")
    sessions.revoke_by_refresh(refresh)
    assert TOKEN_B in eligible_tokens()  # that was a different session than the one the device is bound to


def test_changing_your_password_elsewhere_keeps_the_current_phone(env):
    user, session_id = signed_in()
    other_session, _ = sessions.create(user["id"], "127.0.0.1", "pytest", "web")
    sessions.revoke_all(user["id"], except_session=session_id)
    assert eligible_tokens() == [TOKEN_A]


def test_disabling_an_account_stops_pushes_even_if_nothing_revoked_its_session(env):
    user, _ = signed_in()
    users.set_disabled(user["id"], True)
    assert eligible_tokens() == []


def test_an_expired_session_or_reviewer_account_gets_nothing(env):
    user, session_id = signed_in()
    with auth_db.tx() as conn:
        conn.execute("UPDATE admin_sessions SET last_seen_at = ? WHERE id = ?", (time.time() - sessions.IDLE_SECONDS - 10, session_id))
    assert eligible_tokens() == []
    reviewer, _ = signed_in("reviewer", token=TOKEN_B, install="install-2")
    assert eligible_tokens() == []  # reviewers never hold push.receive
    with auth_db.tx() as conn:
        conn.execute("UPDATE admin_users SET role = 'staff', expires_at = ? WHERE id = ?", (time.time() - 5, reviewer["id"]))
    assert eligible_tokens() == []  # an expired account, whatever its role


def test_a_role_that_loses_the_permission_stops_receiving_without_any_other_change(env, monkeypatch):
    signed_in("partner")
    assert eligible_tokens() == [TOKEN_A]
    monkeypatch.setitem(permissions.ROLE_PERMISSIONS, "partner", permissions.ROLE_PERMISSIONS["partner"] - {permissions.PUSH_RECEIVE})
    assert eligible_tokens() == []


# ------------------------------------------------------------------------------------------------- what is sent

def capture(monkeypatch, tickets=None):
    sent = []

    def fake(messages):
        sent.append(messages)
        return tickets(messages) if tickets else [{"status": "ok", "id": f"ticket-{i}-{len(sent)}"} for i, _ in enumerate(messages)]

    monkeypatch.setattr(staff_push, "_send_batch", fake)
    return sent


def test_a_push_carries_only_a_title_the_county_and_a_link(env, monkeypatch):
    signed_in()
    sent = capture(monkeypatch)
    alert = {"alert_type": "fire_report", "title": "New fire report: Phelps County", "admin_path": "/admin/fires/42",
             "description": "My neighbour Pat Person at 12 Elm St says the barn is burning, call 573-555-0100",
             "fields": [{"name": "Reporter", "value": "bystander"}]}
    assert staff_push.dispatch(alert, background=False) is True
    message = sent[0][0]
    assert message["to"] == TOKEN_A and message["channelId"] == "staff-alerts"
    assert (message["title"], message["body"]) == ("New fire report", "Phelps County")
    assert message["data"] == {"url": "/staff/reports/42", "eventType": "staff_alert", "kind": "fire_report"}
    text = str(message)
    for private in ("Pat", "Elm St", "573-555", "barn", "bystander"):
        assert private not in text


def test_only_moderation_alerts_become_pushes(env, monkeypatch):
    signed_in()
    sent = capture(monkeypatch)
    for alert_type in ("system_issue", "api_key_request", "form_response", "archive_ready", "site_feedback"):
        assert staff_push.dispatch({"alert_type": alert_type, "title": "x"}, background=False) is False
    assert sent == []
    assert staff_push.content_for({"alert_type": "incident_feedback", "title": "Fire incident feedback: Reynolds", "admin_path": "/admin/fires"}) == {
        "alert_type": "incident_feedback", "title": "New incident feedback", "body": "Reynolds", "url": "/staff/incidents"}
    assert staff_push.content_for({"alert_type": "burn_ban", "title": "Burn ban", "admin_path": "/admin/burn-bans"})["url"] == "/staff/burn-bans"


def test_it_is_off_unless_switched_on_and_never_runs_in_the_sandbox(env, monkeypatch):
    signed_in()
    sent = capture(monkeypatch)
    alert = {"alert_type": "fire_report", "title": "New fire report: Phelps County", "admin_path": "/admin/fires/1"}
    monkeypatch.setenv("STAFF_PUSH_ENABLED", "false")
    assert staff_push.dispatch(alert, background=False) is False
    monkeypatch.setenv("STAFF_PUSH_ENABLED", "true")
    with sandbox.sandbox_scope(sandbox.sandbox_db_path()):
        assert staff_push.dispatch(alert, background=False) is False
        assert staff_push.send("fire_report", "t", "b", "/staff/reports/1") == 0
    assert sent == []


def test_only_eligible_devices_are_sent_to_and_a_dead_device_is_forgotten(env, monkeypatch):
    signed_in()
    gone, gone_session = signed_in(email="gone@example.org", token=TOKEN_B, install="install-2")
    users.set_disabled(gone["id"], True)
    live, _ = signed_in(email="live@example.org", token=TOKEN_C, install="install-3")

    def tickets(messages):
        return [{"status": "error", "details": {"error": "DeviceNotRegistered"}} if m["to"] == TOKEN_C else {"status": "ok", "id": "t1"} for m in messages]

    sent = capture(monkeypatch, tickets)
    assert staff_push.send("fire_report", "New fire report", "Phelps County", "/staff/reports/1") == 1
    assert sorted(m["to"] for m in sent[0]) == [TOKEN_A, TOKEN_C]  # the disabled account's phone is not in it
    with auth_db.tx() as conn:
        assert {r[0] for r in conn.execute("SELECT expo_token FROM staff_push_devices")} == {TOKEN_A, TOKEN_B}  # C was pruned
        assert conn.execute("SELECT recipients, sent, failed FROM staff_push_events").fetchone()[:] == (2, 1, 0)


def test_the_same_alert_is_not_sent_twice(env, monkeypatch):
    signed_in()
    sent = capture(monkeypatch)
    assert staff_push.send("fire_report", "t", "b", "/staff/reports/1", event_key="same") == 1
    assert staff_push.send("fire_report", "t", "b", "/staff/reports/1", event_key="same") == 0
    assert len(sent) == 1


# ------------------------------------------------------------------------------------------------- failures stay separate

def test_a_staff_push_failure_cannot_break_the_discord_alert(env, monkeypatch):
    posted = []
    monkeypatch.setattr(discord_notifier, "build_staff_alert_payload", lambda **kw: {"event_type": "staff_alert", **kw})
    monkeypatch.setattr(discord_notifier, "_send_event", lambda payload: posted.append(payload) or True)

    def explode(alert, **kwargs):
        raise RuntimeError("expo is down")

    monkeypatch.setattr(staff_push, "dispatch", explode)
    assert discord_notifier.notify_staff_alert(background=False, alert_type="fire_report", title="t") is True
    assert len(posted) == 1


def test_a_discord_failure_cannot_stop_the_staff_push(env, monkeypatch):
    signed_in()
    sent = capture(monkeypatch)

    def explode(**kwargs):
        raise RuntimeError("discord is down")

    monkeypatch.setattr(discord_notifier, "build_staff_alert_payload", explode)
    monkeypatch.setattr(staff_push.threading, "Thread", lambda target, args, daemon: type("T", (), {"start": lambda self: target(*args)})())
    assert discord_notifier.notify_staff_alert(background=False, alert_type="fire_report", title="New fire report: Texas County",
                                               admin_path="/admin/fires/7") is False
    assert sent and sent[0][0]["data"]["url"] == "/staff/reports/7"


def test_staff_push_works_when_no_discord_channel_is_configured(env, monkeypatch):
    signed_in()
    sent = capture(monkeypatch)
    monkeypatch.setattr(discord_notifier, "get_discord_admin_settings", lambda: {})  # no staff channel at all
    monkeypatch.setattr(staff_push.threading, "Thread", lambda target, args, daemon: type("T", (), {"start": lambda self: target(*args)})())
    discord_notifier.notify_staff_alert(background=False, alert_type="burn_ban", title="Burn ban: Phelps", admin_path="/admin/burn-bans")
    assert len(sent) == 1


# ------------------------------------------------------------------------------------------------- upkeep

def test_receipts_drop_devices_expo_says_are_gone_and_old_records_are_purged(env, monkeypatch):
    signed_in()
    now = time.time()
    with auth_db.tx() as conn:
        device = conn.execute("SELECT id FROM staff_push_devices").fetchone()[0]
        conn.execute("INSERT INTO staff_push_tickets (ticket_id, device_id, event_key, created_at) VALUES ('old', ?, 'e', ?)", (device, now - 3600))
        conn.execute("INSERT INTO staff_push_tickets (ticket_id, device_id, event_key, created_at) VALUES ('new', ?, 'e', ?)", (device, now))

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": {"old": {"status": "error", "details": {"error": "DeviceNotRegistered"}}}}

    class Client:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

        def post(self, url, headers=None, json=None):
            assert json == {"ids": ["old"]}  # only tickets older than 15 minutes are asked about
            return Response()

    monkeypatch.setattr(staff_push.httpx, "Client", Client)
    assert staff_push.check_receipts() == 1
    with auth_db.tx() as conn:
        assert conn.execute("SELECT COUNT(*) FROM staff_push_devices").fetchone()[0] == 0
        assert conn.execute("SELECT status, error FROM staff_push_tickets WHERE ticket_id = 'old'").fetchone()[:] == ("error", "DeviceNotRegistered")
        conn.execute("UPDATE staff_push_tickets SET created_at = ? WHERE ticket_id = 'old'", (now - 9 * 86400,))
    assert staff_push.purge() >= 1
    with auth_db.tx() as conn:
        assert [r[0] for r in conn.execute("SELECT ticket_id FROM staff_push_tickets")] == ["new"]
