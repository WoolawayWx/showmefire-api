"""Sign-in records hold IP addresses and device descriptions, so they are deleted on a schedule the privacy policy can state."""
import time

import pytest

from core import security
from core.admin_auth import db, events, sessions, users

PASSWORD = "correct horse battery 9"
DAY = 86400


@pytest.fixture
def account(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(security, "ADMIN_EMAIL", "owner@example.org")
    monkeypatch.setattr(security, "ADMIN_PASSWORD_HASH", security.hash_password(PASSWORD))
    sessions._forget()
    users.ensure_seeded()
    return users.get_by_email("owner@example.org")


def count(table):
    with db.tx() as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def test_sessions_that_ended_long_ago_are_deleted_and_live_or_recent_ones_are_kept(account):
    live, _ = sessions.create(account["id"], "203.0.113.1", "live phone", "mobile")
    recently_revoked, _ = sessions.create(account["id"], "203.0.113.2", "recent", "mobile")
    long_revoked, _ = sessions.create(account["id"], "203.0.113.3", "old revoked", "mobile")
    long_expired, _ = sessions.create(account["id"], "203.0.113.4", "old expired", "mobile")
    now = time.time()
    with db.tx() as conn:
        conn.execute("UPDATE admin_sessions SET revoked_at = ? WHERE id = ?", (now - 5 * DAY, recently_revoked))
        conn.execute("UPDATE admin_sessions SET revoked_at = ? WHERE id = ?", (now - 120 * DAY, long_revoked))
        conn.execute("UPDATE admin_sessions SET expires_at = ? WHERE id = ?", (now - 100 * DAY, long_expired))
    assert sessions.purge_old() == 2
    with db.tx() as conn:
        left = {row[0] for row in conn.execute("SELECT id FROM admin_sessions")}
    assert left == {live, recently_revoked}


def test_events_with_ip_addresses_go_after_a_year(account):
    events.record("login_ok", account["id"], "203.0.113.9", "password")
    with db.tx() as conn:
        conn.execute("INSERT INTO admin_auth_events (admin_id, kind, ip, detail, ts) VALUES (?, 'login_fail', '198.51.100.7', '', ?)",
                     (account["id"], time.time() - 400 * DAY))
    assert events.purge() == 1
    assert [e["kind"] for e in events.recent(account["id"])] == ["login_ok"]


def test_the_scheduled_job_runs_both_and_reports_counts(account):
    sid, _ = sessions.create(account["id"], "203.0.113.5", "phone", "mobile")
    sessions.revoke(sid)
    with db.tx() as conn:
        conn.execute("UPDATE admin_sessions SET revoked_at = ?", (time.time() - 200 * DAY,))
        conn.execute("INSERT INTO admin_auth_events (admin_id, kind, ip, detail, ts) VALUES (?, 'login_ok', '198.51.100.8', '', ?)",
                     (account["id"], time.time() - 500 * DAY))
    assert events.purge_old_records() == {"sessions": 1, "events": 1}
    assert events.purge_old_records() == {"sessions": 0, "events": 0}
    assert count("admin_users") == 1  # accounts themselves are not touched


def test_the_moderation_log_is_not_part_of_the_retention_job(account, tmp_path):
    """Moderation history is kept permanently (append-only), and the policy says so."""
    from services import audit

    with db.tx() as conn:
        audit.ensure_table(conn)
        audit.record(conn, action="site_feedback.resolved", subject_type="site_feedback", subject_id=1, actor="a@example.org")
    events.purge_old_records()
    with db.tx() as conn:
        assert conn.execute("SELECT COUNT(*) FROM moderation_audit").fetchone()[0] == 1
