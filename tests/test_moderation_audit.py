"""The moderation audit log: every moderation action leaves an append-only row with who, when, why, in the same transaction."""
import re
import sqlite3
import time
from pathlib import Path

import pytest

from core import database, security
from core.admin_auth import permissions, sessions, users
from services import audit

PASSWORD = "correct horse battery 9"


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(security, "ADMIN_EMAIL", "owner@example.org")
    monkeypatch.setattr(security, "ADMIN_PASSWORD_HASH", security.hash_password(PASSWORD))
    sessions._forget()
    database.init_database()
    return database.get_db_path()


def sql(path, statement, params=()):
    with sqlite3.connect(path) as conn:
        cursor = conn.execute(statement, params)
        return cursor.lastrowid


def add_event(path, source="user_submission", status="pending", incident_id=None):
    return sql(path, "INSERT INTO fire_events (source, status, latitude, longitude, occurred_at, description, incident_id) "
                     "VALUES (?, ?, 37, -92, '2026-10-01T12:00:00Z', 'Grass fire', ?)", (source, status, incident_id))


def add_incident(path):
    return sql(path, "INSERT INTO fire_incidents (centroid_latitude, centroid_longitude, first_detected_at, last_detected_at) "
                     "VALUES (37, -92, '2026-10-01T12:00:00Z', '2026-10-01T13:00:00Z')")


def entries(**kwargs):
    return audit.list_entries(**kwargs)


def test_approving_and_rejecting_a_report_is_recorded_with_who_and_why(db):
    first, second = add_event(db), add_event(db)
    database.set_fire_event_status(first, "approved", actor="staff@example.org", reason="photo matches", cause_category="wildfire")
    database.set_fire_event_status(second, "rejected", actor="staff@example.org", reason="duplicate of #1")
    approved = entries(subject_type="fire_event", subject_id=str(first))[0]
    assert (approved["action"], approved["actor_email"], approved["reason"]) == ("fire_event.approved", "staff@example.org", "photo matches")
    assert (approved["before"]["status"], approved["after"]["status"]) == ("pending", "approved")
    assert entries(subject_type="fire_event", subject_id=str(second))[0]["action"] == "fire_event.rejected"


def test_a_signed_in_actor_is_named_with_id_role_client_and_session(db, monkeypatch):
    users.ensure_seeded()
    staff = users.create_user("staff@example.org", "Staff", "staff", PASSWORD, must_change=False)
    session_id, _ = sessions.create(staff["id"], "127.0.0.1", "pytest", "mobile")
    token = permissions.access_token_for(users.get_by_id(staff["id"]), session_id, client="mobile")
    event = add_event(db)
    context = security.set_request_token(token)
    try:
        database.set_fire_event_status(event, "approved", actor="staff@example.org")
    finally:
        security.reset_request_token(context)
    row = entries(subject_type="fire_event")[0]
    assert (row["actor_admin_id"], row["actor_role"], row["client"], row["session_id"]) == (staff["id"], "staff", "mobile", session_id)


def test_the_log_cannot_be_changed_or_erased(db):
    database.set_fire_event_status(add_event(db), "approved", actor="a@example.org")
    with sqlite3.connect(db) as conn:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute("UPDATE moderation_audit SET reason = 'edited'")
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute("DELETE FROM moderation_audit")


def test_the_record_and_the_change_stand_or_fall_together(db):
    event = add_event(db)
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    database.record_fire_moderation(cursor, event, action="rejected", actor="a@example.org", from_status="pending", to_status="rejected")
    conn.rollback()  # the change did not happen
    conn.close()
    assert entries(subject_type="fire_event") == []


def test_feed_ingest_and_first_submission_are_not_moderation(db):
    event = add_event(db)
    with sqlite3.connect(db) as conn:
        cursor = conn.cursor()
        database.record_fire_moderation(cursor, event, action="submitted", actor="", to_status="pending")
        database.record_fire_moderation(cursor, event, action="ingested", actor="system:viirs_ingest", to_status="pending")
    assert entries() == []


def test_a_detection_inheriting_its_incidents_decision_is_a_system_action(db):
    event = add_event(db)
    with sqlite3.connect(db) as conn:
        database.record_fire_moderation(conn.cursor(), event, action="approved", actor="chief@example.org", from_status="pending",
                                        to_status="approved", reason="inherited incident decision (admin_bulk_action)")
    row = entries()[0]
    assert (row["client"], row["actor_email"]) == ("system", "chief@example.org")  # nobody's click; the admin who decided is in the reason's source


def test_incident_decisions_keep_every_decision_with_the_note(db):
    incident = add_incident(db)
    add_event(db, source="viirs", incident_id=incident)
    database.set_incident_review_decision(incident, "approved", "a@example.org", note="confirmed by the county", cause_category="wildfire")
    database.set_incident_review_decision(incident, "rejected", "b@example.org", reason="not_a_fire", note="turned out to be a flare")
    decisions = [e for e in entries(subject_type="fire_incident", subject_id=str(incident)) if e["action"].startswith("incident.")]
    assert [(e["action"], e["actor_email"], e["reason"]) for e in reversed(decisions)] == [
        ("incident.approved", "a@example.org", "confirmed by the county"),
        ("incident.rejected", "b@example.org", "turned out to be a flare"),
    ]
    assert decisions[0]["before"]["decision"] == "approved" and decisions[0]["after"]["code"] == "not_a_fire"
    assert all(e["incident_id"] == incident for e in entries(incident_id=incident))  # member rows are findable by incident too


def test_feedback_reviews_are_not_overwritten(db):
    incident = add_incident(db)
    feedback = sql(db, "INSERT INTO fire_incident_feedback (incident_id, classification, note) VALUES (?, 'controlled_burn', 'pasture')", (incident,))
    database.set_fire_incident_feedback_status(feedback, "approved", "a@example.org")
    database.set_fire_incident_feedback_status(feedback, "rejected", "b@example.org", note="seen the video, it is a wildfire")
    history = audit.history("incident_feedback", feedback)
    assert [(e["actor_email"], e["after"]["status"]) for e in history] == [("a@example.org", "approved"), ("b@example.org", "rejected")]
    assert history[1]["reason"] == "seen the video, it is a wildfire" and history[1]["incident_id"] == incident
    assert database.get_db_path()  # the feedback row itself keeps only the latest reviewer, as before


def test_not_a_fire_and_incident_deletion_have_actor_and_reason(db):
    incident = add_incident(db)
    event = add_event(db, source="viirs", incident_id=incident)
    due = database.schedule_incident_deletion(incident, actor="a@example.org")
    result = database.delete_fire_incident_and_members(incident, actor="a@example.org", reason="bogus cluster")
    assert result["incident_deleted"] and due
    by_action = {e["action"]: e for e in entries(subject_type="fire_incident", subject_id=str(incident))}
    assert by_action["incident.deletion_scheduled"]["after"]["delete_after"] == due
    assert (by_action["incident.deleted"]["reason"], by_action["incident.deleted"]["actor_email"]) == ("bogus cluster", "a@example.org")
    assert entries(subject_type="fire_event", subject_id=str(event))[0]["action"] == "fire_event.deleted"


def test_the_scheduled_purge_is_attributed_to_the_system(db):
    incident = add_incident(db)
    sql(db, "UPDATE fire_incidents SET delete_after = datetime('now', '-1 day') WHERE id = ?", (incident,))
    assert database.purge_due_fire_incidents() == 1
    deleted = [e for e in entries(subject_type="fire_incident") if e["action"] == "incident.deleted"][0]
    assert (deleted["client"], deleted["actor_email"]) == ("system", "system:not_a_fire")


def test_burn_ban_history_survives_the_hard_delete(db):
    ban = sql(db, "INSERT INTO burn_ban_submissions (county_fips, county_name, submitter_name, effective_at, upload_token_hash, submitter_ip_hash) "
                  "VALUES ('29161', 'Phelps', 'Pat', '2026-10-20T00:00:00', 'secret-token', 'iphash')")
    database.moderate_burn_ban_submission(ban, to_status="confirmed", actor="a@example.org", reason="county order seen")
    assert database.delete_burn_ban_submission(ban, actor="b@example.org", reason="entered in error")
    history = audit.history("burn_ban", ban)
    assert [e["action"] for e in history] == ["burn_ban.confirmed", "burn_ban.delete"]
    snapshot = history[-1]["before"]["submission"]
    assert snapshot["county_name"] == "Phelps" and snapshot["status"] == "confirmed"
    assert [h["action"] for h in snapshot["moderation_history"]] == ["confirmed"]
    assert "upload_token_hash" not in snapshot and "submitter_ip_hash" not in snapshot  # no secrets in the log
    assert history[-1]["reason"] == "entered in error"
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM burn_ban_submissions").fetchone()[0] == 0


def test_site_feedback_and_recurring_sources_are_recorded(db):
    feedback = sql(db, "INSERT INTO feedback (category, message, status) VALUES ('bug', 'it broke', 'new')")
    database.update_feedback_status(feedback, "resolved")
    source = sql(db, "INSERT INTO recurring_fire_sources (latitude, longitude, status) VALUES (37, -92, 'candidate')")
    database.set_recurring_source_status(source, "confirmed", reviewed_by="a@example.org", label="sawmill")
    assert audit.history("site_feedback", feedback)[0]["after"]["status"] == "resolved"
    recurring = audit.history("recurring_source", source)[0]
    assert (recurring["actor_email"], recurring["after"]["label"]) == ("a@example.org", "sawmill")


def test_filters_and_pagination(db):
    for _ in range(3):
        database.set_fire_event_status(add_event(db), "approved", actor="a@example.org")
    database.set_fire_event_status(add_event(db), "rejected", actor="b@example.org", reason="spam")
    assert len(entries(actor="a@example.org")) == 3 and len(entries(actor="b@example.org")) == 1
    page = entries(limit=2)
    assert len(page) == 2 and page[0]["id"] > page[1]["id"]
    assert len(entries(before_id=page[-1]["id"])) == 2
    assert entries(since=time.time() + 60) == []
    assert entries(subject_type="burn_ban") == []


def test_no_moderation_write_bypasses_the_audit_log():
    """The per-item history tables are only ever written by the two helpers that also write the audit log."""
    source = (Path(__file__).resolve().parents[1] / "core" / "database.py").read_text()
    assert len(re.findall(r"INSERT INTO fire_event_moderation", source)) == 1
    assert len(re.findall(r"INSERT INTO burn_ban_moderation", source)) == 1
    for helper in ("def record_fire_moderation", "def _record_burn_ban_moderation"):
        body = source.split(helper, 1)[1].split("\ndef ", 1)[0]
        assert "audit.record(" in body, helper
    for other in Path(__file__).resolve().parents[1].glob("routers/*.py"):
        assert "INSERT INTO fire_event_moderation" not in other.read_text(), other
        assert "INSERT INTO burn_ban_moderation" not in other.read_text(), other
