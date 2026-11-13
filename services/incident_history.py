"""A public timeline for each fire incident: how it progressed.

Entries are written by `sync_incident_history` (run by the scheduler right after
the incident confidence shapes refresh) and by services/official_incidents.py
(agency listings). Rather than hooking every code path that can change an
incident, the sync compares each active incident with what it last recorded and
writes the differences, so it also fills in history that predates it (feedback,
reports and reviews carry their own timestamps and are de-duplicated by key).

What is public here is deliberately limited to facts about the incident: counts,
confidence, reviewed public feedback by category (never notes or contact
details), approved report arrivals, staff review, official listings. Unreviewed
feedback is never shown.

Kinds: first_detected, detections, confidence, feedback, report, osint, review, official, dispatch, notification.
Feedback, report and osint entries carry a `status` (pending, approved, rejected) that is updated in place as
staff review them; the app colours them yellow, green and red. Everything else is a plain event.

Every entry also carries a `marker` that says how to draw it, decided here so the app and the website cannot disagree
(see `assign_markers`): blue before staff confirm the incident, green after.

Feedback shows what it was submitted as (a fire, a controlled burn, not a fire, not sure) and its review
status; the comment text and any contact details are never shown.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable


logger = logging.getLogger(__name__)


def get_db_path():
    """Resolved at call time so tests (and anything else) that patch core.database.get_db_path also reach this module."""
    from core import database

    return database.get_db_path()

SATELLITE_SOURCES = {"modis", "viirs", "ngfs"}
ACTIVE_WINDOW_DAYS = 14
DETECTION_STEP = 3                      # record a count change once it grows by this many...
DETECTION_MIN_INTERVAL = timedelta(minutes=30)  # ...or when this long has passed since the last note
CONFIDENCE_POINT_STEP = 10              # a score move this large is worth a line even inside one label
PUBLIC_LIMIT = 100

FEEDBACK_AS = {
    "confirmed_fire": "a fire",
    "controlled_burn": "a controlled burn",
    "not_a_fire": "not a fire",
    "unsure": "not sure",
}
FEEDBACK_OUTCOME = {"pending": "awaiting review", "approved": "approved", "rejected": "not accepted"}


def feedback_summary(classification: str, status: str) -> str:
    """"Public feedback submitted as a controlled burn, awaiting review" (the comment is never included)."""
    return f"Public feedback submitted as {FEEDBACK_AS.get(classification, 'something else')}, {FEEDBACK_OUTCOME[status]}"


REPORT_SUMMARY = {
    "pending": "A public fire report was submitted, awaiting review",
    "approved": "A public fire report was added",
    "rejected": "A public fire report was reviewed and not accepted",
}
# Reports that arrived through an ingest feed (OSINT / online monitoring). Only the fact that one arrived is public: never
# the post text, link, author or which feed it came from.
OSINT_SUMMARY = {
    "pending": "An online report was received, awaiting review",
    "approved": "An online report was added",
    "rejected": "An online report was reviewed and not accepted",
}
STATUSES = ("pending", "approved", "rejected")


def _utc_iso(value: Any = None) -> str:
    if isinstance(value, datetime):
        moment = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    text = str(value or "").strip()
    if text:
        # SQLite CURRENT_TIMESTAMP values look like "2026-11-12 18:03:00".
        if "T" not in text and " " in text:
            text = text.replace(" ", "T", 1)
        if not text.endswith("Z") and "+" not in text[10:] and "-" not in text[10:]:
            text += "Z"
        return text
    return _utc_iso(datetime.now(timezone.utc))


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(get_db_path(), timeout=15)
    connection.row_factory = sqlite3.Row
    return connection


def record(
    connection: sqlite3.Connection, incident_id: int, kind: str, summary: str, *,
    occurred_at: Any = None, data: dict | None = None, dedupe_key: str | None = None,
) -> bool:
    """Add one entry. Returns False when `dedupe_key` was already recorded for this incident."""
    cursor = connection.execute(
        "INSERT OR IGNORE INTO fire_incident_history (incident_id, occurred_at, kind, summary, data_json, dedupe_key) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (incident_id, _utc_iso(occurred_at), kind, summary, json.dumps(data or {}, separators=(",", ":")), dedupe_key),
    )
    return cursor.rowcount == 1


def upsert_status_entry(
    connection: sqlite3.Connection, incident_id: int, kind: str, summary: str, *, occurred_at: Any, status: str,
    dedupe_key: str, data: dict | None = None,
) -> bool:
    """One entry per feedback/report whose summary and status follow its review. Returns True when something changed."""
    payload = json.dumps({**(data or {}), "status": status}, separators=(",", ":"))
    existing = connection.execute(
        "SELECT id, summary, data_json FROM fire_incident_history WHERE incident_id = ? AND dedupe_key = ?",
        (incident_id, dedupe_key),
    ).fetchone()
    if existing is None:
        return record(connection, incident_id, kind, summary, occurred_at=occurred_at,
                      data={**(data or {}), "status": status}, dedupe_key=dedupe_key)
    if existing["summary"] == summary and existing["data_json"] == payload:
        return False
    connection.execute("UPDATE fire_incident_history SET summary = ?, data_json = ? WHERE id = ?",
                       (summary, payload, existing["id"]))
    return True


def _moment(value: Any) -> datetime:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except (TypeError, ValueError):
        return datetime.min.replace(tzinfo=timezone.utc)


MARKERS = ("first", "public_report", "update", "confirmed", "update_confirmed", "rejected")


def assign_markers(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add `marker` to each entry (in place, returned for convenience).

    first             blue marker with "!": the incident's first entry (its first satellite detection, or the dispatch /
                      agency record that started it)
    public_report     blue marker with a check: a public report, online report or feedback made before staff confirmed
    update            blue circle: any other status update before staff confirmed
    confirmed         green marker with a check: staff confirmed the incident
    update_confirmed  empty green circle: anything after staff confirmed
    rejected          red cross: a report or feedback that staff did not accept
    """
    if not entries:
        return entries
    ordered = sorted(entries, key=lambda entry: _moment(entry["at"]))
    confirmed_at = next((_moment(e["at"]) for e in ordered if e["kind"] == "review" and e["status"] == "approved"), None)
    first = next((e for e in ordered if e["kind"] == "first_detected"), ordered[0])
    for entry in entries:
        if entry["status"] == "rejected":
            entry["marker"] = "rejected"
        elif entry["kind"] == "review" and entry["status"] == "approved":
            entry["marker"] = "confirmed"
        elif entry is first:
            entry["marker"] = "first"
        elif confirmed_at is not None and _moment(entry["at"]) >= confirmed_at:
            entry["marker"] = "update_confirmed"
        elif entry["kind"] in ("report", "feedback", "osint"):
            entry["marker"] = "public_report"
        else:
            entry["marker"] = "update"
    return entries


def list_history(incident_id: int, limit: int = PUBLIC_LIMIT) -> list[dict[str, Any]]:
    """Newest first, in the shape the public API returns."""
    with _connect() as connection:
        rows = connection.execute(
            "SELECT id, occurred_at, kind, summary, data_json FROM fire_incident_history "
            "WHERE incident_id = ? ORDER BY occurred_at DESC, id DESC LIMIT ?",
            (incident_id, limit),
        ).fetchall()
    entries = []
    for row in rows:
        try:
            data = json.loads(row["data_json"] or "{}")
        except json.JSONDecodeError:
            data = {}
        entries.append({"at": row["occurred_at"], "kind": row["kind"], "summary": row["summary"],
                        "status": data.get("status") if data.get("status") in STATUSES else None, "data": data})
    return assign_markers(entries)


def _confidence(incident: dict[str, Any], members: list[dict[str, Any]]) -> tuple[int, str] | None:
    """Same score/label the map and the v2 API show (v1/v2 scorer dispatch)."""
    try:
        from services.incident_confidence import score_incident, use_v2

        if use_v2():
            scored = score_incident(incident, members)
            return int(scored["score"]), str(scored["label"])
        from services.fire_confidence import _features, _score

        score, label, _method = _score(_features(incident, members))
        return int(round(float(score) * 100)), str(label)
    except Exception:  # noqa: BLE001 - one unscorable incident must not stop the rest
        logger.exception("incident history: could not score incident %s", incident.get("id"))
        return None


def _sync_one(connection: sqlite3.Connection, incident: dict[str, Any], now: datetime) -> int:
    from core.database import list_fire_incident_members

    incident_id = int(incident["id"])
    written = 0
    state_row = connection.execute(
        "SELECT state_json FROM fire_incident_history_state WHERE incident_id = ?", (incident_id,)
    ).fetchone()
    state: dict[str, Any] = json.loads(state_row["state_json"]) if state_row else {}
    members = list_fire_incident_members(incident_id)

    # Satellite wording only once a satellite has actually seen it: an incident that began as a dispatch or agency
    # record has no satellite member yet (its own "Reported to dispatch" / "Listed by" entry is its first line).
    satellite_times = sorted(str(m.get("occurred_at") or "") for m in members
                             if m.get("source") in SATELLITE_SOURCES and m.get("occurred_at"))
    has_satellite = bool(satellite_times)
    if has_satellite and record(connection, incident_id, "first_detected", "First satellite detection",
                                occurred_at=satellite_times[0], dedupe_key="first"):
        written += 1

    # Detection count: note growth, not every single new pixel.
    count = int(incident.get("detection_count") or 0)
    previous = state.get("detections")
    last_noted = state.get("detections_noted_at")
    if not has_satellite:
        pass
    elif previous is None:
        state["detections"], state["detections_noted_at"] = count, _utc_iso(now)
    elif count != previous:
        waited = last_noted is None or now - datetime.fromisoformat(last_noted.replace("Z", "+00:00")) >= DETECTION_MIN_INTERVAL
        if count - previous >= DETECTION_STEP or waited:
            word = "grew" if count > previous else "changed"
            record(connection, incident_id, "detections", f"Satellite detections {word} from {previous} to {count}",
                   occurred_at=now, data={"from": previous, "to": count})
            state["detections"], state["detections_noted_at"] = count, _utc_iso(now)
            written += 1

    # Confidence: a label change, or a big enough score move.
    current = _confidence(incident, members) if has_satellite else None
    if current is not None:
        score, label = current
        before = state.get("confidence")
        if before is None:
            state["confidence"] = {"score": score, "label": label}
        elif label != before["label"] or abs(score - before["score"]) >= CONFIDENCE_POINT_STEP:
            record(connection, incident_id, "confidence",
                   f"Confidence {'rose' if score > before['score'] else 'fell'} from {before['label']} ({before['score']}%) "
                   f"to {label} ({score}%)",
                   occurred_at=now, data={"from": before, "to": {"score": score, "label": label}})
            state["confidence"] = {"score": score, "label": label}
            written += 1

    # Public feedback: one entry each, following its review status (never notes or contact details; the category
    # is only shown once staff approved it).
    for row in connection.execute(
        "SELECT id, classification, status, created_at FROM fire_incident_feedback WHERE incident_id = ?", (incident_id,),
    ).fetchall():
        status = row["status"] if row["status"] in STATUSES else "pending"
        summary = feedback_summary(row["classification"], status)
        data = {"classification": row["classification"]}
        if upsert_status_entry(connection, incident_id, "feedback", summary, occurred_at=row["created_at"],
                               status=status, dedupe_key=f"feedback:{row['id']}", data=data):
            written += 1

    # Public fire reports linked to the incident.
    for member in members:
        if member.get("source") == "user_submission":
            status = member.get("status") if member.get("status") in STATUSES else "pending"
            if upsert_status_entry(connection, incident_id, "report", REPORT_SUMMARY[status],
                                   occurred_at=member.get("occurred_at"), status=status, dedupe_key=f"report:{member['id']}"):
                written += 1

    # Online (OSINT) reports ingested from monitoring feeds and linked to the incident.
    for member in members:
        if member.get("origin_kind") == "osint_api" and member.get("source") != "user_submission":
            status = member.get("status") if member.get("status") in STATUSES else "pending"
            if upsert_status_entry(connection, incident_id, "osint", OSINT_SUMMARY[status],
                                   occurred_at=member.get("occurred_at"), status=status, dedupe_key=f"osint:{member['id']}"):
                written += 1

    if incident.get("review_decision") == "approved" and incident.get("review_decided_at"):
        # Staff approving an incident is a confirmation: green in the app. Updated in place so an entry written
        # before statuses existed picks the status up.
        label = {"wildfire": "a wildfire", "prescribed": "a prescribed burn"}.get(incident.get("review_label") or "", "a fire")
        if upsert_status_entry(connection, incident_id, "review", f"Confirmed by Show Me Fire staff as {label}",
                               occurred_at=incident["review_decided_at"], status="approved",
                               dedupe_key=f"review:{incident['review_decided_at']}"):
            written += 1

    connection.execute(
        "INSERT INTO fire_incident_history_state (incident_id, state_json, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP) "
        "ON CONFLICT(incident_id) DO UPDATE SET state_json = excluded.state_json, updated_at = CURRENT_TIMESTAMP",
        (incident_id, json.dumps(state, separators=(",", ":"))),
    )
    return written


def sync_incident_history(now: datetime | None = None) -> int:
    """Record what changed on every recently active incident. Returns the number of entries written. Never raises."""
    from core.database import list_fire_incidents

    now = now or datetime.now(timezone.utc)
    cutoff = _utc_iso(now - timedelta(days=ACTIVE_WINDOW_DAYS))
    written = 0
    try:
        incidents = [
            incident for incident in list_fire_incidents(limit=500)
            if incident.get("status") not in ("merged", "deleted") and str(incident.get("last_detected_at") or "") >= cutoff[:19]
        ]
        for incident in incidents:
            with _connect() as connection:
                try:
                    written += _sync_one(connection, incident, now)
                except Exception:  # noqa: BLE001
                    logger.exception("incident history: failed for incident %s", incident.get("id"))
    except Exception:  # noqa: BLE001
        logger.exception("incident history sync failed")
    return written
