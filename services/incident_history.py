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

Kinds: first_detected, detections, confidence, feedback, report, review, official.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable

from core.database import get_db_path

logger = logging.getLogger(__name__)

ACTIVE_WINDOW_DAYS = 14
DETECTION_STEP = 3                      # record a count change once it grows by this many...
DETECTION_MIN_INTERVAL = timedelta(minutes=30)  # ...or when this long has passed since the last note
CONFIDENCE_POINT_STEP = 10              # a score move this large is worth a line even inside one label
PUBLIC_LIMIT = 100

FEEDBACK_SUMMARY = {
    "confirmed_fire": "Reviewed public feedback: confirmed as a fire",
    "controlled_burn": "Reviewed public feedback: reported as a controlled burn",
    "not_a_fire": "Reviewed public feedback: reported as not a fire",
    "unsure": "Reviewed public feedback: not sure what it is",
}


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
        entries.append({"at": row["occurred_at"], "kind": row["kind"], "summary": row["summary"], "data": data})
    return entries


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

    if record(connection, incident_id, "first_detected", "First satellite detection",
              occurred_at=incident.get("first_detected_at"), dedupe_key="first"):
        written += 1

    # Detection count: note growth, not every single new pixel.
    count = int(incident.get("detection_count") or 0)
    previous = state.get("detections")
    last_noted = state.get("detections_noted_at")
    if previous is None:
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
    current = _confidence(incident, members)
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

    # Reviewed public feedback, each at the time it was reviewed (never notes or contact details).
    for row in connection.execute(
        "SELECT id, classification, COALESCE(reviewed_at, created_at) AS at FROM fire_incident_feedback "
        "WHERE incident_id = ? AND status = 'approved'", (incident_id,),
    ).fetchall():
        summary = FEEDBACK_SUMMARY.get(row["classification"])
        if summary and record(connection, incident_id, "feedback", summary, occurred_at=row["at"],
                              data={"classification": row["classification"]}, dedupe_key=f"feedback:{row['id']}"):
            written += 1

    # Approved public fire reports that joined the incident.
    for member in members:
        if member.get("source") == "user_submission" and member.get("status") == "approved":
            if record(connection, incident_id, "report", "A public fire report was added",
                      occurred_at=member.get("occurred_at"), dedupe_key=f"report:{member['id']}"):
                written += 1

    if incident.get("review_decision") == "approved" and incident.get("review_decided_at"):
        label = {"wildfire": "wildfire", "prescribed": "prescribed burn"}.get(incident.get("review_label") or "", "")
        if record(connection, incident_id, "review",
                  f"Reviewed by Show Me Fire staff{f' and labelled a {label}' if label else ' and approved'}",
                  occurred_at=incident["review_decided_at"], dedupe_key=f"review:{incident['review_decided_at']}"):
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
