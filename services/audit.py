"""The moderation audit log: who approved, rejected, edited or deleted what, when, from where and why.

It is one append-only table (`moderation_audit`, UPDATE and DELETE are blocked by triggers) written inside the same
transaction as the change it describes, so an action and its record stand or fall together. It sits beside the older
per-item history (`fire_event_moderation`, `burn_ban_moderation`), which keeps working; this adds what those lack: one
place to ask "what happened to this, and who did it", a stable admin id and role next to the email, whether the action came
from the website, the app or the reviewer sandbox, and history for the things that had none (incident decisions, feedback
review, not-a-fire, incident deletes, generic feedback, recurring sources).

`record()` is called from the two functions that write the per-item history (`record_fire_moderation` and
`_record_burn_ban_moderation`, which covers every fire-event and burn-ban change) and next to each of the other moderation
writes in core/database.py. The actor is whoever's token is on the current request; a scheduler job (no request) is recorded
as a system action under the name it passed.

Reading: `list_entries()` and `history()`, used by the staff API.
"""
from __future__ import annotations

import json
import sqlite3
import time
from typing import Any, Dict, List, Optional

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS moderation_audit (
           id INTEGER PRIMARY KEY AUTOINCREMENT,
           ts REAL NOT NULL,
           actor_admin_id INTEGER,
           actor_email TEXT NOT NULL DEFAULT '',
           actor_role TEXT NOT NULL DEFAULT '',
           client TEXT NOT NULL CHECK (client IN ('web', 'mobile', 'sandbox', 'system')),
           session_id TEXT,
           action TEXT NOT NULL,
           subject_type TEXT NOT NULL,
           subject_id TEXT NOT NULL,
           incident_id INTEGER,
           reason TEXT NOT NULL DEFAULT '',
           before_json TEXT,
           after_json TEXT
       )""",
    "CREATE INDEX IF NOT EXISTS idx_moderation_audit_subject ON moderation_audit(subject_type, subject_id, ts)",
    "CREATE INDEX IF NOT EXISTS idx_moderation_audit_actor ON moderation_audit(actor_admin_id, ts)",
    "CREATE INDEX IF NOT EXISTS idx_moderation_audit_incident ON moderation_audit(incident_id, ts)",
    "CREATE INDEX IF NOT EXISTS idx_moderation_audit_ts ON moderation_audit(ts)",
    """CREATE TRIGGER IF NOT EXISTS moderation_audit_no_update BEFORE UPDATE ON moderation_audit
       BEGIN SELECT RAISE(ABORT, 'moderation_audit is append-only'); END""",
    """CREATE TRIGGER IF NOT EXISTS moderation_audit_no_delete BEFORE DELETE ON moderation_audit
       BEGIN SELECT RAISE(ABORT, 'moderation_audit is append-only'); END""",
]

SUBJECT_TYPES = ("fire_event", "fire_incident", "incident_feedback", "burn_ban", "site_feedback", "recurring_source")
MAX_JSON_CHARS = 40000


def ensure_table(cursor: Any) -> None:
    """Create the table and its triggers if missing. `cursor` may be a cursor or a connection."""
    for statement in SCHEMA:
        cursor.execute(statement)


def _json(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = json.dumps(value, default=str, sort_keys=True)
    return text if len(text) <= MAX_JSON_CHARS else json.dumps({"truncated": True, "preview": text[:MAX_JSON_CHARS // 2]})


def _who(actor: str, system: bool = False) -> Dict[str, Any]:
    """The actor columns: the signed-in admin on this request if there is one, else the name the caller passed (a system job).
    `system=True` is for effects no person chose (a detection inheriting its incident's decision): never attributed to
    whoever's request happened to trigger it."""
    from core.admin_auth import permissions

    try:
        current = None if system else permissions.resolve_actor()
    except Exception:  # noqa: BLE001 - never lose the record because the actor could not be worked out
        current = None
    if current is not None:
        return {"admin_id": current.admin_id, "email": current.email, "role": current.role, "client": current.client,
                "session_id": current.session_id}
    text = (actor or "").strip()
    return {"admin_id": None, "email": text or "system", "role": "",
            "client": "system" if (system or not text or text.startswith("system:")) else "web", "session_id": None}


def record(cursor: Any, *, action: str, subject_type: str, subject_id: Any, reason: str = "", before: Any = None, after: Any = None,
           incident_id: Optional[int] = None, actor: str = "", system: bool = False) -> None:
    """Append one row inside the caller's transaction (the caller commits). `cursor` may be a cursor or a connection."""
    if subject_type not in SUBJECT_TYPES:
        raise ValueError(f"unknown audit subject type: {subject_type}")
    who = _who(actor, system)
    if incident_id is None and subject_type == "fire_event":
        row = cursor.execute("SELECT incident_id FROM fire_events WHERE id = ?", (subject_id,)).fetchone()
        incident_id = row[0] if row else None
    values = (time.time(), who["admin_id"], who["email"], who["role"], who["client"], who["session_id"], action, subject_type,
              str(subject_id), incident_id, (reason or "").strip(), _json(before), _json(after))
    sql = ("INSERT INTO moderation_audit (ts, actor_admin_id, actor_email, actor_role, client, session_id, action, subject_type, "
           "subject_id, incident_id, reason, before_json, after_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)")
    try:
        cursor.execute(sql, values)
    except sqlite3.OperationalError as error:
        if "no such table" not in str(error):
            raise
        ensure_table(cursor)  # a database built before the audit existed
        cursor.execute(sql, values)


def _row(row: sqlite3.Row) -> Dict[str, Any]:
    entry = {key: row[key] for key in row.keys()}
    for column in ("before_json", "after_json"):
        raw = entry.pop(column)
        entry[column[:-5]] = json.loads(raw) if raw else None
    return entry


def list_entries(*, subject_type: Optional[str] = None, subject_id: Optional[str] = None, actor: Optional[str] = None,
                 incident_id: Optional[int] = None, since: Optional[float] = None, until: Optional[float] = None,
                 before_id: Optional[int] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """Newest first. Cursor pagination: pass the last id seen as `before_id`."""
    from core.database import get_db_path

    clauses, params = [], []
    for column, value in (("subject_type", subject_type), ("subject_id", None if subject_id is None else str(subject_id)),
                          ("actor_email", actor), ("incident_id", incident_id)):
        if value is not None:
            clauses.append(f"{column} = ?")
            params.append(value)
    for clause, value in (("ts >= ?", since), ("ts < ?", until), ("id < ?", before_id)):
        if value is not None:
            clauses.append(clause)
            params.append(value)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    limit = max(1, min(int(limit), 200))
    with sqlite3.connect(get_db_path(), timeout=15) as conn:
        conn.row_factory = sqlite3.Row
        ensure_table(conn)
        rows = conn.execute(f"SELECT * FROM moderation_audit {where} ORDER BY id DESC LIMIT ?", (*params, limit)).fetchall()
    return [_row(row) for row in rows]


def history(subject_type: str, subject_id: Any, limit: int = 100) -> List[Dict[str, Any]]:
    """Everything recorded about one item, oldest first, so it reads as a story."""
    return list(reversed(list_entries(subject_type=subject_type, subject_id=str(subject_id), limit=limit)))
