"""SQLite storage for staff-built forms and their responses (in the main showmefire.db).

A form row holds its settings and its question list (`fields_json`). `version` goes up whenever the questions change, and each response
records the version it was sent against along with a snapshot of every answer, so old responses still read correctly after edits.
Responses keep a hash of the visitor's IP (never the address) for the per-form abuse cap.
"""
from __future__ import annotations

import json
import secrets
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, Optional

from core.database import get_db_path
from services.api_keys.store import hash_ip
from services.forms import schema

STATUSES = ("draft", "published", "closed")
RESERVED_SLUGS = {"burn-ban", "new", "admin", "api", "v1"}      # burn-ban is a hand-built page under /forms
_lock = threading.Lock()
_ready_for: Optional[str] = None

EDITABLE = ("title", "slug", "status", "intro", "submit_label", "success_message", "require_captcha", "notify_staff", "max_responses", "closes_at")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    global _ready_for
    path = str(get_db_path())
    conn = sqlite3.connect(path, timeout=15)
    conn.row_factory = sqlite3.Row
    try:
        with _lock:
            if _ready_for != path:
                _create_tables(conn)
                _ready_for = path
        yield conn
        conn.commit()
    finally:
        conn.close()


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS forms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            intro TEXT NOT NULL DEFAULT '',
            submit_label TEXT NOT NULL DEFAULT 'Send',
            success_message TEXT NOT NULL DEFAULT 'Thank you. Your response was received.',
            status TEXT NOT NULL DEFAULT 'draft',
            fields_json TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            require_captcha INTEGER NOT NULL DEFAULT 1,
            notify_staff INTEGER NOT NULL DEFAULT 0,
            max_responses INTEGER,
            closes_at TEXT,
            created_at TEXT NOT NULL,
            created_by TEXT,
            updated_at TEXT NOT NULL,
            updated_by TEXT
        );
        CREATE TABLE IF NOT EXISTS form_responses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            form_id INTEGER NOT NULL,
            reference TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            form_version INTEGER NOT NULL,
            answers_json TEXT NOT NULL,
            ip_hash TEXT,
            FOREIGN KEY (form_id) REFERENCES forms(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_form_responses_form ON form_responses(form_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_form_responses_ip ON form_responses(form_id, ip_hash, created_at DESC);
        """
    )


def _form_row(row: sqlite3.Row, *, with_fields: bool = True) -> dict:
    out = {k: row[k] for k in row.keys() if k != "fields_json"}
    for flag in ("require_captcha", "notify_staff"):
        out[flag] = bool(out[flag])
    if with_fields:
        out["fields"] = json.loads(row["fields_json"])
    return out


def _unique_slug(conn: sqlite3.Connection, wanted: str, exclude_id: Optional[int] = None) -> str:
    base = schema.normalize_slug(wanted) or "form"
    if len(base) < 2:
        base = f"form-{base}"
    slug, n = base, 2
    while slug in RESERVED_SLUGS or conn.execute("SELECT 1 FROM forms WHERE slug = ? AND id != ?", (slug, exclude_id or 0)).fetchone():
        slug = f"{base[:56]}-{n}"
        n += 1
    return slug


def starter_fields() -> list[dict]:
    return [{"id": "q_" + secrets.token_hex(3), "type": "text", "label": "Your name", "required": True}]


# --- forms --------------------------------------------------------------------------------------------

def create_form(*, title: str, slug: str = "", fields: Optional[list[dict]] = None, created_by: str = "", **settings: Any) -> dict:
    title = " ".join(str(title or "").split())
    if not title:
        raise schema.FormError("give the form a title")
    cleaned = schema.clean_fields(fields if fields is not None else starter_fields())
    with connect() as conn:
        if slug:
            wanted = schema.check_slug(slug)
            if wanted in RESERVED_SLUGS or conn.execute("SELECT 1 FROM forms WHERE slug = ?", (wanted,)).fetchone():
                raise schema.FormError("that web address is already used")
        else:
            wanted = _unique_slug(conn, title)
        now = _now()
        cursor = conn.execute(
            "INSERT INTO forms (slug, title, intro, submit_label, success_message, status, fields_json, require_captcha, notify_staff,"
            " max_responses, closes_at, created_at, created_by, updated_at, updated_by) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (wanted, title[:200], str(settings.get("intro") or "")[:4000], str(settings.get("submit_label") or "Send")[:60],
             str(settings.get("success_message") or "Thank you. Your response was received.")[:1000],
             settings.get("status") if settings.get("status") in STATUSES else "draft", json.dumps(cleaned),
             0 if settings.get("require_captcha") is False else 1, 1 if settings.get("notify_staff") else 0,
             settings.get("max_responses"), settings.get("closes_at") or None, now, created_by, now, created_by))
        return get_form(cursor.lastrowid, conn=conn)


def get_form(form_id: Optional[int] = None, *, slug: Optional[str] = None, conn: Optional[sqlite3.Connection] = None) -> Optional[dict]:
    def read(c):
        row = (c.execute("SELECT * FROM forms WHERE slug = ?", (slug,)) if slug is not None
               else c.execute("SELECT * FROM forms WHERE id = ?", (form_id,))).fetchone()
        if not row:
            return None
        out = _form_row(row)
        out["response_count"] = int(c.execute("SELECT COUNT(*) FROM form_responses WHERE form_id = ?", (row["id"],)).fetchone()[0])
        return out
    if conn is not None:
        return read(conn)
    with connect() as c:
        return read(c)


def list_forms() -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT f.*, (SELECT COUNT(*) FROM form_responses r WHERE r.form_id = f.id) AS response_count,"
            " (SELECT MAX(created_at) FROM form_responses r WHERE r.form_id = f.id) AS last_response_at"
            " FROM forms f ORDER BY f.updated_at DESC, f.id DESC").fetchall()
        out = []
        for row in rows:
            item = _form_row(row, with_fields=False)
            out.append(item)
        return out


def update_form(form_id: int, changes: dict[str, Any], updated_by: str = "") -> Optional[dict]:
    """Apply the keys present in `changes` (settings and/or `fields`). Raises schema.FormError for a bad value."""
    with connect() as conn:
        current = get_form(form_id, conn=conn)
        if current is None:
            return None
        sets: dict[str, Any] = {}
        if "title" in changes:
            title = " ".join(str(changes["title"] or "").split())
            if not title:
                raise schema.FormError("give the form a title")
            sets["title"] = title[:200]
        if "slug" in changes and changes["slug"] != current["slug"]:
            wanted = schema.check_slug(str(changes["slug"] or ""))
            if wanted in RESERVED_SLUGS or conn.execute("SELECT 1 FROM forms WHERE slug = ? AND id != ?", (wanted, form_id)).fetchone():
                raise schema.FormError("that web address is already used")
            sets["slug"] = wanted
        if "status" in changes:
            if changes["status"] not in STATUSES:
                raise schema.FormError("status must be draft, published or closed")
            sets["status"] = changes["status"]
        for key, limit in (("intro", 4000), ("submit_label", 60), ("success_message", 1000)):
            if key in changes:
                sets[key] = str(changes[key] or "").strip()[:limit] or {"submit_label": "Send", "success_message": "Thank you. Your response was received.", "intro": ""}[key]
        for key in ("require_captcha", "notify_staff"):
            if key in changes:
                sets[key] = 1 if changes[key] else 0
        if "max_responses" in changes:
            cap = changes["max_responses"]
            if cap in (None, ""):
                sets["max_responses"] = None
            elif not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
                raise schema.FormError("the response limit must be a whole number above zero")
            else:
                sets["max_responses"] = cap
        if "closes_at" in changes:
            closes = str(changes["closes_at"] or "").strip()
            if closes:
                try:
                    closes = datetime.fromisoformat(closes.replace("Z", "+00:00")).astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                except ValueError:
                    raise schema.FormError("the closing time is not a valid date") from None
            sets["closes_at"] = closes or None
        if "fields" in changes:
            cleaned = schema.clean_fields(changes["fields"])
            if cleaned != current["fields"]:
                sets["fields_json"] = json.dumps(cleaned)
                sets["version"] = current["version"] + 1
        if not sets:
            return current
        sets["updated_at"], sets["updated_by"] = _now(), updated_by
        conn.execute(f"UPDATE forms SET {', '.join(f'{k} = ?' for k in sets)} WHERE id = ?", (*sets.values(), form_id))
        return get_form(form_id, conn=conn)


def delete_form(form_id: int) -> bool:
    with connect() as conn:
        conn.execute("DELETE FROM form_responses WHERE form_id = ?", (form_id,))
        return conn.execute("DELETE FROM forms WHERE id = ?", (form_id,)).rowcount > 0


def duplicate_form(form_id: int, created_by: str = "") -> Optional[dict]:
    source = get_form(form_id)
    if source is None:
        return None
    return create_form(title=f"{source['title']} (copy)", fields=source["fields"], created_by=created_by, intro=source["intro"],
                       submit_label=source["submit_label"], success_message=source["success_message"], require_captcha=source["require_captcha"],
                       notify_staff=source["notify_staff"], status="draft")


def public_state(form: dict) -> tuple[bool, str]:
    """(open, reason). A form takes responses only while published, before its closing time and under its response limit."""
    if form["status"] == "closed":
        return False, "This form is no longer accepting responses."
    if form["status"] != "published":
        return False, "This form is not available."
    if form.get("closes_at") and form["closes_at"] <= _now():
        return False, "This form is no longer accepting responses."
    if form.get("max_responses") and form["response_count"] >= form["max_responses"]:
        return False, "This form has reached its response limit."
    return True, ""


# --- responses ----------------------------------------------------------------------------------------

def create_response(form: dict, answers: list[dict], ip: str = "") -> dict:
    with connect() as conn:
        reference = "FR-" + secrets.token_hex(3).upper()
        cursor = conn.execute(
            "INSERT INTO form_responses (form_id, reference, created_at, form_version, answers_json, ip_hash) VALUES (?,?,?,?,?,?)",
            (form["id"], reference, _now(), form["version"], json.dumps(answers), hash_ip(ip) if ip else None))
        return _response(conn.execute("SELECT * FROM form_responses WHERE id = ?", (cursor.lastrowid,)).fetchone())


def _response(row: sqlite3.Row) -> dict:
    out = {k: row[k] for k in row.keys() if k not in ("answers_json", "ip_hash")}
    out["answers"] = json.loads(row["answers_json"])
    return out


def list_responses(form_id: int, limit: int = 100, offset: int = 0) -> list[dict]:
    with connect() as conn:
        rows = conn.execute("SELECT * FROM form_responses WHERE form_id = ? ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                            (form_id, limit, offset)).fetchall()
        return [_response(row) for row in rows]


def all_responses(form_id: int) -> list[dict]:
    return list_responses(form_id, limit=1_000_000)


def delete_response(form_id: int, response_id: int) -> bool:
    with connect() as conn:
        return conn.execute("DELETE FROM form_responses WHERE id = ? AND form_id = ?", (response_id, form_id)).rowcount > 0


def count_recent_from_ip(form_id: int, ip: str, hours: int = 24) -> int:
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    with connect() as conn:
        return int(conn.execute("SELECT COUNT(*) FROM form_responses WHERE form_id = ? AND ip_hash = ? AND created_at >= ?",
                                (form_id, hash_ip(ip), since)).fetchone()[0])


def reset_for_tests() -> None:
    global _ready_for
    _ready_for = None
