"""SQLite storage for API key requests and the keys issued from them (in the main showmefire.db).

A key is shown in full exactly once (when it is approved or rotated, and in the email to the applicant). Only its SHA-256 hash is
stored: keys are long random strings, so a plain hash is enough, and a database leak does not reveal them. Requests keep a hash of
the applicant's IP (never the address itself) for abuse limits.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, Optional

from core.database import get_db_path

KEY_PREFIX = "smf_"
DEFAULT_MULTIPLIER = 10
MAX_MULTIPLIER = 100
REQUEST_STATUSES = ("pending", "approved", "denied")
_lock = threading.Lock()
_ready_for: Optional[str] = None


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def hash_ip(ip: str) -> str:
    from core.security import SECRET_KEY

    return hashlib.sha256(f"{SECRET_KEY}|{ip}".encode("utf-8")).hexdigest()[:32]


def generate_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(24)


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
        CREATE TABLE IF NOT EXISTS api_key_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reference TEXT NOT NULL UNIQUE,
            created_at TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            agency TEXT NOT NULL,
            answers_json TEXT NOT NULL,
            form_version INTEGER NOT NULL,
            ip_hash TEXT,
            reviewed_by TEXT,
            reviewed_at TEXT,
            review_note TEXT,
            decision_email_sent INTEGER NOT NULL DEFAULT 0,
            key_id INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_api_key_requests_status ON api_key_requests(status, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_api_key_requests_email ON api_key_requests(email, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_api_key_requests_ip ON api_key_requests(ip_hash, created_at DESC);
        CREATE TABLE IF NOT EXISTS api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id INTEGER,
            key_hash TEXT NOT NULL UNIQUE,
            key_hint TEXT NOT NULL,
            label TEXT NOT NULL,
            email TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            multiplier INTEGER NOT NULL DEFAULT 10,
            created_at TEXT NOT NULL,
            created_by TEXT,
            expires_at TEXT,
            revoked_at TEXT,
            revoked_by TEXT,
            last_used_at TEXT,
            use_count INTEGER NOT NULL DEFAULT 0
        );
        """
    )


def _request_row(row: sqlite3.Row, *, full: bool = True) -> dict:
    out = {k: row[k] for k in row.keys() if k not in ("answers_json", "ip_hash")}
    out["decision_email_sent"] = bool(out["decision_email_sent"])
    if full:
        out["answers"] = json.loads(row["answers_json"])
    return out


def _key_row(row: sqlite3.Row) -> dict:
    out = {k: row[k] for k in row.keys() if k != "key_hash"}
    out["active"] = out["status"] == "active" and not (out["expires_at"] and out["expires_at"] <= _now())
    return out


# --- requests -----------------------------------------------------------------------------------------

def create_request(*, name: str, email: str, agency: str, answers: list[dict], form_version: int, ip: str = "") -> dict:
    with connect() as conn:
        reference = "KR-" + secrets.token_hex(3).upper()
        cursor = conn.execute(
            "INSERT INTO api_key_requests (reference, created_at, status, name, email, agency, answers_json, form_version, ip_hash)"
            " VALUES (?, ?, 'pending', ?, ?, ?, ?, ?, ?)",
            (reference, _now(), name, email.lower(), agency, json.dumps(answers), form_version, hash_ip(ip) if ip else None))
        return get_request(cursor.lastrowid, conn)


def get_request(request_id: int, conn: Optional[sqlite3.Connection] = None) -> Optional[dict]:
    def read(c):
        row = c.execute("SELECT * FROM api_key_requests WHERE id = ?", (request_id,)).fetchone()
        return _request_row(row) if row else None
    if conn is not None:
        return read(conn)
    with connect() as c:
        return read(c)


def list_requests(status: Optional[str] = None, limit: int = 200) -> list[dict]:
    with connect() as conn:
        query, args = "SELECT * FROM api_key_requests", []
        if status in REQUEST_STATUSES:
            query += " WHERE status = ?"
            args.append(status)
        rows = conn.execute(query + " ORDER BY (status = 'pending') DESC, created_at DESC, id DESC LIMIT ?", (*args, limit)).fetchall()
        return [_request_row(row, full=False) for row in rows]


def count_pending() -> int:
    with connect() as conn:
        return int(conn.execute("SELECT COUNT(*) FROM api_key_requests WHERE status = 'pending'").fetchone()[0])


def open_request_for_email(email: str, within_days: int = 30) -> Optional[dict]:
    """A pending request from this address in the last `within_days` days (so people do not file the same one twice)."""
    since = (datetime.now(timezone.utc) - timedelta(days=within_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    with connect() as conn:
        row = conn.execute("SELECT * FROM api_key_requests WHERE email = ? AND status = 'pending' AND created_at >= ?"
                           " ORDER BY created_at DESC LIMIT 1", (email.lower(), since)).fetchone()
        return _request_row(row, full=False) if row else None


def count_recent_from_ip(ip: str, hours: int = 24) -> int:
    since = (datetime.now(timezone.utc) - timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
    with connect() as conn:
        return int(conn.execute("SELECT COUNT(*) FROM api_key_requests WHERE ip_hash = ? AND created_at >= ?",
                                (hash_ip(ip), since)).fetchone()[0])


def decide_request(request_id: int, status: str, reviewer: str, note: str = "", key_id: Optional[int] = None,
                   email_sent: bool = False) -> Optional[dict]:
    if status not in ("approved", "denied"):
        raise ValueError("status must be approved or denied")
    with connect() as conn:
        updated = conn.execute(
            "UPDATE api_key_requests SET status = ?, reviewed_by = ?, reviewed_at = ?, review_note = ?, key_id = ?,"
            " decision_email_sent = ? WHERE id = ? AND status = 'pending'",
            (status, reviewer, _now(), note, key_id, 1 if email_sent else 0, request_id)).rowcount
        return get_request(request_id, conn) if updated else None


def set_decision_email_sent(request_id: int, sent: bool) -> None:
    with connect() as conn:
        conn.execute("UPDATE api_key_requests SET decision_email_sent = ? WHERE id = ?", (1 if sent else 0, request_id))


# --- keys ---------------------------------------------------------------------------------------------

def create_key(*, label: str, email: str, request_id: Optional[int] = None, multiplier: int = DEFAULT_MULTIPLIER,
               expires_days: Optional[int] = None, expires_at: Optional[str] = None, created_by: str = "") -> tuple[str, dict]:
    """(the key in full, its row). The full key is not stored and cannot be shown again."""
    multiplier = max(1, min(MAX_MULTIPLIER, int(multiplier)))
    key = generate_key()
    expires = expires_at or ((datetime.now(timezone.utc) + timedelta(days=int(expires_days))).strftime("%Y-%m-%dT%H:%M:%SZ")
                             if expires_days else None)
    with connect() as conn:
        cursor = conn.execute(
            "INSERT INTO api_keys (request_id, key_hash, key_hint, label, email, status, multiplier, created_at, created_by, expires_at)"
            " VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?, ?)",
            (request_id, hash_key(key), f"{key[:8]}…{key[-4:]}", label, email.lower(), multiplier, _now(), created_by, expires))
        return key, _key_row(conn.execute("SELECT * FROM api_keys WHERE id = ?", (cursor.lastrowid,)).fetchone())


def verify_key(key: str) -> Optional[dict]:
    """The active, unexpired key row for this key text, or None."""
    if not key.startswith(KEY_PREFIX) or len(key) > 200:
        return None
    with connect() as conn:
        row = conn.execute("SELECT * FROM api_keys WHERE key_hash = ?", (hash_key(key),)).fetchone()
        found = _key_row(row) if row else None
    return found if found and found["active"] else None


def touch_key(key_id: int, uses: int = 1) -> None:
    with connect() as conn:
        conn.execute("UPDATE api_keys SET last_used_at = ?, use_count = use_count + ? WHERE id = ?", (_now(), uses, key_id))


def get_key(key_id: int) -> Optional[dict]:
    with connect() as conn:
        row = conn.execute("SELECT * FROM api_keys WHERE id = ?", (key_id,)).fetchone()
        return _key_row(row) if row else None


def list_keys(limit: int = 500) -> list[dict]:
    with connect() as conn:
        return [_key_row(r) for r in conn.execute(
            "SELECT * FROM api_keys ORDER BY (status = 'active') DESC, created_at DESC, id DESC LIMIT ?", (limit,)).fetchall()]


def revoke_key(key_id: int, reviewer: str) -> Optional[dict]:
    with connect() as conn:
        updated = conn.execute("UPDATE api_keys SET status = 'revoked', revoked_at = ?, revoked_by = ? WHERE id = ? AND status = 'active'",
                               (_now(), reviewer, key_id)).rowcount
        row = conn.execute("SELECT * FROM api_keys WHERE id = ?", (key_id,)).fetchone()
        return _key_row(row) if (updated and row) else None


def reset_for_tests() -> None:
    global _ready_for
    _ready_for = None
