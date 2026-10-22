"""SQLite storage for development-site access codes and the sessions opened with them (in the main showmefire.db).

Codes are chosen by an admin (any text, 4 to 64 characters) and handed to an organization. Matching ignores case and extra spaces, so
"Acme Fire 2026" and "acme  fire 2026" are the same code. Only a keyed hash and a short hint are stored, like API keys.

Every time a browser enters a code (or an admin signs in at the gate) a session row is made: that is one "use". It records when and
from where, and how often the website re-confirmed it since (``seen_count`` / ``last_seen_at``). Revoking a code stops new uses and,
by default, ends its sessions; a single session can also be ended on its own.
"""
from __future__ import annotations

import hashlib
import hmac
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Iterator, Optional

from core.database import get_db_path

CODE_MIN_LENGTH = 4
CODE_MAX_LENGTH = 64
DEFAULT_SESSION_DAYS = 30
MAX_SESSION_DAYS = 365
ADMIN_SESSION_DAYS = 30
_lock = threading.Lock()
_ready_for: Optional[str] = None


def _now_dt() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now() -> str:
    return _iso(_now_dt())


def parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def normalize_code(code: str) -> str:
    return " ".join((code or "").split()).casefold()


def code_problem(code: str) -> Optional[str]:
    """Why a chosen code cannot be used, or None."""
    normalized = normalize_code(code)
    if len(normalized) < CODE_MIN_LENGTH or len(normalized) > CODE_MAX_LENGTH:
        return f"The code must be {CODE_MIN_LENGTH} to {CODE_MAX_LENGTH} characters."
    if any(not ch.isprintable() for ch in normalized):
        return "The code cannot contain control characters."
    return None


def hash_code(code: str) -> str:
    from core.security import SECRET_KEY

    return hmac.new(SECRET_KEY.encode("utf-8"), f"dev-access|{normalize_code(code)}".encode("utf-8"), hashlib.sha256).hexdigest()


def code_hint(code: str) -> str:
    shown = " ".join(code.split())
    if len(shown) <= 6:
        return shown[0] + "•" * (len(shown) - 1)
    return f"{shown[:2]}…{shown[-2:]}"


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
        CREATE TABLE IF NOT EXISTS dev_access_codes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code_hash TEXT NOT NULL UNIQUE,
            code_hint TEXT NOT NULL,
            label TEXT NOT NULL,
            note TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'active',
            created_at TEXT NOT NULL,
            created_by TEXT,
            expires_at TEXT,
            max_uses INTEGER,
            session_days INTEGER NOT NULL DEFAULT 30,
            use_count INTEGER NOT NULL DEFAULT 0,
            last_used_at TEXT,
            revoked_at TEXT,
            revoked_by TEXT
        );
        CREATE TABLE IF NOT EXISTS dev_access_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            code_id INTEGER,
            kind TEXT NOT NULL,
            admin_email TEXT,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            seen_count INTEGER NOT NULL DEFAULT 0,
            ip_address TEXT,
            user_agent TEXT,
            ended_at TEXT,
            ended_by TEXT,
            end_reason TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_dev_access_sessions_code ON dev_access_sessions(code_id, created_at DESC);
        """
    )


def _code_state(row: sqlite3.Row | dict, now: Optional[str] = None) -> str:
    now = now or _now()
    if row["status"] == "revoked":
        return "revoked"
    if row["expires_at"] and row["expires_at"] <= now:
        return "expired"
    if row["max_uses"] is not None and row["use_count"] >= row["max_uses"]:
        return "used_up"
    return "active"


def _code_row(row: sqlite3.Row, active_sessions: int = 0) -> dict:
    out = {k: row[k] for k in row.keys() if k != "code_hash"}
    out["state"] = _code_state(row)
    out["active"] = out["state"] == "active"
    out["active_sessions"] = active_sessions
    return out


def _session_row(row: sqlite3.Row, now: Optional[str] = None) -> dict:
    """A session is good while it is not ended or expired and, for a code session, its code is not revoked or expired.

    A code that has merely reached max_uses keeps its existing sessions working; it only stops new ones.
    """
    now = now or _now()
    out = {k: row[k] for k in row.keys() if k not in ("code_status", "code_expires_at")}
    if row["ended_at"]:
        state = "ended"
    elif row["expires_at"] <= now:
        state = "expired"
    elif row["kind"] == "code" and (row["code_status"] is None or row["code_status"] == "revoked"):
        state = "code_revoked"
    elif row["kind"] == "code" and row["code_expires_at"] and row["code_expires_at"] <= now:
        state = "code_expired"
    else:
        state = "active"
    out["state"] = state
    out["active"] = state == "active"
    return out


# --- codes --------------------------------------------------------------------------------------------

def create_code(*, code: str, label: str, note: str = "", expires_at: Optional[datetime] = None, max_uses: Optional[int] = None,
                session_days: int = DEFAULT_SESSION_DAYS, created_by: str = "") -> Optional[dict]:
    """The new code's row, or None when that code (ignoring case) already exists."""
    with connect() as conn:
        try:
            cursor = conn.execute(
                "INSERT INTO dev_access_codes (code_hash, code_hint, label, note, status, created_at, created_by, expires_at, max_uses, session_days)"
                " VALUES (?, ?, ?, ?, 'active', ?, ?, ?, ?, ?)",
                (hash_code(code), code_hint(code), label.strip(), note.strip(), _now(), created_by,
                 _iso(expires_at) if expires_at else None, max_uses, session_days))
        except sqlite3.IntegrityError:
            return None
        return get_code(cursor.lastrowid, conn)


def get_code(code_id: int, conn: Optional[sqlite3.Connection] = None) -> Optional[dict]:
    def read(c):
        row = c.execute("SELECT * FROM dev_access_codes WHERE id = ?", (code_id,)).fetchone()
        if not row:
            return None
        return _code_row(row, _count_active_sessions(c, code_id))
    if conn is not None:
        return read(conn)
    with connect() as c:
        return read(c)


def _count_active_sessions(conn: sqlite3.Connection, code_id: int) -> int:
    return conn.execute("SELECT COUNT(*) FROM dev_access_sessions WHERE code_id = ? AND ended_at IS NULL AND expires_at > ?",
                        (code_id, _now())).fetchone()[0]


def list_codes(limit: int = 500) -> list[dict]:
    with connect() as conn:
        counts = dict(conn.execute("SELECT code_id, COUNT(*) FROM dev_access_sessions WHERE ended_at IS NULL AND expires_at > ?"
                                   " GROUP BY code_id", (_now(),)).fetchall())
        rows = conn.execute("SELECT * FROM dev_access_codes ORDER BY (status = 'active') DESC, created_at DESC, id DESC LIMIT ?",
                            (limit,)).fetchall()
        return [_code_row(r, counts.get(r["id"], 0)) for r in rows]


def update_code(code_id: int, changes: dict) -> Optional[dict]:
    """Change label, note, expires_at, max_uses or session_days. Existing sessions keep their own expiry."""
    allowed = {"label", "note", "expires_at", "max_uses", "session_days"}
    values = {k: v for k, v in changes.items() if k in allowed}
    if "expires_at" in values and isinstance(values["expires_at"], datetime):
        values["expires_at"] = _iso(values["expires_at"])
    with connect() as conn:
        if values:
            assignments = ", ".join(f"{k} = ?" for k in values)
            conn.execute(f"UPDATE dev_access_codes SET {assignments} WHERE id = ?", (*values.values(), code_id))
        return get_code(code_id, conn)


def revoke_code(code_id: int, admin: str, end_sessions: bool = True) -> Optional[dict]:
    with connect() as conn:
        updated = conn.execute("UPDATE dev_access_codes SET status = 'revoked', revoked_at = ?, revoked_by = ? WHERE id = ? AND status = 'active'",
                               (_now(), admin, code_id)).rowcount
        if not updated:
            return None
        if end_sessions:
            _end_sessions_for_code(conn, code_id, admin, "code revoked")
        return get_code(code_id, conn)


def _end_sessions_for_code(conn: sqlite3.Connection, code_id: int, admin: str, reason: str) -> int:
    return conn.execute("UPDATE dev_access_sessions SET ended_at = ?, ended_by = ?, end_reason = ? WHERE code_id = ? AND ended_at IS NULL",
                        (_now(), admin, reason, code_id)).rowcount


def end_sessions_for_code(code_id: int, admin: str) -> int:
    with connect() as conn:
        return _end_sessions_for_code(conn, code_id, admin, "ended by admin")


# --- sessions -----------------------------------------------------------------------------------------

def redeem(code: str, *, ip: str = "", user_agent: str = "") -> Optional[dict]:
    """Open a session for a valid code: ``{"session": row, "code": row}``, or None when the code is unknown or not usable."""
    now_dt = _now_dt()
    now = _iso(now_dt)
    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")                   # use_count check and increment must not race another redemption
        row = conn.execute("SELECT * FROM dev_access_codes WHERE code_hash = ?", (hash_code(code),)).fetchone()
        if row is None or _code_state(row, now) != "active":
            return None
        expires = now_dt + timedelta(days=int(row["session_days"]))
        code_expiry = parse_iso(row["expires_at"])
        if code_expiry and code_expiry < expires:
            expires = code_expiry
        conn.execute("UPDATE dev_access_codes SET use_count = use_count + 1, last_used_at = ? WHERE id = ?", (now, row["id"]))
        cursor = conn.execute(
            "INSERT INTO dev_access_sessions (code_id, kind, created_at, expires_at, last_seen_at, ip_address, user_agent)"
            " VALUES (?, 'code', ?, ?, ?, ?, ?)", (row["id"], now, _iso(expires), now, ip[:64], user_agent[:400]))
        return {"session": get_session(cursor.lastrowid, conn), "code": get_code(row["id"], conn)}


def open_admin_session(email: str, *, ip: str = "", user_agent: str = "") -> dict:
    now_dt = _now_dt()
    with connect() as conn:
        cursor = conn.execute(
            "INSERT INTO dev_access_sessions (kind, admin_email, created_at, expires_at, last_seen_at, ip_address, user_agent)"
            " VALUES ('admin', ?, ?, ?, ?, ?, ?)",
            (email, _iso(now_dt), _iso(now_dt + timedelta(days=ADMIN_SESSION_DAYS)), _iso(now_dt), ip[:64], user_agent[:400]))
        return get_session(cursor.lastrowid, conn)


_SESSION_SELECT = ("SELECT s.*, c.label AS code_label, c.code_hint AS code_hint, c.status AS code_status, c.expires_at AS code_expires_at"
                   " FROM dev_access_sessions s LEFT JOIN dev_access_codes c ON c.id = s.code_id")


def get_session(session_id: int, conn: Optional[sqlite3.Connection] = None) -> Optional[dict]:
    def read(c):
        row = c.execute(f"{_SESSION_SELECT} WHERE s.id = ?", (session_id,)).fetchone()
        return _session_row(row) if row else None
    if conn is not None:
        return read(conn)
    with connect() as c:
        return read(c)


def confirm_session(session_id: int) -> Optional[dict]:
    """The session if it is still good (and mark it seen), else None."""
    with connect() as conn:
        session = get_session(session_id, conn)
        if session is None or not session["active"]:
            return None
        conn.execute("UPDATE dev_access_sessions SET last_seen_at = ?, seen_count = seen_count + 1 WHERE id = ?", (_now(), session_id))
        return get_session(session_id, conn)


def list_sessions(code_id: Optional[int] = None, kind: Optional[str] = None, active_only: bool = False, limit: int = 500) -> list[dict]:
    clauses, params = [], []
    if code_id is not None:
        clauses.append("s.code_id = ?")
        params.append(code_id)
    if kind:
        clauses.append("s.kind = ?")
        params.append(kind)
    if active_only:
        clauses.append("s.ended_at IS NULL AND s.expires_at > ?")
        params.append(_now())
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    with connect() as conn:
        rows = conn.execute(f"{_SESSION_SELECT}{where} ORDER BY s.last_seen_at DESC, s.id DESC LIMIT ?", (*params, limit)).fetchall()
        sessions = [_session_row(r) for r in rows]
    return [s for s in sessions if s["active"]] if active_only else sessions


def end_session(session_id: int, admin: str, reason: str = "ended by admin") -> Optional[dict]:
    with connect() as conn:
        updated = conn.execute("UPDATE dev_access_sessions SET ended_at = ?, ended_by = ?, end_reason = ? WHERE id = ? AND ended_at IS NULL",
                               (_now(), admin, reason, session_id)).rowcount
        return get_session(session_id, conn) if updated else None


def reset_for_tests() -> None:
    global _ready_for
    _ready_for = None
