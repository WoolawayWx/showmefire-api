"""Server-side admin sessions. The refresh cookie is an opaque random value stored only as a hash; the access cookie is a short JWT
carrying the session id, which is checked on every request so a revoked session stops working within seconds.

The refresh value is deliberately not rotated on each renewal: the browser fires several requests in parallel when the access
cookie expires, and rotation with reuse detection would sign people out at random. A session ends by revocation, by its absolute
lifetime, or after sitting idle."""
from __future__ import annotations

import hashlib
import os
import secrets
import threading
import time
from typing import Optional

from core import security

from . import db

REFRESH_SECONDS = security.REFRESH_TOKEN_EXPIRE_DAYS * 86400
IDLE_SECONDS = int(os.getenv("ADMIN_SESSION_IDLE_DAYS", "14")) * 86400
CACHE_SECONDS = 5

_cache: dict[str, tuple[float, bool]] = {}
_cache_lock = threading.Lock()


def _hash(refresh_token: str) -> str:
    return hashlib.sha256(refresh_token.encode("utf-8")).hexdigest()


def _forget(session_id: Optional[str] = None) -> None:
    with _cache_lock:
        if session_id is None:
            _cache.clear()
        else:
            _cache.pop(session_id, None)


def create(admin_id: int, ip: str, user_agent: str, method: str) -> tuple[str, str]:
    """Start a session: (session id, refresh token). The refresh token is only ever returned here."""
    session_id = secrets.token_urlsafe(16)
    refresh_token = secrets.token_urlsafe(32)
    now = time.time()
    with db.tx() as conn:
        conn.execute(
            """INSERT INTO admin_sessions (id, admin_id, refresh_hash, created_at, last_seen_at, expires_at, ip, user_agent, method)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (session_id, admin_id, _hash(refresh_token), now, now, now + REFRESH_SECONDS, ip[:64], (user_agent or "")[:300], method),
        )
    return session_id, refresh_token


def renew(refresh_token: str) -> Optional[tuple[str, str]]:
    """(admin email, session id) for a live refresh token, marking the session as just used. None if it cannot be renewed."""
    if not refresh_token:
        return None
    now = time.time()
    with db.tx() as conn:
        row = conn.execute(
            """SELECT s.id AS sid, u.email AS email FROM admin_sessions s JOIN admin_users u ON u.id = s.admin_id
               WHERE s.refresh_hash = ? AND s.revoked_at IS NULL AND s.expires_at > ? AND s.last_seen_at > ? AND u.disabled_at IS NULL""",
            (_hash(refresh_token), now, now - IDLE_SECONDS),
        ).fetchone()
        if row is None:
            return None
        conn.execute("UPDATE admin_sessions SET last_seen_at = ? WHERE id = ?", (now, row["sid"]))
        return row["email"], row["sid"]


def is_active(session_id: str) -> bool:
    now = time.time()
    with _cache_lock:
        hit = _cache.get(session_id)
        if hit and now - hit[0] < CACHE_SECONDS:
            return hit[1]
    with db.tx() as conn:
        live = conn.execute(
            """SELECT 1 FROM admin_sessions s JOIN admin_users u ON u.id = s.admin_id
               WHERE s.id = ? AND s.revoked_at IS NULL AND s.expires_at > ? AND u.disabled_at IS NULL""",
            (session_id, now),
        ).fetchone() is not None
    with _cache_lock:
        if len(_cache) > 2000:
            _cache.clear()
        _cache[session_id] = (now, live)
    return live


def revoke(session_id: str, admin_id: Optional[int] = None) -> bool:
    query, params = "UPDATE admin_sessions SET revoked_at = ? WHERE id = ? AND revoked_at IS NULL", [time.time(), session_id]
    if admin_id is not None:
        query += " AND admin_id = ?"
        params.append(admin_id)
    with db.tx() as conn:
        changed = conn.execute(query, params).rowcount == 1
    _forget(session_id)
    return changed


def revoke_by_refresh(refresh_token: str) -> None:
    if not refresh_token:
        return
    with db.tx() as conn:
        conn.execute("UPDATE admin_sessions SET revoked_at = ? WHERE refresh_hash = ? AND revoked_at IS NULL", (time.time(), _hash(refresh_token)))
    _forget()


def revoke_all(admin_id: int, except_session: Optional[str] = None) -> int:
    query, params = "UPDATE admin_sessions SET revoked_at = ? WHERE admin_id = ? AND revoked_at IS NULL", [time.time(), admin_id]
    if except_session:
        query += " AND id != ?"
        params.append(except_session)
    with db.tx() as conn:
        count = conn.execute(query, params).rowcount
    _forget()
    return count


def list_for(admin_id: int, current: Optional[str] = None) -> list[dict]:
    now = time.time()
    with db.tx() as conn:
        rows = conn.execute(
            """SELECT id, created_at, last_seen_at, ip, user_agent, method FROM admin_sessions
               WHERE admin_id = ? AND revoked_at IS NULL AND expires_at > ? AND last_seen_at > ? ORDER BY last_seen_at DESC""",
            (admin_id, now, now - IDLE_SECONDS),
        ).fetchall()
    return [{**{key: row[key] for key in row.keys()}, "current": row["id"] == current} for row in rows]


security.register_session_checker(is_active)
