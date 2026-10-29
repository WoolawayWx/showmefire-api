"""Server-side admin sessions. The refresh cookie is an opaque random value stored only as a hash; the access cookie is a short JWT
carrying the session id, which is checked on every request so a revoked session stops working within seconds.

The refresh value is deliberately not rotated on each renewal: the browser fires several requests in parallel when the access
cookie expires, and rotation with reuse detection would sign people out at random. A session ends by revocation, by its absolute
lifetime, or after sitting idle.

The mobile app is different: it keeps its refresh token in the device keychain, refreshes from one place (single-flight), and
a stolen phone token is a bigger risk than a stolen cookie, so `rotate()` replaces the token on every refresh and treats a
replay of the old one as theft."""
from __future__ import annotations

import hashlib
import logging
import os
import secrets
import sqlite3
import threading
import time
from typing import Optional

from core import security

from . import db

REFRESH_SECONDS = security.REFRESH_TOKEN_EXPIRE_DAYS * 86400
IDLE_SECONDS = int(os.getenv("ADMIN_SESSION_IDLE_DAYS", "14")) * 86400
CACHE_SECONDS = 5
ROTATION_GRACE_SECONDS = int(os.getenv("ADMIN_REFRESH_GRACE_SECONDS", "60"))

logger = logging.getLogger(__name__)
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


def _forget_push(session_id: Optional[str] = None, admin_id: Optional[int] = None, except_session: Optional[str] = None) -> None:
    """Delete staff push devices that belonged to sessions just ended. The push eligibility query already refuses revoked
    sessions, so this is tidiness and a second lock; a failure here must never stop a sign-out."""
    try:
        from services import staff_push

        if session_id:
            staff_push.unregister_session(session_id)
        elif admin_id is not None:
            staff_push.unregister_admin(admin_id, except_session)
    except Exception:  # noqa: BLE001
        logger.warning("could not remove staff push devices", exc_info=True)


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
               WHERE s.refresh_hash = ? AND s.revoked_at IS NULL AND s.expires_at > ? AND s.last_seen_at > ? AND u.disabled_at IS NULL
                 AND (u.expires_at IS NULL OR u.expires_at > ?)""",
            (_hash(refresh_token), now, now - IDLE_SECONDS, now),
        ).fetchone()
        if row is None:
            return None
        conn.execute("UPDATE admin_sessions SET last_seen_at = ? WHERE id = ?", (now, row["sid"]))
        return row["email"], row["sid"]


def rotate(refresh_token: str, ip: str = "") -> Optional[tuple[str, str, str]]:
    """Mobile refresh: (admin email, session id, NEW refresh token) for a live refresh token, or None.

    The presented token is replaced. If the app never received the answer (a dropped connection) it presents the old
    token again; within ROTATION_GRACE_SECONDS that is accepted and gets a fresh token. After the grace period an old
    token is a replay: the whole session is revoked and the event recorded."""
    if not refresh_token:
        return None
    now = time.time()
    presented = _hash(refresh_token)
    theft: Optional[sqlite3.Row] = None
    result: Optional[tuple[str, str, str]] = None
    with db.tx() as conn:
        row = conn.execute(
            """SELECT s.id AS sid, s.admin_id AS admin_id, s.refresh_hash AS current_hash, s.prev_refresh_hash AS prev_hash,
                      s.rotated_at AS rotated_at, s.expires_at AS expires_at, s.last_seen_at AS last_seen_at,
                      s.revoked_at AS revoked_at, u.email AS email, u.disabled_at AS disabled_at, u.expires_at AS user_expires
               FROM admin_sessions s JOIN admin_users u ON u.id = s.admin_id
               WHERE s.refresh_hash = ? OR s.prev_refresh_hash = ?""",
            (presented, presented),
        ).fetchone()
        if row is None or row["revoked_at"] or row["disabled_at"] or row["expires_at"] <= now:
            return None
        if row["user_expires"] and row["user_expires"] <= now:
            return None
        if row["current_hash"] == presented:
            if row["last_seen_at"] <= now - IDLE_SECONDS:
                return None
            fresh = secrets.token_urlsafe(32)
            conn.execute(
                "UPDATE admin_sessions SET prev_refresh_hash = refresh_hash, refresh_hash = ?, rotated_at = ?, last_seen_at = ? WHERE id = ?",
                (_hash(fresh), now, now, row["sid"]),
            )
            result = (row["email"], row["sid"], fresh)
        elif row["rotated_at"] and now - row["rotated_at"] <= ROTATION_GRACE_SECONDS:
            # The answer to the last refresh was lost. Hand out another new token; the old one stays acceptable until the
            # grace period ends, and the lost one is simply never used.
            fresh = secrets.token_urlsafe(32)
            conn.execute("UPDATE admin_sessions SET refresh_hash = ?, last_seen_at = ? WHERE id = ?", (_hash(fresh), now, row["sid"]))
            result = (row["email"], row["sid"], fresh)
        else:
            conn.execute("UPDATE admin_sessions SET revoked_at = ? WHERE id = ?", (now, row["sid"]))
            theft = row
    if theft is not None:
        from . import events

        _forget(theft["sid"])
        _forget_push(session_id=theft["sid"])
        events.record("refresh_token_reused", theft["admin_id"], ip, "old refresh token replayed; session revoked")
    return result


def is_active(session_id: str) -> bool:
    now = time.time()
    with _cache_lock:
        hit = _cache.get(session_id)
        if hit and now - hit[0] < CACHE_SECONDS:
            return hit[1]
    with db.tx() as conn:
        live = conn.execute(
            """SELECT 1 FROM admin_sessions s JOIN admin_users u ON u.id = s.admin_id
               WHERE s.id = ? AND s.revoked_at IS NULL AND s.expires_at > ? AND u.disabled_at IS NULL
                 AND (u.expires_at IS NULL OR u.expires_at > ?)""",
            (session_id, now, now),
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
    if changed:
        _forget_push(session_id=session_id)
    return changed


def revoke_by_refresh(refresh_token: str) -> None:
    if not refresh_token:
        return
    with db.tx() as conn:
        row = conn.execute("SELECT id FROM admin_sessions WHERE (refresh_hash = ? OR prev_refresh_hash = ?) AND revoked_at IS NULL",
                           (_hash(refresh_token), _hash(refresh_token))).fetchone()
        conn.execute("UPDATE admin_sessions SET revoked_at = ? WHERE (refresh_hash = ? OR prev_refresh_hash = ?) AND revoked_at IS NULL",
                     (time.time(), _hash(refresh_token), _hash(refresh_token)))
    _forget()
    if row is not None:
        _forget_push(session_id=row["id"])


def revoke_all(admin_id: int, except_session: Optional[str] = None) -> int:
    query, params = "UPDATE admin_sessions SET revoked_at = ? WHERE admin_id = ? AND revoked_at IS NULL", [time.time(), admin_id]
    if except_session:
        query += " AND id != ?"
        params.append(except_session)
    with db.tx() as conn:
        count = conn.execute(query, params).rowcount
    _forget()
    _forget_push(admin_id=admin_id, except_session=except_session)
    return count


SESSION_RECORD_DAYS = int(os.getenv("ADMIN_SESSION_RECORD_DAYS", "90"))


def purge_old(days: int = SESSION_RECORD_DAYS) -> int:
    """Delete session records that ended (revoked, or past their absolute lifetime) more than `days` ago. They hold the IP
    address and device description of a sign-in, which is kept only as long as it is useful for security review.
    Live sessions are never touched. Returns how many rows went."""
    cutoff = time.time() - days * 86400
    with db.tx() as conn:
        return conn.execute(
            "DELETE FROM admin_sessions WHERE (revoked_at IS NOT NULL AND revoked_at < ?) OR expires_at < ?", (cutoff, cutoff)
        ).rowcount


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
