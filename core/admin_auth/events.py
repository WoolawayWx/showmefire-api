"""Record of sign-ins, failures and security changes, shown to the signed-in admin on the Security page. Rows are never edited,
and are deleted after EVENT_RECORD_DAYS by the scheduled retention job (they hold IP addresses)."""
from __future__ import annotations

import os
import time
from typing import Optional

from . import db


def record(kind: str, admin_id: Optional[int] = None, ip: str = "", detail: str = "") -> None:
    with db.tx() as conn:
        conn.execute(
            "INSERT INTO admin_auth_events (admin_id, kind, ip, detail, ts) VALUES (?, ?, ?, ?, ?)",
            (admin_id, kind, ip[:64], detail[:200], time.time()),
        )


def recent(admin_id: int, limit: int = 50) -> list[dict]:
    with db.tx() as conn:
        rows = conn.execute(
            "SELECT id, kind, ip, detail, ts FROM admin_auth_events WHERE admin_id = ? ORDER BY id DESC LIMIT ?",
            (admin_id, max(1, min(200, limit))),
        ).fetchall()
    return [{key: row[key] for key in row.keys()} for row in rows]


EVENT_RECORD_DAYS = int(os.getenv("ADMIN_EVENT_RECORD_DAYS", "365"))


def purge(days: int = EVENT_RECORD_DAYS) -> int:
    """Delete sign-in and security events (with their IP addresses) older than `days`. Returns how many rows went."""
    with db.tx() as conn:
        return conn.execute("DELETE FROM admin_auth_events WHERE ts < ?", (time.time() - days * 86400,)).rowcount


def purge_old_records() -> dict:
    """The scheduled retention job for account sign-in records: ended sessions after 90 days, events after a year."""
    from . import sessions

    return {"sessions": sessions.purge_old(), "events": purge()}
