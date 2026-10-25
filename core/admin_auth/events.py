"""Append-only record of sign-ins, failures and security changes, shown to the signed-in admin on the Security page."""
from __future__ import annotations

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
