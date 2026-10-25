"""Admin view of the email lists: counts, a masked searchable subscriber list, and an audit trail.

Subscribers are identified by their row id so addresses never appear in URLs. Emails are masked everywhere except an explicit,
audited "reveal"; names are only shown on reveal. There is deliberately no export and no bulk read of full addresses.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Dict, List, Optional

from core.database import get_db_path

PAGE_SIZE_MAX = 100


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path(), timeout=15)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """CREATE TABLE IF NOT EXISTS newsletter_admin_audit (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               admin_email TEXT NOT NULL,
               action TEXT NOT NULL,
               subscriber_id INTEGER,
               created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
           )"""
    )
    return conn


def mask_email(email: str) -> str:
    local, _, domain = (email or "").partition("@")
    if not domain:
        return "***"
    return f"{local[:1]}***@{domain}"


def _types(raw: Optional[str]) -> List[str]:
    try:
        value = json.loads(raw or "[]")
    except json.JSONDecodeError:
        return []
    return [str(item) for item in value] if isinstance(value, list) else []


def audit(conn: sqlite3.Connection, admin_email: str, action: str, subscriber_id: Optional[int] = None) -> None:
    conn.execute(
        "INSERT INTO newsletter_admin_audit (admin_email, action, subscriber_id) VALUES (?, ?, ?)",
        (admin_email, action, subscriber_id),
    )


def summary() -> Dict:
    conn = _connect()
    try:
        rows = conn.execute(
            """SELECT s.rowid AS id, s.subscription_types_json AS types, s.unsubscribed_at, s.created_at,
                      (SELECT COUNT(*) FROM newsletter_preferences p WHERE p.email = s.email) AS counties
               FROM newsletter_subscribers s"""
        ).fetchall()
        per_list: Dict[str, int] = {}
        active = unsubscribed = graphic_only = 0
        for row in rows:
            if row["unsubscribed_at"]:
                unsubscribed += 1
                continue
            active += 1
            types = _types(row["types"])
            for item in types:
                per_list[item] = per_list.get(item, 0) + 1
            if "fire-weather-forecasts" in types and not row["counties"]:
                graphic_only += 1
        recent = conn.execute(
            """SELECT
                   SUM(created_at >= datetime('now', '-7 days')) AS signups_7d,
                   SUM(created_at >= datetime('now', '-30 days')) AS signups_30d,
                   SUM(unsubscribed_at >= datetime('now', '-30 days')) AS unsubscribes_30d
               FROM newsletter_subscribers"""
        ).fetchone()
        top_counties = [
            dict(row) for row in conn.execute(
                """SELECT p.county_fips, COUNT(*) AS subscribers
                   FROM newsletter_preferences p JOIN newsletter_subscribers s ON s.email = p.email
                   WHERE s.unsubscribed_at IS NULL
                   GROUP BY p.county_fips ORDER BY subscribers DESC, p.county_fips LIMIT 10"""
            ).fetchall()
        ]
        return {
            "active": active,
            "unsubscribed": unsubscribed,
            "per_list": per_list,
            "graphic_only": graphic_only,
            "signups_7d": recent["signups_7d"] or 0,
            "signups_30d": recent["signups_30d"] or 0,
            "unsubscribes_30d": recent["unsubscribes_30d"] or 0,
            "top_counties": top_counties,
        }
    finally:
        conn.close()


def list_subscribers(
    admin_email: str,
    query: str = "",
    list_id: str = "",
    status: str = "active",
    page: int = 1,
    page_size: int = 25,
) -> Dict:
    page = max(1, page)
    page_size = max(1, min(PAGE_SIZE_MAX, page_size))
    where, params = [], []
    if status == "active":
        where.append("s.unsubscribed_at IS NULL")
    elif status == "unsubscribed":
        where.append("s.unsubscribed_at IS NOT NULL")
    if list_id:
        where.append("EXISTS (SELECT 1 FROM json_each(s.subscription_types_json) WHERE value = ?)")
        params.append(list_id)
    if query.strip():
        like = f"%{query.strip().lower().replace('%', '').replace('_', '')}%"
        where.append("(lower(s.email) LIKE ? OR lower(s.affiliation) LIKE ?)")
        params += [like, like]
    clause = f"WHERE {' AND '.join(where)}" if where else ""
    conn = _connect()
    try:
        total = conn.execute(f"SELECT COUNT(*) FROM newsletter_subscribers s {clause}", params).fetchone()[0]
        rows = conn.execute(
            f"""SELECT s.rowid AS id, s.email, s.affiliation, s.subscription_types_json AS types, s.unsubscribed_at, s.created_at,
                       (SELECT COUNT(*) FROM newsletter_preferences p WHERE p.email = s.email) AS counties
                FROM newsletter_subscribers s {clause}
                ORDER BY s.created_at DESC, s.rowid DESC LIMIT ? OFFSET ?""",
            params + [page_size, (page - 1) * page_size],
        ).fetchall()
        if query.strip():  # a search can confirm an address is on the list, so it is logged (the text itself is not)
            audit(conn, admin_email, "search_list")
            conn.commit()
        return {
            "total": total,
            "page": page,
            "page_size": page_size,
            "subscribers": [
                {
                    "id": row["id"],
                    "email": mask_email(row["email"]),
                    "affiliation": row["affiliation"],
                    "lists": _types(row["types"]),
                    "county_count": row["counties"],
                    "daily_graphic": "fire-weather-forecasts" in _types(row["types"]) and not row["counties"],
                    "unsubscribed": bool(row["unsubscribed_at"]),
                    "created_at": row["created_at"],
                }
                for row in rows
            ],
        }
    finally:
        conn.close()


def reveal(admin_email: str, subscriber_id: int) -> Optional[Dict]:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT rowid AS id, email, name, affiliation FROM newsletter_subscribers WHERE rowid = ?", (subscriber_id,)
        ).fetchone()
        if not row:
            return None
        counties = [
            dict(item) for item in conn.execute(
                "SELECT county_fips, min_danger_level FROM newsletter_preferences WHERE email = ? ORDER BY county_fips",
                (row["email"],),
            ).fetchall()
        ]
        audit(conn, admin_email, "reveal", subscriber_id)
        conn.commit()
        return {"id": row["id"], "email": row["email"], "name": row["name"], "affiliation": row["affiliation"], "counties": counties}
    finally:
        conn.close()


def get_email(subscriber_id: int) -> Optional[str]:
    conn = _connect()
    try:
        row = conn.execute("SELECT email FROM newsletter_subscribers WHERE rowid = ?", (subscriber_id,)).fetchone()
        return row["email"] if row else None
    finally:
        conn.close()


def mark_unsubscribed(admin_email: str, subscriber_id: int) -> bool:
    conn = _connect()
    try:
        cursor = conn.execute(
            """UPDATE newsletter_subscribers SET unsubscribed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
               WHERE rowid = ? AND unsubscribed_at IS NULL""",
            (subscriber_id,),
        )
        audit(conn, admin_email, "unsubscribe", subscriber_id)
        conn.commit()
        return cursor.rowcount == 1
    finally:
        conn.close()


def recent_audit(limit: int = 50) -> List[Dict]:
    conn = _connect()
    try:
        return [
            dict(row) for row in conn.execute(
                "SELECT id, admin_email, action, subscriber_id, created_at FROM newsletter_admin_audit ORDER BY id DESC LIMIT ?",
                (max(1, min(200, limit)),),
            ).fetchall()
        ]
    finally:
        conn.close()
