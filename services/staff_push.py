"""Push notifications for staff in the app: the same alerts Discord gets, delivered to signed-in staff devices.

Deliberately independent of the other two channels. It has its own tables, its own Expo transport, its own scheduler jobs
and its own kill switch (STAFF_PUSH_ENABLED), and `dispatch()` never raises, so a failure here cannot break Discord or the
public push, and the reverse.

Who gets a push is decided when it is SENT, not when a device registered: one query joins the device to its session, its
account and the role's permission. So signing out, a revoked session, a disabled or expired account, or a role that loses
`push.receive` each stop pushes on their own, without depending on the app to unregister. The routes that end sessions also
delete the device rows (belt and braces).

What is sent is small on purpose: a title, the county, and a link. Never reporter text, names or contact details. Opening the
link in the app fetches the item with the person's current permissions, so a push is only ever a doorbell.

All data lives in the REAL database (accounts do too); nothing here ever runs in the reviewer sandbox.
"""
from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

import httpx

from core.admin_auth import db as auth_db
from core.admin_auth import permissions
from core.sandbox import is_sandbox

logger = logging.getLogger(__name__)

EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"
EXPO_RECEIPTS_URL = "https://exp.host/--/api/v2/push/getReceipts"
CHANNEL = "staff-alerts"
TOKEN = re.compile(r"^Expo(nent)?PushToken\[[A-Za-z0-9_\-]{10,200}\]$")
RETENTION_DAYS = 7

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS staff_push_devices (
           id INTEGER PRIMARY KEY AUTOINCREMENT,
           admin_id INTEGER NOT NULL,
           session_id TEXT NOT NULL,
           installation_id TEXT NOT NULL,
           expo_token TEXT NOT NULL UNIQUE,
           platform TEXT NOT NULL DEFAULT '',
           app_version TEXT NOT NULL DEFAULT '',
           enabled INTEGER NOT NULL DEFAULT 1,
           created_at REAL NOT NULL,
           last_seen_at REAL NOT NULL
       )""",
    "CREATE INDEX IF NOT EXISTS idx_staff_push_devices_admin ON staff_push_devices(admin_id)",
    "CREATE INDEX IF NOT EXISTS idx_staff_push_devices_session ON staff_push_devices(session_id)",
    """CREATE TABLE IF NOT EXISTS staff_push_events (
           event_key TEXT PRIMARY KEY,
           alert_type TEXT NOT NULL,
           created_at REAL NOT NULL,
           recipients INTEGER NOT NULL DEFAULT 0,
           sent INTEGER NOT NULL DEFAULT 0,
           failed INTEGER NOT NULL DEFAULT 0
       )""",
    """CREATE TABLE IF NOT EXISTS staff_push_tickets (
           ticket_id TEXT PRIMARY KEY,
           device_id INTEGER NOT NULL,
           event_key TEXT NOT NULL,
           created_at REAL NOT NULL,
           status TEXT,
           error TEXT,
           checked_at REAL
       )""",
    "CREATE INDEX IF NOT EXISTS idx_staff_push_tickets_open ON staff_push_tickets(checked_at, created_at)",
]

# Which alerts become a push, and where the tap goes. Only moderation work: a push is for "something needs a person".
# Everything else (system issues, API key requests, forms) stays Discord-only until someone decides otherwise.
PUSH_TYPES: Dict[str, Dict[str, str]] = {
    "fire_report": {"title": "New fire report", "route": "/staff/reports", "id_pattern": r"/admin/fires/(\d+)"},
    "incident_feedback": {"title": "New incident feedback", "route": "/staff/incidents", "id_pattern": ""},
    "burn_ban": {"title": "New burn-ban request", "route": "/staff/burn-bans", "id_pattern": r"/(\d+)\b"},
}
_unknown_logged: set = set()


def enabled() -> bool:
    return os.getenv("STAFF_PUSH_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")


_ready: set = set()
_ready_lock = threading.Lock()


@contextmanager
def _tx() -> Iterator[sqlite3.Connection]:
    """A connection to the real database with the staff push tables in place (auth_db.tx also makes the account tables)."""
    with auth_db.tx() as conn:
        key = str(conn.execute("PRAGMA database_list").fetchone()[2])
        if key not in _ready:
            with _ready_lock:
                for statement in SCHEMA:
                    conn.execute(statement)
                _ready.add(key)
        yield conn


def _headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    token = os.getenv("EXPO_ACCESS_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _send_batch(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Expo's tickets for a batch, in order; [] when it could not be sent. Retries transient failures."""
    if is_sandbox():
        return []
    for attempt in range(3):
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(EXPO_PUSH_URL, headers=_headers(), json=messages)
            if response.status_code == 429 or response.status_code >= 500:
                raise httpx.HTTPStatusError("transient Expo push error", request=response.request, response=response)
            response.raise_for_status()
            data = response.json().get("data", [])
            return data if isinstance(data, list) else [data]
        except (httpx.HTTPError, ValueError) as error:
            if attempt == 2:
                logger.warning("Staff push batch failed after retries: %s", error)
                return []
            time.sleep(2 ** attempt)
    return []


# ----------------------------------------------------------------------------------------------- devices

def register_device(admin_id: int, session_id: str, installation_id: str, expo_token: str, platform: str = "", app_version: str = "") -> bool:
    """Bind a device's push token to this signed-in session. False if the token is not an Expo push token."""
    if not TOKEN.match(expo_token or "") or not installation_id or len(installation_id) > 100:
        return False
    now = time.time()
    with _tx() as conn:
        # A token belongs to one row, and an install belongs to one account at a time: signing in as someone else on the
        # same phone replaces the earlier account's registration.
        conn.execute("DELETE FROM staff_push_devices WHERE expo_token = ? OR installation_id = ?", (expo_token, installation_id))
        conn.execute(
            """INSERT INTO staff_push_devices (admin_id, session_id, installation_id, expo_token, platform, app_version, created_at, last_seen_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (admin_id, session_id, installation_id, expo_token, platform[:20], app_version[:20], now, now),
        )
    return True


def unregister_session(session_id: str) -> int:
    with _tx() as conn:
        return conn.execute("DELETE FROM staff_push_devices WHERE session_id = ?", (session_id,)).rowcount


def unregister_admin(admin_id: int, except_session: Optional[str] = None) -> int:
    """Forget every device of an account (role change, disable, reset), except one session's if asked."""
    with _tx() as conn:
        if except_session:
            return conn.execute("DELETE FROM staff_push_devices WHERE admin_id = ? AND session_id != ?", (admin_id, except_session)).rowcount
        return conn.execute("DELETE FROM staff_push_devices WHERE admin_id = ?", (admin_id,)).rowcount


def eligible_devices(now: Optional[float] = None) -> List[Dict[str, Any]]:
    """Devices that may receive a push RIGHT NOW: enabled, on a live session, of an account that is active, not expired and
    holds `push.receive`. This is the only place that decides."""
    from core.admin_auth import sessions

    now = now or time.time()
    roles = permissions.roles_with(permissions.PUSH_RECEIVE)
    if not roles:
        return []
    marks = ",".join("?" for _ in roles)
    with _tx() as conn:
        rows = conn.execute(
            f"""SELECT d.id AS id, d.expo_token AS expo_token, d.platform AS platform, d.app_version AS app_version
                FROM staff_push_devices d
                JOIN admin_sessions s ON s.id = d.session_id
                JOIN admin_users u ON u.id = d.admin_id
                WHERE d.enabled = 1
                  AND s.revoked_at IS NULL AND s.expires_at > ? AND s.last_seen_at > ?
                  AND u.disabled_at IS NULL AND (u.expires_at IS NULL OR u.expires_at > ?)
                  AND u.role IN ({marks})""",
            (now, now - sessions.IDLE_SECONDS, now, *roles),
        ).fetchall()
    return [dict(row) for row in rows]


# ----------------------------------------------------------------------------------------------- sending

def send(alert_type: str, title: str, body: str, url: str, event_key: Optional[str] = None) -> int:
    """Push one alert to every eligible device. Returns how many were accepted by Expo. Never raises."""
    if is_sandbox():
        return 0
    try:
        event_key = event_key or str(uuid.uuid4())
        with _tx() as conn:
            if conn.execute("INSERT OR IGNORE INTO staff_push_events (event_key, alert_type, created_at) VALUES (?, ?, ?)",
                            (event_key, alert_type, time.time())).rowcount == 0:
                return 0  # this alert was already sent
        devices = eligible_devices()
        sent = unregistered = 0
        for start in range(0, len(devices), 100):
            batch = devices[start:start + 100]
            messages = [{
                "to": device["expo_token"], "title": title[:100], "body": body[:200], "sound": "default", "priority": "high",
                "channelId": CHANNEL, "data": {"url": url, "eventType": "staff_alert", "kind": alert_type},
            } for device in batch]
            tickets = _send_batch(messages)
            with _tx() as conn:
                for device, ticket in zip(batch, tickets):
                    if ticket.get("status") == "ok" and ticket.get("id"):
                        conn.execute("INSERT OR IGNORE INTO staff_push_tickets (ticket_id, device_id, event_key, created_at) VALUES (?, ?, ?, ?)",
                                     (ticket["id"], device["id"], event_key, time.time()))
                        sent += 1
                    elif (ticket.get("details") or {}).get("error") == "DeviceNotRegistered":
                        conn.execute("DELETE FROM staff_push_devices WHERE id = ?", (device["id"],))
                        unregistered += 1
        with _tx() as conn:
            conn.execute("UPDATE staff_push_events SET recipients = ?, sent = ?, failed = ? WHERE event_key = ?",
                         (len(devices), sent, max(0, len(devices) - sent - unregistered), event_key))
        return sent
    except Exception:  # noqa: BLE001 - a push problem must never reach the code that raised the alert
        logger.warning("Staff push failed for %s", alert_type, exc_info=True)
        return 0


def content_for(alert: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """{title, body, url, alert_type} for an alert dict (the arguments of notify_staff_alert), or None when this alert
    type does not become a push. The body is the county from the alert's title, nothing from the report itself."""
    alert_type = str(alert.get("alert_type") or "")
    spec = PUSH_TYPES.get(alert_type)
    if spec is None:
        if alert_type and alert_type not in _unknown_logged:
            _unknown_logged.add(alert_type)
            logger.info("Staff push: alert type %r is Discord-only", alert_type)
        return None
    original = str(alert.get("title") or "")
    place = original.split(":", 1)[1].strip() if ":" in original else ""
    place = re.sub(r"\s+", " ", place)[:60]
    match = re.search(spec["id_pattern"], str(alert.get("admin_path") or "")) if spec["id_pattern"] else None
    url = f"{spec['route']}/{match.group(1)}" if match else spec["route"]
    return {"alert_type": alert_type, "title": spec["title"], "body": place or "Open the app to review", "url": url}


def dispatch(alert: Dict[str, Any], *, background: bool = True) -> bool:
    """Called from notify_staff_alert for every staff alert. Sends the push for the alert types that have one, on its own
    thread so it neither waits for nor can slow down Discord. Never raises. True if a push was started."""
    try:
        if not enabled() or is_sandbox():  # decided here, in the calling thread
            return False
        content = content_for(alert)
        if content is None:
            return False
        args = (content["alert_type"], content["title"], content["body"], content["url"])
        if background:
            threading.Thread(target=send, args=args, daemon=True).start()
        else:
            send(*args)
        return True
    except Exception:  # noqa: BLE001
        logger.warning("Staff push dispatch failed", exc_info=True)
        return False


# ----------------------------------------------------------------------------------------------- scheduled upkeep

def check_receipts(limit: int = 1000) -> int:
    """Look up Expo's delivery receipts for tickets older than 15 minutes; drop devices Expo says are gone. Returns how many
    were checked. Run by the scheduler."""
    if not enabled():
        return 0
    with _tx() as conn:
        rows = conn.execute("SELECT ticket_id, device_id FROM staff_push_tickets WHERE checked_at IS NULL AND created_at < ? LIMIT ?",
                            (time.time() - 900, limit)).fetchall()
    if not rows:
        return 0
    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.post(EXPO_RECEIPTS_URL, headers=_headers(), json={"ids": [row["ticket_id"] for row in rows]})
        response.raise_for_status()
        receipts = response.json().get("data", {})
    except (httpx.HTTPError, ValueError):
        logger.warning("Staff push receipts could not be fetched", exc_info=True)
        return 0
    now = time.time()
    with _tx() as conn:
        for row in rows:
            receipt = receipts.get(row["ticket_id"])
            if not receipt:
                continue
            error = (receipt.get("details") or {}).get("error") or ""
            conn.execute("UPDATE staff_push_tickets SET status = ?, error = ?, checked_at = ? WHERE ticket_id = ?",
                         (receipt.get("status") or "", error, now, row["ticket_id"]))
            if error == "DeviceNotRegistered":
                conn.execute("DELETE FROM staff_push_devices WHERE id = ?", (row["device_id"],))
    return len(rows)


def purge(days: int = RETENTION_DAYS) -> int:
    """Delete old send records. Devices are never purged here: they live and die with their sessions."""
    if not enabled():
        return 0
    cutoff = time.time() - days * 86400
    with _tx() as conn:
        removed = conn.execute("DELETE FROM staff_push_tickets WHERE created_at < ?", (cutoff,)).rowcount
        removed += conn.execute("DELETE FROM staff_push_events WHERE created_at < ?", (cutoff,)).rowcount
        # Devices on sessions that ended long ago are only clutter (the eligibility query already ignores them).
        removed += conn.execute(
            "DELETE FROM staff_push_devices WHERE session_id NOT IN (SELECT id FROM admin_sessions WHERE revoked_at IS NULL AND expires_at > ?)",
            (time.time(),)).rowcount
    return removed
