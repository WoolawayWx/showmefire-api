"""Failed-attempt limits for the admin login steps.

A lock only ever covers the factor that was guessed at (the password, the authenticator code, the recovery code), for a short
window that clears itself, and never the passkey path. The per-IP count is a wide brake on top: behind a proxy that hides client
addresses everyone shares one count, so it is set well above what a person fumbling a password can reach."""
from __future__ import annotations

import os
import time

from fastapi import HTTPException

from . import db

WINDOW_SECONDS = 15 * 60


def _limit(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(f"ADMIN_LOGIN_MAX_{name.upper()}", str(default))))
    except ValueError:
        return default


def limits() -> dict[str, int]:
    return {"pw": _limit("pw", 8), "totp": _limit("totp", 6), "recovery": _limit("recovery", 10), "ip": _limit("ip", 80)}


def _wait(conn, scope: str, key: str, limit: int, now: float) -> int:
    """Seconds until another attempt is allowed (0 if allowed now)."""
    rows = conn.execute(
        "SELECT ts FROM admin_login_failures WHERE scope = ? AND key = ? AND ts > ? ORDER BY ts",
        (scope, key, now - WINDOW_SECONDS),
    ).fetchall()
    if len(rows) < limit:
        return 0
    # Allowed again once enough of the oldest failures age out of the window.
    return max(1, int(rows[len(rows) - limit]["ts"] + WINDOW_SECONDS - now) + 1)


def check(scope: str, key: str, ip: str) -> None:
    """Raise 429 (with Retry-After) if this account factor or this address has failed too often recently."""
    configured = limits()
    now = time.time()
    with db.tx() as conn:
        wait = max(_wait(conn, scope, key, configured[scope], now), _wait(conn, "ip", ip, configured["ip"], now))
    if wait:
        raise HTTPException(status_code=429, detail="Too many failed attempts. Try again later.", headers={"Retry-After": str(wait)})


def fail(scope: str, key: str, ip: str) -> None:
    now = time.time()
    with db.tx() as conn:
        conn.execute("INSERT INTO admin_login_failures (scope, key, ts) VALUES (?, ?, ?)", (scope, key, now))
        conn.execute("INSERT INTO admin_login_failures (scope, key, ts) VALUES ('ip', ?, ?)", (ip, now))
        conn.execute("DELETE FROM admin_login_failures WHERE ts < ?", (now - 2 * WINDOW_SECONDS,))


def clear(scope: str, key: str) -> None:
    with db.tx() as conn:
        conn.execute("DELETE FROM admin_login_failures WHERE scope = ? AND key = ?", (scope, key))
