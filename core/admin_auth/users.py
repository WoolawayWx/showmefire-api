"""Admin accounts. The first start copies the ADMIN_EMAIL / ADMIN_PASSWORD_HASH environment pair into an owner account, so the
existing login keeps working unchanged; after that the table is the source of truth and the environment pair is only a seed."""
from __future__ import annotations

import re
import secrets
import time
from typing import Optional

from core import security

from . import db

# Every role signs in the same way. Today they all get the same access to the admin area; what each one may do beyond that is not
# decided yet, and owners alone manage accounts. Use core.admin_auth.access.require_role to restrict an endpoint when that changes.
ROLE_INFO = {
    "owner": {"label": "Owner", "description": "Manages admin accounts."},
    "admin": {"label": "Admin", "description": "Staff who run the site."},
    "staff": {"label": "Staff", "description": "Show Me Fire staff."},
    "partner": {"label": "Partner", "description": "Partner organizations."},
}
ROLES = tuple(ROLE_INFO)
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_dummy_hash: Optional[str] = None


class AccountError(ValueError):
    """A change the caller asked for that is not allowed (shown to the admin as-is)."""


def normalize_email(email: str) -> str:
    return (email or "").strip().lower()


def dummy_hash() -> str:
    """A real argon2 hash of a throwaway secret, verified against when an email is unknown so a miss costs the same as a wrong password."""
    global _dummy_hash
    if _dummy_hash is None:
        _dummy_hash = security.hash_password(secrets.token_urlsafe(24))
    return _dummy_hash


def _public(row) -> Optional[dict]:
    if row is None:
        return None
    return {key: row[key] for key in row.keys()}


def ensure_seeded() -> None:
    email = normalize_email(security.ADMIN_EMAIL or "")
    password_hash = security.ADMIN_PASSWORD_HASH
    if not email or not password_hash:
        return
    with db.tx() as conn:
        if conn.execute("SELECT COUNT(*) FROM admin_users").fetchone()[0] == 0:
            conn.execute(
                "INSERT INTO admin_users (email, display_name, password_hash, role, created_at) VALUES (?, '', ?, 'owner', ?)",
                (email, password_hash, time.time()),
            )


def get_by_email(email: str) -> Optional[dict]:
    ensure_seeded()
    with db.tx() as conn:
        return _public(conn.execute("SELECT * FROM admin_users WHERE email = ?", (normalize_email(email),)).fetchone())


def get_by_id(admin_id: int) -> Optional[dict]:
    with db.tx() as conn:
        return _public(conn.execute("SELECT * FROM admin_users WHERE id = ?", (admin_id,)).fetchone())


def list_users() -> list[dict]:
    ensure_seeded()
    with db.tx() as conn:
        return [_public(row) for row in conn.execute("SELECT * FROM admin_users ORDER BY id").fetchall()]


def authenticate(email: str, password: str) -> Optional[dict]:
    """The account for this email and password, or None. Always spends one password verification."""
    user = get_by_email(email)
    stored = user["password_hash"] if user else dummy_hash()
    try:
        matched = security.verify_password(password, stored)
    except Exception:  # a malformed stored hash must read as "wrong password", not a 500
        matched = False
    return user if (user and matched and not user["disabled_at"]) else None


def verify_password_for(user: dict, password: str) -> bool:
    try:
        return security.verify_password(password, user["password_hash"])
    except Exception:
        return False


def _check_email(email: str) -> str:
    cleaned = normalize_email(email)
    if not _EMAIL.match(cleaned) or len(cleaned) > 254:
        raise AccountError("Enter a valid email address.")
    return cleaned


def create_user(email: str, display_name: str, role: str, password: str, must_change: bool = True) -> dict:
    cleaned = _check_email(email)
    if role not in ROLES:
        raise AccountError("Unknown role.")
    with db.tx() as conn:
        if conn.execute("SELECT 1 FROM admin_users WHERE email = ?", (cleaned,)).fetchone():
            raise AccountError("An account with that email already exists.")
        cursor = conn.execute(
            "INSERT INTO admin_users (email, display_name, password_hash, role, must_change_password, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (cleaned, display_name.strip()[:80], security.hash_password(password), role, 1 if must_change else 0, time.time()),
        )
        return _public(conn.execute("SELECT * FROM admin_users WHERE id = ?", (cursor.lastrowid,)).fetchone())


def set_password(admin_id: int, password: str, must_change: bool = False) -> None:
    with db.tx() as conn:
        conn.execute(
            "UPDATE admin_users SET password_hash = ?, must_change_password = ? WHERE id = ?",
            (security.hash_password(password), 1 if must_change else 0, admin_id),
        )


def update_profile(admin_id: int, display_name: Optional[str] = None, email: Optional[str] = None) -> dict:
    with db.tx() as conn:
        if display_name is not None:
            conn.execute("UPDATE admin_users SET display_name = ? WHERE id = ?", (display_name.strip()[:80], admin_id))
        if email is not None:
            cleaned = _check_email(email)
            clash = conn.execute("SELECT id FROM admin_users WHERE email = ? AND id != ?", (cleaned, admin_id)).fetchone()
            if clash:
                raise AccountError("An account with that email already exists.")
            conn.execute("UPDATE admin_users SET email = ? WHERE id = ?", (cleaned, admin_id))
        return _public(conn.execute("SELECT * FROM admin_users WHERE id = ?", (admin_id,)).fetchone())


def _active_owner_count(conn, excluding: Optional[int] = None) -> int:
    query = "SELECT COUNT(*) FROM admin_users WHERE role = 'owner' AND disabled_at IS NULL"
    params: tuple = ()
    if excluding is not None:
        query += " AND id != ?"
        params = (excluding,)
    return conn.execute(query, params).fetchone()[0]


def set_role(admin_id: int, role: str) -> None:
    if role not in ROLES:
        raise AccountError("Unknown role.")
    with db.tx() as conn:
        if role != "owner" and _active_owner_count(conn, excluding=admin_id) == 0:
            raise AccountError("There must always be at least one active owner.")
        conn.execute("UPDATE admin_users SET role = ? WHERE id = ?", (role, admin_id))


def set_disabled(admin_id: int, disabled: bool) -> None:
    with db.tx() as conn:
        if disabled and _active_owner_count(conn, excluding=admin_id) == 0:
            raise AccountError("There must always be at least one active owner.")
        conn.execute("UPDATE admin_users SET disabled_at = ? WHERE id = ?", (time.time() if disabled else None, admin_id))


def touch_login(admin_id: int) -> None:
    with db.tx() as conn:
        conn.execute("UPDATE admin_users SET last_login_at = ? WHERE id = ?", (time.time(), admin_id))


def temporary_password() -> str:
    return secrets.token_urlsafe(12)


def public_view(user: dict) -> dict:
    return {
        "id": user["id"],
        "email": user["email"],
        "display_name": user["display_name"],
        "role": user["role"],
        "disabled": bool(user["disabled_at"]),
        "must_change_password": bool(user["must_change_password"]),
        "created_at": user["created_at"],
        "last_login_at": user["last_login_at"],
    }
