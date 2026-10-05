"""Read-only kiosk display auth (/display/dashboard on the frontend).

Separate from the admin and graphics login systems: a department admin
mints a shared 6-digit code from the admin UI (POST /api/admin/display-codes),
good for months, and anyone with physical/network access to the display
enters it once at /display/login. There is no per-user identity here - the
code itself is the credential, and it can be revoked at any time.
"""
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from core import security
from core.database import (
    count_burn_ban_submissions,
    count_fire_events,
    create_display_access_code,
    get_display_code_settings,
    list_active_display_access_codes,
    list_display_access_codes,
    revoke_display_access_code,
    set_display_code_settings,
    touch_display_access_code_usage,
)
from core.security import hash_password, verify_password, verify_token

router = APIRouter(tags=["display-auth"])

DISPLAY_MAX_VALID_DAYS = 400


def _require_admin(token: Optional[str] = None) -> str:
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return email


def _set_display_cookie(response: Response, value: str, max_age: int) -> None:
    response.set_cookie(
        key=security.DISPLAY_ACCESS_COOKIE_NAME, value=value, max_age=max_age,
        httponly=True, secure=security.AUTH_COOKIE_SECURE,
        samesite=security.AUTH_COOKIE_SAMESITE, domain=security.AUTH_COOKIE_DOMAIN,
        path="/",
    )


def _clear_display_cookie(response: Response) -> None:
    response.delete_cookie(
        key=security.DISPLAY_ACCESS_COOKIE_NAME, path="/",
        domain=security.AUTH_COOKIE_DOMAIN,
    )


class DisplayLoginRequest(BaseModel):
    code: str = Field(pattern=r"^\d{6}$")


@router.post("/api/display/auth/login")
def display_login(payload: DisplayLoginRequest, response: Response):
    now = datetime.now(timezone.utc)
    for candidate in list_active_display_access_codes():
        if not verify_password(payload.code, candidate["code_hash"]):
            continue
        expires_at = datetime.fromisoformat(candidate["expires_at"]).replace(tzinfo=timezone.utc)
        remaining = expires_at - now
        if remaining.total_seconds() <= 0:
            continue
        token = security.create_display_access_token(candidate["id"], remaining)
        _set_display_cookie(response, token, int(remaining.total_seconds()))
        touch_display_access_code_usage(candidate["id"])
        return {"success": True}
    raise HTTPException(status_code=401, detail="Invalid or expired code")


@router.post("/api/display/auth/verify")
def display_verify(request: Request):
    token = request.cookies.get(security.DISPLAY_ACCESS_COOKIE_NAME)
    code_id = security.verify_display_access_token(token)
    return {"valid": code_id is not None}


@router.post("/api/display/auth/logout")
def display_logout(response: Response):
    _clear_display_cookie(response)
    return {"success": True}


def _require_display_code_id(request: Request) -> int:
    token = request.cookies.get(security.DISPLAY_ACCESS_COOKIE_NAME)
    code_id = security.verify_display_access_token(token)
    if code_id is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return code_id


@router.get("/api/display/review-queue")
def display_review_queue(request: Request):
    """Pending-item counts (unreviewed fire reports, unreviewed burn-ban
    submissions) for the kiosk sidebar. Reuses the same data the admin
    badge counts draw from, but exposes only counts - not report contents
    or submitter contact info - and is gated by the display code instead of
    a full admin session, so the kiosk can show "N awaiting review" without
    needing an admin logged in."""
    _require_display_code_id(request)
    pending_reports = count_fire_events(status="pending")
    pending_burn_bans = count_burn_ban_submissions(status="pending")
    return {
        "success": True,
        "pending_fire_reports": pending_reports,
        "pending_burn_bans": pending_burn_bans,
    }


@router.get("/api/display/settings")
def display_settings(request: Request):
    """Per-screen config (which views/tiles to show, timing, confidence
    threshold) so one code system can drive multiple kiosks that each want
    something different - see core.database.DEFAULT_DISPLAY_SETTINGS.
    code_id is included so the kiosk's own in-screen settings editor (gated
    behind a fresh admin login, see pages/display/dashboard.vue) knows which
    code's settings to fetch/save via the /api/admin/display-codes endpoints."""
    code_id = _require_display_code_id(request)
    return {"success": True, "code_id": code_id, "settings": get_display_code_settings(code_id)}


class DisplayCodeCreateRequest(BaseModel):
    label: str = ""
    valid_days: int = Field(default=180, ge=1, le=DISPLAY_MAX_VALID_DAYS)


@router.get("/api/admin/display-codes")
def admin_list_display_codes(token: Optional[str] = None):
    _require_admin(token)
    codes = list_display_access_codes()
    for item in codes:
        item.pop("code_hash", None)
    return {"success": True, "codes": codes}


@router.post("/api/admin/display-codes", status_code=201)
def admin_create_display_code(payload: DisplayCodeCreateRequest, token: Optional[str] = None):
    admin_email = _require_admin(token)
    code = f"{secrets.randbelow(1_000_000):06d}"
    expires_at = datetime.now(timezone.utc) + timedelta(days=payload.valid_days)
    row = create_display_access_code(
        label=payload.label.strip(),
        code_hash=hash_password(code),
        created_by=admin_email,
        expires_at=expires_at.isoformat(),
    )
    return {
        "success": True,
        "id": row["id"],
        "label": row["label"],
        "code": code,
        "expires_at": row["expires_at"],
    }


@router.delete("/api/admin/display-codes/{code_id}")
def admin_revoke_display_code(code_id: int, token: Optional[str] = None):
    _require_admin(token)
    if not revoke_display_access_code(code_id):
        raise HTTPException(status_code=404, detail="Code not found or already revoked")
    return {"success": True}


@router.get("/api/admin/display-codes/{code_id}/settings")
def admin_get_display_code_settings(code_id: int, token: Optional[str] = None):
    _require_admin(token)
    return {"success": True, "settings": get_display_code_settings(code_id)}


@router.put("/api/admin/display-codes/{code_id}/settings")
def admin_set_display_code_settings(code_id: int, settings: Dict[str, Any], token: Optional[str] = None):
    """Body is the complete settings object (the admin UI always submits the
    full, already-merged-with-defaults form) - stored as-is; new default keys
    added later still apply to old codes via merge_display_settings at read
    time, so this never needs a migration when the schema grows."""
    _require_admin(token)
    if not set_display_code_settings(code_id, settings):
        raise HTTPException(status_code=404, detail="Code not found")
    return {"success": True, "settings": get_display_code_settings(code_id)}
