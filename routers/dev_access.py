"""Access codes for the development website build (the `dev` branch deployment is closed to everyone without one).

Public side (/v1/dev-access, called by the gate page and the website server):
  POST /redeem    a code -> a signed session token the website keeps in an HttpOnly cookie
  POST /admin     admin email + password -> the same kind of token, without a code
  POST /validate  the website re-confirms a token every few minutes; a revoked code or ended session answers valid=false
  POST /logout    ends the session behind a token

Admin side (/admin/dev-access-codes; the page route /admin/dev-access must not be shadowed by it): make codes with any text the admin
chooses, edit them, see every use, revoke a code (ending its sessions) or end single sessions.

Tokens are signed with DEV_ACCESS_SECRET (services/dev_access/tokens.py). Guessing is slowed by a per-minute limit and a cap on
failed attempts per address.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Deque, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from core.admin_auth import mfa as admin_mfa, throttle as admin_throttle, users as admin_users
from core.rate_limit import RateLimiter, client_ip
from core.security import require_admin
from services.dev_access import store, tokens

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/dev-access", tags=["dev-access"])
admin_router = APIRouter(prefix="/admin/dev-access-codes", tags=["dev-access-admin"])
redeem_limiter = RateLimiter("dev_access_redeem", 10)
MAX_FAILURES = 10
FAILURE_WINDOW_SECONDS = 15 * 60
_failures: Dict[str, Deque[float]] = {}
_failures_lock = threading.Lock()


def _recent_failures(ip: str, now: float) -> Deque[float]:
    hits = _failures.setdefault(ip, deque())
    while hits and hits[0] <= now - FAILURE_WINDOW_SECONDS:
        hits.popleft()
    return hits


def _guard(ip: str) -> None:
    with _failures_lock:
        if len(_recent_failures(ip, time.monotonic())) >= MAX_FAILURES:
            raise HTTPException(status_code=429, detail="Too many wrong attempts. Wait 15 minutes and try again.")


def _failed(ip: str) -> None:
    with _failures_lock:
        _recent_failures(ip, time.monotonic()).append(time.monotonic())


def reset_failures() -> None:
    with _failures_lock:
        _failures.clear()


def _require_configured() -> None:
    if not tokens.configured():
        raise HTTPException(status_code=503, detail="Development access is not set up on the server (DEV_ACCESS_SECRET).")


def _issued(session: dict) -> dict:
    expires = store.parse_iso(session["expires_at"])
    return {"success": True, "token": tokens.issue(session["id"], int(expires.timestamp())), "expires_at": session["expires_at"]}


# --- public -------------------------------------------------------------------------------------------

class RedeemBody(BaseModel):
    code: str = Field(..., max_length=200)


class AdminBody(BaseModel):
    email: str = Field(..., max_length=320)
    password: str = Field(..., max_length=500)
    code: str = Field("", max_length=64)


class TokenBody(BaseModel):
    token: str = Field(..., max_length=1024)


@router.post("/redeem", dependencies=[Depends(redeem_limiter)])
def redeem(body: RedeemBody, request: Request):
    _require_configured()
    ip = client_ip(request)
    _guard(ip)
    result = store.redeem(body.code, ip=ip, user_agent=request.headers.get("user-agent", ""))
    if result is None:
        _failed(ip)
        raise HTTPException(status_code=401, detail="That code is not valid, or it has expired or been retracted.")
    logger.info("Dev access code %s used (session %s)", result["code"]["id"], result["session"]["id"])
    return {**_issued(result["session"]), "label": result["code"]["label"]}


@router.post("/admin", dependencies=[Depends(redeem_limiter)])
def admin_sign_in(body: AdminBody, request: Request):
    _require_configured()
    ip = client_ip(request)
    _guard(ip)
    email = admin_users.normalize_email(body.email)
    admin_throttle.check("pw", email, ip)
    user = admin_users.authenticate(email, body.password)
    if user is None:
        _failed(ip)
        admin_throttle.fail("pw", email, ip)
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    if admin_mfa.enforcement_active() and admin_mfa.has_factor(user["id"]):
        # Same bar as the admin sign-in: a password alone is not enough once second factors are enforced.
        code = body.code.strip()
        if not code:
            raise HTTPException(status_code=401, detail="Enter the code from your authenticator app or a recovery code.")
        admin_throttle.check("totp", str(user["id"]), ip)
        if not (admin_mfa.totp_verify(user["id"], code) or admin_mfa.recovery_use(user["id"], code)):
            _failed(ip)
            admin_throttle.fail("totp", str(user["id"]), ip)
            raise HTTPException(status_code=401, detail="That code is not right.")
    admin_throttle.clear("pw", email)
    session = store.open_admin_session(user["email"], ip=ip, user_agent=request.headers.get("user-agent", ""))
    return _issued(session)


@router.post("/validate")
def validate(body: TokenBody):
    """For the website server. Answers 200 either way so the caller can tell "no longer valid" from "API unreachable"."""
    _require_configured()
    payload = tokens.read(body.token)
    if payload is None:
        return {"valid": False, "reason": "invalid"}
    session = store.confirm_session(payload["sid"])
    if session is None:
        return {"valid": False, "reason": "ended"}
    return {"valid": True, **_issued(session)}


@router.post("/logout")
def logout(body: TokenBody):
    payload = tokens.read(body.token)
    if payload is not None:
        store.end_session(payload["sid"], "visitor", "signed out")
    return {"success": True}


# --- admin --------------------------------------------------------------------------------------------

def _future_utc(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    value = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    if value <= datetime.now(timezone.utc):
        raise HTTPException(status_code=422, detail="The expiry must be in the future.")
    return value


class CreateCodeBody(BaseModel):
    code: str = Field(..., max_length=200, description="Any text the admin chooses; matched ignoring case and extra spaces")
    label: str = Field(..., min_length=1, max_length=200, description="Who the code is for, e.g. the organization's name")
    note: str = Field("", max_length=1000)
    expires_at: Optional[datetime] = Field(None, description="When the code and every session from it stop working (none = never)")
    max_uses: Optional[int] = Field(None, ge=1, le=100000, description="How many times the code can be entered (none = unlimited)")
    session_days: int = Field(store.DEFAULT_SESSION_DAYS, ge=1, le=store.MAX_SESSION_DAYS,
                              description="How long a browser stays signed in after entering the code")


class UpdateCodeBody(BaseModel):
    label: Optional[str] = Field(None, min_length=1, max_length=200)
    note: Optional[str] = Field(None, max_length=1000)
    expires_at: Optional[datetime] = None
    clear_expiry: bool = False
    max_uses: Optional[int] = Field(None, ge=1, le=100000)
    clear_max_uses: bool = False
    session_days: Optional[int] = Field(None, ge=1, le=store.MAX_SESSION_DAYS)


class RevokeBody(BaseModel):
    end_sessions: bool = True


@admin_router.get("")
def list_codes(_admin: str = Depends(require_admin)):
    codes = store.list_codes()
    return {"codes": codes, "configured": tokens.configured(),
            "active_codes": sum(1 for c in codes if c["active"]), "active_sessions": len(store.list_sessions(active_only=True))}


@admin_router.post("", status_code=201)
def create_code(body: CreateCodeBody, admin: str = Depends(require_admin)):
    problem = store.code_problem(body.code)
    if problem:
        raise HTTPException(status_code=422, detail=problem)
    row = store.create_code(code=body.code, label=body.label, note=body.note, expires_at=_future_utc(body.expires_at),
                            max_uses=body.max_uses, session_days=body.session_days, created_by=admin)
    if row is None:
        raise HTTPException(status_code=409, detail="That code is already in use (codes ignore case). Choose a different one.")
    return {"success": True, "code": row}


@admin_router.patch("/{code_id}")
def update_code(code_id: int, body: UpdateCodeBody, _admin: str = Depends(require_admin)):
    if store.get_code(code_id) is None:
        raise HTTPException(status_code=404, detail="Code not found")
    changes = {k: v for k, v in body.model_dump(exclude={"clear_expiry", "clear_max_uses"}).items() if v is not None}
    if "expires_at" in changes:
        changes["expires_at"] = _future_utc(changes["expires_at"])
    if body.clear_expiry:
        changes["expires_at"] = None
    if body.clear_max_uses:
        changes["max_uses"] = None
    return {"success": True, "code": store.update_code(code_id, changes)}


@admin_router.post("/{code_id}/revoke")
def revoke_code(code_id: int, body: RevokeBody, admin: str = Depends(require_admin)):
    row = store.revoke_code(code_id, admin, end_sessions=body.end_sessions)
    if row is None:
        raise HTTPException(status_code=404, detail="Code not found or already retracted")
    return {"success": True, "code": row}


@admin_router.post("/{code_id}/end-sessions")
def end_code_sessions(code_id: int, admin: str = Depends(require_admin)):
    if store.get_code(code_id) is None:
        raise HTTPException(status_code=404, detail="Code not found")
    return {"success": True, "ended": store.end_sessions_for_code(code_id, admin)}


@admin_router.get("/sessions")
def list_sessions(code_id: Optional[int] = Query(None), kind: Optional[str] = Query(None, pattern="^(code|admin)$"),
                  active: bool = Query(False), _admin: str = Depends(require_admin)):
    return {"sessions": store.list_sessions(code_id=code_id, kind=kind, active_only=active)}


@admin_router.post("/sessions/{session_id}/end")
def end_session(session_id: int, admin: str = Depends(require_admin)):
    row = store.end_session(session_id, admin)
    if row is None:
        raise HTTPException(status_code=404, detail="Session not found or already ended")
    return {"success": True, "session": row}
