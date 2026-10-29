"""Staff sign-in for the mobile app: the same accounts, password checks, second step, throttling and sessions as the website,
with tokens returned in the response body instead of cookies.

    POST /api/admin/mobile/login            email + password -> tokens, or a second step is needed
    POST /api/admin/mobile/login/totp       the authenticator code for that second step
    POST /api/admin/mobile/login/recovery   or a recovery code
    POST /api/admin/mobile/login/passkey/options   a passkey (Face ID, fingerprint, device PIN): the challenge to sign
    POST /api/admin/mobile/login/passkey/verify    the signed answer -> tokens (replaces both the password and the second step)
    POST /api/admin/mobile/refresh          a new access token; the refresh token is replaced every time (sessions.rotate)
    POST /api/admin/mobile/logout           ends the session (and with it that phone's staff push)

Everything is off (404) unless STAFF_API_ENABLED is set. Nothing here is reachable without credentials except the sign-in
steps themselves, which share the website's lockout counters.

The app keeps the refresh token in the device keychain and the access token in memory. Access tokens are short (15 minutes);
revocation is checked on every request regardless, so signing out or disabling an account works at once.
"""
from __future__ import annotations

import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from core import sandbox_data, security
from core.admin_auth import events, mfa, permissions, sessions, throttle, users
from core.rate_limit import client_ip
from routers.admin_auth import PasskeyVerify, passkey_limiter  # the website's passkey sign-in shares this body model and its rate limit

ACCESS_MINUTES = int(os.getenv("MOBILE_ACCESS_TOKEN_MINUTES", "15"))
GENERIC_LOGIN_ERROR = "Invalid credentials"


def staff_api_enabled() -> bool:
    return os.getenv("STAFF_API_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")


def _require_enabled() -> None:
    if not staff_api_enabled():
        raise HTTPException(status_code=404, detail="Not found")


router = APIRouter(prefix="/api/admin/mobile", tags=["admin-mobile"], dependencies=[Depends(_require_enabled)])


class LoginRequest(BaseModel):
    email: str = Field(..., max_length=320)
    password: str = Field(..., max_length=500)


class MfaCode(BaseModel):
    mfa_token: str
    code: str = Field(..., max_length=64)


class RefreshRequest(BaseModel):
    refresh_token: str = Field(..., max_length=200)


class LogoutRequest(BaseModel):
    refresh_token: Optional[str] = Field(None, max_length=200)


def account_view(user: dict) -> dict:
    return {**users.public_view(user), "permissions": sorted(permissions.permissions_for(user["role"]))}


def _guarded(scope: str, key: str, ip: str, admin_id: Optional[int] = None) -> None:
    try:
        throttle.check(scope, key, ip)
    except HTTPException as error:
        events.record("locked_out", admin_id, ip, str(error.headers.get("Retry-After", "")) if error.headers else "")
        raise


def _fail(scope: str, key: str, ip: str, admin_id: Optional[int], kind: str) -> None:
    throttle.fail(scope, key, ip)
    events.record(kind, admin_id, ip)


def _start(user: dict, request: Request, method: str) -> dict:
    """Open a mobile session and return the tokens."""
    ip = client_ip(request)
    session_id, refresh_token = sessions.create(user["id"], ip, request.headers.get("user-agent", ""), method)
    users.touch_login(user["id"])
    events.record("login_ok", user["id"], ip, method)
    return {
        "success": True,
        "access_token": permissions.access_token_for(user, session_id, client="mobile", minutes=ACCESS_MINUTES),
        "refresh_token": refresh_token,
        "expires_in": ACCESS_MINUTES * 60,
        "user": account_view(user),
        "sandbox": users.is_sandbox_role(user["role"]),
    }


def _ready_for_session(user: dict) -> Optional[dict]:
    """A reason this account cannot get a session yet (as a response body), or None."""
    if users.is_sandbox_role(user["role"]) and not sandbox_data.sandbox_enabled():
        raise HTTPException(status_code=503, detail="Reviewer access is not available right now.")
    if user["must_change_password"]:
        # The app has no password-change screen: a temporary password is exchanged for a permanent one on the website.
        return {"success": False, "must_change_password": True,
                "message": "Sign in on the website once to choose a new password, then come back."}
    return None


@router.post("/login")
def login(data: LoginRequest, request: Request):
    ip = client_ip(request)
    email = users.normalize_email(data.email)
    _guarded("pw", email, ip)
    user = users.authenticate(email, data.password, client="mobile")
    if user is None:
        found = users.get_by_email(email)
        _fail("pw", email, ip, found["id"] if found else None, "login_fail")
        raise HTTPException(status_code=401, detail=GENERIC_LOGIN_ERROR)
    throttle.clear("pw", email)
    blocked = _ready_for_session(user)
    if blocked:
        return blocked
    if users.is_mfa_exempt(user["role"]):  # only the sandbox role; see users.ROLE_INFO
        return _start(user, request, "mobile")
    available = mfa.methods(user["id"])
    if available:
        return {"success": False, "mfa_required": True, "mfa_token": mfa.pending_token(user["id"]), "methods": available}
    if mfa.enforcement_active():
        return {"success": False, "mfa_enrollment_required": True,
                "message": "Set up two-step sign-in on the website first, then sign in here."}
    return _start(user, request, "mobile")


def _pending_user(token: str) -> dict:
    admin_id = mfa.admin_id_from_pending(token)
    user = users.get_by_id(admin_id) if admin_id is not None else None
    if user is None or user["disabled_at"] or users.is_expired(user):
        raise HTTPException(status_code=401, detail="That sign-in step expired. Start again.")
    return user


@router.post("/login/totp")
def login_totp(data: MfaCode, request: Request):
    ip = client_ip(request)
    user = _pending_user(data.mfa_token)
    key = str(user["id"])
    _guarded("totp", key, ip, user["id"])
    if not mfa.totp_verify(user["id"], data.code):
        _fail("totp", key, ip, user["id"], "mfa_fail")
        raise HTTPException(status_code=401, detail="That code is not right.")
    throttle.clear("totp", key)
    return _start(user, request, "mobile+totp")


@router.post("/login/recovery")
def login_recovery(data: MfaCode, request: Request):
    ip = client_ip(request)
    user = _pending_user(data.mfa_token)
    key = str(user["id"])
    _guarded("recovery", key, ip, user["id"])
    if not mfa.recovery_use(user["id"], data.code):
        _fail("recovery", key, ip, user["id"], "mfa_fail")
        raise HTTPException(status_code=401, detail="That recovery code is not valid or was already used.")
    throttle.clear("recovery", key)
    events.record("recovery_used", user["id"], ip)
    body = _start(user, request, "mobile+recovery")
    body["recovery_remaining"] = mfa.recovery_remaining(user["id"])
    return body


@router.post("/login/passkey/options", dependencies=[Depends(passkey_limiter)])
def login_passkey_options():
    """A challenge for the phone to sign with a passkey. Any passkey for this site will do (they are discoverable), so no
    email is needed first."""
    return mfa.passkey_auth_options()


@router.post("/login/passkey/verify", dependencies=[Depends(passkey_limiter)])
def login_passkey_verify(data: PasskeyVerify, request: Request):
    """A passkey is proof of possession plus the person's fingerprint, face or PIN (user verification is required), so it
    replaces both the password and the second step, exactly as on the website. Everything else a password sign-in checks still
    applies: the account must exist, be enabled and not expired, reviewer accounts need the sandbox switched on, and an
    account still on a temporary password is sent to the website first."""
    ip = client_ip(request)
    admin_id = mfa.passkey_auth_verify(data.challenge_id, data.credential)
    user = users.get_by_id(admin_id) if admin_id is not None else None
    if user is None or user["disabled_at"] or users.is_expired(user):
        events.record("passkey_fail", admin_id, ip)
        raise HTTPException(status_code=401, detail="That passkey was not accepted.")
    blocked = _ready_for_session(user)
    if blocked:
        return blocked
    return _start(user, request, "mobile+passkey")


@router.post("/refresh")
def refresh(data: RefreshRequest, request: Request):
    """Exchange the refresh token for a new access token AND a new refresh token; the old one stops working (after a short
    grace period for a lost response). A token that was already replaced and is replayed later ends the whole session."""
    rotated = sessions.rotate(data.refresh_token, client_ip(request))
    user = users.get_by_email(rotated[0]) if rotated else None
    if rotated is None or user is None or user["disabled_at"] or users.is_expired(user):
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")
    _, session_id, new_refresh = rotated
    return {
        "success": True,
        "access_token": permissions.access_token_for(user, session_id, client="mobile", minutes=ACCESS_MINUTES),
        "refresh_token": new_refresh,
        "expires_in": ACCESS_MINUTES * 60,
        "user": account_view(user),
        "sandbox": users.is_sandbox_role(user["role"]),
    }


@router.post("/logout")
def logout(data: Optional[LogoutRequest] = None):
    """End this session. Works with the access token, the refresh token, or both, so a phone whose access token has
    already expired can still sign out properly. Always succeeds: signing out must never be blocked."""
    session_id = security.current_session_id()
    if session_id:
        sessions.revoke(session_id)
    if data and data.refresh_token:
        sessions.revoke_by_refresh(data.refresh_token)
    return {"success": True}
