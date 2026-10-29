"""Admin sign-in, sessions, account settings, second factors and (for owners) other admin accounts.

Sign-in is a password step and, once the account has an authenticator app or passkey, a second step; a passkey can also sign in on
its own. Everything lives under /api/admin so the dev-server proxy and CORS settings already cover it."""
from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from core import security
from core.admin_auth import events, mfa, permissions, sessions, throttle, users
from core.admin_session import set_auth_cookie
from core.rate_limit import RateLimiter, client_ip
from core.security import require_admin

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin-auth"])
passkey_limiter = RateLimiter("admin_passkey", 30)

GENERIC_LOGIN_ERROR = "Invalid credentials"


# ----------------------------------------------------------------------------- helpers

def _clear_cookies(response: Response) -> None:
    for name in (security.ACCESS_COOKIE_NAME, security.REFRESH_COOKIE_NAME):
        response.delete_cookie(
            key=name, httponly=True, secure=security.AUTH_COOKIE_SECURE, samesite=security.AUTH_COOKIE_SAMESITE,
            domain=security.AUTH_COOKIE_DOMAIN, path="/",
        )


def _start_session(response: Response, request: Request, user: dict, method: str) -> None:
    ip = client_ip(request)
    session_id, refresh_token = sessions.create(user["id"], ip, request.headers.get("user-agent", ""), method)
    access = permissions.access_token_for(user, session_id)
    set_auth_cookie(response, security.ACCESS_COOKIE_NAME, access, security.ACCESS_TOKEN_EXPIRE_MINUTES * 60)
    set_auth_cookie(response, security.REFRESH_COOKIE_NAME, refresh_token, security.REFRESH_TOKEN_EXPIRE_DAYS * 86400)
    users.touch_login(user["id"])
    events.record("login_ok", user["id"], ip, method)


def _current_user(email: str = Depends(require_admin)) -> dict:
    user = users.get_by_email(email)
    if user is None or user["disabled_at"]:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user


def _owner(user: dict = Depends(_current_user)) -> dict:
    if user["role"] != "owner":
        raise HTTPException(status_code=403, detail="Only an owner can do that.")
    return user


def _fail_login(scope: str, key: str, ip: str, admin_id: Optional[int], kind: str) -> None:
    throttle.fail(scope, key, ip)
    events.record(kind, admin_id, ip)


def _locked_event(error: HTTPException, admin_id: Optional[int], ip: str) -> None:
    events.record("locked_out", admin_id, ip, str(error.headers.get("Retry-After", "")) if error.headers else "")


def _guarded(scope: str, key: str, ip: str, admin_id: Optional[int] = None) -> None:
    try:
        throttle.check(scope, key, ip)
    except HTTPException as error:
        _locked_event(error, admin_id, ip)
        raise


# ----------------------------------------------------------------------------- sign-in

class LoginRequest(BaseModel):
    email: str = Field(..., max_length=320)
    password: str = Field(..., max_length=500)


class MfaCode(BaseModel):
    mfa_token: str
    code: str = Field(..., max_length=64)


class PasskeyVerify(BaseModel):
    challenge_id: str
    credential: dict[str, Any]


@router.post("/login")
def login(data: LoginRequest, request: Request, response: Response):
    ip = client_ip(request)
    email = users.normalize_email(data.email)
    _guarded("pw", email, ip)
    user = users.authenticate(email, data.password)
    if user is None:
        found = users.get_by_email(email)
        _fail_login("pw", email, ip, found["id"] if found else None, "login_fail")
        raise HTTPException(status_code=401, detail=GENERIC_LOGIN_ERROR)
    throttle.clear("pw", email)
    available = mfa.methods(user["id"])
    if available:
        return {"success": False, "mfa_required": True, "mfa_token": mfa.pending_token(user["id"]), "methods": available}
    if mfa.enforcement_active():
        return {"success": False, "mfa_enrollment_required": True, "enroll_token": mfa.enroll_token(user["id"])}
    _start_session(response, request, user, "password")
    return {"success": True, "message": "Login successful", "role": user["role"],
            "must_change_password": bool(user["must_change_password"]), "mfa_setup_recommended": True}


def _pending_user(token: str) -> dict:
    admin_id = mfa.admin_id_from_pending(token)
    user = users.get_by_id(admin_id) if admin_id is not None else None
    if user is None or user["disabled_at"]:
        raise HTTPException(status_code=401, detail="That sign-in step expired. Start again.")
    return user


@router.post("/login/totp")
def login_totp(data: MfaCode, request: Request, response: Response):
    ip = client_ip(request)
    user = _pending_user(data.mfa_token)
    key = str(user["id"])
    _guarded("totp", key, ip, user["id"])
    if not mfa.totp_verify(user["id"], data.code):
        _fail_login("totp", key, ip, user["id"], "mfa_fail")
        raise HTTPException(status_code=401, detail="That code is not right.")
    throttle.clear("totp", key)
    _start_session(response, request, user, "password+totp")
    return {"success": True, "role": user["role"], "must_change_password": bool(user["must_change_password"])}


@router.post("/login/recovery")
def login_recovery(data: MfaCode, request: Request, response: Response):
    ip = client_ip(request)
    user = _pending_user(data.mfa_token)
    key = str(user["id"])
    _guarded("recovery", key, ip, user["id"])
    if not mfa.recovery_use(user["id"], data.code):
        _fail_login("recovery", key, ip, user["id"], "mfa_fail")
        raise HTTPException(status_code=401, detail="That recovery code is not valid or was already used.")
    throttle.clear("recovery", key)
    events.record("recovery_used", user["id"], ip)
    _start_session(response, request, user, "password+recovery")
    return {"success": True, "role": user["role"], "recovery_remaining": mfa.recovery_remaining(user["id"]),
            "must_change_password": bool(user["must_change_password"])}


@router.post("/login/passkey/options", dependencies=[Depends(passkey_limiter)])
def login_passkey_options():
    return mfa.passkey_auth_options()


@router.post("/login/passkey/verify", dependencies=[Depends(passkey_limiter)])
def login_passkey_verify(data: PasskeyVerify, request: Request, response: Response):
    admin_id = mfa.passkey_auth_verify(data.challenge_id, data.credential)
    user = users.get_by_id(admin_id) if admin_id is not None else None
    if user is None or user["disabled_at"]:
        events.record("passkey_fail", None, client_ip(request))
        raise HTTPException(status_code=401, detail="That passkey was not accepted.")
    _start_session(response, request, user, "passkey")
    return {"success": True, "role": user["role"], "must_change_password": bool(user["must_change_password"])}


# ----------------------------------------------------------------------------- session

class TokenProbe(BaseModel):
    token: Optional[str] = None


@router.post("/verify")
def verify_session(data: Optional[TokenProbe] = None):
    """Session probe used by the website. Unauthenticated callers get 200 with valid=false."""
    email = security.verify_token(data.token if data else None)
    user = users.get_by_email(email) if email else None
    if user is None or user["disabled_at"]:
        return {"valid": False}
    return {"valid": True, "email": user["email"], "display_name": user["display_name"], "role": user["role"],
            "must_change_password": bool(user["must_change_password"]),
            # What this account may do, so the website can hide actions it would be refused. The server still checks every request.
            "permissions": sorted(permissions.permissions_for(user["role"]))}


@router.post("/refresh")
def refresh_session(request: Request, response: Response):
    """Renew the access cookie from the refresh cookie."""
    renewed = sessions.renew(request.cookies.get(security.REFRESH_COOKIE_NAME, ""))
    if renewed is None:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")
    email, session_id = renewed
    access = permissions.access_token_for(users.get_by_email(email), session_id)
    set_auth_cookie(response, security.ACCESS_COOKIE_NAME, access, security.ACCESS_TOKEN_EXPIRE_MINUTES * 60)
    return {"success": True, "message": "Token refreshed"}


@router.post("/logout")
def logout(request: Request, response: Response):
    sessions.revoke_by_refresh(request.cookies.get(security.REFRESH_COOKIE_NAME, ""))
    _clear_cookies(response)
    return {"success": True, "message": "Logged out"}


@router.post("/logout-all")
def logout_all(request: Request, response: Response, user: dict = Depends(_current_user)):
    count = sessions.revoke_all(user["id"])
    events.record("logout_all", user["id"], client_ip(request), f"{count} sessions")
    _clear_cookies(response)
    return {"success": True, "revoked": count}


# ----------------------------------------------------------------------------- account

@router.get("/roles")
def list_roles(all: bool = False, _user: dict = Depends(_current_user)):
    """The roles an owner can pick when creating or editing an account. `?all=true` also lists roles with their own
    management screen (reviewers), for showing a label."""
    return {"roles": [{"id": key, **info} for key, info in users.ROLE_INFO.items() if all or info.get("assignable", True)]}


@router.get("/me")
def me(user: dict = Depends(_current_user)):
    return {**users.public_view(user), "mfa": mfa.status(user["id"]), "permissions": sorted(permissions.permissions_for(user["role"]))}


class PasswordChange(BaseModel):
    current_password: str = Field(..., max_length=500)
    new_password: str = Field(..., max_length=500)


@router.post("/password")
def change_password(data: PasswordChange, request: Request, user: dict = Depends(_current_user)):
    ip = client_ip(request)
    key = users.normalize_email(user["email"])
    _guarded("pw", key, ip, user["id"])
    if not users.verify_password_for(user, data.current_password):
        _fail_login("pw", key, ip, user["id"], "password_check_fail")
        raise HTTPException(status_code=401, detail="Your current password is not right.")
    problem = security.password_strength_error(data.new_password)
    if problem:
        raise HTTPException(status_code=422, detail=problem)
    users.set_password(user["id"], data.new_password)
    others = sessions.revoke_all(user["id"], except_session=security.current_session_id())
    events.record("password_changed", user["id"], ip, f"{others} other sessions ended")
    return {"success": True, "other_sessions_ended": others}


class ProfileUpdate(BaseModel):
    display_name: Optional[str] = Field(None, max_length=80)
    email: Optional[str] = Field(None, max_length=254)
    current_password: str = Field("", max_length=500)


@router.patch("/profile")
def update_profile(data: ProfileUpdate, request: Request, user: dict = Depends(_current_user)):
    ip = client_ip(request)
    changing_email = data.email is not None and users.normalize_email(data.email) != user["email"]
    if changing_email:
        key = users.normalize_email(user["email"])
        _guarded("pw", key, ip, user["id"])
        if not users.verify_password_for(user, data.current_password):
            _fail_login("pw", key, ip, user["id"], "password_check_fail")
            raise HTTPException(status_code=401, detail="Enter your current password to change your email.")
    try:
        updated = users.update_profile(user["id"], data.display_name, data.email if changing_email else None)
    except users.AccountError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    if changing_email:
        # The open session's token still names the old address; end the others and ask for a fresh sign-in on the next renewal.
        sessions.revoke_all(user["id"], except_session=security.current_session_id())
        events.record("email_changed", user["id"], ip)
    return users.public_view(updated)


@router.get("/sessions")
def list_sessions(user: dict = Depends(_current_user)):
    return {"sessions": sessions.list_for(user["id"], security.current_session_id())}


@router.delete("/sessions/{session_id}")
def end_session(session_id: str, request: Request, user: dict = Depends(_current_user)):
    if not sessions.revoke(session_id, user["id"]):
        raise HTTPException(status_code=404, detail="Session not found")
    events.record("session_revoked", user["id"], client_ip(request))
    return {"success": True}


@router.get("/events")
def list_events(limit: int = 50, user: dict = Depends(_current_user)):
    return {"events": events.recent(user["id"], limit)}


# ----------------------------------------------------------------------------- second factors

class Confirmed(BaseModel):
    confirm_token: str = ""


class TotpConfirm(Confirmed):
    code: str = Field(..., max_length=16)


class PasskeyRegister(Confirmed):
    challenge_id: str
    credential: dict[str, Any]
    name: str = Field("", max_length=60)


def _session_email(token: Optional[str] = None) -> Optional[str]:
    """The signed-in admin's email if there is a valid session (cookie, or the `token` query parameter the tests use)."""
    return security.verify_token(token)


def _factor_actor(request: Request, confirm_token: str, email: Optional[str]) -> dict:
    """Who is changing their second factors: an admin session that just re-confirmed with their password, or a sign-in that is
    waiting to enrol its first factor (password already checked, so no extra confirmation)."""
    enrol = request.headers.get("x-admin-enroll-token")
    if enrol:
        admin_id = mfa.admin_id_from_enroll(enrol)
        user = users.get_by_id(admin_id) if admin_id is not None else None
        if user is None or user["disabled_at"]:
            raise HTTPException(status_code=401, detail="That sign-in step expired. Start again.")
        return user
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    if security.verify_confirm_token(confirm_token, "mfa_manage") != email:
        raise HTTPException(status_code=403, detail="Confirm with your password first.")
    user = users.get_by_email(email)
    if user is None or user["disabled_at"]:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user


def _first_factor_codes(user: dict, had_factor: bool) -> Optional[list[str]]:
    """Adding the first factor also issues the recovery codes, so there is never a factor without a way back in."""
    if had_factor or mfa.recovery_remaining(user["id"]):
        return None
    return mfa.recovery_regenerate(user["id"])


def _may_drop_factor(user: dict, removing_passkey: Optional[int] = None, removing_totp: bool = False) -> None:
    remaining = len(mfa.passkeys_for(user["id"])) - (1 if removing_passkey is not None else 0) + (
        1 if mfa.totp_enabled(user["id"]) and not removing_totp else 0)
    if remaining <= 0 and mfa.enforcement_active():
        raise HTTPException(status_code=409, detail="Add another sign-in method before removing your last one.")


@router.post("/mfa/totp/setup")
def totp_setup(data: Confirmed, request: Request, email: Optional[str] = Depends(_session_email)):
    user = _factor_actor(request, data.confirm_token, email)
    return mfa.totp_begin(user["id"], user["email"])


@router.post("/mfa/totp/confirm")
def totp_confirm(data: TotpConfirm, request: Request, email: Optional[str] = Depends(_session_email)):
    user = _factor_actor(request, data.confirm_token, email)
    had_factor = mfa.has_factor(user["id"])
    if not mfa.totp_confirm(user["id"], data.code):
        raise HTTPException(status_code=422, detail="That code is not right. Check the time on your phone and try again.")
    events.record("totp_added", user["id"], client_ip(request))
    return {"success": True, "recovery_codes": _first_factor_codes(user, had_factor)}


@router.post("/mfa/totp/remove")
def totp_remove(data: Confirmed, request: Request, email: Optional[str] = Depends(_session_email)):
    user = _factor_actor(request, data.confirm_token, email)
    _may_drop_factor(user, removing_totp=True)
    mfa.totp_remove(user["id"])
    events.record("totp_removed", user["id"], client_ip(request))
    return {"success": True}


@router.post("/mfa/passkeys/options")
def passkey_register_options(data: Confirmed, request: Request, email: Optional[str] = Depends(_session_email)):
    return mfa.passkey_register_options(_factor_actor(request, data.confirm_token, email))


@router.post("/mfa/passkeys/verify")
def passkey_register_verify(data: PasskeyRegister, request: Request, email: Optional[str] = Depends(_session_email)):
    user = _factor_actor(request, data.confirm_token, email)
    had_factor = mfa.has_factor(user["id"])
    result = mfa.passkey_register_verify(user, data.challenge_id, data.credential, data.name)
    events.record("passkey_added", user["id"], client_ip(request))
    return {**result, "recovery_codes": _first_factor_codes(user, had_factor)}


@router.post("/mfa/passkeys/{passkey_id}/remove")
def passkey_remove(passkey_id: int, data: Confirmed, request: Request, email: Optional[str] = Depends(_session_email)):
    user = _factor_actor(request, data.confirm_token, email)
    _may_drop_factor(user, removing_passkey=passkey_id)
    if not mfa.passkey_remove(user["id"], passkey_id):
        raise HTTPException(status_code=404, detail="Passkey not found")
    events.record("passkey_removed", user["id"], client_ip(request))
    return {"passkeys": mfa.passkeys_for(user["id"])}


@router.post("/mfa/recovery/regenerate")
def recovery_regenerate(data: Confirmed, request: Request, email: Optional[str] = Depends(_session_email)):
    user = _factor_actor(request, data.confirm_token, email)
    if not mfa.has_factor(user["id"]):
        raise HTTPException(status_code=409, detail="Set up a passkey or an authenticator app first.")
    events.record("recovery_regenerated", user["id"], client_ip(request))
    return {"recovery_codes": mfa.recovery_regenerate(user["id"])}


# ----------------------------------------------------------------------------- step-up confirmation by passkey

@router.post("/confirm/passkey/options", dependencies=[Depends(passkey_limiter)])
def confirm_passkey_options(user: dict = Depends(_current_user)):
    if not mfa.passkeys_for(user["id"]):
        raise HTTPException(status_code=409, detail="No passkey is set up.")
    return mfa.passkey_auth_options(user["id"])


class ConfirmPasskey(PasskeyVerify):
    action: str = Field(..., max_length=120)


@router.post("/confirm/passkey/verify", dependencies=[Depends(passkey_limiter)])
def confirm_passkey_verify(data: ConfirmPasskey, user: dict = Depends(_current_user)):
    if mfa.passkey_auth_verify(data.challenge_id, data.credential, user["id"]) != user["id"]:
        raise HTTPException(status_code=401, detail="That passkey was not accepted.")
    return {"confirm_token": security.create_confirm_token(user["email"], data.action)}


# ----------------------------------------------------------------------------- owner: other admin accounts

class NewUser(BaseModel):
    email: str = Field(..., max_length=254)
    display_name: str = Field("", max_length=80)
    role: str = "admin"


class UserUpdate(BaseModel):
    display_name: Optional[str] = Field(None, max_length=80)
    role: Optional[str] = None
    disabled: Optional[bool] = None


def _target(admin_id: int) -> dict:
    user = users.get_by_id(admin_id)
    if user is None:
        raise HTTPException(status_code=404, detail="Account not found")
    return user


@router.get("/users")
def list_users(_owner: dict = Depends(_owner)):
    return {"users": [{**users.public_view(u), "factors": {"passkeys": len(mfa.passkeys_for(u["id"])), "totp": mfa.totp_enabled(u["id"])}} for u in users.list_users()]}


@router.post("/users")
def create_user(data: NewUser, request: Request, owner: dict = Depends(_owner)):
    temporary = users.temporary_password()
    try:
        if not users.role_flag(data.role, "assignable", True):
            raise users.AccountError("Reviewer accounts are created from the Reviewer accounts panel.")
        created = users.create_user(data.email, data.display_name, data.role, temporary, created_by=owner["email"])
    except users.AccountError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    events.record("user_created", owner["id"], client_ip(request), created["email"])
    return {"user": users.public_view(created), "temporary_password": temporary}


@router.patch("/users/{admin_id}")
def update_user(admin_id: int, data: UserUpdate, request: Request, owner: dict = Depends(_owner)):
    target = _target(admin_id)
    try:
        if data.display_name is not None:
            users.update_profile(admin_id, display_name=data.display_name)
        if data.role is not None and data.role != target["role"]:
            users.set_role(admin_id, data.role)
            # A new role means new permissions: end the account's sessions (and so its staff push) so it signs in again
            # under the new role instead of carrying on with the old one. Keep the owner's own current session.
            sessions.revoke_all(admin_id, except_session=security.current_session_id() if admin_id == owner["id"] else None)
            events.record("role_changed", owner["id"], client_ip(request), f"{target['email']}: {target['role']} -> {data.role}")
        if data.disabled is not None and data.disabled != bool(target["disabled_at"]):
            if admin_id == owner["id"]:
                raise users.AccountError("You cannot disable your own account.")
            users.set_disabled(admin_id, data.disabled)
            if data.disabled:
                sessions.revoke_all(admin_id)
            events.record("user_disabled" if data.disabled else "user_enabled", owner["id"], client_ip(request), target["email"])
    except users.AccountError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return users.public_view(_target(admin_id))


@router.post("/users/{admin_id}/reset")
def reset_user(admin_id: int, request: Request, owner: dict = Depends(_owner)):
    """For a locked-out admin: a new temporary password, their second factors cleared, every session ended."""
    target = _target(admin_id)
    temporary = users.temporary_password()
    users.set_password(admin_id, temporary, must_change=True)
    mfa.clear_all(admin_id)
    sessions.revoke_all(admin_id)
    events.record("user_reset", owner["id"], client_ip(request), target["email"])
    return {"user": users.public_view(_target(admin_id)), "temporary_password": temporary}
