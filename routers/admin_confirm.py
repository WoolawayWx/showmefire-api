"""Re-authentication step for sensitive admin actions (manual model/forecast
runs, activate/rollback) - a logged-in admin re-enters their password to get
a short-lived, action-scoped token, which the sensitive endpoint itself then
requires. Deliberately does not touch the session cookies set by
POST /api/admin/login - this is a pure verify-and-mint step.

Once second-factor enforcement is active the password alone is not enough: send a code from the authenticator app (or a recovery
code) too, or confirm with a passkey instead (routers/admin_auth.py, /confirm/passkey/*)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from core.admin_auth import mfa, throttle, users
from core.rate_limit import client_ip
from core.security import ADMIN_PASSWORD_HASH, create_confirm_token, require_admin, verify_password

router = APIRouter(prefix="/api/admin", tags=["admin-confirm"])


class ConfirmPasswordRequest(BaseModel):
    password: str
    action: str
    code: str = ""


@router.post("/verify-password")
def verify_admin_password(payload: ConfirmPasswordRequest, request: Request, email: str = Depends(require_admin)):
    ip = client_ip(request)
    key = users.normalize_email(email)
    throttle.check("pw", key, ip)
    user = users.get_by_email(email)
    stored = user["password_hash"] if user else ADMIN_PASSWORD_HASH
    if not stored or not verify_password(payload.password, stored):
        throttle.fail("pw", key, ip)
        raise HTTPException(status_code=401, detail="Incorrect password")
    if user and mfa.enforcement_active() and mfa.has_factor(user["id"]):
        if not payload.code.strip():
            raise HTTPException(status_code=401, detail="Enter a code from your authenticator app, or confirm with your passkey.")
        throttle.check("totp", str(user["id"]), ip)
        if not (mfa.totp_verify(user["id"], payload.code) or mfa.recovery_use(user["id"], payload.code)):
            throttle.fail("totp", str(user["id"]), ip)
            raise HTTPException(status_code=401, detail="That code is not right.")
    throttle.clear("pw", key)
    return {"confirm_token": create_confirm_token(email, payload.action)}
