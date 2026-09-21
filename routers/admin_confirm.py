"""Re-authentication step for sensitive admin actions (manual model/forecast
runs, activate/rollback) - a logged-in admin re-enters their password to get
a short-lived, action-scoped token, which the sensitive endpoint itself then
requires. Deliberately does not touch the session cookies set by
POST /api/admin/login - this is a pure verify-and-mint step."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.security import ADMIN_PASSWORD_HASH, create_confirm_token, verify_password, verify_token

router = APIRouter(prefix="/api/admin", tags=["admin-confirm"])


def _require_admin(token: Optional[str] = None) -> str:
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return email


class ConfirmPasswordRequest(BaseModel):
    password: str
    action: str


@router.post("/verify-password")
async def verify_admin_password(payload: ConfirmPasswordRequest, token: Optional[str] = None):
    email = _require_admin(token)
    if not ADMIN_PASSWORD_HASH or not verify_password(payload.password, ADMIN_PASSWORD_HASH):
        raise HTTPException(status_code=401, detail="Incorrect password")
    return {"confirm_token": create_confirm_token(email, payload.action)}
