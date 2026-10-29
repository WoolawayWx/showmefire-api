"""Owner tools for app-store reviewer accounts and the sandbox they use (the website's "Reviewer accounts" panel).

    GET    /api/admin/reviewers                       reviewer accounts and the sandbox's state
    POST   /api/admin/reviewers                       make one (a label, a note, how many days); the password is shown once
    POST   /api/admin/reviewers/{id}/reset-password   a new password; ends that account's sessions
    POST   /api/admin/reviewers/{id}/extend           push the expiry date out
    POST   /api/admin/sandbox/reset                   put the demo data back to how it started

Owners only (`accounts.manage`). Disabling or re-enabling a reviewer uses the normal account route
(PATCH /api/admin/users/{id}), which also ends their sessions. These routes live under /api/admin, so a reviewer's own session
can never reach them (core/admin_auth/lockdown.py).
"""
from __future__ import annotations

import os
import re
import secrets
import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from core import sandbox_data
from core.admin_auth import events, permissions, sessions, users
from core.admin_auth.permissions import require_permission
from core.rate_limit import client_ip

router = APIRouter(prefix="/api/admin", tags=["admin-reviewers"])

OWNER_ONLY = require_permission(permissions.ACCOUNTS_MANAGE)
EMAIL_DOMAIN = os.getenv("REVIEWER_EMAIL_DOMAIN", "reviewer.showmefire.org")
DEFAULT_DAYS = 30
MAX_DAYS = 180


class NewReviewer(BaseModel):
    label: str = Field(..., min_length=3, max_length=80, description="Who it is for, e.g. 'Apple review, 1.3.0'")
    note: str = Field("", max_length=300)
    days: int = Field(DEFAULT_DAYS, ge=1, le=MAX_DAYS)


class Extension(BaseModel):
    days: int = Field(DEFAULT_DAYS, ge=1, le=MAX_DAYS)


def _owner() -> dict:
    actor = permissions.resolve_actor()
    user = users.get_by_id(actor.admin_id) if actor and actor.admin_id else None
    if user is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user


def _reviewer(admin_id: int) -> dict:
    user = users.get_by_id(admin_id)
    if user is None or not users.is_sandbox_role(user["role"]):
        raise HTTPException(status_code=404, detail="Reviewer account not found")
    return user


def _login_for(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")[:20] or "review"
    return f"{slug}-{secrets.token_hex(2)}@{EMAIL_DOMAIN}"


def _password() -> str:
    return secrets.token_urlsafe(14)


def _view(user: dict) -> dict:
    return {**users.public_view(user), "expired": users.is_expired(user)}


@router.get("/reviewers")
def list_reviewers(_email: str = Depends(OWNER_ONLY)):
    idle = sandbox_data.seconds_idle()
    return {
        "reviewers": [_view(u) for u in users.list_users() if users.is_sandbox_role(u["role"])],
        "sandbox": {
            "enabled": sandbox_data.sandbox_enabled(),
            "seconds_since_last_use": None if idle is None else int(idle),
            "has_unreset_changes": sandbox_data._marker("dirty").exists(),
            "auto_reset_after_minutes": sandbox_data.IDLE_RESET_SECONDS // 60,
        },
    }


@router.post("/reviewers")
def create_reviewer(data: NewReviewer, request: Request, _email: str = Depends(OWNER_ONLY)):
    owner = _owner()
    password = _password()
    try:
        created = users.create_user(
            _login_for(data.label), data.label, "reviewer", password, must_change=False,
            expires_at=time.time() + data.days * 86400, note=data.note, created_by=owner["email"],
        )
    except users.AccountError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    events.record("reviewer_created", owner["id"], client_ip(request), f"{created['email']} ({data.days} days)")
    return {"reviewer": _view(created), "email": created["email"], "password": password}


@router.post("/reviewers/{admin_id}/reset-password")
def reset_reviewer_password(admin_id: int, request: Request, _email: str = Depends(OWNER_ONLY)):
    owner = _owner()
    target = _reviewer(admin_id)
    password = _password()
    users.set_password(admin_id, password, must_change=False)
    sessions.revoke_all(admin_id)
    events.record("reviewer_password_reset", owner["id"], client_ip(request), target["email"])
    return {"reviewer": _view(_reviewer(admin_id)), "email": target["email"], "password": password}


@router.post("/reviewers/{admin_id}/extend")
def extend_reviewer(admin_id: int, data: Extension, request: Request, _email: str = Depends(OWNER_ONLY)):
    owner = _owner()
    target = _reviewer(admin_id)
    users.set_expiry(admin_id, time.time() + data.days * 86400)
    events.record("reviewer_extended", owner["id"], client_ip(request), f"{target['email']} (+{data.days} days)")
    return {"reviewer": _view(_reviewer(admin_id))}


@router.post("/sandbox/reset")
def reset_sandbox(request: Request, _email: str = Depends(OWNER_ONLY)):
    """Back to the demo data as it started. A reviewer partway through a review will see their changes disappear, so this is
    for between reviews; the automatic reset waits until nobody has used the sandbox for a while."""
    owner = _owner()
    if not sandbox_data.sandbox_enabled():
        raise HTTPException(status_code=409, detail="The reviewer sandbox is switched off (SANDBOX_ENABLED).")
    sandbox_data.reset_sandbox()
    events.record("sandbox_reset", owner["id"], client_ip(request), "")
    return {"success": True}
