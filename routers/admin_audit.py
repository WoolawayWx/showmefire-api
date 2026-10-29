"""The moderation log for the website's admin area: who approved, rejected, edited or deleted what, when and why.

    GET /api/admin/moderation-log      newest first; filter by subject, actor, incident or time; page with before_id

The same data the staff API serves to the app (/api/staff/v1/audit), here behind the website's own session so the log page works
whether or not the staff API is switched on. Needs `audit.view`.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from core.admin_auth import permissions
from services import audit

router = APIRouter(prefix="/api/admin", tags=["admin-audit"])


@router.get("/moderation-log")
def moderation_log(subject_type: Optional[str] = None, subject_id: Optional[str] = None, actor: Optional[str] = None,
                   incident_id: Optional[int] = None, since: Optional[float] = None, until: Optional[float] = None,
                   before_id: Optional[int] = None, limit: int = 50,
                   _email: str = Depends(permissions.require_permission(permissions.AUDIT_VIEW))):
    if subject_type is not None and subject_type not in audit.SUBJECT_TYPES:
        raise HTTPException(status_code=400, detail="Unknown subject type")
    entries = audit.list_entries(subject_type=subject_type, subject_id=subject_id, actor=actor, incident_id=incident_id,
                                 since=since, until=until, before_id=before_id, limit=limit)
    return {"success": True, "entries": entries, "next_before_id": entries[-1]["id"] if len(entries) >= max(1, min(limit, 200)) else None,
            "subject_types": list(audit.SUBJECT_TYPES)}
