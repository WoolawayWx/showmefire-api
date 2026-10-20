"""Admin side of the API key application: review requests, approve (which creates the key and emails it) or deny, and manage issued keys.

The key is returned in full exactly once, in the approve/rotate response, so the admin page can show it for copying if the email could
not be sent. Only its hash is stored afterwards.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from core.security import require_admin
from services.api_keys import mail, store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/api-key-requests", tags=["api-keys-admin"])


class ApproveBody(BaseModel):
    note: str = Field("", max_length=1000, description="Shown to the applicant in the email and kept on the request")
    multiplier: int = Field(store.DEFAULT_MULTIPLIER, ge=1, le=store.MAX_MULTIPLIER, description="Times the public rate limits")
    expires_days: Optional[int] = Field(None, ge=1, le=3650, description="Days until the key stops working (none = no expiry)")
    send_email: bool = True


class DenyBody(BaseModel):
    note: str = Field("", max_length=1000, description="Internal reason, not sent")
    message: str = Field("", max_length=1000, description="Optional explanation included in the email to the applicant")
    send_email: bool = True


class RotateBody(BaseModel):
    send_email: bool = True


def _send(to: str, parts: tuple[str, str, str]) -> tuple[bool, Optional[str]]:
    try:
        mail.send(to, *parts)
        return True, None
    except Exception as error:  # noqa: BLE001 - reported to the admin, who can pass the key on by hand
        logger.warning("API key email failed: %s", error)
        return False, str(error)[:300]


def _pending_or_404(request_id: int) -> dict:
    row = store.get_request(request_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Request not found")
    return row


@router.get("/summary")
def summary(_admin: str = Depends(require_admin)):
    """Counts for the admin menu badge."""
    keys = store.list_keys()
    return {"pending": store.count_pending(), "active_keys": sum(1 for k in keys if k["active"]), "email_configured": mail.configured()}


@router.get("/requests")
def list_requests(status: Optional[str] = Query(None, pattern="^(pending|approved|denied)$"), _admin: str = Depends(require_admin)):
    return {"requests": store.list_requests(status)}


@router.get("/requests/{request_id}")
def get_request(request_id: int, _admin: str = Depends(require_admin)):
    row = _pending_or_404(request_id)
    key = store.get_key(row["key_id"]) if row.get("key_id") else None
    return {"request": row, "key": key}


@router.post("/requests/{request_id}/approve")
def approve(request_id: int, body: ApproveBody, admin: str = Depends(require_admin)):
    row = _pending_or_404(request_id)
    if row["status"] != "pending":
        raise HTTPException(status_code=409, detail=f"This request was already {row['status']}.")
    key, key_row = store.create_key(label=row["agency"], email=row["email"], request_id=request_id, multiplier=body.multiplier,
                                    expires_days=body.expires_days, created_by=admin)
    decided = store.decide_request(request_id, "approved", admin, body.note, key_id=key_row["id"])
    if decided is None:                                   # someone else decided it a moment ago: do not leave a stray key
        store.revoke_key(key_row["id"], admin)
        raise HTTPException(status_code=409, detail="This request was just decided by someone else.")
    sent, error = (False, None)
    if body.send_email:
        sent, error = _send(row["email"], mail.approved(row["name"], row["agency"], key, key_row["multiplier"], key_row["expires_at"], body.note))
        store.set_decision_email_sent(request_id, sent)
    return {"success": True, "key": key, "key_info": key_row, "email_sent": sent, "email_error": error,
            "request": store.get_request(request_id)}


@router.post("/requests/{request_id}/deny")
def deny(request_id: int, body: DenyBody, admin: str = Depends(require_admin)):
    row = _pending_or_404(request_id)
    decided = store.decide_request(request_id, "denied", admin, body.note)
    if decided is None:
        raise HTTPException(status_code=409, detail=f"This request was already {row['status']}.")
    sent, error = (False, None)
    if body.send_email:
        sent, error = _send(row["email"], mail.denied(row["name"], body.message))
        store.set_decision_email_sent(request_id, sent)
    return {"success": True, "email_sent": sent, "email_error": error, "request": store.get_request(request_id)}


@router.get("/keys")
def list_keys(_admin: str = Depends(require_admin)):
    return {"keys": store.list_keys()}


@router.post("/keys/{key_id}/revoke")
def revoke(key_id: int, admin: str = Depends(require_admin)):
    row = store.revoke_key(key_id, admin)
    if row is None:
        raise HTTPException(status_code=404, detail="Key not found or already revoked")
    from core.rate_limit import clear_database_key_cache

    clear_database_key_cache()
    return {"success": True, "key_info": row}


@router.post("/keys/{key_id}/rotate")
def rotate(key_id: int, body: RotateBody, admin: str = Depends(require_admin)):
    """Replace a key (for example after it was exposed): the old one stops working now, a new one with the same limits and expiry is made."""
    old = store.get_key(key_id)
    if old is None or old["status"] != "active":
        raise HTTPException(status_code=404, detail="Key not found or already revoked")
    request = store.get_request(old["request_id"]) if old.get("request_id") else None
    key, new = store.create_key(label=old["label"], email=old["email"], request_id=old.get("request_id"), multiplier=old["multiplier"],
                                expires_at=old["expires_at"], created_by=admin)
    store.revoke_key(key_id, admin)
    from core.rate_limit import clear_database_key_cache

    clear_database_key_cache()
    sent, error = (False, None)
    if body.send_email:
        sent, error = _send(old["email"], mail.approved((request or {}).get("name") or old["label"], old["label"], key, new["multiplier"],
                                                      new["expires_at"], "This key replaces your previous one, which no longer works."))
    return {"success": True, "key": key, "key_info": new, "email_sent": sent, "email_error": error}
