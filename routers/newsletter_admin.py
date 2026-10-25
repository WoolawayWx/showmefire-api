"""Admin endpoints for the email lists: counts, masked list, reveal, unsubscribe, audit log. Every route requires an admin session."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from core.security import require_admin
from services import newsletter_admin as service
from services.graphics_email import set_audience_contact_unsubscribed

router = APIRouter(prefix="/api/admin/newsletter", tags=["newsletter-admin"])


@router.get("/summary")
def newsletter_summary(_admin: str = Depends(require_admin)):
    return service.summary()


@router.get("/subscribers")
def newsletter_subscribers(
    q: str = Query("", max_length=100),
    list_id: str = Query("", alias="list", max_length=60),
    status: str = Query("active", pattern="^(active|unsubscribed|all)$"),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=service.PAGE_SIZE_MAX),
    admin: str = Depends(require_admin),
):
    return service.list_subscribers(admin, q, list_id, status, page, page_size)


@router.post("/subscribers/{subscriber_id}/reveal")
def newsletter_reveal(subscriber_id: int, admin: str = Depends(require_admin)):
    result = service.reveal(admin, subscriber_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Subscriber not found")
    return result


@router.post("/subscribers/{subscriber_id}/unsubscribe")
def newsletter_unsubscribe(subscriber_id: int, admin: str = Depends(require_admin)):
    email = service.get_email(subscriber_id)
    if email is None:
        raise HTTPException(status_code=404, detail="Subscriber not found")
    try:
        set_audience_contact_unsubscribed(email, True)
    except Exception as exc:  # same behavior as the public unsubscribe: keep Resend and the site in step
        raise HTTPException(status_code=502, detail=f"Unable to synchronize unsubscribe: {exc}") from exc
    service.mark_unsubscribed(admin, subscriber_id)
    return {"unsubscribed": True}


@router.get("/audit")
def newsletter_audit(limit: int = Query(50, ge=1, le=200), _admin: str = Depends(require_admin)):
    return {"entries": service.recent_audit(limit)}
