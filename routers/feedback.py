"""
Public feedback form and the admin review queue.

POST /api/feedback is the only unauthenticated write path here - honeypot
field, per-IP + global rate limiting, and no CAPTCHA (this is a much lower
stakes surface than fires.py's public reporting, see that file for the
heavier anti-abuse pattern this borrows a reduced version of). Every
submission lands as status='new'; nothing here is publicly readable.
"""
import hashlib
import hmac
import logging
import os
import re
from datetime import datetime, timezone
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, field_validator, model_validator

from core.database import (
    consume_feedback_submission_quota,
    create_feedback_submission,
    list_feedback,
    update_feedback_status,
)
from core.security import SECRET_KEY, verify_token

logger = logging.getLogger(__name__)

router = APIRouter(tags=["feedback"])

FEEDBACK_LIMIT_PER_HOUR = int(os.getenv("FEEDBACK_LIMIT_PER_HOUR", "5"))
FEEDBACK_LIMIT_PER_DAY = int(os.getenv("FEEDBACK_LIMIT_PER_DAY", "15"))
FEEDBACK_GLOBAL_LIMIT_PER_DAY = int(os.getenv("FEEDBACK_GLOBAL_LIMIT_PER_DAY", "500"))

# Same distrust-proxy-headers-by-default posture as fires.py - see that
# file's TRUST_PROXY_HEADERS comment for why this isn't safe to flip until
# the origin is confirmed firewalled to Cloudflare.
TRUST_PROXY_HEADERS = os.getenv("TRUST_PROXY_HEADERS", "false").lower() == "true"

CATEGORIES = {"general", "website", "app", "forecast", "data", "other"}
# Categories that reveal a platform sub-dropdown, and what it's allowed to contain.
PLATFORM_OPTIONS = {
    "website": {"desktop", "mobile"},
    "app": {"ios", "android"},
}
ALLOWED_DETAIL_KEYS = {"platform"}
STATUSES = {"new", "read", "archived"}

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _require_admin(token: Optional[str] = None) -> str:
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return email


def _client_ip(request: Request) -> str:
    """Resolve the visitor IP from Cloudflare/proxy headers - mirrors fires.py::_client_ip."""
    if TRUST_PROXY_HEADERS:
        cf_ip = (request.headers.get("cf-connecting-ip") or "").strip()
        if cf_ip:
            return cf_ip
        forwarded = (request.headers.get("x-forwarded-for") or "").split(",")
        if forwarded and forwarded[0].strip():
            return forwarded[0].strip()
    return (request.client.host if request.client else "") or "unknown"


def _ip_bucket_key(ip: str) -> str:
    """HMAC the client IP so no raw address is ever persisted - mirrors fires.py::_ip_bucket_key."""
    secret = os.getenv("FEEDBACK_IP_SALT", "").strip() or SECRET_KEY
    return hmac.new(secret.encode(), ip.encode(), hashlib.sha256).hexdigest()


def _clean_text(value: Optional[str], *, required: bool, field: str) -> str:
    text = _CONTROL_CHARS.sub("", str(value or "")).strip()
    if required and not text:
        raise ValueError(f"{field} must not be empty")
    return text


class FeedbackCreate(BaseModel):
    name: str = Field(default="", max_length=120)
    email: str = Field(default="", max_length=254)
    category: str = Field(min_length=1, max_length=32)
    details: Dict[str, str] = Field(default_factory=dict)
    message: str = Field(min_length=5, max_length=4000)
    website: str = Field(default="", max_length=200)  # honeypot; must stay empty

    @field_validator("name")
    @classmethod
    def _clean_name(cls, value: str) -> str:
        return _clean_text(value, required=False, field="name")

    @field_validator("email")
    @classmethod
    def _validate_email(cls, value: str) -> str:
        text = str(value or "").strip()
        if text and not _EMAIL_RE.fullmatch(text):
            raise ValueError("email must be a valid address")
        return text

    @field_validator("category")
    @classmethod
    def _validate_category(cls, value: str) -> str:
        text = str(value or "").strip().lower()
        if text not in CATEGORIES:
            raise ValueError(f"unknown category: {text!r}")
        return text

    @field_validator("message")
    @classmethod
    def _clean_message(cls, value: str) -> str:
        text = _clean_text(value, required=True, field="message")
        if len(text) < 5:
            raise ValueError("message must be at least 5 characters")
        return text

    @field_validator("website")
    @classmethod
    def _reject_honeypot(cls, value: str) -> str:
        if str(value or "").strip():
            raise ValueError("invalid submission")
        return ""

    @model_validator(mode="after")
    def _validate_details(self) -> "FeedbackCreate":
        unknown_keys = set(self.details) - ALLOWED_DETAIL_KEYS
        if unknown_keys:
            raise ValueError(f"unsupported details field(s): {', '.join(sorted(unknown_keys))}")
        platform = self.details.get("platform")
        if platform is not None:
            allowed = PLATFORM_OPTIONS.get(self.category)
            if not allowed or platform not in allowed:
                raise ValueError(f"invalid platform {platform!r} for category {self.category!r}")
        return self


class FeedbackStatusUpdate(BaseModel):
    status: str

    @field_validator("status")
    @classmethod
    def _validate_status(cls, value: str) -> str:
        text = str(value or "").strip().lower()
        if text not in STATUSES:
            raise ValueError(f"unknown status: {text!r}")
        return text


@router.post("/api/feedback", status_code=201)
def submit_feedback(payload: FeedbackCreate, request: Request):
    """Submit anonymous site feedback (public endpoint)."""
    client_ip = _client_ip(request)
    ip_hash = _ip_bucket_key(client_ip)
    now = datetime.now(timezone.utc)

    quota = consume_feedback_submission_quota(ip_hash, now, FEEDBACK_LIMIT_PER_HOUR, FEEDBACK_LIMIT_PER_DAY)
    if not quota["allowed"]:
        raise HTTPException(
            status_code=429,
            detail="Too much feedback from this network. Please try again later.",
            headers={"Retry-After": str(quota["retry_after"])},
        )

    global_quota = consume_feedback_submission_quota("__global__", now, 10**9, FEEDBACK_GLOBAL_LIMIT_PER_DAY)
    if not global_quota["allowed"]:
        raise HTTPException(
            status_code=429,
            detail="Too much feedback right now. Please try again later.",
            headers={"Retry-After": str(global_quota["retry_after"])},
        )

    feedback = create_feedback_submission(
        name=payload.name,
        email=payload.email,
        category=payload.category,
        details=payload.details,
        message=payload.message,
        submitter_ip_hash=ip_hash,
    )

    return {"success": True, "feedback": {"id": feedback["id"], "status": feedback["status"],
                                          "submitted_at": feedback["created_at"]}}


@router.get("/api/admin/feedback")
def admin_list_feedback(token: Optional[str] = None, status: Optional[str] = None,
                        limit: int = 50, offset: int = 0):
    _require_admin(token)
    if status and status not in STATUSES:
        raise HTTPException(status_code=400, detail=f"unknown status: {status!r}")
    try:
        items = list_feedback(status=status, limit=limit, offset=offset)
    except Exception as e:
        logger.error(f"Failed to list feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to load feedback")
    return {"feedback": items}


@router.patch("/api/admin/feedback/{feedback_id}")
def admin_update_feedback_status(feedback_id: int, payload: FeedbackStatusUpdate, token: Optional[str] = None):
    _require_admin(token)
    try:
        updated = update_feedback_status(feedback_id, payload.status)
    except Exception as e:
        logger.error(f"Failed to update feedback {feedback_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to update feedback")
    if updated is None:
        raise HTTPException(status_code=404, detail="Feedback not found")
    return {"success": True, "feedback": updated}
