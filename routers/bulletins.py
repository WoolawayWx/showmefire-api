"""FireWx bulletin signup, preference, and administrative endpoints."""
from __future__ import annotations

import re
import os
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, field_validator

from core.database import (
    create_bulletin,
    get_bulletin,
    get_newsletter_account_by_token,
    get_newsletter_preferences,
    get_db_path,
    add_bulletin_attachment,
    list_newsletter_subscribers,
    list_bulletins,
    remove_bulletin_attachment,
    replace_newsletter_preferences,
    unsubscribe_newsletter,
    update_newsletter_account,
    schedule_bulletin,
    unschedule_bulletin,
    update_bulletin,
    upsert_newsletter_subscriber,
)
from core.security import require_admin
from services.graphics_email import (
    set_audience_contact_properties,
    send_bulletin_email,
    set_audience_contact_unsubscribed,
    upsert_audience_contact,
)
from services import newsletter_web
from services.bulletin_send import BulletinSendError, send_bulletin
from services.mobile_content import county_catalog

router = APIRouter(tags=["bulletins"])
ATTACHMENT_MAX_BYTES = 10 * 1024 * 1024
ATTACHMENT_MAX_TOTAL_BYTES = 25 * 1024 * 1024
ATTACHMENT_TYPES = {"application/pdf"}
DIRECT_RECIPIENT_MAX = 25

LEVEL_LABELS = {
    1: "Low",
    2: "Moderate",
    3: "Elevated",
    4: "Critical",
    5: "Extreme",
}
LIST_OPTIONS = {
    "show-me-fire-newsletter": "Show Me Fire Newsletter",
    "fire-weather-forecasts": "Fire Weather Forecasts",
    "show-me-fire-briefing-packet": "Operations Briefing Packet",
}
# The graphic-only list was folded into Fire Weather Forecasts (no counties selected = daily graphic).
# Old pages/clients may still send it; treat it as the forecast list.
LEGACY_LIST_ALIASES = {"forecast-graphic": "fire-weather-forecasts"}


class CountyPreference(BaseModel):
    county_fips: str
    level: int = Field(ge=1, le=5)

    @field_validator("county_fips")
    @classmethod
    def normalize_fips(cls, value: str) -> str:
        value = str(value).strip()
        if not re.fullmatch(r"29\d{3}", value):
            raise ValueError("county_fips must be a Missouri county FIPS")
        return value


class NewsletterSignup(BaseModel):
    email: str
    name: str = Field(min_length=1, max_length=200)
    affiliation: str = Field(default="", max_length=200)
    lists: List[str] = Field(min_length=1, max_length=len(LIST_OPTIONS))
    counties: List[CountyPreference] = Field(default_factory=list, max_length=115)

    @field_validator("email")
    @classmethod
    def validate_email(cls, value: str) -> str:
        normalized = str(value).strip().lower()
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", normalized):
            raise ValueError("must be a valid email address")
        return normalized

    @field_validator("name", "affiliation")
    @classmethod
    def normalize_profile_field(cls, value: str) -> str:
        return " ".join(str(value or "").strip().split())

    @field_validator("lists")
    @classmethod
    def validate_lists(cls, values: List[str]) -> List[str]:
        normalized = list(dict.fromkeys(LEGACY_LIST_ALIASES.get(str(value).strip(), str(value).strip()) for value in values))
        invalid = [value for value in normalized if value not in LIST_OPTIONS]
        if invalid:
            raise ValueError(f"Unknown email list: {invalid[0]}")
        if not normalized:
            raise ValueError("Select at least one email list")
        return normalized


class BulletinCreate(BaseModel):
    subject: str = Field(min_length=1, max_length=255)
    html_body: str = Field(min_length=1)
    text_body: str = Field(min_length=1)
    audience_list_id: str = ""
    direct_recipients: List[str] = Field(default_factory=list, max_length=DIRECT_RECIPIENT_MAX)

    @field_validator("audience_list_id")
    @classmethod
    def validate_audience_list(cls, value: str) -> str:
        value = str(value).strip()
        if not value:
            return ""
        if value not in LIST_OPTIONS:
            raise ValueError(f"Unknown email list: {value}")
        return value

    @field_validator("direct_recipients")
    @classmethod
    def validate_direct_recipients(cls, values: List[str]) -> List[str]:
        normalized = []
        for value in values:
            email = str(value).strip().lower()
            if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
                raise ValueError(f"Invalid recipient email: {value}")
            if email not in normalized:
                normalized.append(email)
        return normalized


class NewsletterManage(NewsletterSignup):
    token: str = Field(min_length=20)


def _validate_preferences(counties: List[CountyPreference]) -> List[dict]:
    valid_fips = {item["fips"] for item in county_catalog()}
    seen = set()
    normalized = []
    for preference in counties:
        if preference.county_fips not in valid_fips:
            raise HTTPException(status_code=422, detail=f"Unknown Missouri county: {preference.county_fips}")
        if preference.county_fips in seen:
            raise HTTPException(status_code=422, detail="Each county may be selected only once")
        seen.add(preference.county_fips)
        normalized.append({
            "county_fips": preference.county_fips,
            "min_danger_level": preference.level - 1,
        })
    return normalized


def _manage_url(token: str) -> str:
    base = os.getenv("PUBLIC_WEB_URL", "https://showmefire.org").rstrip("/")
    return f"{base}/comms/emails/manage?token={quote(token, safe='')}"


@router.post("/api/newsletter/signup")
def signup_newsletter(payload: NewsletterSignup):
    preferences = _validate_preferences(payload.counties)
    email = str(payload.email).strip().lower()
    try:
        contact_id = upsert_audience_contact(email)
        set_audience_contact_unsubscribed(email, False)
        subscriber = upsert_newsletter_subscriber(
            email,
            contact_id,
            payload.lists,
            payload.name,
            payload.affiliation,
        )
        saved = replace_newsletter_preferences(email, preferences)
        set_audience_contact_properties(
            email,
            {"manage_url": _manage_url(subscriber["manage_token"])},
        )
        return {
            "email": subscriber["email"],
            "name": subscriber["name"],
            "affiliation": subscriber["affiliation"],
            "manage_token": subscriber["manage_token"],
            "lists": [
                {"id": list_id, "label": LIST_OPTIONS[list_id]}
                for list_id in payload.lists
            ],
            "counties": [
                {
                    "county_fips": row["county_fips"],
                    "level": row["min_danger_level"] + 1,
                    "level_name": LEVEL_LABELS[row["min_danger_level"] + 1],
                }
                for row in saved
            ],
        }
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Newsletter signup failed: {exc}") from exc


@router.put("/api/newsletter/preferences")
def update_newsletter_preferences(payload: NewsletterSignup):
    return signup_newsletter(payload)


def _validated_newsletter_payload(payload: NewsletterSignup) -> List[dict]:
    return _validate_preferences(payload.counties)


@router.get("/api/newsletter/manage")
def get_newsletter_management(token: str):
    account = get_newsletter_account_by_token(token)
    if not account:
        raise HTTPException(status_code=404, detail="Management link is invalid or expired")
    return account


@router.put("/api/newsletter/manage")
def manage_newsletter(payload: NewsletterManage):
    preferences = _validated_newsletter_payload(payload)
    account = get_newsletter_account_by_token(payload.token)
    if not account:
        raise HTTPException(status_code=404, detail="Management link is invalid or expired")
    try:
        set_audience_contact_unsubscribed(account["email"], False)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Unable to synchronize subscription: {exc}") from exc
    account = update_newsletter_account(
        payload.token,
        payload.name,
        payload.affiliation,
        payload.lists,
        preferences,
    )
    if not account:
        raise HTTPException(status_code=404, detail="Management link is invalid or expired")
    return account


@router.post("/api/newsletter/unsubscribe")
def unsubscribe_newsletter_endpoint(token: str):
    account = get_newsletter_account_by_token(token)
    if not account:
        raise HTTPException(status_code=404, detail="Management link is invalid or expired")
    try:
        set_audience_contact_unsubscribed(account["email"], True)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Unable to synchronize unsubscribe: {exc}") from exc
    unsubscribe_newsletter(token)
    return {"unsubscribed": True}


@router.get("/api/newsletter/counties")
def newsletter_counties():
    return {
        "counties": county_catalog(),
        "levels": LEVEL_LABELS,
        "lists": [{"id": key, "label": value} for key, value in LIST_OPTIONS.items()],
    }


@router.get("/api/admin/bulletins")
def admin_list_bulletins(_admin: str = Depends(require_admin)):
    return {"bulletins": list_bulletins(), "lists": [{"id": key, "label": value} for key, value in LIST_OPTIONS.items()]}


@router.post("/api/admin/bulletins")
def admin_create_bulletin(payload: BulletinCreate, _admin: str = Depends(require_admin)):
    return create_bulletin(
        payload.subject, payload.html_body, payload.text_body, payload.audience_list_id, payload.direct_recipients
    )


@router.get("/api/admin/bulletins/{bulletin_id}")
def admin_get_bulletin(bulletin_id: int, _admin: str = Depends(require_admin)):
    bulletin = get_bulletin(bulletin_id)
    if not bulletin:
        raise HTTPException(status_code=404, detail="Bulletin not found")
    return bulletin


@router.put("/api/admin/bulletins/{bulletin_id}")
def admin_update_bulletin(bulletin_id: int, payload: BulletinCreate, _admin: str = Depends(require_admin)):
    bulletin = update_bulletin(
        bulletin_id, payload.subject, payload.html_body, payload.text_body,
        payload.audience_list_id, payload.direct_recipients,
    )
    if not bulletin:
        raise HTTPException(status_code=404, detail="Draft bulletin not found")
    return bulletin


@router.post("/api/admin/bulletins/{bulletin_id}/preview")
def admin_preview_bulletin(bulletin_id: int, _admin: str = Depends(require_admin)):
    bulletin = get_bulletin(bulletin_id)
    if not bulletin:
        raise HTTPException(status_code=404, detail="Bulletin not found")
    return {
        "id": bulletin["id"],
        "subject": bulletin["subject"],
        "html_body": bulletin["html_body"],
        "text_body": bulletin["text_body"],
        "attachments": bulletin.get("attachments", []),
        "direct_recipients": bulletin.get("direct_recipients", []),
    }


@router.post("/api/admin/bulletins/{bulletin_id}/attachments")
async def admin_upload_bulletin_attachment(
    bulletin_id: int,
    file: UploadFile = File(...),
    _admin: str = Depends(require_admin),
):
    bulletin = get_bulletin(bulletin_id)
    if not bulletin or bulletin["status"] != "draft":
        raise HTTPException(status_code=404, detail="Draft bulletin not found")
    content_type = (file.content_type or "").lower().strip()
    filename = (file.filename or "").strip()
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    allowed_image = content_type.startswith("image/") and suffix in {"jpg", "jpeg", "png", "gif", "webp"}
    allowed_pdf = content_type in ATTACHMENT_TYPES and suffix == "pdf"
    if not filename or not (allowed_image or allowed_pdf):
        raise HTTPException(status_code=415, detail="Only image files and PDFs are supported.")
    existing_total = sum(int(item.get("size", 0)) for item in bulletin.get("attachments", []))
    content = await file.read(ATTACHMENT_MAX_BYTES + 1)
    if len(content) > ATTACHMENT_MAX_BYTES:
        raise HTTPException(status_code=413, detail="Each attachment must be 10 MB or smaller.")
    if existing_total + len(content) > ATTACHMENT_MAX_TOTAL_BYTES:
        raise HTTPException(status_code=413, detail="Attachments for one email may total 25 MB or less.")
    attachment_id = secrets.token_urlsafe(12)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", filename)[:120] or f"attachment.{suffix}"
    attachment_dir = get_db_path().parent / "bulletin_attachments" / str(bulletin_id)
    attachment_dir.mkdir(parents=True, exist_ok=True)
    path = attachment_dir / f"{attachment_id}-{safe_name}"
    path.write_bytes(content)
    metadata = {
        "id": attachment_id,
        "filename": safe_name,
        "content_type": content_type,
        "size": len(content),
        "path": str(path),
    }
    result = add_bulletin_attachment(bulletin_id, metadata)
    if result is None:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=404, detail="Draft bulletin not found")
    return result


@router.delete("/api/admin/bulletins/{bulletin_id}/attachments/{attachment_id}")
def admin_delete_bulletin_attachment(
    bulletin_id: int,
    attachment_id: str,
    _admin: str = Depends(require_admin),
):
    bulletin = get_bulletin(bulletin_id)
    if not bulletin or bulletin["status"] != "draft":
        raise HTTPException(status_code=404, detail="Draft bulletin not found")
    attachment = next((item for item in bulletin.get("attachments", []) if item.get("id") == attachment_id), None)
    if not attachment:
        raise HTTPException(status_code=404, detail="Attachment not found")
    result = remove_bulletin_attachment(bulletin_id, attachment_id)
    Path(attachment["path"]).unlink(missing_ok=True)
    return result


@router.post("/api/admin/bulletins/{bulletin_id}/send")
def admin_send_bulletin(bulletin_id: int, _admin: str = Depends(require_admin)):
    try:
        return send_bulletin(bulletin_id)
    except BulletinSendError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


class BulletinSchedule(BaseModel):
    send_at: datetime  # must carry a timezone (the website sends UTC, e.g. 2026-12-01T14:00:00Z)

    @field_validator("send_at")
    @classmethod
    def _needs_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("send_at must include a timezone")
        return value.astimezone(timezone.utc)


@router.put("/api/admin/bulletins/{bulletin_id}/schedule")
def admin_schedule_bulletin(bulletin_id: int, payload: BulletinSchedule, admin: str = Depends(require_admin)):
    """Send this draft at a set time (the scheduler checks every minute). Calling it again changes the time."""
    if payload.send_at < datetime.now(timezone.utc) + timedelta(minutes=1):
        raise HTTPException(status_code=422, detail="Choose a time at least a minute from now.")
    bulletin = get_bulletin(bulletin_id)
    if not bulletin:
        raise HTTPException(status_code=404, detail="Bulletin not found")
    if not bulletin.get("audience_list_id") and not bulletin.get("direct_recipients"):
        raise HTTPException(status_code=409, detail="Choose an email list or add direct recipients before scheduling.")
    scheduled = schedule_bulletin(bulletin_id, payload.send_at.strftime("%Y-%m-%dT%H:%M:%SZ"), str(admin))
    if not scheduled:
        raise HTTPException(status_code=409, detail="Only a draft can be scheduled; this one has been sent or is sending.")
    return scheduled


@router.delete("/api/admin/bulletins/{bulletin_id}/schedule")
def admin_unschedule_bulletin(bulletin_id: int, _admin: str = Depends(require_admin)):
    bulletin = unschedule_bulletin(bulletin_id)
    if not bulletin:
        raise HTTPException(status_code=409, detail="Only a draft can be unscheduled.")
    return bulletin


@router.get("/api/admin/newsletter/preferences")
def admin_newsletter_preferences(_admin: str = Depends(require_admin)):
    return {"preferences": get_newsletter_preferences()}


@router.post("/api/admin/newsletter/sync-manage-urls")
def admin_sync_newsletter_manage_urls(_admin: str = Depends(require_admin)):
    """Backfill Resend's manage_url property for existing subscribers."""
    synced = 0
    failed = []
    for subscriber in list_newsletter_subscribers():
        try:
            set_audience_contact_properties(
                subscriber["email"],
                {"manage_url": _manage_url(subscriber["manage_token"])},
            )
            synced += 1
        except Exception as exc:
            failed.append({"email": subscriber["email"], "error": str(exc)})
    return {"synced": synced, "failed": failed}
