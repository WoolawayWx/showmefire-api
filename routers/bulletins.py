"""FireWx bulletin signup, preference, and administrative endpoints."""
from __future__ import annotations

import re
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from core.database import (
    create_bulletin,
    claim_bulletin_send,
    get_bulletin,
    get_newsletter_account_by_token,
    get_newsletter_preferences,
    list_bulletins,
    mark_bulletin_sent,
    replace_newsletter_preferences,
    set_bulletin_error,
    unsubscribe_newsletter,
    update_newsletter_account,
    update_bulletin,
    upsert_newsletter_subscriber,
)
from core.security import verify_token
from services.graphics_email import (
    send_bulletin_broadcast,
    set_audience_contact_unsubscribed,
    upsert_audience_contact,
)
from services.mobile_content import county_catalog

router = APIRouter(tags=["bulletins"])

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
        normalized = list(dict.fromkeys(str(value).strip() for value in values))
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


class NewsletterManage(NewsletterSignup):
    token: str = Field(min_length=20)


def _require_admin(token: Optional[str] = None) -> str:
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return email


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


@router.post("/api/newsletter/signup")
def signup_newsletter(payload: NewsletterSignup):
    if "fire-weather-forecasts" in payload.lists and not payload.counties:
        raise HTTPException(status_code=422, detail="Select at least one county for Fire Weather Forecasts")
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
    if "fire-weather-forecasts" in payload.lists and not payload.counties:
        raise HTTPException(status_code=422, detail="Select at least one county for Fire Weather Forecasts")
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
def admin_list_bulletins(token: Optional[str] = None):
    _require_admin(token)
    return {"bulletins": list_bulletins()}


@router.post("/api/admin/bulletins")
def admin_create_bulletin(payload: BulletinCreate, token: Optional[str] = None):
    _require_admin(token)
    return create_bulletin(payload.subject, payload.html_body, payload.text_body)


@router.get("/api/admin/bulletins/{bulletin_id}")
def admin_get_bulletin(bulletin_id: int, token: Optional[str] = None):
    _require_admin(token)
    bulletin = get_bulletin(bulletin_id)
    if not bulletin:
        raise HTTPException(status_code=404, detail="Bulletin not found")
    return bulletin


@router.put("/api/admin/bulletins/{bulletin_id}")
def admin_update_bulletin(bulletin_id: int, payload: BulletinCreate, token: Optional[str] = None):
    _require_admin(token)
    bulletin = update_bulletin(bulletin_id, payload.subject, payload.html_body, payload.text_body)
    if not bulletin:
        raise HTTPException(status_code=404, detail="Draft bulletin not found")
    return bulletin


@router.post("/api/admin/bulletins/{bulletin_id}/preview")
def admin_preview_bulletin(bulletin_id: int, token: Optional[str] = None):
    _require_admin(token)
    bulletin = get_bulletin(bulletin_id)
    if not bulletin:
        raise HTTPException(status_code=404, detail="Bulletin not found")
    return {
        "id": bulletin["id"],
        "subject": bulletin["subject"],
        "html_body": bulletin["html_body"],
        "text_body": bulletin["text_body"],
    }


@router.post("/api/admin/bulletins/{bulletin_id}/send")
def admin_send_bulletin(bulletin_id: int, token: Optional[str] = None):
    _require_admin(token)
    bulletin = get_bulletin(bulletin_id)
    if not bulletin:
        raise HTTPException(status_code=404, detail="Bulletin not found")
    if bulletin["status"] != "draft":
        raise HTTPException(status_code=409, detail="Bulletin has already been sent")
    if not claim_bulletin_send(bulletin_id):
        raise HTTPException(status_code=409, detail="Bulletin send is already in progress")
    try:
        broadcast_id = send_bulletin_broadcast(
            bulletin["subject"], bulletin["html_body"], bulletin["text_body"]
        )
        return mark_bulletin_sent(bulletin_id, broadcast_id)
    except Exception as exc:
        set_bulletin_error(bulletin_id, str(exc))
        raise HTTPException(status_code=502, detail=f"Bulletin send failed: {exc}") from exc


@router.get("/api/admin/newsletter/preferences")
def admin_newsletter_preferences(token: Optional[str] = None):
    _require_admin(token)
    return {"preferences": get_newsletter_preferences()}
