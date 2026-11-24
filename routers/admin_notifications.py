"""Staff-sent mobile push notifications.

Reuses the existing push pipeline (services/mobile_push.py): a manual message
goes out on one of the app's four existing channels, so it respects each
user's existing toggle for that channel, lands in the Android channel the app
already registers, and deep-links through the generic `data.url` handler. No
mobile release is needed.
"""
import json
import logging
import sqlite3
import uuid
from urllib.parse import urlsplit
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, field_validator

from core.database import connect as connect_db
from core.database import get_db_path
from core.security import require_admin
from services.mobile_content import county_catalog
from services.mobile_push import _eligible_subscriptions, push_stats, send_mobile_event

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/admin/notifications", tags=["admin-notifications"])

# Screens the shipped app already routes to (mobile/src/app).
APP_DESTINATIONS = {
    "/": "Home",
    "/forecasts": "Forecasts",
    "/alerts": "Alerts",
    "/map": "Map",
    "/sitrep": "SitRep",
    "/burn-bans": "Burn bans",
    "/report-fire": "Report a fire",
}


WEB_HOSTS = {"showmefire.org", "www.showmefire.org"}


def validate_web_url(value: str) -> str:
    """A showmefire.org page the app opens in its in-app browser. https on our own host only, no credentials or port;
    keep in step with `notificationWebUrl` in mobile/src/notifications/navigation.ts."""
    value = value.strip()
    try:
        parts = urlsplit(value)
        port = parts.port
    except ValueError:
        raise ValueError("web_url is not a valid address")
    if (parts.scheme != "https" or (parts.hostname or "") not in WEB_HOSTS or port is not None
            or parts.username is not None or parts.password is not None):
        raise ValueError("web_url must be an https://showmefire.org page")
    return value


class ManualNotification(BaseModel):
    channel: Literal["forecast", "sitrep", "fire_weather", "fire_detection", "general"]
    title: str = Field(min_length=1, max_length=100)
    body: str = Field(min_length=1, max_length=500)
    url: str = Field(default="/", max_length=200)
    # Optional showmefire.org page (e.g. a newsletter) the app opens in its in-app browser when the alert is tapped.
    # Sent as `webUrl` beside the normal `url`, so an app without support just opens `url` (Home).
    web_url: Optional[str] = Field(default=None, max_length=500)
    # Empty = everyone subscribed to the channel (statewide). County-scoped
    # channels (fire_weather, fire_detection) otherwise need a county match.
    countyFips: list[str] = Field(default_factory=list, max_length=115)

    @field_validator("title", "body")
    @classmethod
    def strip_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be blank")
        return value

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        value = value.strip() or "/"
        # The app only follows in-app paths (see mobile notifications/navigation.ts).
        if not value.startswith("/") or value.startswith("//"):
            raise ValueError("url must be an in-app path starting with a single '/'")
        return value

    @field_validator("web_url")
    @classmethod
    def check_web_url(cls, value: Optional[str]) -> Optional[str]:
        if value is None or not value.strip():
            return None
        return validate_web_url(value)

    @field_validator("countyFips")
    @classmethod
    def validate_counties(cls, values: list[str]) -> list[str]:
        known = {county["fips"] for county in county_catalog()}
        normalized = sorted(set(values))
        unknown = [value for value in normalized if value not in known]
        if unknown:
            raise ValueError(f"Unknown Missouri county FIPS: {', '.join(unknown[:3])}")
        return normalized


class AudienceRequest(BaseModel):
    """Just who would receive it: the audience count does not depend on the text, so a blank form must not be rejected."""
    channel: Literal["forecast", "sitrep", "fire_weather", "fire_detection", "general"]
    countyFips: list[str] = Field(default_factory=list, max_length=115)

    @field_validator("countyFips")
    @classmethod
    def validate_counties(cls, values: list[str]) -> list[str]:
        return ManualNotification.validate_counties(values)

    model_config = {"extra": "ignore"}


def _recipients(payload: "ManualNotification | AudienceRequest") -> list[dict]:
    return _eligible_subscriptions(payload.channel, payload.countyFips, all_counties=True)


@router.get("/options")
def notification_options(_admin: str = Depends(require_admin)):
    return {
        "success": True,
        "channels": [
            {"value": "forecast", "label": "Forecast", "county_scoped": False},
            {"value": "sitrep", "label": "SitRep", "county_scoped": False},
            {"value": "fire_weather", "label": "Fire weather", "county_scoped": True},
            {"value": "fire_detection", "label": "Fire detection", "county_scoped": True},
            {"value": "general", "label": "General", "county_scoped": False},
        ],
        "destinations": [{"value": path, "label": label} for path, label in APP_DESTINATIONS.items()],
        "counties": county_catalog(),
    }


@router.post("/preview")
def preview_notification(payload: AudienceRequest, _admin: str = Depends(require_admin)):
    """How many devices would receive this, so staff see the audience before sending."""
    return {"success": True, "recipients": len(_recipients(payload))}


@router.post("/send")
def send_notification(payload: ManualNotification, email: str = Depends(require_admin)):
    recipients = len(_recipients(payload))
    if recipients == 0:
        raise HTTPException(status_code=409, detail="No subscribed devices match this audience; nothing was sent.")

    event_key = f"manual:{uuid.uuid4().hex}"
    sent = send_mobile_event(
        event_type=payload.channel,
        event_key=event_key,
        title=payload.title,
        body=payload.body,
        url=payload.url,
        county_fips=payload.countyFips,
        extra_data={"manual": True, **({"webUrl": payload.web_url} if payload.web_url else {})},
        all_counties=True,
    )
    with connect_db(get_db_path()) as connection:
        connection.execute(
            """INSERT INTO manual_notifications
               (event_key, channel, title, body, url, county_fips_json, recipients, sent, sent_by)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (event_key, payload.channel, payload.title, payload.body, payload.web_url or payload.url,
             json.dumps(payload.countyFips), recipients, sent, email),
        )
    logger.info("Manual notification by %s on %s: %d/%d accepted by Expo", email, payload.channel, sent, recipients)
    return {"success": True, "recipients": recipients, "sent": sent}


@router.get("/stats")
def notification_stats(days: int = 30, recent: int = 25, _admin: str = Depends(require_admin)):
    """Is the app alerting working? Audience size, and per alert type how many were sent, accepted by Expo,
    confirmed delivered to Apple/Google, failed, or reached nobody; plus a per-day series and the latest alerts."""
    return {"success": True, **push_stats(days=days, recent=recent)}


@router.get("/history")
def notification_history(limit: int = 50, _admin: str = Depends(require_admin)):
    with connect_db(get_db_path()) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT id, channel, title, body, url, county_fips_json, recipients, sent, sent_by, created_at "
            "FROM manual_notifications ORDER BY id DESC LIMIT ?",
            (max(1, min(limit, 200)),),
        ).fetchall()
    items = []
    for row in rows:
        item = dict(row)
        item["county_fips"] = json.loads(item.pop("county_fips_json") or "[]")
        items.append(item)
    return {"success": True, "notifications": items}
