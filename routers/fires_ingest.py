"""
Machine ingestion of OSINT/social-monitoring fire leads (e.g. Muse scanning
Facebook for department posts so staff don't have to scroll manually).

POST /fires/ingest/reports is a Bearer-API-key-authenticated sibling to the
public POST /api/fires/reports flow: every row lands status='pending',
verification_tier='unverified' and goes through the exact same moderation
queue documented in api/docs/fire_report_moderation_runbook.md. Approving it
runs the same correlate_report_with_incident() every other report uses, so
an approved lead is clustered into fire_incidents (and therefore the public
map) without any extra code path.

Deliberately has no '/api/' path segment - this API is served at
api.showmefire.org, so repeating '/api/' in the path is redundant. The
historical fires.py router keeps '/api/' for backward compatibility with
existing callers; this new surface does not need to.
"""
import logging
from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Response
from pydantic import BaseModel, Field

from core.database import (
    add_ingest_media_links,
    consume_fire_submission_quota,
    create_fire_ingest_source,
    create_ingest_api_key_for_source,
    get_ingest_source_by_key,
    list_fire_ingest_sources,
    revoke_ingest_api_key,
    upsert_ingest_report,
)
from core.fire_events import MO_LAT_MAX, MO_LAT_MIN, MO_LON_MAX, MO_LON_MIN
from core.security import verify_token
from routers.fires import _GeocodeUnavailable, _forward_geocode
from services.county_lookup import county_for_point

logger = logging.getLogger(__name__)

router = APIRouter(tags=["fires-ingest"])


def _require_admin(token: Optional[str] = None) -> str:
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return email


def _ingest_source(authorization: Optional[str] = Header(default=None)) -> Dict:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Bearer API key required")
    source = get_ingest_source_by_key(authorization[7:].strip())
    if not source:
        raise HTTPException(status_code=401, detail="Invalid or revoked API key")
    return source


class IngestSourceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    slug: str = Field(min_length=1, max_length=50, pattern=r"^[a-z0-9_-]+$")
    daily_limit: int = Field(default=200, ge=1, le=10000)
    contact_email: str = Field(default="", max_length=200)


class IngestReportCreate(BaseModel):
    external_id: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=300)
    reported_at: str = Field(min_length=8, max_length=40)
    occurred_at: Optional[str] = Field(default=None, max_length=40)
    occurred_at_precision: Literal["minute", "hour", "day"] = "day"
    latitude: Optional[float] = Field(default=None, ge=MO_LAT_MIN, le=MO_LAT_MAX)
    longitude: Optional[float] = Field(default=None, ge=MO_LON_MIN, le=MO_LON_MAX)
    location_text: Optional[str] = Field(default=None, max_length=500)
    size_acres: Optional[float] = Field(default=None, gt=0, le=200000)
    status_hint: Literal["unknown", "contained", "out"] = "unknown"
    description: str = Field(default="", max_length=4000)
    source_url: str = Field(min_length=1, max_length=2000)
    source_name: str = Field(default="", max_length=120)
    media_urls: List[str] = Field(default_factory=list, max_length=10)


@router.post("/fires/ingest/reports")
def ingest_fire_report(payload: IngestReportCreate, response: Response, ingest_source: Dict = Depends(_ingest_source)):
    """Upsert an OSINT fire lead, keyed on (source slug, external_id). A
    repost of the same external_id updates the description/acres but never
    silently moves the pin, occurred_at, or moderation status - matching the
    guarantee upsert_detection_event already gives satellite re-ingests."""
    quota = consume_fire_submission_quota(
        f"ingest:{ingest_source['source_slug']}", datetime.now(timezone.utc),
        per_hour_limit=10**9, per_day_limit=ingest_source["daily_limit"],
    )
    if not quota["allowed"]:
        raise HTTPException(
            status_code=429,
            detail="Daily ingest quota exceeded for this source",
            headers={"Retry-After": str(quota["retry_after"])},
        )

    latitude, longitude = payload.latitude, payload.longitude
    if (latitude is None or longitude is None) and payload.location_text:
        try:
            candidates = _forward_geocode(payload.location_text)
        except _GeocodeUnavailable:
            candidates = []
        if candidates:
            latitude, longitude = candidates[0]["latitude"], candidates[0]["longitude"]

    if latitude is None or longitude is None:
        raise HTTPException(
            status_code=422,
            detail="Could not resolve a location - provide latitude/longitude or a location_text that geocodes",
        )

    county_fips, county_name = county_for_point(latitude, longitude)

    event, created = upsert_ingest_report(
        source=ingest_source["source_slug"],
        external_id=payload.external_id,
        title=payload.title,
        description=payload.description,
        source_url=payload.source_url,
        source_name=payload.source_name,
        latitude=latitude,
        longitude=longitude,
        occurred_at=payload.occurred_at or payload.reported_at,
        occurred_at_precision=payload.occurred_at_precision if payload.occurred_at else "day",
        acres=payload.size_acres,
        county_fips=county_fips,
        county_name=county_name,
    )
    if payload.media_urls:
        add_ingest_media_links(event["id"], payload.media_urls)

    response.status_code = 201 if created else 200
    return {"success": True, "event_id": event["id"], "status": event["status"], "created": created}


@router.get("/api/admin/fires/ingest-sources")
def admin_list_ingest_sources(token: Optional[str] = None):
    """List ingest sources and their most recent key's status (admin only).
    Raw keys are never returned here - only whether one is active, when it
    was last used, and when it was issued."""
    _require_admin(token)
    return {"success": True, "sources": list_fire_ingest_sources()}


@router.post("/api/admin/fires/ingest-sources")
def admin_create_ingest_source(payload: IngestSourceCreate, token: Optional[str] = None):
    """Create a new ingest source + its API key (admin only). The raw key is
    returned exactly once - only its SHA-256 hash is ever stored."""
    _require_admin(token)
    try:
        source = create_fire_ingest_source(
            name=payload.name, slug=payload.slug,
            daily_limit=payload.daily_limit, contact_email=payload.contact_email,
        )
    except Exception as exc:
        raise HTTPException(status_code=409, detail="A source with that name or slug already exists") from exc
    return {"success": True, "source": source}


@router.post("/api/admin/fires/ingest-sources/{source_id}/keys")
def admin_rotate_ingest_source_key(source_id: int, token: Optional[str] = None):
    """Issue a new key for an existing ingest source (admin only). Any
    previously issued key keeps working until separately revoked - rotation
    doesn't lock out an in-flight integration."""
    _require_admin(token)
    raw_key = create_ingest_api_key_for_source(source_id)
    return {"success": True, "api_key": raw_key}


@router.post("/api/admin/fires/ingest-sources/{source_id}/revoke")
def admin_revoke_ingest_source(source_id: int, token: Optional[str] = None):
    """Revoke an ingest source's current API key (admin only)."""
    _require_admin(token)
    revoked = revoke_ingest_api_key(source_id)
    if not revoked:
        raise HTTPException(status_code=404, detail="No active key found for this source")
    return {"success": True}
