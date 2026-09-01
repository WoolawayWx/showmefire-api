"""
Public fire reporting and the unified fire-event store.

POST /api/fires/reports is the only unauthenticated write path in this
API. It requires a Cloudflare Turnstile token, enforces per-IP and global
rate limits, and stores every submission as status='pending' - nothing is
publicly readable or label-eligible until an administrator approves it.

GET /api/fires/events and /api/fires/events.geojson return approved
events only. The GeoJSON properties intentionally match the uppercase
shape already served by /fires/satdet so existing map clients
(website/app/components/FireDetections.vue, detectionPopup.vue) need no
changes.
"""
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Literal, Optional
from zoneinfo import ZoneInfo

from fastapi import APIRouter, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from core.database import (
    add_ip_to_blocklist,
    add_fire_event_media,
    consume_fire_submission_quota,
    count_fire_event_media,
    create_fire_report,
    delete_fire_event,
    export_fire_labels,
    get_fire_event,
    get_fire_incident,
    get_fire_upload_token_hash,
    is_ip_blocked,
    list_fire_events,
    list_fire_incident_members,
    list_fire_incidents,
    list_nearby_fire_events,
    set_fire_event_status,
    update_fire_event,
)
from core.fire_events import FUEL_TYPES, MO_LAT_MAX, MO_LAT_MIN, MO_LON_MAX, MO_LON_MIN, VERIFICATION_TIERS
from core.security import SECRET_KEY, verify_token
from services.county_lookup import county_for_point
from services.turnstile import verify_turnstile

logger = logging.getLogger(__name__)

router = APIRouter(tags=["fires"])

MAX_REPORT_AGE_DAYS = int(os.getenv("FIRE_REPORT_MAX_AGE_DAYS", "14"))
FUTURE_SKEW_MINUTES = 10
CONSENT_VERSION = "2026-08-fire-report-v1"
CENTRAL = ZoneInfo("America/Chicago")

# Defaults to distrusting proxy headers everywhere, including production:
# trusting CF-Connecting-IP/X-Forwarded-For is only safe once the origin is
# firewalled to Cloudflare's IP ranges, and that hasn't been confirmed for
# this deployment (api.showmefire.org's origin was directly reachable by IP
# in an August 2026 check, bypassing Cloudflare entirely - a directly
# reachable origin lets an attacker forge either header to spoof any source
# IP for rate-limiting/blocklisting). Set TRUST_PROXY_HEADERS=true once the
# firewall restriction is in place and verified.
TRUST_PROXY_HEADERS = os.getenv("TRUST_PROXY_HEADERS", "false").lower() == "true"

FIRE_REPORT_LIMIT_PER_HOUR = int(os.getenv("FIRE_REPORT_LIMIT_PER_HOUR", "3"))
FIRE_REPORT_LIMIT_PER_DAY = int(os.getenv("FIRE_REPORT_LIMIT_PER_DAY", "10"))
FIRE_REPORT_GLOBAL_LIMIT_PER_DAY = int(os.getenv("FIRE_REPORT_GLOBAL_LIMIT_PER_DAY", "300"))
FIRE_LABEL_ADMIN_REVIEWED_WEIGHT = float(os.getenv("FIRE_LABEL_ADMIN_REVIEWED_WEIGHT", "0.3"))
FIRE_GEOCODE_LIMIT_PER_HOUR = int(os.getenv("FIRE_GEOCODE_LIMIT_PER_HOUR", "20"))
FIRE_GEOCODE_LIMIT_PER_DAY = int(os.getenv("FIRE_GEOCODE_LIMIT_PER_DAY", "60"))

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_CONTACT_RE = re.compile(r"^[A-Za-z0-9@._%+\-\s()]{5,120}$")
_NAME_RE = re.compile(r"^[^\r\n]{1,120}$")
MEDIA_TYPES = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
DOCUMENT_TYPES = {**MEDIA_TYPES, "application/pdf": ".pdf"}
MAX_MEDIA_BYTES = 10 * 1024 * 1024
MAX_MEDIA_COUNT = 5
MAX_DOCUMENT_COUNT = 1
MEDIA_DIR = Path(os.getenv("FIRE_MEDIA_DIR", str(Path(os.getenv("DATA_DIR", ".")) / "fire-media")))


def _require_admin(token: Optional[str] = None) -> str:
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return email


def _client_ip(request: Request) -> str:
    """Resolve the visitor IP from Cloudflare/proxy headers."""
    if TRUST_PROXY_HEADERS:
        cf_ip = (request.headers.get("cf-connecting-ip") or "").strip()
        if cf_ip:
            return cf_ip
        forwarded = (request.headers.get("x-forwarded-for") or "").split(",")
        if forwarded and forwarded[0].strip():
            return forwarded[0].strip()
    return (request.client.host if request.client else "") or "unknown"


def _ip_bucket_key(ip: str) -> str:
    """HMAC the client IP so no raw address is ever persisted."""
    secret = os.getenv("FIRE_REPORT_IP_SALT", "").strip() or SECRET_KEY
    return hmac.new(secret.encode(), ip.encode(), hashlib.sha256).hexdigest()


def _clean_text(value: Optional[str], *, required: bool, field: str) -> str:
    text = _CONTROL_CHARS.sub("", str(value or "")).strip()
    if required and not text:
        raise ValueError(f"{field} must not be empty")
    return text


def _reverse_geocode(latitude: float, longitude: float) -> str:
    """Best-effort address label; never blocks or rejects a report."""
    try:
        query = urllib.parse.urlencode({"lat": latitude, "lon": longitude, "format": "json", "zoom": 18})
        request = urllib.request.Request(
            f"https://nominatim.openstreetmap.org/reverse?{query}",
            headers={"User-Agent": "ShowMeFire/1.0 fire-report-address"},
        )
        with urllib.request.urlopen(request, timeout=0.75) as response:
            data = json.loads(response.read().decode("utf-8"))
        return _clean_text(data.get("display_name"), required=False, field="address_text")[:500]
    except Exception:
        logger.info("Reverse geocoding unavailable", exc_info=False)
        return ""


class _GeocodeUnavailable(Exception):
    """Nominatim rejected/failed the request (rate limit, timeout, 5xx) - distinct
    from a clean "no results", so the caller can tell a reporter to retry
    shortly instead of telling them their address doesn't exist."""


def _forward_geocode(address: str) -> Optional[Dict]:
    """
    Best-effort address -> point lookup for the "I don't know the exact
    coordinates" reporting flow. Returns an approximate point bounded to
    Missouri, or None when Nominatim genuinely has no match/only an
    out-of-state one. Raises _GeocodeUnavailable when the lookup itself
    failed (rate limited, timed out, 5xx) so the caller doesn't conflate
    "temporarily can't check" with "not a real address". Coordinates from
    this are always approximate; staff correct them via FireEventUpdate
    during review like any other report field.
    """
    try:
        query = urllib.parse.urlencode({
            "q": address,
            "format": "json",
            "limit": 1,
            "countrycodes": "us",
            "viewbox": f"{MO_LON_MIN},{MO_LAT_MAX},{MO_LON_MAX},{MO_LAT_MIN}",
            "bounded": 1,
        })
        request = urllib.request.Request(
            f"https://nominatim.openstreetmap.org/search?{query}",
            headers={"User-Agent": "ShowMeFire/1.0 fire-report-address"},
        )
        with urllib.request.urlopen(request, timeout=3.0) as response:
            results = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code in (429, 503):
            raise _GeocodeUnavailable() from exc
        logger.info("Forward geocoding unavailable", exc_info=False)
        return None
    except urllib.error.URLError:
        raise _GeocodeUnavailable()
    except Exception:
        logger.info("Forward geocoding unavailable", exc_info=False)
        return None

    if not results:
        return None
    latitude = float(results[0]["lat"])
    longitude = float(results[0]["lon"])
    if not (MO_LAT_MIN <= latitude <= MO_LAT_MAX and MO_LON_MIN <= longitude <= MO_LON_MAX):
        return None
    return {
        "latitude": round(latitude, 4),
        "longitude": round(longitude, 4),
        "display_name": _clean_text(results[0].get("display_name"), required=False, field="address_text")[:500],
    }


class AddressGeocodeRequest(BaseModel):
    address: str = Field(min_length=3, max_length=300)


class FireReportCreate(BaseModel):
    latitude: float = Field(ge=MO_LAT_MIN, le=MO_LAT_MAX)
    longitude: float = Field(ge=MO_LON_MIN, le=MO_LON_MAX)
    occurred_at: str = Field(min_length=8, max_length=40)
    occurred_at_precision: Literal["minute", "hour", "day"] = "minute"
    acres: float = Field(gt=0, le=100000)
    acres_is_estimate: bool = True
    fuel_types: List[str] = Field(min_length=1, max_length=6)
    description: str = Field(min_length=20, max_length=4000)
    out_of_ordinary: str = Field(default="", max_length=2000)
    reporter_contact: str = Field(min_length=1, max_length=120)
    reporter_name: str = Field(min_length=1, max_length=120)
    reporter_org: str = Field(default="", max_length=120)
    address_text: str = Field(default="", max_length=500)
    consent_acknowledged: bool
    turnstile_token: str = Field(min_length=1, max_length=4096)
    website: str = Field(default="", max_length=200)  # honeypot; must stay empty

    @field_validator("fuel_types", mode="before")
    @classmethod
    def _normalize_fuels(cls, value):
        seen, out = set(), []
        for raw in value or []:
            item = str(raw or "").strip().lower()
            if item and item not in seen:
                seen.add(item)
                out.append(item)
        return out

    @field_validator("fuel_types")
    @classmethod
    def _validate_fuels(cls, value: List[str]) -> List[str]:
        unknown = [item for item in value if item not in FUEL_TYPES]
        if unknown:
            raise ValueError(f"unknown fuel type(s): {', '.join(unknown)}")
        return value

    @field_validator("description")
    @classmethod
    def _clean_description(cls, value: str) -> str:
        text = _clean_text(value, required=True, field="description")
        if len(text) < 20:
            raise ValueError("description must be at least 20 characters")
        if text.count("http") > 3:
            raise ValueError("description contains too many links")
        return text

    @field_validator("out_of_ordinary")
    @classmethod
    def _clean_out_of_ordinary(cls, value: str) -> str:
        return _clean_text(value, required=False, field="out_of_ordinary")

    @field_validator("reporter_contact")
    @classmethod
    def _clean_contact(cls, value: str) -> str:
        text = _clean_text(value, required=True, field="reporter_contact")
        if not _CONTACT_RE.fullmatch(text):
            raise ValueError("reporter_contact must be an email or phone number")
        return text

    @field_validator("reporter_name")
    @classmethod
    def _clean_reporter_name(cls, value: str) -> str:
        text = _clean_text(value, required=True, field="reporter_name")
        if not _NAME_RE.fullmatch(text):
            raise ValueError("reporter_name contains invalid line breaks")
        return text

    @field_validator("reporter_org", "address_text")
    @classmethod
    def _clean_reporter_fields(cls, value: str) -> str:
        text = _clean_text(value, required=False, field="reporter details")
        if text and not _NAME_RE.fullmatch(text):
            raise ValueError("reporter details contain invalid line breaks")
        return text

    @field_validator("occurred_at")
    @classmethod
    def _normalize_occurred_at(cls, value: str) -> str:
        raw = str(value or "").strip()
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("occurred_at must be ISO 8601") from exc
        if parsed.tzinfo is None:
            # Naive input is read as America/Chicago: a Missouri reporter
            # typing "14:30" means local time, and the whole scheduler
            # already runs on Central.
            parsed = parsed.replace(tzinfo=CENTRAL)
        parsed = parsed.astimezone(timezone.utc)
        now = datetime.now(timezone.utc)
        if parsed > now + timedelta(minutes=FUTURE_SKEW_MINUTES):
            raise ValueError("occurred_at must not be in the future")
        if parsed < now - timedelta(days=MAX_REPORT_AGE_DAYS):
            raise ValueError(f"occurred_at must be within the last {MAX_REPORT_AGE_DAYS} days")
        return parsed.strftime("%Y-%m-%dT%H:%M:%SZ")

    @field_validator("consent_acknowledged")
    @classmethod
    def _require_consent(cls, value: bool) -> bool:
        if not value:
            raise ValueError("consent_acknowledged must be true")
        return value

    @field_validator("website")
    @classmethod
    def _reject_honeypot(cls, value: str) -> str:
        if str(value or "").strip():
            raise ValueError("invalid submission")
        return ""


class FireReportModeration(BaseModel):
    verification_tier: Literal["admin_reviewed", "official_source_confirmed"] = "admin_reviewed"
    official_source_ref: str = Field(default="", max_length=500)
    moderator_note: str = Field(default="", max_length=2000)

    @model_validator(mode="after")
    def _require_ref_for_official(self):
        if self.verification_tier == "official_source_confirmed" and not self.official_source_ref.strip():
            raise ValueError("official_source_ref is required for official_source_confirmed")
        return self


class FireReportRejection(BaseModel):
    reason: str = Field(min_length=3, max_length=1000)


class FireEventUpdate(BaseModel):
    latitude: Optional[float] = Field(default=None, ge=MO_LAT_MIN, le=MO_LAT_MAX)
    longitude: Optional[float] = Field(default=None, ge=MO_LON_MIN, le=MO_LON_MAX)
    acres: Optional[float] = Field(default=None, gt=0, le=100000)
    fuel_types: Optional[List[str]] = None
    description: Optional[str] = Field(default=None, max_length=4000)
    out_of_ordinary: Optional[str] = Field(default=None, max_length=2000)
    verification_tier: Optional[Literal[VERIFICATION_TIERS]] = None
    official_source_ref: Optional[str] = Field(default=None, max_length=500)
    cause_category: Optional[str] = None
    reporter_name: Optional[str] = Field(default=None, max_length=120)
    reporter_org: Optional[str] = Field(default=None, max_length=120)
    address_text: Optional[str] = Field(default=None, max_length=500)
    redact_reporter_contact: bool = False
    edit_reason: str = Field(min_length=3, max_length=500)

    @field_validator("fuel_types")
    @classmethod
    def _validate_fuels(cls, value):
        if value is None:
            return None
        unknown = [item for item in value if item not in FUEL_TYPES]
        if unknown:
            raise ValueError(f"unknown fuel type(s): {', '.join(unknown)}")
        return value


class BlocklistCreate(BaseModel):
    ip_hash: str = Field(min_length=8, max_length=128)
    reason: str = Field(default="", max_length=500)


def _parse_bbox(bbox: Optional[str]) -> Optional[tuple]:
    if not bbox:
        return None
    try:
        min_lon, min_lat, max_lon, max_lat = (float(part) for part in bbox.split(","))
    except (ValueError, AttributeError):
        raise HTTPException(status_code=400, detail="bbox must be minlon,minlat,maxlon,maxlat")
    if min_lon > max_lon or min_lat > max_lat:
        raise HTTPException(status_code=400, detail="bbox min values must not exceed max values")
    return (min_lon, min_lat, max_lon, max_lat)


def _confidence_for_tier(tier: str) -> str:
    return {"official_source_confirmed": "high", "admin_reviewed": "nominal"}.get(tier, "low")


def _event_to_geojson_feature(event: dict) -> dict:
    tier = event.get("verification_tier", "unverified")
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [event["longitude"], event["latitude"]]},
        "properties": {
            "SOURCE": "SHOWMEFIRE_STORE",
            "TYPENAME": event.get("source", "unknown"),
            "SOURCE_ID": event.get("external_id") or f"smf-event-{event['id']}",
            "EVENT_ID": event["id"],
            "LATITUDE": event["latitude"],
            "LONGITUDE": event["longitude"],
            "COUNTY": event.get("county_name"),
            "STATE": "Missouri",
            "COUNTRY": "United States",
            "ACQ_DATE_TIME": event.get("occurred_at"),
            "FRP": event.get("frp"),
            "BRIGHT_T7": None,
            "CONFIDENCE": _confidence_for_tier(tier),
            "SATELLITE": event.get("satellite"),
            "TYPE_DESCRIPTION": "Public fire report" if event.get("source") == "user_submission" else "Fire detection",
            "FUEL": ", ".join(event.get("fuel_types") or []),
            "LAND_COVER": "Unknown",
            "VERIFICATION_TIER": tier,
            "ACRES": event.get("acres"),
            "DESCRIPTION": event.get("description"),
            "OUT_OF_ORDINARY": event.get("out_of_ordinary"),
        },
    }


# --- Public: geocode ---

@router.post("/api/fires/geocode")
def geocode_fire_report_address(payload: AddressGeocodeRequest, request: Request):
    """
    Best-effort address -> point lookup for reporters who don't know exact
    coordinates. Returns an approximate point for the frontend to place on
    the map for the reporter to confirm/adjust before submitting; staff can
    still correct it during moderation review like any other report field.
    """
    client_ip = _client_ip(request)
    ip_hash = _ip_bucket_key(client_ip)

    if is_ip_blocked(ip_hash):
        raise HTTPException(status_code=403, detail="Not allowed from this network")

    quota = consume_fire_submission_quota(
        f"geocode:{ip_hash}", datetime.now(timezone.utc), FIRE_GEOCODE_LIMIT_PER_HOUR, FIRE_GEOCODE_LIMIT_PER_DAY
    )
    if not quota["allowed"]:
        raise HTTPException(
            status_code=429,
            detail="Too many address lookups. Please try again shortly.",
            headers={"Retry-After": str(quota["retry_after"])},
        )

    address = _clean_text(payload.address, required=True, field="address")
    try:
        result = _forward_geocode(address)
    except _GeocodeUnavailable:
        raise HTTPException(
            status_code=503,
            detail="Location lookup is busy right now. Please try again in a moment, or place the pin on the map directly.",
        )
    if not result:
        raise HTTPException(
            status_code=404,
            detail="Couldn't find that address in Missouri. Try adding more detail, or place the point on the map instead.",
        )
    return {"success": True, "location": result}


# --- Public: submit ---

@router.post("/api/fires/reports", status_code=201)
def submit_fire_report(payload: FireReportCreate, request: Request):
    """Submit an anonymous fire report for moderation (public endpoint)"""
    client_ip = _client_ip(request)
    ip_hash = _ip_bucket_key(client_ip)

    if is_ip_blocked(ip_hash):
        raise HTTPException(status_code=403, detail="Submission not allowed from this network")

    quota = consume_fire_submission_quota(
        ip_hash, datetime.now(timezone.utc), FIRE_REPORT_LIMIT_PER_HOUR, FIRE_REPORT_LIMIT_PER_DAY
    )
    if not quota["allowed"]:
        raise HTTPException(
            status_code=429,
            detail="Too many fire reports from this network. Please try again later.",
            headers={"Retry-After": str(quota["retry_after"])},
        )

    global_quota = consume_fire_submission_quota(
        "__global__", datetime.now(timezone.utc), 10**9, FIRE_REPORT_GLOBAL_LIMIT_PER_DAY
    )
    if not global_quota["allowed"]:
        raise HTTPException(
            status_code=429,
            detail="Too many fire reports right now. Please try again later.",
            headers={"Retry-After": str(global_quota["retry_after"])},
        )

    success, verdict = verify_turnstile(payload.turnstile_token, client_ip)
    if not success:
        raise HTTPException(status_code=403, detail="Captcha verification failed. Please try again.")

    county_fips, county_name = county_for_point(payload.latitude, payload.longitude)
    upload_token = secrets.token_urlsafe(32)
    address_text = payload.address_text or _reverse_geocode(payload.latitude, payload.longitude)

    event = create_fire_report(
        latitude=payload.latitude,
        longitude=payload.longitude,
        occurred_at=payload.occurred_at,
        occurred_at_precision=payload.occurred_at_precision,
        acres=payload.acres,
        acres_is_estimate=payload.acres_is_estimate,
        fuel_types=payload.fuel_types,
        description=payload.description,
        out_of_ordinary=payload.out_of_ordinary,
        reporter_contact=payload.reporter_contact,
        reporter_name=payload.reporter_name,
        reporter_org=payload.reporter_org,
        address_text=address_text,
        submitter_ip_hash=ip_hash,
        upload_token_hash=hashlib.sha256(upload_token.encode()).hexdigest(),
        consent_version=CONSENT_VERSION,
        captcha_verdict=verdict,
        county_fips=county_fips,
        county_name=county_name,
    )

    return {
        "success": True,
        "report": {
            "id": event["id"],
            "status": event["status"],
            "submitted_at": event["created_at"],
            "county_name": event.get("county_name"),
            "upload_token": upload_token,
        },
    }


def _matches_file_signature(content_type: str, content: bytes) -> bool:
    signatures = {
        "image/jpeg": content[:3] == b"\xff\xd8\xff",
        "image/png": content[:8] == b"\x89PNG\r\n\x1a\n",
        "image/webp": content[:4] == b"RIFF" and content[8:12] == b"WEBP",
        "application/pdf": content[:5] == b"%PDF-",
    }
    return signatures.get(content_type, False)


async def _save_fire_report_upload(
    event_id: int,
    upload_token: str,
    media: UploadFile,
    *,
    kind: str,
    allowed_types: Dict[str, str],
    max_count: int,
    max_count_message: str,
    type_error_message: str,
) -> Dict:
    """Shared validate-and-store logic for the post-submission upload step.

    Photos (kind="photo") and department-report documents (kind="document")
    share fire_event_media but are counted and capped independently, and
    documents must stay admin-only - see get_fire_event(admin=...) gating.
    """
    if not upload_token or len(upload_token) > 256:
        raise HTTPException(status_code=403, detail="Invalid upload token")
    event = get_fire_event(event_id, admin=True)
    if not event or event.get("status") != "pending" or event.get("source") != "user_submission":
        raise HTTPException(status_code=404, detail="Report not found")
    expected = get_fire_upload_token_hash(event_id) or ""
    supplied = hashlib.sha256(upload_token.encode()).hexdigest()
    if not expected or not hmac.compare_digest(expected, supplied):
        raise HTTPException(status_code=403, detail="Invalid upload token")
    if count_fire_event_media(event_id, kind=kind) >= max_count:
        raise HTTPException(status_code=413, detail=max_count_message)

    content_type = (media.content_type or "").lower().split(";")[0].strip()
    if content_type not in allowed_types:
        raise HTTPException(status_code=415, detail=type_error_message)
    filename = Path(media.filename or "upload").name[:180]
    if not filename:
        raise HTTPException(status_code=400, detail="filename required")

    content = await media.read(MAX_MEDIA_BYTES + 1)
    if len(content) > MAX_MEDIA_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 10 MB")
    if not _matches_file_signature(content_type, content):
        raise HTTPException(status_code=415, detail="File content does not match its MIME type")

    digest = hashlib.sha256(content).hexdigest()
    stored_name = f"{event_id}-{secrets.token_hex(16)}{allowed_types[content_type]}"
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    target = MEDIA_DIR / stored_name
    target.write_bytes(content)
    try:
        record = add_fire_event_media(event_id, stored_name, filename, content_type, len(content), digest, kind=kind)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return {k: v for k, v in record.items() if k != "sha256"}


@router.post("/api/fires/reports/{event_id}/media", status_code=201)
async def upload_fire_report_media(
    event_id: int,
    upload_token: str,
    media: UploadFile = File(...),
):
    """Attach a photo during the private, post-submission upload step."""
    record = await _save_fire_report_upload(
        event_id, upload_token, media,
        kind="photo",
        allowed_types=MEDIA_TYPES,
        max_count=MAX_MEDIA_COUNT,
        max_count_message=f"Maximum of {MAX_MEDIA_COUNT} media files allowed",
        type_error_message="Only JPEG, PNG, and WebP images are allowed",
    )
    return {"success": True, "media": record}


@router.post("/api/fires/reports/{event_id}/document", status_code=201)
async def upload_fire_report_document(
    event_id: int,
    upload_token: str,
    media: UploadFile = File(...),
):
    """Attach an optional department report (PDF or image) - admin-review only, never public."""
    record = await _save_fire_report_upload(
        event_id, upload_token, media,
        kind="document",
        allowed_types=DOCUMENT_TYPES,
        max_count=MAX_DOCUMENT_COUNT,
        max_count_message=f"Maximum of {MAX_DOCUMENT_COUNT} department report file allowed",
        type_error_message="Only PDF, JPEG, PNG, or WebP files are allowed",
    )
    return {"success": True, "document": record}


@router.get("/api/admin/fires/reports/{event_id}/media/{media_id}")
def admin_get_fire_report_media(event_id: int, media_id: int, token: Optional[str] = None):
    """Serve report media only to authenticated administrators."""
    _require_admin(token)
    event = get_fire_event(event_id, admin=True)
    item = next((row for row in (event or {}).get("media", []) if row["id"] == media_id), None)
    if not item:
        raise HTTPException(status_code=404, detail="Media not found")
    target = MEDIA_DIR / Path(item["stored_filename"]).name
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Media file unavailable")
    return FileResponse(target, media_type=item["content_type"], filename=item["original_filename"])


# --- Public: read ---

@router.get("/api/fires/events")
def list_public_fire_events(
    response: Response,
    since: Optional[str] = None,
    until: Optional[str] = None,
    source: Optional[str] = None,
    verification_tier: Optional[str] = None,
    county_fips: Optional[str] = None,
    bbox: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
):
    """Approved fire events (public endpoint)"""
    response.headers["Cache-Control"] = "public, max-age=60"
    events = list_fire_events(
        status="approved", source=source, verification_tier=verification_tier,
        county_fips=county_fips, since=since, until=until,
        bbox=_parse_bbox(bbox), limit=limit, offset=offset,
    )
    return {"success": True, "events": events, "count": len(events)}


@router.get("/api/fires/events.geojson")
def list_public_fire_events_geojson(
    response: Response,
    since: Optional[str] = None,
    until: Optional[str] = None,
    source: Optional[str] = None,
    verification_tier: Optional[str] = None,
    county_fips: Optional[str] = None,
    bbox: Optional[str] = None,
    limit: int = 200,
    offset: int = 0,
):
    """Approved fire events as GeoJSON (public endpoint)"""
    response.headers["Cache-Control"] = "public, max-age=60"
    events = list_fire_events(
        status="approved", source=source, verification_tier=verification_tier,
        county_fips=county_fips, since=since, until=until,
        bbox=_parse_bbox(bbox), limit=limit, offset=offset,
    )
    return {
        "type": "FeatureCollection",
        "features": [_event_to_geojson_feature(event) for event in events],
        "metadata": {
            "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "feature_count": len(events),
            "source": "fire_events store",
        },
    }


@router.get("/api/fires/events/{event_id}")
def get_public_fire_event(event_id: int, response: Response):
    """A single approved fire event (public endpoint)"""
    event = get_fire_event(event_id, admin=False)
    if not event or event.get("status") != "approved":
        raise HTTPException(status_code=404, detail="Fire event not found")
    response.headers["Cache-Control"] = "public, max-age=60"
    return {"success": True, "event": event}


# --- Admin: moderation ---

@router.get("/api/admin/fires/reports")
def admin_list_fire_reports(
    token: Optional[str] = None,
    status: Optional[str] = "pending",
    source: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """List fire reports for moderation (admin only)"""
    _require_admin(token)
    events = list_fire_events(status=status, source=source, limit=limit, offset=offset, admin=True)
    return {"success": True, "reports": events, "count": len(events)}


@router.get("/api/admin/fires/reports/{event_id}")
def admin_get_fire_report(event_id: int, token: Optional[str] = None):
    """Fire report detail, including moderation history and contact info (admin only)"""
    _require_admin(token)
    event = get_fire_event(event_id, admin=True)
    if not event:
        raise HTTPException(status_code=404, detail="Fire event not found")
    event["nearby_reports"] = list_nearby_fire_events(event["latitude"], event["longitude"], radius_km=2.0, hours=6.0)
    return {"success": True, "report": event}


@router.post("/api/admin/fires/reports/{event_id}/approve")
def admin_approve_fire_report(event_id: int, payload: FireReportModeration, token: Optional[str] = None):
    """Approve a pending fire report and assign its verification tier (admin only)"""
    actor = _require_admin(token)
    result = set_fire_event_status(
        event_id, to_status="approved", actor=actor,
        to_tier=payload.verification_tier, official_source_ref=payload.official_source_ref,
        reason=payload.moderator_note,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Fire event not found")
    if result.get("already_moderated"):
        raise HTTPException(status_code=409, detail="Report already moderated")
    return {"success": True, "report": result}


@router.post("/api/admin/fires/reports/{event_id}/reject")
def admin_reject_fire_report(event_id: int, payload: FireReportRejection, token: Optional[str] = None):
    """Reject a pending fire report (admin only)"""
    actor = _require_admin(token)
    result = set_fire_event_status(event_id, to_status="rejected", actor=actor, reason=payload.reason)
    if result is None:
        raise HTTPException(status_code=404, detail="Fire event not found")
    if result.get("already_moderated"):
        raise HTTPException(status_code=409, detail="Report already moderated")
    return {"success": True, "report": result}


@router.put("/api/admin/fires/events/{event_id}")
def admin_update_fire_event(event_id: int, payload: FireEventUpdate, token: Optional[str] = None):
    """Edit a fire event; every edit requires a reason (admin only)"""
    actor = _require_admin(token)
    fields = payload.model_dump(exclude={"edit_reason"}, exclude_none=True)
    event = update_fire_event(event_id, actor=actor, edit_reason=payload.edit_reason, **fields)
    if not event:
        raise HTTPException(status_code=404, detail="Fire event not found")
    return {"success": True, "event": event}


@router.delete("/api/admin/fires/events/{event_id}")
def admin_delete_fire_event(event_id: int, token: Optional[str] = None, reason: str = ""):
    """Soft-delete a fire event; moderation history is retained (admin only)"""
    actor = _require_admin(token)
    deleted = delete_fire_event(event_id, actor=actor, reason=reason)
    if not deleted:
        raise HTTPException(status_code=404, detail="Fire event not found")
    return {"success": True}


@router.get("/api/admin/fires/labels")
def admin_export_fire_labels(token: Optional[str] = None, min_tier: str = "admin_reviewed", since: Optional[str] = None, until: Optional[str] = None):
    """Fire events eligible as model training labels (admin only)"""
    _require_admin(token)
    if min_tier not in VERIFICATION_TIERS:
        raise HTTPException(status_code=400, detail=f"min_tier must be one of {VERIFICATION_TIERS}")
    rows = export_fire_labels(min_tier=min_tier, since=since, until=until)
    for row in rows:
        row["label_weight"] = {
            "official_source_confirmed": 1.0,
            "admin_reviewed": FIRE_LABEL_ADMIN_REVIEWED_WEIGHT,
        }.get(row["verification_tier"], 0.0)
    return {"success": True, "labels": rows, "count": len(rows)}


@router.post("/api/admin/fires/blocklist")
def admin_add_to_blocklist(payload: BlocklistCreate, token: Optional[str] = None):
    """Block an IP hash from submitting further reports (admin only)"""
    actor = _require_admin(token)
    add_ip_to_blocklist(payload.ip_hash, payload.reason, actor)
    return {"success": True}


# --- Admin: satellite detection incidents ---

@router.get("/api/admin/fires/incidents")
def admin_list_fire_incidents(
    token: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    source: Optional[str] = None,
    bbox: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    """List satellite-detection incidents, most recently active first (admin only)"""
    _require_admin(token)
    incidents = list_fire_incidents(
        since=since, until=until, source=source, bbox=_parse_bbox(bbox), limit=limit, offset=offset,
    )
    return {"success": True, "incidents": incidents, "count": len(incidents)}


@router.get("/api/admin/fires/incidents/{incident_id}")
def admin_get_fire_incident(incident_id: int, token: Optional[str] = None):
    """Incident detail, including every member detection for map plotting (admin only)"""
    _require_admin(token)
    incident = get_fire_incident(incident_id)
    if not incident:
        raise HTTPException(status_code=404, detail="Fire incident not found")
    incident["detections"] = list_fire_incident_members(incident_id)
    return {"success": True, "incident": incident}
