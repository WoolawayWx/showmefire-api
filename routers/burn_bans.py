"""
County burn-ban public submissions and admin moderation.

POST /api/burn-bans/submissions is the unauthenticated write path. It requires
Cloudflare Turnstile, honeypot, and rate limits. Submissions start as
status='pending' until an administrator confirms or denies them.

GET /api/burn-bans/active and /api/burn-bans/active.geojson expose only
confirmed, in-effect bans. Submitter contact and uploaded proof files are
admin-only.
"""
import hashlib
import hmac
import logging
import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional
from zoneinfo import ZoneInfo

from fastapi import APIRouter, File, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from core.database import (
    consume_burn_ban_submission_quota,
    create_burn_ban_submission,
    delete_burn_ban_submission,
    expire_confirmed_burn_bans_for_county,
    get_burn_ban_submission,
    get_burn_ban_upload_token_hash,
    list_active_burn_bans,
    list_burn_ban_submissions,
    moderate_burn_ban_submission,
    purge_burn_ban_submission_pii,
    purge_burn_ban_throttle_rows,
    set_burn_ban_proof_file,
    update_burn_ban_submission,
)
from core.security import SECRET_KEY, verify_token
from services.mobile_content import county_catalog
from services.turnstile import verify_turnstile

logger = logging.getLogger(__name__)

router = APIRouter(tags=["burn-bans"])

CENTRAL = ZoneInfo("America/Chicago")
CONSENT_VERSION = "2026-08-burn-ban-v1"

TRUST_PROXY_HEADERS = os.getenv("TRUST_PROXY_HEADERS", "false").lower() == "true"
BURN_BAN_LIMIT_PER_HOUR = int(os.getenv("BURN_BAN_LIMIT_PER_HOUR", "3"))
BURN_BAN_LIMIT_PER_DAY = int(os.getenv("BURN_BAN_LIMIT_PER_DAY", "10"))
BURN_BAN_GLOBAL_LIMIT_PER_DAY = int(os.getenv("BURN_BAN_GLOBAL_LIMIT_PER_DAY", "200"))

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_CONTACT_RE = re.compile(r"^[A-Za-z0-9@._%+\-\s()]{5,120}$")
_NAME_RE = re.compile(r"^[^\r\n]{1,120}$")
_URL_RE = re.compile(r"^https?://[^\s]+$", re.IGNORECASE)

PROOF_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "application/pdf": ".pdf",
}
MAX_PROOF_BYTES = 10 * 1024 * 1024
PROOF_DIR = Path(os.getenv("BURN_BAN_PROOF_DIR", str(Path(os.getenv("DATA_DIR", ".")) / "burn-ban-proofs")))

STATUSES = {"pending", "confirmed", "denied", "expired"}
REQUEST_TYPES = {"issue", "lift"}


def _require_admin(token: Optional[str] = None) -> str:
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return email


def _client_ip(request: Request) -> str:
    if TRUST_PROXY_HEADERS:
        cf_ip = (request.headers.get("cf-connecting-ip") or "").strip()
        if cf_ip:
            return cf_ip
        forwarded = (request.headers.get("x-forwarded-for") or "").split(",")
        if forwarded and forwarded[0].strip():
            return forwarded[0].strip()
    return (request.client.host if request.client else "") or "unknown"


def _ip_bucket_key(ip: str) -> str:
    secret = os.getenv("BURN_BAN_IP_SALT", "").strip() or SECRET_KEY
    return hmac.new(secret.encode(), ip.encode(), hashlib.sha256).hexdigest()


def _clean_text(value: Optional[str], *, required: bool, field: str) -> str:
    text = _CONTROL_CHARS.sub("", str(value or "")).strip()
    if required and not text:
        raise ValueError(f"{field} must not be empty")
    return text


def _known_county_fips() -> Dict[str, str]:
    return {item["fips"]: item["name"] for item in county_catalog()}


def _parse_optional_ban_datetime(value: Optional[str]) -> Optional[str]:
    text = str(value or "").strip()
    if not text:
        return None
    return _parse_ban_datetime(text)


def _parse_ban_datetime(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("date must not be empty")
    try:
        if text.endswith("Z"):
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        elif "+" in text or text.count("-") > 2:
            parsed = datetime.fromisoformat(text)
        else:
            parsed = datetime.strptime(text[:19], "%Y-%m-%dT%H:%M:%S")
            parsed = parsed.replace(tzinfo=CENTRAL)
    except ValueError as exc:
        raise ValueError("invalid datetime format") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=CENTRAL)
    return parsed.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _has_proof(submission: Dict) -> bool:
    return bool((submission.get("proof_url") or "").strip() or (submission.get("proof_stored_filename") or "").strip())


def _public_ban_payload(submission: Dict) -> Dict:
    return {
        "id": submission["id"],
        "county_fips": submission["county_fips"],
        "county_name": submission["county_name"],
        "request_type": submission.get("request_type") or "issue",
        "effective_at": submission["effective_at"],
        "expires_at": submission.get("expires_at") or None,
        "proof_url": submission.get("proof_url") or None,
        "published_at": submission.get("published_at"),
        "updated_at": submission.get("updated_at"),
    }


def _ban_to_geojson_feature(submission: Dict) -> Dict:
    return {
        "type": "Feature",
        "geometry": None,
        "properties": _public_ban_payload(submission),
    }


def _matches_file_signature(content_type: str, content: bytes) -> bool:
    signatures = {
        "image/jpeg": content[:3] == b"\xff\xd8\xff",
        "image/png": content[:8] == b"\x89PNG\r\n\x1a\n",
        "image/webp": content[:4] == b"RIFF" and content[8:12] == b"WEBP",
        "application/pdf": content[:5] == b"%PDF-",
    }
    return signatures.get(content_type, False)


def _maybe_regenerate_map() -> None:
    try:
        from services.burn_ban_map import generate_burn_ban_map
        generate_burn_ban_map()
    except Exception as exc:
        logger.warning("Burn-ban map regeneration failed: %s", exc, exc_info=True)


class BurnBanCreate(BaseModel):
    county_fips: str = Field(min_length=5, max_length=5)
    request_type: str = Field(default="issue", min_length=4, max_length=8)
    submitter_name: str = Field(min_length=1, max_length=120)
    submitter_contact: str = Field(min_length=1, max_length=120)
    proof_url: str = Field(default="", max_length=2000)
    effective_at: str = Field(min_length=8, max_length=40)
    expires_at: str = Field(default="", max_length=40)
    consent_acknowledged: bool
    turnstile_token: str = Field(min_length=1, max_length=4096)
    website: str = Field(default="", max_length=200)

    @field_validator("county_fips")
    @classmethod
    def _validate_county(cls, value: str) -> str:
        fips = str(value or "").strip()
        if fips not in _known_county_fips():
            raise ValueError("county_fips must be a Missouri county FIPS code")
        return fips

    @field_validator("submitter_name")
    @classmethod
    def _clean_name(cls, value: str) -> str:
        text = _clean_text(value, required=True, field="submitter_name")
        if not _NAME_RE.fullmatch(text):
            raise ValueError("submitter_name contains invalid characters")
        return text

    @field_validator("submitter_contact")
    @classmethod
    def _clean_contact(cls, value: str) -> str:
        text = _clean_text(value, required=True, field="submitter_contact")
        if not _CONTACT_RE.fullmatch(text):
            raise ValueError("submitter_contact must be an email or phone number")
        return text

    @field_validator("proof_url")
    @classmethod
    def _clean_proof_url(cls, value: str) -> str:
        text = _clean_text(value, required=False, field="proof_url")
        if text and not _URL_RE.fullmatch(text):
            raise ValueError("proof_url must be an http or https URL")
        return text

    @field_validator("website")
    @classmethod
    def _reject_honeypot(cls, value: str) -> str:
        if str(value or "").strip():
            raise ValueError("invalid submission")
        return ""

    @field_validator("consent_acknowledged")
    @classmethod
    def _require_consent(cls, value: bool) -> bool:
        if not value:
            raise ValueError("consent must be acknowledged")
        return True

    @field_validator("request_type")
    @classmethod
    def _validate_request_type(cls, value: str) -> str:
        kind = str(value or "issue").strip().lower()
        if kind not in REQUEST_TYPES:
            raise ValueError("request_type must be issue or lift")
        return kind

    @model_validator(mode="after")
    def _validate_dates(self) -> "BurnBanCreate":
        effective = _parse_ban_datetime(self.effective_at)
        expires = _parse_optional_ban_datetime(self.expires_at)
        if self.request_type == "lift":
            expires = None
        if expires and expires <= effective:
            raise ValueError("expires_at must be after effective_at")
        self.effective_at = effective
        self.expires_at = expires or ""
        return self


class BurnBanModeration(BaseModel):
    moderator_note: str = Field(default="", max_length=2000)
    effective_at: Optional[str] = None
    expires_at: Optional[str] = None

    @field_validator("effective_at", "expires_at")
    @classmethod
    def _normalize_optional_dates(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if not str(value).strip():
            return ""
        return _parse_ban_datetime(value)


class BurnBanUpdate(BaseModel):
    edit_reason: str = Field(min_length=1, max_length=2000)
    effective_at: Optional[str] = None
    expires_at: Optional[str] = None
    proof_url: Optional[str] = Field(default=None, max_length=2000)
    county_fips: Optional[str] = Field(default=None, min_length=5, max_length=5)

    @field_validator("effective_at", "expires_at")
    @classmethod
    def _normalize_optional_dates(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        if not str(value).strip():
            return ""
        return _parse_ban_datetime(value)

    @field_validator("proof_url")
    @classmethod
    def _clean_proof_url(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        text = _clean_text(value, required=False, field="proof_url")
        if text and not _URL_RE.fullmatch(text):
            raise ValueError("proof_url must be an http or https URL")
        return text

    @field_validator("county_fips")
    @classmethod
    def _validate_county(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        fips = str(value).strip()
        if fips not in _known_county_fips():
            raise ValueError("county_fips must be a Missouri county FIPS code")
        return fips


class BurnBanRejection(BaseModel):
    reason: str = Field(min_length=1, max_length=2000)


class BurnBanAdminCreate(BaseModel):
    county_fips: str = Field(min_length=5, max_length=5)
    effective_at: str = Field(min_length=8, max_length=40)
    expires_at: str = Field(default="", max_length=40)
    proof_url: str = Field(default="", max_length=2000)
    submitter_name: str = Field(default="Show Me Fire Staff", max_length=120)
    submitter_contact: str = Field(default="admin@showmefire.org", max_length=120)
    moderator_note: str = Field(default="", max_length=2000)

    @field_validator("county_fips")
    @classmethod
    def _validate_county(cls, value: str) -> str:
        fips = str(value or "").strip()
        if fips not in _known_county_fips():
            raise ValueError("county_fips must be a Missouri county FIPS code")
        return fips

    @field_validator("proof_url")
    @classmethod
    def _clean_proof_url(cls, value: str) -> str:
        text = _clean_text(value, required=False, field="proof_url")
        if text and not _URL_RE.fullmatch(text):
            raise ValueError("proof_url must be an http or https URL")
        return text

    @field_validator("submitter_name")
    @classmethod
    def _clean_name(cls, value: str) -> str:
        text = _clean_text(value, required=True, field="submitter_name")
        if not _NAME_RE.fullmatch(text):
            raise ValueError("submitter_name contains invalid characters")
        return text

    @field_validator("submitter_contact")
    @classmethod
    def _clean_contact(cls, value: str) -> str:
        text = _clean_text(value, required=True, field="submitter_contact")
        if not _CONTACT_RE.fullmatch(text):
            raise ValueError("submitter_contact must be an email or phone number")
        return text

    @model_validator(mode="after")
    def _validate_dates(self) -> "BurnBanAdminCreate":
        effective = _parse_ban_datetime(self.effective_at)
        expires = _parse_optional_ban_datetime(self.expires_at)
        if expires and expires <= effective:
            raise ValueError("expires_at must be after effective_at")
        self.effective_at = effective
        self.expires_at = expires or ""
        return self


@router.get("/api/burn-bans/counties")
def list_burn_ban_counties():
    return {"success": True, "counties": county_catalog()}


@router.get("/api/burn-bans/active")
def list_public_active_burn_bans(response: Response):
    response.headers["Cache-Control"] = "public, max-age=300"
    bans = list_active_burn_bans()
    from services.burn_ban_map import ensure_burn_ban_map
    return {
        "success": True,
        "count": len(bans),
        "bans": [_public_ban_payload(item) for item in bans],
        "map": ensure_burn_ban_map(),
    }


@router.get("/api/burn-bans/active.geojson")
def list_public_active_burn_bans_geojson(response: Response):
    response.headers["Cache-Control"] = "public, max-age=300"
    bans = list_active_burn_bans()
    return {
        "type": "FeatureCollection",
        "features": [_ban_to_geojson_feature(item) for item in bans],
        "metadata": {
            "fetched_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "feature_count": len(bans),
            "join_key": "county_fips",
        },
    }


@router.post("/api/burn-bans/submissions", status_code=201)
def submit_burn_ban(payload: BurnBanCreate, request: Request):
    client_ip = _client_ip(request)
    ip_hash = _ip_bucket_key(client_ip)
    now = datetime.now(timezone.utc)

    quota = consume_burn_ban_submission_quota(
        ip_hash, now, BURN_BAN_LIMIT_PER_HOUR, BURN_BAN_LIMIT_PER_DAY,
    )
    if not quota["allowed"]:
        raise HTTPException(
            status_code=429,
            detail="Too many burn-ban submissions from this network. Please try again later.",
            headers={"Retry-After": str(quota["retry_after"])},
        )

    global_quota = consume_burn_ban_submission_quota(
        "__global__", now, 10**9, BURN_BAN_GLOBAL_LIMIT_PER_DAY,
    )
    if not global_quota["allowed"]:
        raise HTTPException(
            status_code=429,
            detail="Too many burn-ban submissions right now. Please try again later.",
            headers={"Retry-After": str(global_quota["retry_after"])},
        )

    success, verdict = verify_turnstile(payload.turnstile_token, client_ip)
    if not success:
        raise HTTPException(status_code=403, detail="Captcha verification failed. Please try again.")

    counties = _known_county_fips()
    upload_token = secrets.token_urlsafe(32)
    submission = create_burn_ban_submission(
        county_fips=payload.county_fips,
        county_name=counties[payload.county_fips],
        submitter_name=payload.submitter_name,
        submitter_contact=payload.submitter_contact,
        proof_url=payload.proof_url,
        effective_at=payload.effective_at,
        expires_at=payload.expires_at,
        submitter_ip_hash=ip_hash,
        upload_token_hash=hashlib.sha256(upload_token.encode()).hexdigest(),
        captcha_verdict=verdict,
        consent_version=CONSENT_VERSION,
        request_type=payload.request_type,
    )

    return {
        "success": True,
        "submission": {
            "id": submission["id"],
            "status": submission["status"],
            "submitted_at": submission["created_at"],
            "county_name": submission["county_name"],
            "request_type": submission.get("request_type") or payload.request_type,
            "upload_token": upload_token,
            "proof_required": not bool(payload.proof_url.strip()),
        },
    }


@router.post("/api/burn-bans/submissions/{submission_id}/proof", status_code=201)
async def upload_burn_ban_proof(
    submission_id: int,
    upload_token: str,
    media: UploadFile = File(...),
):
    if not upload_token or len(upload_token) > 256:
        raise HTTPException(status_code=403, detail="Invalid upload token")
    submission = get_burn_ban_submission(submission_id, admin=True)
    if not submission or submission.get("status") != "pending":
        raise HTTPException(status_code=404, detail="Submission not found")
    expected = get_burn_ban_upload_token_hash(submission_id) or ""
    supplied = hashlib.sha256(upload_token.encode()).hexdigest()
    if not expected or not hmac.compare_digest(expected, supplied):
        raise HTTPException(status_code=403, detail="Invalid upload token")
    if submission.get("proof_stored_filename"):
        raise HTTPException(status_code=409, detail="Proof file already uploaded")

    content_type = (media.content_type or "").lower().split(";")[0].strip()
    if content_type not in PROOF_TYPES:
        raise HTTPException(status_code=415, detail="Only PDF, JPEG, PNG, or WebP files are allowed")
    filename = Path(media.filename or "proof").name[:180]
    if not filename:
        raise HTTPException(status_code=400, detail="filename required")

    content = await media.read(MAX_PROOF_BYTES + 1)
    if len(content) > MAX_PROOF_BYTES:
        raise HTTPException(status_code=413, detail="File exceeds 10 MB")
    if not _matches_file_signature(content_type, content):
        raise HTTPException(status_code=415, detail="File content does not match its MIME type")

    stored_name = f"{submission_id}-{secrets.token_hex(16)}{PROOF_TYPES[content_type]}"
    PROOF_DIR.mkdir(parents=True, exist_ok=True)
    target = PROOF_DIR / stored_name
    target.write_bytes(content)
    try:
        record = set_burn_ban_proof_file(
            submission_id, stored_name, filename, content_type,
        )
    except Exception:
        target.unlink(missing_ok=True)
        raise
    if not record:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=404, detail="Submission not found")
    return {
        "success": True,
        "proof": {
            "original_filename": record["proof_original_filename"],
            "content_type": record["proof_content_type"],
        },
    }


@router.get("/api/admin/burn-bans")
def admin_list_burn_bans(
    token: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
):
    _require_admin(token)
    if status and status not in STATUSES:
        raise HTTPException(status_code=400, detail=f"unknown status: {status!r}")
    items = list_burn_ban_submissions(status=status, limit=limit, offset=offset, admin=True)
    return {"success": True, "submissions": items, "count": len(items)}


@router.post("/api/admin/burn-bans", status_code=201)
def admin_create_burn_ban(payload: BurnBanAdminCreate, token: Optional[str] = None):
    actor = _require_admin(token)
    county_name = _known_county_fips()[payload.county_fips]
    submission = create_burn_ban_submission(
        county_fips=payload.county_fips,
        county_name=county_name,
        submitter_name=payload.submitter_name,
        submitter_contact=payload.submitter_contact,
        proof_url=payload.proof_url,
        effective_at=payload.effective_at,
        expires_at=payload.expires_at,
        submitter_ip_hash="admin",
        upload_token_hash="",
        captcha_verdict="admin",
        consent_version=CONSENT_VERSION,
    )
    result = moderate_burn_ban_submission(
        submission["id"],
        to_status="confirmed",
        actor=actor,
        reason=payload.moderator_note or "Created by administrator",
        effective_at=payload.effective_at,
        expires_at=payload.expires_at,
    )
    if result is None:
        raise HTTPException(status_code=500, detail="Failed to publish burn ban")
    _maybe_regenerate_map()
    return {"success": True, "submission": result}


@router.get("/api/admin/burn-bans/map")
def admin_get_burn_ban_map(token: Optional[str] = None):
    _require_admin(token)
    from services.burn_ban_map import burn_ban_map_public_meta
    meta = burn_ban_map_public_meta()
    active = list_active_burn_bans()
    return {
        "success": True,
        "map": meta,
        "active_count": len(active),
    }


@router.post("/api/admin/burn-bans/map/regenerate")
def admin_regenerate_burn_ban_map(token: Optional[str] = None):
    _require_admin(token)
    from services.burn_ban_map import generate_burn_ban_map
    result = generate_burn_ban_map()
    return {
        "success": True,
        "map": {
            "image_path": result["image_path"],
            "url": "/images/mo-burnban.png",
            "updated_at": result["updated_at"],
        },
        "active_count": result["active_counties"],
    }


@router.get("/api/admin/burn-bans/{submission_id}")
def admin_get_burn_ban(submission_id: int, token: Optional[str] = None):
    _require_admin(token)
    submission = get_burn_ban_submission(submission_id, admin=True)
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    return {"success": True, "submission": submission}


@router.get("/api/admin/burn-bans/{submission_id}/proof")
def admin_get_burn_ban_proof(submission_id: int, token: Optional[str] = None):
    _require_admin(token)
    submission = get_burn_ban_submission(submission_id, admin=True)
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    stored = submission.get("proof_stored_filename") or ""
    if not stored:
        raise HTTPException(status_code=404, detail="Proof file not found")
    target = PROOF_DIR / Path(stored).name
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Proof file unavailable")
    return FileResponse(
        target,
        media_type=submission.get("proof_content_type") or "application/octet-stream",
        filename=submission.get("proof_original_filename") or target.name,
    )


@router.post("/api/admin/burn-bans/{submission_id}/confirm")
def admin_confirm_burn_ban(
    submission_id: int,
    payload: BurnBanModeration,
    token: Optional[str] = None,
):
    actor = _require_admin(token)
    submission = get_burn_ban_submission(submission_id, admin=True)
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    request_type = submission.get("request_type") or "issue"
    effective_at = payload.effective_at or submission["effective_at"]
    if payload.expires_at is None:
        expires_at = submission.get("expires_at") or ""
    else:
        expires_at = payload.expires_at
    if request_type == "lift":
        expires_at = ""
    if expires_at and expires_at <= effective_at:
        raise HTTPException(status_code=400, detail="expires_at must be after effective_at")
    check = {**submission, "effective_at": effective_at, "expires_at": expires_at}
    if request_type == "issue" and not _has_proof(check):
        raise HTTPException(status_code=400, detail="A proof URL or uploaded document is required")

    result = moderate_burn_ban_submission(
        submission_id,
        to_status="confirmed",
        actor=actor,
        reason=payload.moderator_note,
        effective_at=effective_at,
        expires_at=expires_at,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Submission not found")
    if result.get("already_moderated"):
        raise HTTPException(status_code=409, detail="Submission already moderated")
    if request_type == "lift":
        expire_confirmed_burn_bans_for_county(
            submission["county_fips"],
            actor=actor,
            reason=payload.moderator_note or "Public lift request confirmed",
            exclude_id=submission_id,
        )
    _maybe_regenerate_map()
    return {"success": True, "submission": result}


@router.post("/api/admin/burn-bans/{submission_id}/deny")
def admin_deny_burn_ban(
    submission_id: int,
    payload: BurnBanRejection,
    token: Optional[str] = None,
):
    actor = _require_admin(token)
    result = moderate_burn_ban_submission(
        submission_id,
        to_status="denied",
        actor=actor,
        reason=payload.reason,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Submission not found")
    if result.get("already_moderated"):
        raise HTTPException(status_code=409, detail="Submission already moderated")
    return {"success": True, "submission": result}


@router.put("/api/admin/burn-bans/{submission_id}")
def admin_update_burn_ban(
    submission_id: int,
    payload: BurnBanUpdate,
    token: Optional[str] = None,
):
    actor = _require_admin(token)
    county_name = None
    if payload.county_fips:
        county_name = _known_county_fips()[payload.county_fips]
    result = update_burn_ban_submission(
        submission_id,
        actor=actor,
        edit_reason=payload.edit_reason,
        effective_at=payload.effective_at,
        expires_at=payload.expires_at,
        proof_url=payload.proof_url,
        county_fips=payload.county_fips,
        county_name=county_name,
    )
    if not result:
        raise HTTPException(status_code=404, detail="Submission not found")
    if result.get("status") == "confirmed":
        _maybe_regenerate_map()
    return {"success": True, "submission": result}


@router.delete("/api/admin/burn-bans/{submission_id}")
def admin_delete_burn_ban(
    submission_id: int,
    token: Optional[str] = None,
    reason: str = "",
):
    actor = _require_admin(token)
    submission = get_burn_ban_submission(submission_id, admin=True)
    if not submission:
        raise HTTPException(status_code=404, detail="Submission not found")
    was_confirmed = submission.get("status") == "confirmed"
    deleted = delete_burn_ban_submission(submission_id, actor=actor, reason=reason)
    if not deleted:
        raise HTTPException(status_code=404, detail="Submission not found")
    if was_confirmed:
        _maybe_regenerate_map()
    return {"success": True}


def run_burn_ban_maintenance() -> Dict[str, int]:
    from core.database import expire_stale_burn_bans
    from services.burn_ban_map import ensure_burn_ban_map
    expired = expire_stale_burn_bans()
    purged = purge_burn_ban_submission_pii()
    throttle_purged = purge_burn_ban_throttle_rows()
    if expired:
        _maybe_regenerate_map()
    else:
        try:
            ensure_burn_ban_map()
        except Exception as exc:
            logger.warning("Burn-ban map ensure failed: %s", exc, exc_info=True)
    return {"expired": expired, "pii_purged": purged, "throttle_purged": throttle_purged}
