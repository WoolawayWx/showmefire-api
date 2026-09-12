"""Department static graphic API."""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import secrets
import sqlite3
import time
import tempfile
import uuid
import zipfile
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import geopandas as gpd
from PIL import Image, UnidentifiedImageError

from fastapi import APIRouter, File, Header, HTTPException, Request, Response, UploadFile
from pydantic import BaseModel, Field, field_validator

from core.database import get_db_path
from core.security import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    AUTH_COOKIE_DOMAIN,
    AUTH_COOKIE_SAMESITE,
    AUTH_COOKIE_SECURE,
    GRAPHICS_ACCESS_COOKIE_NAME,
    GRAPHICS_REFRESH_COOKIE_NAME,
    REFRESH_TOKEN_EXPIRE_DAYS,
    create_graphics_access_token,
    create_graphics_refresh_token,
    hash_password,
    password_strength_error,
    verify_graphics_token,
    verify_password,
    verify_token,
)
from services.archive_bundler import _r2_client
from services.graphic_renderer import PRODUCT_IDS, center_zoom_for_bounds, render_graphic

router = APIRouter(prefix="/api/graphics", tags=["graphics"])
_executor = None
ASSET_ROOT = Path(os.getenv("SMF_GRAPHICS_ASSET_ROOT", "data/graphics/assets"))
CDN_BASE_URL = os.getenv("CDN_BASE_URL", "https://cdn.showmefire.org").rstrip("/")
GRAPHICS_CDN_BASE_URL = os.getenv("SMF_GRAPHICS_CDN_BASE_URL", f"{CDN_BASE_URL}/imggen").rstrip("/")


def _image_url(bundle_id: str) -> str:
    return f"{GRAPHICS_CDN_BASE_URL}/{bundle_id}/image.png"


EMAIL_PATTERN = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"


class DepartmentCreate(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    slug: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{1,63}$")

    @field_validator("slug")
    @classmethod
    def valid_slug(cls, value):
        return value


class InviteCreate(BaseModel):
    email: str = Field(pattern=EMAIL_PATTERN, max_length=254)


class InviteVerify(BaseModel):
    email: str = Field(pattern=EMAIL_PATTERN, max_length=254)
    token: str = Field(min_length=8, max_length=200)


class PasswordSet(BaseModel):
    email: str = Field(pattern=EMAIL_PATTERN, max_length=254)
    token: str = Field(min_length=8, max_length=200)
    password: str = Field(min_length=1, max_length=200)


class PasswordLogin(BaseModel):
    email: str = Field(pattern=EMAIL_PATTERN, max_length=254)
    password: str = Field(min_length=1, max_length=200)


class BundleCreate(BaseModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{2,63}$")
    name: str = Field(min_length=2, max_length=120)
    product_id: str
    header_text: str = Field(default="Show Me Fire Weather Graphics", max_length=180)
    subtitle: str = Field(default="", max_length=240)
    jurisdiction_asset_id: Optional[int] = None
    department_logo_asset_id: Optional[int] = None
    center: tuple[float, float] = (-92.45, 38.343121)
    zoom: float = Field(default=7.5, ge=1, le=14)
    # Legacy compatibility for bundles created before center/zoom support.
    map_extent: Optional[tuple[float, float, float, float]] = None
    background_color: str = Field(default="#e8e8e8", pattern=r"^#[0-9a-fA-F]{6}$")
    basemap_style: str = "rastertiles/voyager"

    @field_validator("product_id")
    @classmethod
    def valid_product(cls, value):
        if value not in PRODUCT_IDS:
            raise ValueError("unsupported graphic product")
        return value

    @field_validator("map_extent")
    @classmethod
    def valid_extent(cls, value):
        if value is not None and (value[0] >= value[1] or value[2] >= value[3]):
            raise ValueError("map_extent must be [west, east, south, north]")
        return value

    @field_validator("center")
    @classmethod
    def valid_center(cls, value):
        if not (-180 <= value[0] <= 180 and -85 <= value[1] <= 85):
            raise ValueError("center must be [longitude, latitude]")
        return value

    @field_validator("basemap_style")
    @classmethod
    def valid_basemap(cls, value):
        if value not in {"rastertiles/voyager", "light_all", "dark_all"}:
            raise ValueError("unsupported basemap style")
        return value


@contextmanager
def _db():
    db = sqlite3.connect(get_db_path(), timeout=30)
    db.row_factory = sqlite3.Row
    try:
        with db:
            yield db
    finally:
        db.close()


def _get_executor():
    global _executor
    if _executor is None:
        _executor = ProcessPoolExecutor(max_workers=max(1, min(2, int(os.getenv("SMF_GRAPHICS_WORKERS", "1")))))
    return _executor


def _admin(token: Optional[str]):
    if not verify_token(token):
        raise HTTPException(status_code=401, detail="Unauthorized")


def _set_graphics_cookies(response: Response, email: str, department_id: int, user_id: int) -> None:
    access = create_graphics_access_token(email, department_id, user_id)
    refresh = create_graphics_refresh_token(email, department_id, user_id)
    for name, value, max_age in (
        (GRAPHICS_ACCESS_COOKIE_NAME, access, ACCESS_TOKEN_EXPIRE_MINUTES * 60),
        (GRAPHICS_REFRESH_COOKIE_NAME, refresh, REFRESH_TOKEN_EXPIRE_DAYS * 24 * 60 * 60),
    ):
        response.set_cookie(
            key=name, value=value, max_age=max_age, httponly=True,
            secure=AUTH_COOKIE_SECURE, samesite=AUTH_COOKIE_SAMESITE,
            domain=AUTH_COOKIE_DOMAIN, path="/",
        )


def _clear_graphics_cookies(response: Response) -> None:
    for name in (GRAPHICS_ACCESS_COOKIE_NAME, GRAPHICS_REFRESH_COOKIE_NAME):
        response.delete_cookie(
            key=name, httponly=True, secure=AUTH_COOKIE_SECURE,
            samesite=AUTH_COOKIE_SAMESITE, domain=AUTH_COOKIE_DOMAIN, path="/",
        )


def _department_key(authorization: Optional[str]):
    if authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
        digest = hashlib.sha256(supplied.encode()).hexdigest()
        with _db() as db:
            row = db.execute("""SELECT k.*, d.name department_name, d.daily_limit, d.monthly_limit
                FROM graphic_api_keys k JOIN graphic_departments d ON d.id=k.department_id
                WHERE k.key_hash=? AND k.revoked_at IS NULL""", (digest,)).fetchone()
            if not row:
                raise HTTPException(status_code=401, detail="Invalid or revoked API key")
            db.execute("UPDATE graphic_api_keys SET last_used_at=CURRENT_TIMESTAMP WHERE id=?", (row["id"],))
        return row
    payload = verify_graphics_token(None)
    if not payload:
        raise HTTPException(status_code=401, detail="Sign in required")
    with _db() as db:
        row = db.execute("""SELECT k.*, d.name department_name, d.daily_limit, d.monthly_limit
            FROM graphic_department_users u
            JOIN graphic_api_keys k ON k.id = u.api_key_id
            JOIN graphic_departments d ON d.id = u.department_id
            WHERE u.id=? AND k.revoked_at IS NULL""", (payload["user_id"],)).fetchone()
        if not row:
            raise HTTPException(status_code=401, detail="Sign in required")
        db.execute("UPDATE graphic_api_keys SET last_used_at=CURRENT_TIMESTAMP WHERE id=?", (row["id"],))
    return row


@router.get("/products")
def products():
    return {"products": [
        {"id": "spc_cat", "label": "Day 1 Categorical"},
        {"id": "spc_tor", "label": "Day 1 Tornado"},
        {"id": "spc_wind", "label": "Day 1 Wind"},
        {"id": "spc_hail", "label": "Day 1 Hail"},
        {"id": "spc_four_panel", "label": "Day 1 Four Panel"},
        {"id": "mo_alerts", "label": "Missouri Weather Alerts"},
    ]}


@router.get("/me")
def graphics_session(authorization: Optional[str] = Header(default=None)):
    key = _department_key(authorization)
    with _db() as db:
        department = db.execute(
            "SELECT id,name,slug,daily_limit,monthly_limit FROM graphic_departments WHERE id=?",
            (key["department_id"],),
        ).fetchone()
        today = db.execute(
            "SELECT COUNT(*) FROM graphic_usage_events WHERE department_id=? AND created_at >= date('now')",
            (key["department_id"],),
        ).fetchone()[0]
        month = db.execute(
            "SELECT COUNT(*) FROM graphic_usage_events WHERE department_id=? AND created_at >= date('now','start of month')",
            (key["department_id"],),
        ).fetchone()[0]
    return {
        "department": dict(department),
        "token_prefix": key["key_prefix"],
        "usage": {"today": today, "month": month},
    }


def _find_invited_user(db, email: str, token: str):
    digest = hashlib.sha256(token.encode()).hexdigest()
    return db.execute(
        """SELECT u.*, k.key_hash, k.revoked_at FROM graphic_department_users u
           JOIN graphic_api_keys k ON k.id = u.api_key_id
           WHERE u.email=? AND k.key_hash=?""",
        (email, digest),
    ).fetchone()


@router.post("/auth/verify-invite")
def verify_invite(payload: InviteVerify):
    with _db() as db:
        row = _find_invited_user(db, payload.email, payload.token)
    if not row or row["revoked_at"] is not None or row["password_hash"] is not None:
        raise HTTPException(status_code=401, detail="This invite link is invalid, expired, or already used")
    return {"valid": True}


@router.post("/auth/set-password")
def set_password(payload: PasswordSet, response: Response):
    with _db() as db:
        row = _find_invited_user(db, payload.email, payload.token)
        if not row or row["revoked_at"] is not None or row["password_hash"] is not None:
            raise HTTPException(status_code=401, detail="This invite link is invalid, expired, or already used")
        strength_error = password_strength_error(payload.password)
        if strength_error:
            raise HTTPException(status_code=400, detail=strength_error)
        db.execute(
            "UPDATE graphic_department_users SET password_hash=?, password_set_at=CURRENT_TIMESTAMP, last_login_at=CURRENT_TIMESTAMP WHERE id=?",
            (hash_password(payload.password), row["id"]),
        )
    _set_graphics_cookies(response, payload.email, row["department_id"], row["id"])
    return {"success": True}


@router.post("/auth/login")
def graphics_login(payload: PasswordLogin, response: Response):
    with _db() as db:
        row = db.execute("SELECT * FROM graphic_department_users WHERE email=?", (payload.email,)).fetchone()
        if not row or not row["password_hash"] or not verify_password(payload.password, row["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        db.execute("UPDATE graphic_department_users SET last_login_at=CURRENT_TIMESTAMP WHERE id=?", (row["id"],))
    _set_graphics_cookies(response, payload.email, row["department_id"], row["id"])
    return {"success": True}


@router.post("/auth/verify")
def graphics_verify():
    return {"valid": verify_graphics_token(None) is not None}


@router.post("/auth/refresh")
def graphics_refresh(request: Request, response: Response):
    payload = verify_graphics_token(request.cookies.get(GRAPHICS_REFRESH_COOKIE_NAME), expected_type="graphics_refresh")
    if not payload:
        raise HTTPException(status_code=401, detail="No active session")
    _set_graphics_cookies(response, payload["sub"], payload["department_id"], payload["user_id"])
    return {"success": True}


@router.post("/auth/logout")
def graphics_logout(response: Response):
    _clear_graphics_cookies(response)
    return {"success": True}


@router.post("/admin/departments")
def create_department(payload: DepartmentCreate, token: Optional[str] = None):
    _admin(token)
    with _db() as db:
        try:
            cursor = db.execute("INSERT INTO graphic_departments(name,slug) VALUES (?,?)", (payload.name, payload.slug))
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=409, detail="Department slug already exists")
    return {"id": cursor.lastrowid, "name": payload.name, "slug": payload.slug}


@router.post("/admin/departments/{department_id}/keys")
def issue_key(department_id: int, token: Optional[str] = None):
    _admin(token)
    secret = "smf_" + secrets.token_urlsafe(32)
    prefix = secret[:12]
    with _db() as db:
        if not db.execute("SELECT 1 FROM graphic_departments WHERE id=?", (department_id,)).fetchone():
            raise HTTPException(status_code=404, detail="Department not found")
        db.execute("INSERT INTO graphic_api_keys(department_id,key_prefix,key_hash) VALUES (?,?,?)", (department_id, prefix, hashlib.sha256(secret.encode()).hexdigest()))
    return {"key": secret, "prefix": prefix, "warning": "Store this key now; it cannot be displayed again."}


@router.post("/admin/departments/{department_id}/invite")
def invite_department_user(department_id: int, payload: InviteCreate, token: Optional[str] = None):
    _admin(token)
    secret = "smf_" + secrets.token_urlsafe(32)
    prefix = secret[:12]
    with _db() as db:
        if not db.execute("SELECT 1 FROM graphic_departments WHERE id=?", (department_id,)).fetchone():
            raise HTTPException(status_code=404, detail="Department not found")
        if db.execute("SELECT 1 FROM graphic_department_users WHERE email=?", (payload.email,)).fetchone():
            raise HTTPException(status_code=409, detail="An account already exists for this email")
        cursor = db.execute(
            "INSERT INTO graphic_api_keys(department_id,key_prefix,key_hash) VALUES (?,?,?)",
            (department_id, prefix, hashlib.sha256(secret.encode()).hexdigest()),
        )
        db.execute(
            "INSERT INTO graphic_department_users(department_id,api_key_id,email) VALUES (?,?,?)",
            (department_id, cursor.lastrowid, payload.email),
        )
    return {
        "email": payload.email,
        "token": secret,
        "warning": "Give this token to the department now; it cannot be shown again.",
    }


@router.post("/admin/keys/{key_id}/revoke")
def revoke_key(key_id: int, token: Optional[str] = None):
    _admin(token)
    with _db() as db:
        changed = db.execute("UPDATE graphic_api_keys SET revoked_at=CURRENT_TIMESTAMP WHERE id=? AND revoked_at IS NULL", (key_id,)).rowcount
    if not changed:
        raise HTTPException(status_code=404, detail="Active key not found")
    return {"revoked": True, "key_id": key_id}


@router.get("/admin/departments/{department_id}/usage")
def department_usage(department_id: int, token: Optional[str] = None):
    _admin(token)
    with _db() as db:
        department = db.execute("SELECT id,name,daily_limit,monthly_limit FROM graphic_departments WHERE id=?", (department_id,)).fetchone()
        if not department:
            raise HTTPException(status_code=404, detail="Department not found")
        rows = db.execute("SELECT status,product_id,COUNT(*) count,COALESCE(SUM(bytes),0) bytes FROM graphic_usage_events WHERE department_id=? AND created_at >= date('now','-30 day') GROUP BY status,product_id", (department_id,)).fetchall()
    return {"department": dict(department), "last_30_days": [dict(row) for row in rows]}


@router.get("/bundles")
def list_bundles(authorization: Optional[str] = Header(default=None)):
    key = _department_key(authorization)
    with _db() as db:
        rows = db.execute("SELECT id,name,version,config_json,updated_at FROM graphic_bundles WHERE department_id=? AND active=1 ORDER BY name", (key["department_id"],)).fetchall()
    return {"bundles": [
        {**dict(row), "config": json.loads(row["config_json"]), "image_url": _image_url(row["id"])}
        for row in rows
    ]}


def _validate_bundle_assets(db, payload: BundleCreate, department_id: int):
    if payload.jurisdiction_asset_id is not None and not db.execute(
        "SELECT 1 FROM graphic_assets WHERE id=? AND department_id=? AND content_type='application/geo+json'",
        (payload.jurisdiction_asset_id, department_id),
    ).fetchone():
        raise HTTPException(status_code=400, detail="Jurisdiction asset is not available to this department")
    if payload.department_logo_asset_id is not None and not db.execute(
        "SELECT 1 FROM graphic_assets WHERE id=? AND department_id=? AND content_type='image/png'",
        (payload.department_logo_asset_id, department_id),
    ).fetchone():
        raise HTTPException(status_code=400, detail="Department logo is not available to this department")


@router.post("/bundles")
def create_bundle(payload: BundleCreate, authorization: Optional[str] = Header(default=None)):
    key = _department_key(authorization)
    config = payload.model_dump()
    with _db() as db:
        _validate_bundle_assets(db, payload, key["department_id"])
        try:
            db.execute("INSERT INTO graphic_bundles(id,department_id,name,config_json) VALUES (?,?,?,?)", (payload.id, key["department_id"], payload.name, json.dumps(config)))
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=409, detail="Bundle ID already exists")
    return {"bundle_id": payload.id, "config": config, "image_url": _image_url(payload.id)}


@router.put("/bundles/{bundle_id}")
def update_bundle(bundle_id: str, payload: BundleCreate, authorization: Optional[str] = Header(default=None)):
    key = _department_key(authorization)
    if payload.id != bundle_id:
        raise HTTPException(status_code=400, detail="Bundle ID in the path and body must match")
    config = payload.model_dump()
    with _db() as db:
        _validate_bundle_assets(db, payload, key["department_id"])
        changed = db.execute(
            """UPDATE graphic_bundles SET name=?,config_json=?,version=version+1,updated_at=CURRENT_TIMESTAMP
               WHERE id=? AND department_id=? AND active=1""",
            (payload.name, json.dumps(config), bundle_id, key["department_id"]),
        ).rowcount
        if not changed:
            raise HTTPException(status_code=404, detail="Bundle not found")
        version = db.execute("SELECT version FROM graphic_bundles WHERE id=?", (bundle_id,)).fetchone()[0]
    return {"bundle_id": bundle_id, "version": version, "config": config, "image_url": _image_url(bundle_id)}


@router.post("/assets")
async def upload_asset(upload: UploadFile = File(...), authorization: Optional[str] = Header(default=None)):
    key = _department_key(authorization)
    filename = Path(upload.filename or "asset").name
    content = await upload.read()
    if len(content) > 25 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Asset is larger than 25 MB")
    content_type = upload.content_type or "application/octet-stream"
    if content_type not in {"application/geo+json", "application/json", "application/zip", "application/x-zip-compressed"}:
        raise HTTPException(status_code=415, detail="Upload GeoJSON or a Shapefile ZIP")
    digest = hashlib.sha256(content).hexdigest()
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    path = ASSET_ROOT / f"{key['department_id']}-{digest[:16]}.geojson"
    try:
        with tempfile.TemporaryDirectory(prefix="smf-graphics-asset-") as temporary:
            temporary_path = Path(temporary)
            if content_type.startswith("application/zip") or filename.lower().endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(content)) as archive:
                    for member in archive.infolist():
                        member_path = Path(member.filename)
                        if member_path.is_absolute() or ".." in member_path.parts:
                            raise ValueError("unsafe archive path")
                    archive.extractall(temporary_path)
                shapefiles = list(temporary_path.rglob("*.shp"))
                if len(shapefiles) != 1:
                    raise ValueError("archive must contain exactly one Shapefile")
                frame = gpd.read_file(shapefiles[0])
            else:
                source = temporary_path / "upload.geojson"
                source.write_bytes(content)
                frame = gpd.read_file(source)
            if frame.empty or frame.crs is None:
                raise ValueError("jurisdiction must contain geometry with a declared CRS")
            frame = frame.to_crs("EPSG:4326")
            if not frame.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
                raise ValueError("jurisdiction geometry must be Polygon or MultiPolygon")
            if not frame.geometry.is_valid.all():
                raise ValueError("jurisdiction contains invalid geometry")
            frame.to_file(path, driver="GeoJSON")
            suggested_center, suggested_zoom = center_zoom_for_bounds(frame.total_bounds)
    except (zipfile.BadZipFile, ValueError, OSError) as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid jurisdiction asset: {exc}")
    with _db() as db:
        cursor = db.execute("INSERT INTO graphic_assets(department_id,filename,content_type,sha256,path) VALUES (?,?,?,?,?)", (key["department_id"], filename, "application/geo+json", digest, str(path)))
    return {"asset_id": cursor.lastrowid, "filename": filename, "sha256": digest,
            "suggested_center": suggested_center, "suggested_zoom": suggested_zoom}


@router.post("/assets/logos")
async def upload_logo(upload: UploadFile = File(...), authorization: Optional[str] = Header(default=None)):
    """Store a validated department PNG for optional placement in the header."""
    key = _department_key(authorization)
    filename = Path(upload.filename or "logo.png").name
    content = await upload.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Logo is larger than 5 MB")
    if upload.content_type != "image/png" and not filename.lower().endswith(".png"):
        raise HTTPException(status_code=415, detail="Department logo must be a PNG")
    try:
        with Image.open(io.BytesIO(content)) as image:
            image.verify()
        with Image.open(io.BytesIO(content)) as image:
            if image.format != "PNG":
                raise ValueError("file is not a PNG")
            if image.width > 4096 or image.height > 4096:
                raise ValueError("logo dimensions may not exceed 4096x4096")
            normalized = image.convert("RGBA")
            output = io.BytesIO()
            normalized.save(output, format="PNG", optimize=True)
            content = output.getvalue()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"Invalid department logo: {exc}")
    digest = hashlib.sha256(content).hexdigest()
    ASSET_ROOT.mkdir(parents=True, exist_ok=True)
    path = ASSET_ROOT / f"{key['department_id']}-{digest[:16]}.png"
    path.write_bytes(content)
    with _db() as db:
        cursor = db.execute(
            "INSERT INTO graphic_assets(department_id,filename,content_type,sha256,path) VALUES (?,?,?,?,?)",
            (key["department_id"], filename, "image/png", digest, str(path)),
        )
    return {"asset_id": cursor.lastrowid, "filename": filename, "sha256": digest}


async def _run_job(job_id: str, bundle: dict, department_id: int, api_key_id: int):
    started = time.monotonic()
    with _db() as db:
        db.execute("UPDATE graphic_jobs SET status='running', started_at=CURRENT_TIMESTAMP WHERE id=?", (job_id,))
    try:
        asset_id = bundle.get("jurisdiction_asset_id")
        if asset_id is not None:
            with _db() as db:
                asset = db.execute("SELECT path FROM graphic_assets WHERE id=? AND department_id=?", (asset_id, department_id)).fetchone()
            if not asset:
                raise ValueError("configured jurisdiction asset no longer exists")
            bundle["jurisdiction_path"] = asset["path"]
        logo_asset_id = bundle.get("department_logo_asset_id")
        if logo_asset_id is not None:
            with _db() as db:
                logo_asset = db.execute(
                    "SELECT path FROM graphic_assets WHERE id=? AND department_id=? AND content_type='image/png'",
                    (logo_asset_id, department_id),
                ).fetchone()
            if not logo_asset:
                raise ValueError("configured department logo asset no longer exists")
            bundle["department_logo_path"] = logo_asset["path"]
        result = await asyncio.get_running_loop().run_in_executor(_get_executor(), render_graphic, bundle)
        data = result["bytes"]
        if len(data) < 1000:
            raise ValueError("rendered image failed content validation")
        with _db() as db:
            previous = db.execute("SELECT source_fingerprint FROM graphic_jobs WHERE bundle_id=? AND status='completed' ORDER BY finished_at DESC LIMIT 1", (bundle["id"],)).fetchone()
        if previous and previous["source_fingerprint"] == result["source_fingerprint"]:
            with _db() as db:
                db.execute("UPDATE graphic_jobs SET status='skipped',source_fingerprint=?,manifest_json=?,image_url=?,finished_at=CURRENT_TIMESTAMP WHERE id=?", (result["source_fingerprint"], json.dumps({"reason": "source unchanged", **{k: result[k] for k in ("source_urls", "renderer_version", "generated_at")}}), _image_url(bundle["id"]), job_id))
                db.execute("INSERT INTO graphic_usage_events(department_id,api_key_id,job_id,status,product_id,latency_ms,bytes) VALUES (?,?,?,?,?,?,?)", (department_id, api_key_id, job_id, "skipped", bundle["product_id"], int((time.monotonic()-started)*1000), 0))
            return
        bucket = os.getenv("R2_BUCKET", "cdn-showmefire")
        key = f"imggen/{bundle['id']}/image.png"
        if not all(os.getenv(name) for name in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")):
            raise RuntimeError("graphics R2 credentials are not configured")
        _r2_client().put_object(Bucket=bucket, Key=key, Body=data, ContentType="image/png", CacheControl="public,max-age=60,stale-while-revalidate=300")
        _purge_cdn(key)
        manifest = {k: result[k] for k in ("source_fingerprint", "source_urls", "renderer_version", "generated_at")}
        url = _image_url(bundle["id"])
        with _db() as db:
            db.execute("UPDATE graphic_jobs SET status='completed', source_fingerprint=?,manifest_json=?,image_url=?,finished_at=CURRENT_TIMESTAMP WHERE id=?", (result["source_fingerprint"], json.dumps(manifest), url, job_id))
            db.execute("INSERT INTO graphic_usage_events(department_id,api_key_id,job_id,status,product_id,latency_ms,bytes) VALUES (?,?,?,?,?,?,?)", (department_id, api_key_id, job_id, "completed", bundle["product_id"], int((time.monotonic()-started)*1000), len(data)))
    except Exception as exc:
        with _db() as db:
            db.execute("UPDATE graphic_jobs SET status='failed',error=?,finished_at=CURRENT_TIMESTAMP WHERE id=?", (str(exc)[:1000], job_id))
            db.execute("INSERT INTO graphic_usage_events(department_id,api_key_id,job_id,status,product_id,latency_ms) VALUES (?,?,?,?,?,?)", (department_id, api_key_id, job_id, "failed", bundle["product_id"], int((time.monotonic()-started)*1000)))


def _purge_cdn(key: str):
    zone_id = os.getenv("CLOUDFLARE_ZONE_ID", "").strip()
    endpoint = os.getenv("CLOUDFLARE_ZONE_PURGE_URL") or (
        f"https://api.cloudflare.com/client/v4/zones/{zone_id}/purge_cache" if zone_id else None
    )
    token = os.getenv("CLOUDFLARE_API_TOKEN")
    if not endpoint or not token:
        return
    import requests
    url = f"{CDN_BASE_URL}/{key}"
    response = requests.post(endpoint, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}, json={"files": [url]}, timeout=15)
    response.raise_for_status()


@router.post("/jobs", status_code=202)
async def create_job(bundle_id: str, authorization: Optional[str] = Header(default=None)):
    key = _department_key(authorization)
    with _db() as db:
        bundle_row = db.execute("SELECT * FROM graphic_bundles WHERE id=? AND department_id=? AND active=1", (bundle_id, key["department_id"])).fetchone()
        if not bundle_row:
            raise HTTPException(status_code=404, detail="Bundle not found")
        bundle = json.loads(bundle_row["config_json"])
        bundle["id"] = bundle_id
        today = db.execute("SELECT COUNT(*) FROM graphic_usage_events WHERE department_id=? AND created_at >= date('now')", (key["department_id"],)).fetchone()[0]
        month = db.execute("SELECT COUNT(*) FROM graphic_usage_events WHERE department_id=? AND created_at >= date('now','start of month')", (key["department_id"],)).fetchone()[0]
        if today >= key["daily_limit"] or month >= key["monthly_limit"]:
            raise HTTPException(status_code=429, detail="Graphics usage quota exceeded")
        job_id = str(uuid.uuid4())
        db.execute("INSERT INTO graphic_jobs(id,bundle_id,department_id,status,config_json) VALUES (?,?,?,?,?)", (job_id, bundle_id, key["department_id"], "queued", json.dumps(bundle)))
    asyncio.create_task(_run_job(job_id, bundle, key["department_id"], key["id"]))
    return {"job_id": job_id, "status": "queued", "image_url": _image_url(bundle_id)}


@router.get("/jobs/{job_id}")
def get_job(job_id: str, authorization: Optional[str] = Header(default=None)):
    key = _department_key(authorization)
    with _db() as db:
        row = db.execute("SELECT * FROM graphic_jobs WHERE id=? AND department_id=?", (job_id, key["department_id"])).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Job not found")
    result = dict(row)
    result["manifest"] = json.loads(result.pop("manifest_json") or "{}")
    result.pop("config_json", None)
    return result
