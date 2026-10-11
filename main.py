from dotenv import load_dotenv
load_dotenv()

from fastapi import Depends, FastAPI, HTTPException, UploadFile, File, Request, Header
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, FileResponse
from contextlib import asynccontextmanager
from services.synoptic import (
    get_station_data, 
    fetch_historical_station_data, 
    save_raw_data_to_archive, 
    fetch_synoptic_data
)
from services.timeseries import get_timeseries_data, fetchtimeseriesdata
from services.banner import BannerData, load_banner_config, save_banner_config
from services.file_manager import list_files, view_file
from services.log_maintenance import purge_old_logs
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import logging
import json
import asyncio
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response
from datetime import datetime, timedelta
from typing import Optional
from services.rss import write_rss_feed
from services import fm_explain
from maps.station_danger_history import (
    get_recent_fire_danger_history,
    normalize_fire_danger_counts,
    CLASS_VALUE_TO_LABEL,
)
from maps.observed_peak_history import STATE_FILE as OBSERVED_PEAK_STATE_FILE
from pathlib import Path
from pytz import timezone
import pandas as pd
from core.database import (
    init_database,
    get_latest_forecast,
    get_forecast_by_time,
    get_recent_forecasts,
    get_forecast_count,
    get_db_path,
    list_dev_projects,
    create_dev_project,
    update_dev_project,
    delete_dev_project,
    get_briefing_config,
    create_briefing
)
import sqlite3
from core.security import (
    verify_password,
    create_access_token,
    create_refresh_token,
    verify_token,
    require_admin,
    set_request_token,
    reset_request_token,
    ACCESS_TOKEN_EXPIRE_MINUTES,
    REFRESH_TOKEN_EXPIRE_DAYS,
    ACCESS_COOKIE_NAME,
    REFRESH_COOKIE_NAME,
    AUTH_COOKIE_SECURE,
    AUTH_COOKIE_SAMESITE,
    AUTH_COOKIE_DOMAIN,
    ADMIN_EMAIL,
    ADMIN_PASSWORD_HASH,
    ACCESS_TOKEN_EXPIRE_HOURS,
    SECRET_KEY,
    INSECURE_DEVELOPMENT_SECRET,
)
from core.scheduler import (
    create_scheduler,
    start_scheduler_jobs,
    run_initial_fetches,
    raws_station_data
)
from core.executors import shutdown_process_pool
from services.beta_products import BETA_ROOT
from core.config import (
    IMAGES_DIR,
    GIS_DIR,
    FORECAST_V1_DIR,
    PUBLIC_DIR,
    REPORTS_DIR,
    LOGS_DIR,
    ARCHIVE_RAW_DATA_DIR,
    MISSOURI_FIRES_JSON,
    MISSOURI_FIRES_GEOJSON
)
from routers import archive_admin, tiles, outlook, discord_admin, afds, spatial_model, mobile, posts, post_media, fires, fires_v2, fires_ingest, verification, feedback, model_admin, verification_admin, forecast_discussions, rtma_peak_admin, burn_bans, testbed, forecast_admin, forecast_admin_09z, forecast_09z_metrics, spread_rate_admin, fire_weather_alerts, forecast_v1, forecast_v1_admin, forecast_models_admin, fuel_sensor_admin, graphics, admin_confirm, admin_notifications, home_video, conditions, bulletins, arcgis_feature_service, display_auth, health, field
from forecast_v1.repository import ensure_schema as ensure_forecast_v1_schema
from core.field_store import ensure_field_schema
from core.public_cors import PublicTileCorsMiddleware

IS_PRODUCTION = os.getenv("ENVIRONMENT", "development").lower() == "production"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

from services.system_alerts import install_discord_log_handler, notify_system_issue

# Forwards any ERROR+ log record app-wide to the Discord staff-alert channel -
# this is what turns core/scheduler.py's ~25 existing
# logger.error(..., exc_info=True) job-failure calls into a Discord ping,
# with no need to touch each job individually. See services/system_alerts.py.
install_discord_log_handler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    run_scheduler = os.getenv("run_sch", "false").lower() == "true"
    scheduler_local = None

    missing_security = []
    if not ADMIN_EMAIL:
        missing_security.append("ADMIN_EMAIL")
    if not ADMIN_PASSWORD_HASH:
        missing_security.append("ADMIN_PASSWORD_HASH")
    if SECRET_KEY == INSECURE_DEVELOPMENT_SECRET:
        missing_security.append("JWT_SECRET")
    if not os.getenv("TURNSTILE_SECRET_KEY", "").strip():
        missing_security.append("TURNSTILE_SECRET_KEY")
    if missing_security:
        message = f"Missing secure runtime configuration: {', '.join(missing_security)}"
        if IS_PRODUCTION:
            raise RuntimeError(message)
        logger.warning("%s; development mode only", message)

    try:
        init_database()
        ensure_forecast_v1_schema()
        ensure_field_schema()
    except Exception as exc:
        logger.error("Database initialization failed: %s", exc, exc_info=True)
        raise

    try:
        from models.shadow_bundles import reapply_active_env_vars
        reapply_active_env_vars()
    except Exception as exc:
        logger.warning("Failed to reapply guarded-shadow active bundle pointers (non-fatal): %s", exc)

    if run_scheduler:
        logger.info("Starting scheduler...")
        scheduler_local = create_scheduler()
        start_scheduler_jobs(scheduler_local)
    else:
        logger.info("Scheduler disabled via run_sch environment variable")

    # Reachable by routers (e.g. GET /api/admin/models/schedule) for
    # read-only introspection - the scheduler instance was previously only
    # a bare local variable here, unreachable from anywhere else in the app.
    app.state.scheduler = scheduler_local

    # Start the fetch in the background so the API starts immediately
    asyncio.create_task(run_initial_fetches())
    
    yield
    
    if scheduler_local:
        logger.info("Shutting down scheduler...")
        scheduler_local.shutdown()
        shutdown_process_pool()

app = FastAPI(
    title="Show Me Fire Weather API",
    description="API for fetching synoptic weather data",
    lifespan=lifespan,
    docs_url=None if IS_PRODUCTION else "/docs",
    redoc_url=None if IS_PRODUCTION else "/redoc",
    openapi_url=None if IS_PRODUCTION else "/openapi.json"
)


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Last-resort handler for anything a route didn't already turn into an
    HTTPException - logs it (as before) and alerts staff on Discord via
    services/system_alerts.py. Starlette dispatches HTTPException to its own
    more-specific default handler first, so deliberate `raise
    HTTPException(...)` calls never reach this."""
    logger.error("Unhandled exception on %s %s", request.method, request.url.path, exc_info=True)
    try:
        notify_system_issue(
            source=f"{request.method} {request.url.path}",
            message=f"{type(exc).__name__}: {exc}",
        )
    except Exception:
        logger.warning("Failed to send system issue alert for unhandled exception", exc_info=True)
    return JSONResponse(status_code=500, content={"success": False, "error": "Internal server error"})


class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        response: Response = await super().get_response(path, scope)

        try:
            is_outlook_image = path.startswith("mo-outlook-day") and (
                path.endswith(".png") or path.endswith(".webp")
            )

            # Only apply long-lived headers to successful outlook image responses.
            if is_outlook_image and response.status_code == 200:
                response.headers["Cache-Control"] = "public, max-age=900"
                response.headers["Pragma"] = "public"
                response.headers["Expires"] = ""
                return response

            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        except Exception:
            # Never fail static file delivery because of cache-header customization.
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"

        return response


class ForecastStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        response: Response = await super().get_response(path, scope)
        if response.status_code == 200 and path.startswith("runs/"):
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        else:
            response.headers["Cache-Control"] = "no-cache"
        return response

# A fresh deployment may not have generated maps or reports yet. StaticFiles
# validates directories at construction time, so create its writable roots
# before mounting them instead of making API startup depend on old artifacts.
for static_directory in (IMAGES_DIR, GIS_DIR, FORECAST_V1_DIR, REPORTS_DIR, PUBLIC_DIR, BETA_ROOT):
    static_directory.mkdir(parents=True, exist_ok=True)

# Use this instead of the default StaticFiles
app.mount("/images", NoCacheStaticFiles(directory=str(IMAGES_DIR)), name="images")
app.mount("/gis", NoCacheStaticFiles(directory=str(GIS_DIR)), name="gis")
app.mount("/forecast-v1-assets", ForecastStaticFiles(directory=str(FORECAST_V1_DIR)), name="forecast-v1-assets")
app.mount("/home-video", StaticFiles(directory=str(home_video.HOME_VIDEO_DIR)), name="home-video")
app.mount("/reports", NoCacheStaticFiles(directory=str(REPORTS_DIR)), name="reports")
app.mount("/testbed-assets", NoCacheStaticFiles(directory=str(BETA_ROOT)), name="testbed-assets")
OPSBRIEF_DIR = Path(__file__).resolve().parent / "files" / "opsbrief"
OPSBRIEF_DIR.mkdir(parents=True, exist_ok=True)
OPSBRIEF_FALLBACK_FILE = "notactive.pdf"

# Include tile router for GeoTIFF rendering
app.include_router(health.router)
app.include_router(admin_confirm.router)
app.include_router(archive_admin.router)
app.include_router(tiles.router)
app.include_router(conditions.router)
app.include_router(outlook.router)
app.include_router(discord_admin.router)
app.include_router(afds.router)
app.include_router(spatial_model.router)
app.include_router(model_admin.router)
app.include_router(verification_admin.router)
app.include_router(rtma_peak_admin.router)
app.include_router(mobile.router)
app.include_router(admin_notifications.router)
app.include_router(home_video.router)
app.include_router(posts.router)
app.include_router(post_media.router)
app.include_router(fires.router)
app.include_router(fires_v2.router)
app.include_router(fires_ingest.router)
app.include_router(verification.router)
app.include_router(feedback.router)
app.include_router(forecast_discussions.router)
app.include_router(burn_bans.router)
app.include_router(arcgis_feature_service.router)
app.include_router(fire_weather_alerts.router)
app.include_router(testbed.router)
app.include_router(forecast_admin.router)
app.include_router(forecast_admin_09z.router)
app.include_router(forecast_09z_metrics.router)
app.include_router(spread_rate_admin.router)
app.include_router(forecast_v1.router)
app.include_router(forecast_v1_admin.router)
app.include_router(forecast_models_admin.router)
app.include_router(fuel_sensor_admin.router)
app.include_router(graphics.router)
app.include_router(bulletins.router)
app.include_router(display_auth.router)
app.include_router(field.router)

origins = [
    "http://localhost:3000",        # For local development of a React/Vue frontend
    "http://127.0.0.1:3000",       # Vite/localhost variant
    "http://localhost:5173",       # Vite default alternative
    "http://127.0.0.1:5173",       # Vite variant
    # "http://192.168.1.100:8080",    # Example of a local IP for testing
    "https://showmefire.org",
    "https://preview.showmefire.org",# Your production frontend domain
    # You can also add specific port numbers if your frontend is served from one
]
# Extra browser origins that may call the API with credentials, comma separated, for example the
# address the ShowMeFire Field web app is hosted at: CORS_EXTRA_ORIGINS=https://field.showmefire.org
origins += [o.strip() for o in os.getenv("CORS_EXTRA_ORIGINS", "").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    # Nuxt may select another port when the default dev port is occupied.
    # Keep local development origins limited to loopback hosts while allowing
    # any local port, instead of making developers change the API for each run.
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Added after CORSMiddleware so it runs outside it: public map tiles always answer
# "Access-Control-Allow-Origin: *" so a CDN cache can never hand out a copy that some site rejects.
app.add_middleware(PublicTileCorsMiddleware)

class LoginRequest(BaseModel):
    email:str
    password:str

from core.admin_session import admin_cookie_context, set_auth_cookie as _set_auth_cookie

def _clear_auth_cookie(response: Response, name: str) -> None:
    response.delete_cookie(
        key=name,
        httponly=True,
        secure=AUTH_COOKIE_SECURE,
        samesite=AUTH_COOKIE_SAMESITE,
        domain=AUTH_COOKIE_DOMAIN,
        path="/",
    )

app.middleware("http")(admin_cookie_context)

from core.graphics_session import graphics_cookie_context

app.middleware("http")(graphics_cookie_context)

class DevProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    timeline: Optional[str] = None
    status: str = "planned"
    sort_order: Optional[int] = None


class DevProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    timeline: Optional[str] = None
    status: Optional[str] = None
    sort_order: Optional[int] = None


class OpsBriefConfigUpdate(BaseModel):
    title: Optional[str] = None
    file_path: Optional[str] = None
    is_active: Optional[bool] = None
    expires_at: Optional[str] = None

@app.get('/')
def hello():
    return {'message': 'Show Me Fire Weather API', 'status': 'running'}

@app.get('/stations')
def get_stations():
    """Get all stations with weather data and metadata combined"""
    return get_station_data()

@app.get('/stations/refresh')
async def refresh_stations():
    """Manually trigger a data refresh"""
    await fetch_synoptic_data()
    data = get_station_data()
    return {"message": "Station data refreshed", "last_updated": data["last_updated"]}

@app.get('/stations/timeseries')
async def timeseries():
    return get_timeseries_data()

@app.get('/stations/timeseries/refresh')
async def timeseries_refresh():
    """Manually trigger a timeseries data refresh"""
    try:
        await fetchtimeseriesdata()
        data = get_timeseries_data()
        
        if data.get("error"):
            return {"error": data["error"], "status": "failed"}
        
        station_count = len(data.get("stations", []))
        return {
            "message": "Timeseries data refreshed",
            "status": "success",
            "last_updated": data.get("last_updated"),
            "station_count": station_count
        }
    except Exception as e:
        return {"error": str(e), "status": "failed"}

@app.get('/stations/timeseries/{stid}')
async def timeseries_by_stid(stid: str):
    """Get timeseries data for a specific station by STID"""
    data = get_timeseries_data()
    
    if data.get("error"):
        return {"error": data["error"]}
    
    stations = data.get("stations", [])
    for station in stations:
        if station.get("stid") == stid.upper():
            return {
                "stid": stid.upper(),
                "observations": station.get("observations", {}),
                "last_updated": data.get("last_updated")
            }
    
    return {"error": f"Station {stid} not found"}

@app.get('/status')
def status():
    """Get API status and data freshness"""
    synoptic_data = get_station_data()
    timeseries_data = get_timeseries_data()
    
    return {
        "status": "running",
        "synoptic": {
            "last_updated": synoptic_data.get("last_updated"),
            "station_count": len(synoptic_data.get("stations", [])) if synoptic_data.get("stations") else 0,
            "error": synoptic_data.get("error")
        },
        "timeseries": {
            "last_updated": timeseries_data.get("last_updated"),
            "station_count": len(timeseries_data.get("stations", [])) if timeseries_data.get("stations") else 0,
            "error": timeseries_data.get("error")
        }
    }

@app.get('/dashboard', response_class=HTMLResponse)
def dashboard():
    """Simple HTML dashboard showing API status with WebSocket updates"""
    return HTMLResponse(content=(PUBLIC_DIR / "dashboard.html").read_text())

@app.get("/list-images")
def list_images():
    files = []
    images_dir = IMAGES_DIR
    for fname in os.listdir(images_dir):
        fpath = os.path.join(images_dir, fname)
        if os.path.isfile(fpath):
            files.append(fname)
    return JSONResponse(content={"files": files})

@app.get("/list-gis")
def list_gis():
    files = []
    gis_dir = GIS_DIR
    for fname in os.listdir(gis_dir):
        fpath = os.path.join(gis_dir, fname)
        if os.path.isfile(fpath):
            files.append(fname)
    return JSONResponse(content={"files": files})

@app.post("/api/admin/login")
async def login(data: LoginRequest, response: Response):
    logger.info(f"Login attempt from: {data.email}")
    
    # Verify email and password
    if data.email != ADMIN_EMAIL:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if not ADMIN_PASSWORD_HASH or not verify_password(data.password, ADMIN_PASSWORD_HASH):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Create JWT token
    access_token = create_access_token(
        data={"sub": data.email},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    refresh_token = create_refresh_token(data={"sub": data.email})
    _set_auth_cookie(response, ACCESS_COOKIE_NAME, access_token, ACCESS_TOKEN_EXPIRE_MINUTES * 60)
    _set_auth_cookie(response, REFRESH_COOKIE_NAME, refresh_token, REFRESH_TOKEN_EXPIRE_DAYS * 86400)

    return {
        "success": True,
        "message": "Login successful"
    }

class TokenVerify(BaseModel):
    token: str

@app.post("/api/admin/verify")
async def verify_admin_token(data: Optional[TokenVerify] = None):
    """Session probe used by the public site. Unauthenticated callers get 200."""
    email = verify_token(data.token if data else None)
    if email:
        return {"valid": True, "email": email}
    return {"valid": False}

@app.post("/api/admin/refresh")
async def refresh_admin_token(request: Request, response: Response):
    """Rotate the access cookie using a valid refresh cookie."""
    email = verify_token(request.cookies.get(REFRESH_COOKIE_NAME), expected_type="refresh")
    if not email:
        raise HTTPException(status_code=401, detail="Invalid or expired refresh token")
    access_token = create_access_token(
        data={"sub": email},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    _set_auth_cookie(response, ACCESS_COOKIE_NAME, access_token, ACCESS_TOKEN_EXPIRE_MINUTES * 60)
    return {"success": True, "message": "Token refreshed"}

@app.post("/api/admin/logout")
async def logout_admin(response: Response):
    _clear_auth_cookie(response, ACCESS_COOKIE_NAME)
    _clear_auth_cookie(response, REFRESH_COOKIE_NAME)
    return {"success": True, "message": "Logged out"}

@app.get("/api/banner")
async def get_banner_public():
    """Get current banner (public endpoint)"""
    # Always load from file for latest value
    return load_banner_config()

@app.get("/api/admin/banner")
async def get_banner_admin(email: str = Depends(require_admin)):
    """Get banner data (admin only)"""
    return load_banner_config()

@app.post("/api/admin/banner")
async def update_banner(banner: BannerData, email: str = Depends(require_admin)):
    """Update banner (admin only)"""
    save_banner_config(banner)
    logger.info(f"Banner updated by {email}: enabled={banner.enabled}, type={banner.type}")
    return {"success": True, "message": "Banner updated successfully"}


@app.get("/api/admin/ignored_stations")
def admin_list_ignored(email: str = Depends(require_admin)):
    """List ignored stations (admin only)"""

    db_path = get_db_path()
    try:
        conn = sqlite3.connect(db_path, timeout=15)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT stid, reason, added_at FROM ignored_stations ORDER BY added_at DESC')
        rows = cursor.fetchall()
        conn.close()
        return {"success": True, "ignored": [dict(r) for r in rows]}
    except Exception as e:
        logger.error(f"Error listing ignored stations: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/ignored_stations")
def admin_add_ignored(payload: dict, email: str = Depends(require_admin)):
    """Add an ignored station. Payload: { stid: 'MBGM7', reason: '...' }"""

    stid = (payload.get('stid') or '').strip().upper()
    reason = payload.get('reason') or None
    if not stid:
        raise HTTPException(status_code=400, detail="stid required")

    db_path = get_db_path()
    try:
        conn = sqlite3.connect(db_path, timeout=15)
        cursor = conn.cursor()
        cursor.execute('INSERT OR IGNORE INTO ignored_stations (stid, reason) VALUES (?, ?)', (stid, reason))
        conn.commit()
        conn.close()
        # Invalidate cache if module loaded
        try:
            from core.ignored_stations import refresh_ignored_stations
            refresh_ignored_stations()
        except Exception:
            pass
        return {"success": True, "stid": stid}
    except Exception as e:
        logger.error(f"Error adding ignored station {stid}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/admin/ignored_stations/{stid}")
def admin_remove_ignored(stid: str, email: str = Depends(require_admin)):
    """Remove an ignored station by STID"""

    stid = stid.strip().upper()
    db_path = get_db_path()
    try:
        conn = sqlite3.connect(db_path, timeout=15)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM ignored_stations WHERE stid = ?', (stid,))
        conn.commit()
        conn.close()
        try:
            from core.ignored_stations import refresh_ignored_stations
            refresh_ignored_stations()
        except Exception:
            pass
        return {"success": True, "stid": stid}
    except Exception as e:
        logger.error(f"Error removing ignored station {stid}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/website/version')
def api_get_website_version():
    """Public endpoint returning website version"""
    try:
        from core.database import get_website_version
        info = get_website_version()
        return {"version": info.get('version')}
    except Exception as e:
        logger.error(f"Error fetching website version: {e}")
        return PlainTextResponse('1', status_code=200)


@app.post('/api/admin/website/version')
def admin_set_website_version(payload: dict, email: str = Depends(require_admin)):
    """Admin-only: set website version. Payload: { version: '1.2.3' }"""

    version = (payload.get('version') or '').strip()
    if not version:
        raise HTTPException(status_code=400, detail="version required")

    try:
        from core.database import set_website_version
        ok = set_website_version(version)
        if ok:
            return {"success": True, "version": version}
        raise HTTPException(status_code=500, detail="Failed to update version")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting website version: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/api/projects')
def get_public_projects():
    """Public endpoint: list development projects for the website."""
    try:
        return {"projects": list_dev_projects()}
    except Exception as e:
        logger.error(f"Error fetching projects: {e}")
        raise HTTPException(status_code=500, detail="Failed to load projects")


@app.get('/api/admin/projects')
def admin_list_projects(email: str = Depends(require_admin)):
    return {"success": True, "projects": list_dev_projects()}


@app.post('/api/admin/projects')
def admin_create_project(payload: DevProjectCreate, email: str = Depends(require_admin)):

    name = (payload.name or '').strip()
    if not name:
        raise HTTPException(status_code=400, detail="name is required")

    project = create_dev_project(
        name=name,
        description=payload.description,
        timeline=payload.timeline,
        status=payload.status,
        sort_order=payload.sort_order
    )
    return {"success": True, "project": project}


@app.put('/api/admin/projects/{project_id}')
def admin_update_project(project_id: int, payload: DevProjectUpdate, email: str = Depends(require_admin)):

    ok = update_dev_project(
        project_id=project_id,
        name=payload.name,
        description=payload.description,
        timeline=payload.timeline,
        status=payload.status,
        sort_order=payload.sort_order
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Project not found")

    projects = list_dev_projects()
    updated = next((p for p in projects if p['id'] == project_id), None)
    return {"success": True, "project": updated}


@app.delete('/api/admin/projects/{project_id}')
def admin_delete_project(project_id: int, email: str = Depends(require_admin)):

    if not delete_dev_project(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    return {"success": True, "deleted": project_id}


@app.get("/api/admin/opsbrief/files")
def admin_list_opsbrief_files(email: str = Depends(require_admin)):

    files = []
    for f in sorted(OPSBRIEF_DIR.glob("*")):
        if f.is_file():
            stat = f.stat()
            files.append({
                "name": f.name,
                "size_bytes": stat.st_size,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
    return {"success": True, "files": files}


@app.post("/api/admin/opsbrief/upload")
async def admin_upload_opsbrief(email: str = Depends(require_admin), upload: UploadFile = File(...)):

    filename = Path(upload.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="filename required")
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    target = OPSBRIEF_DIR / filename
    if filename.lower() == OPSBRIEF_FALLBACK_FILE and target.exists():
        raise HTTPException(
            status_code=400,
            detail=f"{OPSBRIEF_FALLBACK_FILE} already exists and cannot be overwritten"
        )

    content = await upload.read()
    with open(target, "wb") as f:
        f.write(content)

    return {"success": True, "filename": filename}


@app.delete("/api/admin/opsbrief/files/{filename}")
def admin_delete_opsbrief_file(filename: str, email: str = Depends(require_admin)):

    safe_name = Path(filename).name
    if safe_name.lower() == OPSBRIEF_FALLBACK_FILE:
        raise HTTPException(status_code=400, detail=f"{OPSBRIEF_FALLBACK_FILE} cannot be deleted")

    target = OPSBRIEF_DIR / safe_name
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    target.unlink()
    return {"success": True, "deleted": safe_name}


@app.get("/api/admin/opsbrief/config")
def admin_get_opsbrief_config(email: str = Depends(require_admin)):

    config = get_briefing_config()
    return {"success": True, "config": config}


@app.post("/api/admin/opsbrief/config")
def admin_set_opsbrief_config(payload: OpsBriefConfigUpdate, email: str = Depends(require_admin)):

    if payload.file_path is not None:
        safe_name = Path(payload.file_path).name
        if safe_name:
            file_path = OPSBRIEF_DIR / safe_name
            if not file_path.exists():
                raise HTTPException(status_code=400, detail="Selected file does not exist")
        else:
            safe_name = ""
    else:
        safe_name = None

    # Upsert singleton config row (always overwrite provided values cleanly)
    current = get_briefing_config() or {}
    updated = create_briefing(
        file_path=(safe_name if safe_name is not None else (current.get("file_path") or "")),
        title=(payload.title if payload.title is not None else current.get("title")),
        is_active=(payload.is_active if payload.is_active is not None else bool(current.get("is_active"))),
        expires_at=payload.expires_at
    )

    new_config = updated or get_briefing_config() or {}
    became_active = bool(new_config.get("is_active")) and (
        not bool(current.get("is_active"))
        or str(new_config.get("file_path") or "") != str(current.get("file_path") or "")
        or str(new_config.get("title") or "") != str(current.get("title") or "")
    )
    if became_active and new_config.get("file_path"):
        try:
            from services.mobile_push import notify_sitrep
            identity = f"{new_config.get('file_path')}:{new_config.get('updated_at') or new_config.get('title') or ''}"
            notify_sitrep(title=str(new_config.get("title") or "New situation report available"), identity=identity)
        except Exception as exc:
            logger.warning("Failed to emit mobile SitRep event: %s", exc)

    return {"success": True, "config": new_config}


@app.get("/api/opsbrief")
def public_opsbrief_info():
    config = get_briefing_config()
    if not config:
        return {"active": False, "url": None}

    is_active = bool(config.get("is_active"))
    file_name = config.get("file_path") or ""
    expires_at = config.get("expires_at")
    if not is_active or not file_name:
        return {"active": False, "url": None}

    if expires_at:
        try:
            expires_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            now_utc = datetime.now(expires_dt.tzinfo) if expires_dt.tzinfo else datetime.utcnow()
            if expires_dt <= now_utc:
                return {"active": False, "url": None}
        except Exception:
            pass

    file_path = OPSBRIEF_DIR / Path(file_name).name
    if not file_path.exists():
        return {"active": False, "url": None}

    return {
        "active": True,
        "title": config.get("title"),
        "url": "/api/opsbrief/view"
    }


@app.get("/api/opsbrief/view")
def public_opsbrief_view():
    fallback_file = OPSBRIEF_DIR / OPSBRIEF_FALLBACK_FILE

    def _serve_fallback():
        if fallback_file.exists():
            return FileResponse(
                fallback_file,
                media_type="application/pdf",
                headers={"Content-Disposition": f'inline; filename="{OPSBRIEF_FALLBACK_FILE}"'}
            )
        raise HTTPException(status_code=404, detail="Fallback briefing file not found")

    config = get_briefing_config()
    if not config or not bool(config.get("is_active")):
        return _serve_fallback()

    file_name = config.get("file_path") or ""
    if not file_name:
        return _serve_fallback()

    expires_at = config.get("expires_at")
    if expires_at:
        try:
            expires_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
            now_utc = datetime.now(expires_dt.tzinfo) if expires_dt.tzinfo else datetime.utcnow()
            if expires_dt <= now_utc:
                return _serve_fallback()
        except ValueError:
            pass

    file_path = OPSBRIEF_DIR / Path(file_name).name
    if not file_path.exists():
        return _serve_fallback()

    return FileResponse(
        file_path,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{Path(file_name).name}"'}
    )

@app.get('/stations/raws')
def get_raws_stations():
    """
    Get all RAWS stations in MO, OK, AR, TN, KY, IL, IA, NE, KS.
    """
    return raws_station_data


@app.get('/api/stations/fire-danger-counts')
def get_station_fire_danger_counts():
    """Return station danger counts from latest realtime artifacts."""
    status_path = Path(__file__).resolve().parent / 'status.json'

    # Preferred source: map update output written every refresh.
    if status_path.exists():
        try:
            with open(status_path, 'r') as f:
                status_data = json.load(f)

            realtime_fire = status_data.get('RealtimeFireDanger') or {}
            if realtime_fire.get('station_fire_danger_counts'):
                return {
                    'source': 'status_file_realtime',
                    'last_update': realtime_fire.get('last_update'),
                    'counts': realtime_fire.get('station_fire_danger_counts'),
                    'total_mo_stations': realtime_fire.get('total_mo_stations'),
                    'classified_mo_stations': realtime_fire.get('classified_mo_stations'),
                    'unclassified_mo_stations': realtime_fire.get('unclassified_mo_stations'),
                }

            realtime_fm = status_data.get('RealTimeFuelMoisture') or {}
            if realtime_fm.get('station_fire_danger_counts'):
                return {
                    'source': 'status_file_fuelmoisture',
                    'last_update': realtime_fm.get('last_update'),
                    'counts': realtime_fm.get('station_fire_danger_counts'),
                    'total_mo_stations': realtime_fm.get('total_mo_stations'),
                    'classified_mo_stations': realtime_fm.get('classified_mo_stations'),
                    'unclassified_mo_stations': realtime_fm.get('unclassified_mo_stations'),
                }
        except Exception as e:
            logger.warning(f"Unable to read station counts from status file: {e}")

    # Fallback source: most recent persisted history snapshot.
    snapshots = get_recent_fire_danger_history(last_hours=168)
    if not snapshots:
        raise HTTPException(status_code=503, detail='No realtime fire danger count artifacts available')

    latest_snapshot = snapshots[-1]
    return {
        'source': 'history_snapshot',
        'last_update': latest_snapshot.get('timestamp_local') or latest_snapshot.get('timestamp_utc'),
        'counts': normalize_fire_danger_counts(latest_snapshot.get('counts', {})),
        'total_mo_stations': latest_snapshot.get('total_mo_stations'),
        'classified_mo_stations': latest_snapshot.get('classified_mo_stations'),
        'unclassified_mo_stations': latest_snapshot.get('unclassified_mo_stations'),
    }


@app.get('/api/fire-danger/peak-today')
def get_fire_danger_peak_today():
    """Return the highest fire-danger class reached anywhere in MO today.

    Cheap: reads the small observed-peak state.json (kept up to date by
    realtimefiredanger.py's update_observed_peak_grid) rather than the
    raster itself. Never raises for missing/stale data -- the frontend
    badge should simply hide itself when `available` is false.
    """
    today_local = datetime.now(timezone('America/Chicago')).strftime('%Y-%m-%d')

    last_update = None
    status_path = Path(__file__).resolve().parent / 'status.json'
    if status_path.exists():
        try:
            with open(status_path, 'r') as f:
                last_update = (json.load(f).get('RealtimeFireDanger') or {}).get('last_update')
        except (json.JSONDecodeError, OSError):
            last_update = None

    if not OBSERVED_PEAK_STATE_FILE.exists():
        return {'available': False, 'reason': 'no observed peak data yet', 'last_update': last_update}

    try:
        with open(OBSERVED_PEAK_STATE_FILE, 'r', encoding='utf-8') as f:
            state = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {'available': False, 'reason': 'observed peak state unreadable', 'last_update': last_update}

    if state.get('date_local') != today_local:
        return {'available': False, 'reason': 'observed peak data is not from today', 'last_update': last_update}

    peak_class = state.get('peak_class')
    if peak_class is None:
        return {'available': False, 'reason': 'no classified pixels yet today', 'last_update': last_update}

    return {
        'available': True,
        'date_local': state.get('date_local'),
        'peak_class': peak_class,
        'peak_label': CLASS_VALUE_TO_LABEL.get(int(peak_class)),
        'peak_reached_at_utc': state.get('peak_reached_at_utc'),
        'last_update': last_update,
    }


@app.get('/api/stations/fire-danger-counts/history')
def get_station_fire_danger_counts_history(last_hours: int = 48):
    """Return station fire danger count snapshots for the trailing time window."""
    hours = max(1, min(last_hours, 168))
    snapshots = get_recent_fire_danger_history(last_hours=hours)
    return {
        'source': 'history_file',
        'last_hours': hours,
        'snapshot_count': len(snapshots),
        'snapshots': snapshots,
    }

@app.get("/api/historical/fetch")
async def fetch_historical_data(
    days_back: int = 1,
    states: str = "MO,OK,AR,TN,KY,IL,IA,NE,KS",
    networks: str = "1,2,156",
    save_to_archive: bool = False
):
    '''
    Fetch historical weather station data.
    
    Query params:
    - days_back: Number of days to fetch (default: 1)
    - states: Comma-separated state codes (default: MO and surrounding)
    - networks: Comma-separated network IDs (default: 1,2,156)
    - save_to_archive: Whether to save raw data to archive (default: False)
    
    Returns: Raw Synoptic API response with metadata
    '''
    states_list = states.split(",")
    networks_list = [int(n) for n in networks.split(",")]
    
    try:
        data = await fetch_historical_station_data(
            states=states_list,
            days_back=days_back,
            networks=networks_list
        )
        
        if save_to_archive:
            filepath = await save_raw_data_to_archive(
                days_back=days_back,
                archive_dir=str(ARCHIVE_RAW_DATA_DIR)
            )
            data['_saved_to'] = str(filepath) if filepath else None
        
        return {
            "success": True,
            "data": data,
            "stats": get_raw_data_stats(data)
        }
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        return {
            "success": False,
            "error": str(e)
        }
        
@app.get("/api/historical/stats")
async def get_historical_stats(days_back: int = 7):
    '''
    Get statistics about available historical data.
    
    Query params:
    - days_back: Number of days to analyze (default: 7)
    
    Returns: Summary statistics
    '''
    try:
        data = await fetch_historical_station_data(days_back=days_back)
        stats = get_raw_data_stats(data)
        return {"success": True, "stats": stats}
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/historical/archive/save")
async def save_historical_to_archive(days_back: int = 1):
    '''
    Save historical data to archive folder.
    
    Body params:
    - days_back: Number of days to fetch and save (default: 1)
    
    Returns: Path to saved file
    '''
    try:
        filepath = await save_raw_data_to_archive(days_back=days_back)
        if filepath:
            return {
                "success": True,
                "filepath": str(filepath),
                "message": f"Saved {days_back} days of data"
            }
        else:
            return {
                "success": False,
                "error": "No data to save"
            }
    except Exception as e:
        logger.error(f"Error saving to archive: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/historical/archive/list")
async def list_archived_files(_admin: str = Depends(require_admin)):
    '''
    List all archived data files in the fixed raw-data archive directory.

    Returns: List of available archive files with metadata
    '''
    archive_path = ARCHIVE_RAW_DATA_DIR

    files = []
    from services.archive_store import ArchiveStore
    store = ArchiveStore()
    for m in await asyncio.to_thread(store.list_observations):
        files.append({"filename": m['name'], "size_mb": round(m['size'] / 1024 / 1024, 2),
                      "created": m['captured_at'],
                      "modified": datetime.fromtimestamp(m['mtime_ns'] / 1e9).isoformat(),
                      "storage": "r2"})
    for filepath in archive_path.glob("raw_data_*.json"):
        stat = filepath.stat()
        files.append({
            "filename": filepath.name,
            "size_mb": round(stat.st_size / 1024 / 1024, 2),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
        })

    files = list({item['filename']: item for item in files}.values())
    files.sort(key=lambda x: x['modified'], reverse=True)

    return {
        "success": True,
        "files": files,
        "count": len(files)
    }


@app.get("/api/historical/archive/load/{filename}")
async def load_archived_file(filename: str, _admin: str = Depends(require_admin)):
    '''
    Load a specific archived data file from the fixed raw-data archive directory.

    Path params:
    - filename: Name of the file to load

    Returns: Archived data with statistics
    '''
    base_dir = ARCHIVE_RAW_DATA_DIR.resolve()
    archive_path = (base_dir / filename).resolve()
    try:
        archive_path.relative_to(base_dir)
    except ValueError:
        raise HTTPException(status_code=403, detail="Forbidden: Path traversal not allowed")

    try:
        if archive_path.exists():
            with open(archive_path, 'r') as f:
                data = json.load(f)
        else:
            from services.archive_store import ArchiveStore
            data = await asyncio.to_thread(ArchiveStore().read_observation, filename)
            if data is None:
                return {"success": False, "error": "File not found"}

        return {
            "success": True,
            "data": data,
            "stats": get_raw_data_stats(data)
        }
    except Exception as e:
        logger.error(f"Error loading archive: {e}")
        return {"success": False, "error": str(e)}
    
@app.get("/api/fires/missouri")
async def get_missouri_fires():
    """Get current Missouri fire detections as JSON"""
    json_file = MISSOURI_FIRES_JSON
    if os.path.exists(json_file):
        return FileResponse(json_file, media_type='application/json')
    else:
        raise HTTPException(status_code=404, detail="Fire data not yet available")
    
@app.get("/fires/satdet")
async def get_satdet_fires():
    """Get current Missouri satellite fire detections as GeoJSON"""
    geojson_file = Path(GIS_DIR) / 'satfiredetection.geojson'
    if geojson_file.exists():
        return FileResponse(geojson_file, media_type='application/geo+json')
    else:
        raise HTTPException(status_code=404, detail="Fire data not yet available")

@app.get("/api/fires/missouri/geojson")
async def get_missouri_fires_geojson():
    """Get current Missouri fire detections as GeoJSON"""
    geojson_file = MISSOURI_FIRES_GEOJSON
    if os.path.exists(geojson_file):
        return FileResponse(geojson_file, media_type='application/geo+json')
    else:
        raise HTTPException(status_code=404, detail="Fire GeoJSON not yet available")

@app.get("/fires/detections/advanced")
async def get_advanced_fire_detections():
    """Get advanced NGFS fire detections for CONUS-EAST as GeoJSON"""
    from pathlib import Path
    gis_dir = Path(GIS_DIR)
    geojson_file = gis_dir / 'ngfs_advanced_fire_detections_missouri.geojson'
    if geojson_file.exists():
        return FileResponse(geojson_file, media_type='application/geo+json')
    else:
        raise HTTPException(status_code=404, detail="Advanced fire detection data not yet available")

@app.get("/api/images/timeline")
async def get_image_timeline():
    """Get image timeline data for the past 48 hours"""
    timeline_file = PUBLIC_DIR / "image-timeline.json"
    if timeline_file.exists():
        return FileResponse(timeline_file, media_type="application/json")
    else:
        raise HTTPException(status_code=404, detail="Timeline file not found")    
@app.get("/forecast/latest")
def api_latest_forecast(cycle: int = 12):
    forecast = get_latest_forecast(cycle=cycle)
    if not forecast:
        raise HTTPException(status_code=404, detail="No forecast found")
    return forecast

@app.get("/api/admin/reports/list")
async def list_reports(email: str = Depends(require_admin)):
    """List all reports (files and folders) in the reports directory (Admin/Logged-in only)"""
    return list_files(str(REPORTS_DIR))

@app.get("/api/admin/reports/list/{path:path}")
async def list_reports_in_path(path: str, email: str = Depends(require_admin)):
    """List contents of a specific folder within reports directory"""
    return list_files(str(REPORTS_DIR), path)

@app.get("/api/admin/reports/view/{filepath:path}")
async def view_report(filepath: str, email: str = Depends(require_admin)):
    """Serve a specific report file securely from reports directory"""
    return view_file(str(REPORTS_DIR), filepath, email)

@app.get("/api/admin/logs/list")
async def list_logs(email: str = Depends(require_admin)):
    """List all logs (files and folders) in the logs directory (Admin/Logged-in only)"""
    return list_files(str(LOGS_DIR))

@app.get("/api/admin/logs/list/{path:path}")
async def list_logs_in_path(path: str, email: str = Depends(require_admin)):
    """List contents of a specific folder within logs directory"""
    return list_files(str(LOGS_DIR), path)

@app.get("/api/admin/logs/view/{filepath:path}")
async def view_log(filepath: str, email: str = Depends(require_admin)):
    """Serve a specific log file securely from logs directory"""
    return view_file(str(LOGS_DIR), filepath, email)

@app.post("/api/admin/logs/cleanup")
async def cleanup_logs(email: str = Depends(require_admin)):
    """Run the log purge (delete stale dated logs, truncate oversized ever-growing ones) immediately."""
    result = await asyncio.to_thread(purge_old_logs)
    return {"success": True, **result}

@app.get("/api/training/latest-stats")
async def get_latest_training_stats():
    """Returns the JSON stats from the most recent training folder"""
    reports_dir = REPORTS_DIR
    # Get the most recent date folder
    date_folders = sorted([d for d in reports_dir.iterdir() if d.is_dir()], reverse=True)
    
    if not date_folders:
        raise HTTPException(status_code=404, detail="No training data found")
        
    stats_path = date_folders[0] / "stats.json"
    if stats_path.exists():
        with open(stats_path, "r") as f:
            return json.load(f)
    
    raise HTTPException(status_code=404, detail="Stats file missing")

@app.get("/api/fuel-moisture/morning")
async def get_morning_fuel_moisture(
    date: Optional[str] = None,
    hour_utc: Optional[int] = None,
):
    """
    Fetch fuel moisture observations near 7 AM Central Time by default.
    The secondary forecast may explicitly request a UTC date and hour.
    Uses default states (MO and surrounding) and network 2 (RAWS).
    
    Returns: Fuel moisture observations from stations
    """
    from services.synoptic import fetch_fuel_moisture_at_time, resolve_fuel_moisture_target_time
    
    try:
        try:
            target_time = resolve_fuel_moisture_target_time(date, hour_utc)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

        data = await fetch_fuel_moisture_at_time(
            target_time=target_time,
            states=None,       # Defaults to MO and surrounding states
            networks=None      # Defaults to network 2
        )
        
        return {
            "success": True,
            "data": data
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching morning fuel moisture: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching fuel moisture data: {str(e)}"
        )



@app.post("/api/fuel-sensor/readings")
async def ingest_fuel_sensor_readings(payload: dict, x_device_key: Optional[str] = Header(None)):
    """Ingest fuel-moisture readings from our own dowel sensor hardware
    (SMF_FuelMoistureSensor), not the RAWS network. Devices authenticate
    with a shared secret in the X-Device-Key header, checked against
    FUEL_SENSOR_API_KEY, since these are unauthenticated ESP32s posting
    over the open internet.

    Payload shape:
    {
      "site_id": "...", "device_id": "...",
      "diagnostics": {"battery_v": null, "rssi_dbm": -55, "uptime_s": 1234, "firmware_version": "0.1.0"},
      "readings": [{"recorded_at": "2026-09-08T14:00:00Z", "air_temp_c": 22.3,
                     "relative_humidity_pct": 41.2, "fuel_moisture_pct": 9.8}]
    }
    """
    configured_key = os.getenv("FUEL_SENSOR_API_KEY", "").strip()
    if not configured_key:
        raise HTTPException(status_code=503, detail="Fuel sensor ingest is not configured (FUEL_SENSOR_API_KEY unset)")
    if not x_device_key or x_device_key != configured_key:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Device-Key")

    site_id = (payload.get("site_id") or "").strip()
    device_id = (payload.get("device_id") or "").strip()
    if not site_id or not device_id:
        raise HTTPException(status_code=400, detail="site_id and device_id are required")

    readings = payload.get("readings")
    if not isinstance(readings, list) or not readings:
        raise HTTPException(status_code=400, detail="readings must be a non-empty array")

    diagnostics = payload.get("diagnostics") or {}
    battery_v = diagnostics.get("battery_v")
    rssi_dbm = diagnostics.get("rssi_dbm")
    uptime_s = diagnostics.get("uptime_s")
    firmware_version = diagnostics.get("firmware_version")
    enclosure_state = diagnostics.get("enclosure_state")

    try:
        stored = await asyncio.to_thread(
            _ingest_fuel_sensor_readings_sync,
            site_id, device_id, readings,
            battery_v, rssi_dbm, uptime_s, firmware_version, enclosure_state,
        )
        return {"success": True, "stored": stored, "received": len(readings)}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ingesting fuel sensor readings from {device_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _ingest_fuel_sensor_readings_sync(
    site_id: str, device_id: str, readings: list,
    battery_v, rssi_dbm, uptime_s, firmware_version, enclosure_state,
) -> int:
    db_path = get_db_path()
    stored = 0
    conn = sqlite3.connect(db_path, timeout=15)
    try:
        cursor = conn.cursor()
        for reading in readings:
            recorded_at = reading.get("recorded_at")
            if not recorded_at:
                raise HTTPException(status_code=400, detail="each reading requires recorded_at")
            cursor.execute('''
                INSERT OR IGNORE INTO fuel_moisture_sensor_readings (
                    site_id, device_id, recorded_at, air_temp_c, relative_humidity_pct,
                    fuel_moisture_pct, battery_v, rssi_dbm, uptime_s, firmware_version, enclosure_state
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                site_id, device_id, recorded_at,
                reading.get("air_temp_c"), reading.get("relative_humidity_pct"), reading.get("fuel_moisture_pct"),
                battery_v, rssi_dbm, uptime_s, firmware_version, enclosure_state,
            ))
            stored += cursor.rowcount
        conn.commit()
        return stored
    finally:
        conn.close()


@app.get("/api/fuel-sensor/readings")
def list_fuel_sensor_readings(site_id: Optional[str] = None, device_id: Optional[str] = None, limit: int = 100):
    """Recent readings from our own dowel sensor hardware, newest upload first."""
    limit = max(1, min(limit, 1000))
    db_path = get_db_path()
    try:
        conn = sqlite3.connect(db_path, timeout=15)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        query = "SELECT * FROM fuel_moisture_sensor_readings WHERE 1=1"
        params: list = []
        if site_id:
            query += " AND site_id = ?"
            params.append(site_id)
        if device_id:
            query += " AND device_id = ?"
            params.append(device_id)
        # ``recorded_at`` is device time and bench firmware currently reports
        # uptime there. ``received_at`` is the reliable upload ordering.
        query += " ORDER BY received_at DESC, id DESC LIMIT ?"
        params.append(limit)
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        return {"success": True, "readings": [dict(r) for r in rows]}
    except Exception as e:
        logger.error(f"Error listing fuel sensor readings: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models/history")
async def get_model_history():
    """
    Get the history of all trained models including archived versions.
    
    Returns: Dictionary containing current models and their history
    """
    try:
        config_path = Path("models/config.json")
        archive_dir = Path("models/archive")
        
        if not config_path.exists():
            return {"success": False, "error": "No model configuration found"}
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Get archived files
        archived_models = []
        if archive_dir.exists():
            for file_path in archive_dir.glob("*"):
                if file_path.is_file():
                    stat = file_path.stat()
                    archived_models.append({
                        "filename": file_path.name,
                        "path": str(file_path),
                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                    })
        
        # Sort archived models by modification time (newest first)
        archived_models.sort(key=lambda x: x['modified'], reverse=True)
        
        return {
            "success": True,
            "current_models": config,
            "archived_models": archived_models,
            "archive_count": len(archived_models),
            "total_history_entries": sum(len(model.get('history', [])) for model in config.values() if isinstance(model, dict))
        }
    except Exception as e:
        logger.error(f"Error retrieving model history: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving model history: {str(e)}")

@app.get("/api/models/formulas")
async def get_model_formulas():
    """
    Get the formulas and criteria used by the fuel moisture and fire danger models.
    
    Returns: Dictionary containing formulas for both models
    """
    try:
        fuel_moisture_formula = {
            "model_type": "XGBoost Regression",
            "description": "Machine learning model that predicts 10-hour fuel moisture percentage",
            "features": [
                "temp_c - Temperature in Celsius",
                "rel_humidity - Relative humidity percentage", 
                "wind_speed_ms - Wind speed in meters per second",
                "hour - Hour of day (0-23)",
                "month - Month of year (1-12)",
                "emc_baseline - Compatibility moisture baseline (RH / 5)",
                "temp_mean_3h - 3-hour rolling mean temperature",
                "rh_mean_3h - 3-hour rolling mean relative humidity",
                "temp_mean_6h - 6-hour rolling mean temperature", 
                "rh_mean_6h - 6-hour rolling mean relative humidity",
                "precip_1h - 1-hour precipitation accumulation (mm)",
                "precip_3h - 3-hour precipitation accumulation (mm)",
                "precip_6h - 6-hour precipitation accumulation (mm)",
                "precip_24h - 24-hour precipitation accumulation (mm)",
                "hours_since_rain - Hours since last significant rain (>0.1mm)",
                "hour_sin/hour_cos - Cyclical hour encoding (schema v2 candidates)",
                "day_of_year_sin/day_of_year_cos - Cyclical seasonal encoding (schema v2 candidates)"
            ],
            "output": "Predicted 10-hour fuel moisture percentage (1-40%)",
            "baseline_equation": "Compatibility baseline = RH / 5",
            "fallback_estimate": "FM ≈ 3 + 0.25 × RH (clamped to 3-30%)"
        }
        
        fire_danger_formula = {
            "model_type": "Criteria-based Classification",
            "description": "Fire danger classification based on Missouri fire weather criteria",
            "criteria_levels": [
                {
                    "level": "Low",
                    "score": 0,
                    "condition": "FM ≥ 15% (Fuels too wet to carry fire effectively)"
                },
                {
                    "level": "Moderate", 
                    "score": 1,
                    "condition": "FM < 15% AND (RH < 45% OR Wind ≥ 10 kts)"
                },
                {
                    "level": "Elevated",
                    "score": 2, 
                    "condition": "FM < 9% AND ((RH < 35% AND Wind ≥ 12 kts) OR (RH < 25% AND Wind ≥ 5 kts))"
                },
                {
                    "level": "Critical",
                    "score": 3,
                    "condition": "FM < 9% AND RH < 25% AND Wind ≥ 15 kts"
                },
                {
                    "level": "Extreme",
                    "score": 4,
                    "condition": "FM < 7% AND RH < 20% AND Wind ≥ 25 kts"
                }
            ],
            "input_variables": {
                "FM": "10-hour fuel moisture percentage",
                "RH": "Relative humidity percentage", 
                "Wind": "Wind speed in knots (kts)"
            },
            "output": "Fire danger level (0-4) corresponding to Low-Extreme",
            "evaluation_order": "Extreme → Critical → Elevated → Moderate → Low (most restrictive first)"
        }
        
        try:
            fuel_moisture_formula["feature_importance"] = fm_explain.global_importance()
        except Exception as importance_error:
            # The static prose above (thresholds, category definitions) is
            # genuine policy, not model-derived - it must still be served
            # even if the live importance lookup fails for some reason.
            logger.error(f"Error computing live feature importance: {importance_error}")
            fuel_moisture_formula["feature_importance"] = None

        return {
            "success": True,
            "fuel_moisture_model": fuel_moisture_formula,
            "fire_danger_model": fire_danger_formula,
            "last_updated": "2026-08-14",
            "version": "1.1.0"
        }
    except Exception as e:
        logger.error(f"Error retrieving model formulas: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving model formulas: {str(e)}")

@app.get("/web/issues")
async def get_issues():
    """Get GitHub issues data as JSON"""
    issues_file = Path("data/issues.json")
    if issues_file.exists():
        return FileResponse(issues_file, media_type='application/json')
    else:
        raise HTTPException(status_code=404, detail="Issues data not available")


def get_verificationcsv ():
    try:
        df = pd.read_csv('reports/verification_history.csv')    
        csv_string = df.to_csv(index=False)
        return csv_string
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    
@app.get("/get-csv/")
async def get_csv():
    try:
        csv_data = get_verificationcsv()
        return csv_data
    except HTTPException as e:
        return {"error": str(e)}
    
def _rss_file_response():
    """Serve the feed from its canonical location, even before first generation."""
    rss_file = Path(PUBLIC_DIR) / "rss.xml"
    if not rss_file.exists():
        write_rss_feed()
    return FileResponse(rss_file, media_type="application/rss+xml")


@app.get("/rss.xml", include_in_schema=False)
@app.get("/public/rss.xml", include_in_schema=False)
async def public_rss_feed():
    return _rss_file_response()


app.mount("/", StaticFiles(directory=str(PUBLIC_DIR)), name="public")
    

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
