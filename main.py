from fastapi import FastAPI, HTTPException
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
from services.rss import generate_rss_feed
from pathlib import Path
from pytz import timezone
from core.database import (
    get_latest_forecast,
    get_forecast_by_time,
    get_recent_forecasts,
    get_forecast_count,
    get_db_path,
    list_dev_projects,
    create_dev_project,
    update_dev_project,
    delete_dev_project
)
import sqlite3
from core.security import (
    verify_password,
    create_access_token,
    verify_token,
    ADMIN_EMAIL,
    ADMIN_PASSWORD_HASH,
    ACCESS_TOKEN_EXPIRE_HOURS
)
from core.scheduler import (
    create_scheduler,
    start_scheduler_jobs,
    run_initial_fetches,
    raws_station_data
)
from core.config import (
    IMAGES_DIR,
    GIS_DIR,
    PUBLIC_DIR,
    REPORTS_DIR,
    LOGS_DIR,
    ARCHIVE_RAW_DATA_DIR,
    MISSOURI_FIRES_JSON,
    MISSOURI_FIRES_GEOJSON
)

IS_PRODUCTION = os.getenv("ENVIRONMENT", "development").lower() == "production"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    run_scheduler = os.getenv("run_sch", "false").lower() == "true"
    scheduler_local = None

    if run_scheduler:
        logger.info("Starting scheduler...")
        scheduler_local = create_scheduler()
        start_scheduler_jobs(scheduler_local)
    else:
        logger.info("Scheduler disabled via run_sch environment variable")
    
    # Start the fetch in the background so the API starts immediately
    asyncio.create_task(run_initial_fetches())
    
    yield
    
    if scheduler_local:
        logger.info("Shutting down scheduler...")
        scheduler_local.shutdown()

app = FastAPI(
    title="Show Me Fire Weather API",
    description="API for fetching synoptic weather data",
    lifespan=lifespan,
    docs_url=None if IS_PRODUCTION else "/docs",
    redoc_url=None if IS_PRODUCTION else "/redoc",
    openapi_url=None if IS_PRODUCTION else "/openapi.json"
)

class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        response: Response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

# Use this instead of the default StaticFiles
app.mount("/images", NoCacheStaticFiles(directory=str(IMAGES_DIR)), name="images")
app.mount("/gis", NoCacheStaticFiles(directory=str(GIS_DIR)), name="gis")
app.mount("/reports", NoCacheStaticFiles(directory=str(REPORTS_DIR)), name="reports")
app.mount("/public", StaticFiles(directory=str(PUBLIC_DIR)), name="public")

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LoginRequest(BaseModel):
    email:str
    password:str


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
async def login(data: LoginRequest):
    logger.info(f"Login attempt from: {data.email}")
    
    # Verify email and password
    if data.email != ADMIN_EMAIL:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if not verify_password(data.password, ADMIN_PASSWORD_HASH):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Create JWT token
    access_token = create_access_token(
        data={"sub": data.email},
        expires_delta=timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    )
    
    return {
        "success": True,
        "token": access_token,
        "message": "Login successful"
    }

class TokenVerify(BaseModel):
    token: str

@app.post("/api/admin/verify")
async def verify_admin_token(data: TokenVerify):
    """Verify if a token is still valid"""
    email = verify_token(data.token)
    if email:
        return {"valid": True, "email": email}
    else:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

@app.get("/api/banner")
async def get_banner_public():
    """Get current banner (public endpoint)"""
    # Always load from file for latest value
    return load_banner_config()

@app.get("/api/admin/banner")
async def get_banner_admin(token: str):
    """Get banner data (admin only)"""
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return load_banner_config()

@app.post("/api/admin/banner")
async def update_banner(banner: BannerData, token: str):
    """Update banner (admin only)"""
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    save_banner_config(banner)
    logger.info(f"Banner updated by {email}: enabled={banner.enabled}, type={banner.type}")
    return {"success": True, "message": "Banner updated successfully"}


@app.get("/api/admin/ignored_stations")
def admin_list_ignored(token: str):
    """List ignored stations (admin only)"""
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")

    db_path = get_db_path()
    try:
        conn = sqlite3.connect(db_path)
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
def admin_add_ignored(payload: dict, token: str):
    """Add an ignored station. Payload: { stid: 'MBGM7', reason: '...' }"""
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")

    stid = (payload.get('stid') or '').strip().upper()
    reason = payload.get('reason') or None
    if not stid:
        raise HTTPException(status_code=400, detail="stid required")

    db_path = get_db_path()
    try:
        conn = sqlite3.connect(db_path)
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
def admin_remove_ignored(stid: str, token: str):
    """Remove an ignored station by STID"""
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")

    stid = stid.strip().upper()
    db_path = get_db_path()
    try:
        conn = sqlite3.connect(db_path)
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
def admin_set_website_version(payload: dict, token: str):
    """Admin-only: set website version. Payload: { version: '1.2.3' }"""
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")

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
def admin_list_projects(token: str):
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return {"success": True, "projects": list_dev_projects()}


@app.post('/api/admin/projects')
def admin_create_project(payload: DevProjectCreate, token: str):
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")

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
def admin_update_project(project_id: int, payload: DevProjectUpdate, token: str):
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")

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
def admin_delete_project(project_id: int, token: str):
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")

    if not delete_dev_project(project_id):
        raise HTTPException(status_code=404, detail="Project not found")
    return {"success": True, "deleted": project_id}

@app.get('/stations/raws')
def get_raws_stations():
    """
    Get all RAWS stations in MO, OK, AR, TN, KY, IL, IA, NE, KS.
    """
    return raws_station_data

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
async def list_archived_files(archive_dir: str = str(ARCHIVE_RAW_DATA_DIR)):
    '''
    List all archived data files.
    
    Query params:
    - archive_dir: Directory to list (default: archive/raw_data)
    
    Returns: List of available archive files with metadata
    '''
    archive_path = Path(archive_dir)
    
    if not archive_path.exists():
        return {"success": True, "files": []}
    
    files = []
    for filepath in archive_path.glob("raw_data_*.json"):
        stat = filepath.stat()
        files.append({
            "filename": filepath.name,
            "size_mb": round(stat.st_size / 1024 / 1024, 2),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
        })
    
    files.sort(key=lambda x: x['modified'], reverse=True)
    
    return {
        "success": True,
        "files": files,
        "count": len(files)
    }


@app.get("/api/historical/archive/load/{filename}")
async def load_archived_file(filename: str, archive_dir: str = str(ARCHIVE_RAW_DATA_DIR)):
    '''
    Load a specific archived data file.
    
    Path params:
    - filename: Name of the file to load
    
    Query params:
    - archive_dir: Directory to load from (default: archive/raw_data)
    
    Returns: Archived data with statistics
    '''
    archive_path = Path(archive_dir) / filename
    
    if not archive_path.exists():
        return {"success": False, "error": "File not found"}
    
    try:
        with open(archive_path, 'r') as f:
            data = json.load(f)
        
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

@app.get("/api/fires/missouri/geojson")
async def get_missouri_fires_geojson():
    """Get current Missouri fire detections as GeoJSON"""
    geojson_file = MISSOURI_FIRES_GEOJSON
    if os.path.exists(geojson_file):
        return FileResponse(geojson_file, media_type='application/geo+json')
    else:
        raise HTTPException(status_code=404, detail="Fire GeoJSON not yet available")
    
@app.get("/forecast/latest")
def api_latest_forecast():
    forecast = get_latest_forecast()
    if not forecast:
        raise HTTPException(status_code=404, detail="No forecast found")
    return forecast

@app.get("/api/admin/reports/list")
async def list_reports(token: str):
    """List all reports (files and folders) in the reports directory (Admin/Logged-in only)"""
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return list_files(str(REPORTS_DIR))

@app.get("/api/admin/reports/list/{path:path}")
async def list_reports_in_path(path: str, token: str):
    """List contents of a specific folder within reports directory"""
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return list_files(str(REPORTS_DIR), path)

@app.get("/api/admin/reports/view/{filepath:path}")
async def view_report(filepath: str, token: str):
    """Serve a specific report file securely from reports directory"""
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return view_file(str(REPORTS_DIR), filepath, email)

@app.get("/api/admin/logs/list")
async def list_logs(token: str):
    """List all logs (files and folders) in the logs directory (Admin/Logged-in only)"""
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return list_files(str(LOGS_DIR))

@app.get("/api/admin/logs/list/{path:path}")
async def list_logs_in_path(path: str, token: str):
    """List contents of a specific folder within logs directory"""
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return list_files(str(LOGS_DIR), path)

@app.get("/api/admin/logs/view/{filepath:path}")
async def view_log(filepath: str, token: str):
    """Serve a specific log file securely from logs directory"""
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return view_file(str(LOGS_DIR), filepath, email)

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
async def get_morning_fuel_moisture():
    """
    Fetch fuel moisture observations near 7 AM Central Time for today.
    Uses default states (MO and surrounding) and network 2 (RAWS).
    
    Returns: Fuel moisture observations from stations
    """
    from services.synoptic import fetch_fuel_moisture_at_time
    
    try:
        # Use defaults: current day 7 AM CT, multi-state, network 2
        data = await fetch_fuel_moisture_at_time(
            target_time=None,  # Defaults to 7 AM CT today
            states=None,       # Defaults to MO and surrounding states
            networks=None      # Defaults to network 2
        )
        
        return {
            "success": True,
            "data": data
        }
    except Exception as e:
        logger.error(f"Error fetching morning fuel moisture: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching fuel moisture data: {str(e)}"
        )



@app.get("/api/fuel-moisture/morning/debug")
async def get_morning_fuel_moisture_debug(
    date: Optional[str] = None,
    states: Optional[str] = "MO",
    networks: Optional[str] = "2"
):
    """
    Debug endpoint - returns raw Synoptic API response for fuel moisture.
    Helps diagnose data extraction issues.
    """
    from services.synoptic import SYNOPTIC_API_TOKEN
    from pytz import timezone as pytz_timezone
    import aiohttp
    
    try:
        # Parse date if provided
        if date:
            from datetime import datetime as dt
            central = pytz_timezone('America/Chicago')
            parsed_date = dt.strptime(date, "%Y-%m-%d")
            target_central = central.localize(parsed_date.replace(hour=7, minute=0, second=0))
            target_time = target_central.astimezone(pytz_timezone('UTC'))
        else:
            from pytz import timezone as pytz_timezone
            central = pytz_timezone('America/Chicago')
            now_central = datetime.now(central)
            target_central = now_central.replace(hour=7, minute=0, second=0, microsecond=0)
            target_time = target_central.astimezone(pytz_timezone('UTC'))
        
        attime = target_time.strftime("%Y%m%d%H%M")
        
        url = "https://api.synopticdata.com/v2/stations/nearesttime"
        params = {
            "token": SYNOPTIC_API_TOKEN,
            "state": states,
            "attime": attime,
            "within": "60",
            "network": networks,
            "vars": "fuel_moisture",
            "obtimezone": "local"
        }
        
        async with aiohttp.ClientSession() as session:
            response = await session.get(url, params=params, timeout=60)
            response.raise_for_status()
            raw_data = await response.json()
        
        return {
            "success": True,
            "query_params": params,
            "raw_response": raw_data
        }
    except Exception as e:
        logger.error(f"Error in debug endpoint: {e}")
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
                "emc_baseline - Equilibrium moisture content (Simard 1968: 0.03229 + 0.281073*RH - 0.000578*RH*Temp)",
                "temp_mean_3h - 3-hour rolling mean temperature",
                "rh_mean_3h - 3-hour rolling mean relative humidity",
                "temp_mean_6h - 6-hour rolling mean temperature", 
                "rh_mean_6h - 6-hour rolling mean relative humidity",
                "precip_1h - 1-hour precipitation accumulation (mm)",
                "precip_3h - 3-hour precipitation accumulation (mm)",
                "precip_6h - 6-hour precipitation accumulation (mm)",
                "precip_24h - 24-hour precipitation accumulation (mm)",
                "hours_since_rain - Hours since last significant rain (>0.1mm)"
            ],
            "output": "Predicted 10-hour fuel moisture percentage (1-40%)",
            "baseline_equation": "EMC = 0.03229 + (0.281073 × RH) - (0.000578 × RH × Temp)",
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
                    "condition": "9% ≤ FM < 15% AND RH < 50% AND Wind ≥ 10 kts"
                },
                {
                    "level": "Elevated",
                    "score": 2, 
                    "condition": "FM < 9% AND (RH < 45% OR Wind ≥ 10 kts)"
                },
                {
                    "level": "Critical",
                    "score": 3,
                    "condition": "FM < 9% AND RH < 25% AND Wind ≥ 15 kts"
                },
                {
                    "level": "Extreme",
                    "score": 4,
                    "condition": "FM < 7% AND RH < 20% AND Wind ≥ 30 kts"
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
        
        return {
            "success": True,
            "fuel_moisture_model": fuel_moisture_formula,
            "fire_danger_model": fire_danger_formula,
            "last_updated": "2024-01-28",
            "version": "1.0"
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
    
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)