from fastapi import FastAPI, HTTPException # , WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, FileResponse
from contextlib import asynccontextmanager
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from synoptic import fetch_synoptic_data, get_station_data, fetch_raws_stations_multi_state, fetch_historical_station_data, save_raw_data_to_archive, get_raw_data_stats
from timeseries import fetchtimeseriesdata, get_timeseries_data
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
# from broadcast import add_client, remove_client, broadcast_update, connected_clients
import os
import logging
import json
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional
from rss_feed import generate_rss_feed
from pathlib import Path
from tools.nfgs_firedetect import main as firedetect
from pytz import timezone
from database import (
    get_latest_forecast,
    get_forecast_by_time,
    get_recent_forecasts,
    get_forecast_count
)


# Security Configuration
SECRET_KEY = os.getenv("JWT_SECRET", "CHANGE-THIS-TO-A-RANDOM-SECRET-KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

ADMIN_EMAIL = os.getenv("ADMIN_EMAIL")
ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH")

IS_PRODUCTION = os.getenv("ENVIRONMENT", "development").lower() == "production"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Add a global storage for RAWS stations
raws_station_data = {
    "stations": None,
    "last_updated": None,
    "error": None
}

central_tz = timezone('America/Chicago')
scheduler = AsyncIOScheduler(timezone=central_tz)

async def fetch_and_store_raws_stations():
    """Fetch RAWS stations and store in global variable"""
    try:
        raws_stations = await fetch_raws_stations_multi_state()
        raws_station_data["stations"] = raws_stations
        raws_station_data["last_updated"] = datetime.now().isoformat()
        raws_station_data["error"] = None
    except Exception as e:
        raws_station_data["error"] = str(e)
        raws_station_data["stations"] = []
        raws_station_data["last_updated"] = datetime.now().isoformat()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("Starting scheduler...")
    scheduler = AsyncIOScheduler()
    # Run synoptic at :00, :05, :10, etc.
    scheduler.add_job(fetch_synoptic_data, 'interval', minutes=5, id='fetch_synoptic')
    # Run timeseries at :02, :07, :12, etc. (2 minutes offset)
    scheduler.add_job(fetchtimeseriesdata, 'interval', minutes=5, seconds=60, id='fetch_timeseries')
    # Run RAWS fetch at :00, :05, :10, etc.
    scheduler.add_job(fetch_and_store_raws_stations, 'interval', minutes=5, id='fetch_raws_stations')
    
    scheduler.add_job(
        firedetect, 
        'cron', 
        minute='0,5,10,15,20,25,30,35,40,45,50,55',
        hour='10-22',  # 10 AM through 10 PM
        id='fetch_fire_detections'
    )
    scheduler.start()
    logger.info("Scheduler started")
    
    await fetch_synoptic_data()
    await fetchtimeseriesdata()
    await fetch_and_store_raws_stations()
    
    yield
    
    logger.info("Shutting down scheduler...")
    scheduler.shutdown()

app = FastAPI(
    title="Show Me Fire Weather API",
    description="API for fetching synoptic weather data",
    lifespan=lifespan,
    docs_url=None if IS_PRODUCTION else "/docs",
    redoc_url=None if IS_PRODUCTION else "/redoc",
    openapi_url=None if IS_PRODUCTION else "/openapi.json"
)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[str]:
    """Verify a JWT token and return the email"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        return email
    except JWTError:
        return None

class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        response: Response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

# Use this instead of the default StaticFiles
app.mount("/images", NoCacheStaticFiles(directory="images"), name="images")
app.mount("/gis", NoCacheStaticFiles(directory="gis"), name="gis")
app.mount("/public", StaticFiles(directory="public"), name="public")

origins = [
    "http://localhost:3000",        # For local development of a React/Vue frontend
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


# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     """WebSocket endpoint for real-time data updates"""
#     await websocket.accept()
#     add_client(websocket)
#     logger.info("Client connected")
#     
#     try:
#         # Send initial data
#         await websocket.send_json({
#             "type": "initial",
#             "synoptic": get_station_data(),
#             "timeseries": get_timeseries_data()
#         })
#         
#         # Broadcast connection event to all clients
#         await broadcast_update("connection", {
#             "message": "New client connected",
#             "total_clients": len(connected_clients)
#         })
#         
#         # Keep connection open
#         while True:
#             await websocket.receive_text()
#     except WebSocketDisconnect:
#         remove_client(websocket)
#         logger.info("Client disconnected")
#         
#         # Broadcast disconnection event to all remaining clients
#         await broadcast_update("disconnection", {
#             "message": "Client disconnected",
#             "total_clients": len(connected_clients)
#         })

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
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Show Me Fire - API Status</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1000px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            h1 { color: #333; }
            .status-box {
                background: white;
                padding: 20px;
                margin: 20px 0;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .healthy { color: #28a745; font-weight: bold; }
            .error { color: #dc3545; font-weight: bold; }
            .info { color: #666; font-size: 14px; }
            .connected { color: #28a745; }
            .disconnected { color: #dc3545; }
            .refresh { 
                display: inline-block; 
                margin-top: 10px; 
                padding: 8px 12px; 
                background: #007bff; 
                color: white; 
                border: none; 
                border-radius: 4px; 
                cursor: pointer;
            }
            .refresh:hover { background: #0056b3; }
        </style>
        <script>
            let ws;
            
            function connectWebSocket() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(protocol + '//' + window.location.host + '/ws');
                
                ws.onopen = () => {
                    console.log('WebSocket connected');
                    document.getElementById('ws-status').textContent = 'Connected';
                    document.getElementById('ws-status').className = 'connected';
                };
                
                ws.onmessage = (event) => {
                    const message = JSON.parse(event.data);
                    console.log('Update received:', message);
                    updateStatus(message);
                };
                
                ws.onclose = () => {
                    console.log('WebSocket disconnected');
                    document.getElementById('ws-status').textContent = 'Disconnected';
                    document.getElementById('ws-status').className = 'disconnected';
                    // Reconnect after 3 seconds
                    setTimeout(connectWebSocket, 3000);
                };
            }
            
            function updateStatus(message) {
                if (message.type === 'initial' || message.type === 'synoptic') {
                    const data = message.synoptic || message.data;
                    document.getElementById('synoptic-status').innerHTML = formatSynopticStatus(data);
                }
                if (message.type === 'initial' || message.type === 'timeseries') {
                    const data = message.timeseries || message.data;
                    document.getElementById('timeseries-status').innerHTML = formatTimeseriesStatus(data);
                }
            }
            
            function formatSynopticStatus(data) {
                return `
                    <p>Status: <span class="${data.error ? 'error' : 'healthy'}">${data.error ? 'ERROR: ' + data.error : '✓ Healthy'}</span></p>
                    <p>Stations: <span class="info">${data.station_count}</span></p>
                    <p>Last Updated: <span class="info">${new Date(data.last_updated).toLocaleString()}</span></p>
                `;
            }
            
            function formatTimeseriesStatus(data) {
                return `
                    <p>Status: <span class="${data.error ? 'error' : 'healthy'}">${data.error ? 'ERROR: ' + data.error : '✓ Healthy'}</span></p>
                    <p>Stations: <span class="info">${data.station_count}</span></p>
                    <p>Last Updated: <span class="info">${new Date(data.last_updated).toLocaleString()}</span></p>
                `;
            }
            
            connectWebSocket();
        </script>
    </head>
    <body>
        <h1>🔥 Show Me Fire - API Status</h1>
        <p>WebSocket: <span id="ws-status" class="disconnected">Connecting...</span></p>
        <div class="status-box">
            <h2>Synoptic Weather Data</h2>
            <div id="synoptic-status">Loading...</div>
        </div>
        <div class="status-box">
            <h2>Timeseries Data</h2>
            <div id="timeseries-status">Loading...</div>
        </div>
    </body>
    </html>
    """

@app.get("/list-images")
def list_images():
    files = []
    images_dir = "images"
    for fname in os.listdir(images_dir):
        fpath = os.path.join(images_dir, fname)
        if os.path.isfile(fpath):
            files.append(fname)
    return JSONResponse(content={"files": files})

@app.get("/list-gis")
def list_gis():
    files = []
    gis_dir = "gis"
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
    
    
class BannerData(BaseModel):
    enabled: bool = False
    type: str = "info"  # info, warning, danger, success
    message: str = ""
    link: Optional[str] = None

# Persistent banner config file
BANNER_CONFIG_PATH = Path("banner_config.json")

def load_banner_config() -> BannerData:
    if BANNER_CONFIG_PATH.exists():
        try:
            with open(BANNER_CONFIG_PATH, "r") as f:
                data = json.load(f)
            return BannerData(**data)
        except Exception as e:
            logger.error(f"Failed to load banner config: {e}")
    return BannerData()

def save_banner_config(banner: BannerData):
    try:
        with open(BANNER_CONFIG_PATH, "w") as f:
            json.dump(banner.dict(), f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save banner config: {e}")

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
                archive_dir="archive/raw_data"
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
async def list_archived_files(archive_dir: str = "archive/raw_data"):
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
async def load_archived_file(filename: str, archive_dir: str = "archive/raw_data"):
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
    json_file = 'data/missouri_fires_coords.json'
    if os.path.exists(json_file):
        return FileResponse(json_file, media_type='application/json')
    else:
        raise HTTPException(status_code=404, detail="Fire data not yet available")

@app.get("/api/fires/missouri/geojson")
async def get_missouri_fires_geojson():
    """Get current Missouri fire detections as GeoJSON"""
    geojson_file = 'data/missouri_fires.geojson'
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
    
    reports_dir = Path("reports")
    if not reports_dir.exists():
        return {"success": True, "items": [], "path": ""}
    
    items = []
    for filepath in sorted(reports_dir.glob("*")):
        stat = filepath.stat()
        if filepath.is_dir():
            # For folders, count the items inside
            item_count = len(list(filepath.glob("*")))
            items.append({
                "name": filepath.name,
                "type": "folder",
                "item_count": item_count,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        else:
            # For files, show size
            items.append({
                "name": filepath.name,
                "type": "file",
                "size_kb": round(stat.st_size / 1024, 2),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
    
    items.sort(key=lambda x: x['modified'], reverse=True)
    return {"success": True, "items": items, "path": ""}

@app.get("/api/admin/reports/list/{path:path}")
async def list_reports_in_path(path: str, token: str):
    """List contents of a specific folder within reports directory"""
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    reports_dir = Path("reports")
    target_path = reports_dir / path
    
    # Security check: ensure the user isn't trying to access files outside the folder
    if not target_path.resolve().is_relative_to(reports_dir.resolve()):
        raise HTTPException(status_code=403, detail="Forbidden")
    
    if not target_path.exists():
        raise HTTPException(status_code=404, detail="Folder not found")
    
    if not target_path.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")
    
    items = []
    for filepath in sorted(target_path.glob("*")):
        stat = filepath.stat()
        if filepath.is_dir():
            item_count = len(list(filepath.glob("*")))
            items.append({
                "name": filepath.name,
                "type": "folder",
                "item_count": item_count,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        else:
            items.append({
                "name": filepath.name,
                "type": "file",
                "size_kb": round(stat.st_size / 1024, 2),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
    
    items.sort(key=lambda x: x['modified'], reverse=True)
    return {"success": True, "items": items, "path": path}

@app.get("/api/admin/reports/view/{filepath:path}")
async def view_report(filepath: str, token: str):
    """Serve a specific report file securely from reports directory"""
    # Verify token
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    # Construct the file path
    report_path = Path("reports") / filepath
    
    # Security check: ensure the user isn't trying to access files outside the folder
    try:
        report_path.resolve().relative_to(Path("reports").resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Forbidden: Path traversal not allowed")

    # Check if file exists
    if not report_path.exists():
        logger.warning(f"File not found: {report_path} (requested by {email})")
        raise HTTPException(status_code=404, detail=f"File not found: {filepath}")
    
    # Ensure it's a file, not a directory
    if report_path.is_dir():
        raise HTTPException(status_code=400, detail="Path is a directory, not a file")
    
    # Serve the file
    logger.info(f"Serving report {filepath} to {email}")
    return FileResponse(report_path)

@app.get("/api/admin/logs/list")
async def list_logs(token: str):
    """List all logs (files and folders) in the logs directory (Admin/Logged-in only)"""
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    logs_dir = Path("logs")
    if not logs_dir.exists():
        return {"success": True, "items": [], "path": ""}
    
    items = []
    for filepath in sorted(logs_dir.glob("*")):
        stat = filepath.stat()
        if filepath.is_dir():
            # For folders, count the items inside
            item_count = len(list(filepath.glob("*")))
            items.append({
                "name": filepath.name,
                "type": "folder",
                "item_count": item_count,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        else:
            # For files, show size
            items.append({
                "name": filepath.name,
                "type": "file",
                "size_kb": round(stat.st_size / 1024, 2),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
    
    items.sort(key=lambda x: x['modified'], reverse=True)
    return {"success": True, "items": items, "path": ""}

@app.get("/api/admin/logs/list/{path:path}")
async def list_logs_in_path(path: str, token: str):
    """List contents of a specific folder within logs directory"""
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    logs_dir = Path("logs")
    target_path = logs_dir / path
    
    # Security check: ensure the user isn't trying to access files outside the folder
    if not target_path.resolve().is_relative_to(logs_dir.resolve()):
        raise HTTPException(status_code=403, detail="Forbidden")
    
    if not target_path.exists():
        raise HTTPException(status_code=404, detail="Folder not found")
    
    if not target_path.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a directory")
    
    items = []
    for filepath in sorted(target_path.glob("*")):
        stat = filepath.stat()
        if filepath.is_dir():
            item_count = len(list(filepath.glob("*")))
            items.append({
                "name": filepath.name,
                "type": "folder",
                "item_count": item_count,
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
        else:
            items.append({
                "name": filepath.name,
                "type": "file",
                "size_kb": round(stat.st_size / 1024, 2),
                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
            })
    
    items.sort(key=lambda x: x['modified'], reverse=True)
    return {"success": True, "items": items, "path": path}

@app.get("/api/admin/logs/view/{filepath:path}")
async def view_log(filepath: str, token: str):
    """Serve a specific log file securely from logs directory"""
    # Verify token
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    # Construct the file path
    log_path = Path("logs") / filepath
    
    # Security check: ensure the user isn't trying to access files outside the folder
    try:
        log_path.resolve().relative_to(Path("logs").resolve())
    except ValueError:
        raise HTTPException(status_code=403, detail="Forbidden: Path traversal not allowed")

    # Check if file exists
    if not log_path.exists():
        logger.warning(f"Log file not found: {log_path} (requested by {email})")
        raise HTTPException(status_code=404, detail=f"Log file not found: {filepath}")
    
    # Ensure it's a file, not a directory
    if log_path.is_dir():
        raise HTTPException(status_code=400, detail="Path is a directory, not a file")
    
    # Serve the file
    logger.info(f"Serving log {filepath} to {email}")
    return FileResponse(log_path)

@app.get("/api/training/latest-stats")
async def get_latest_training_stats():
    """Returns the JSON stats from the most recent training folder"""
    reports_dir = Path("reports")
    # Get the most recent date folder
    date_folders = sorted([d for d in reports_dir.iterdir() if d.is_dir()], reverse=True)
    
    if not date_folders:
        raise HTTPException(status_code=404, detail="No training data found")
        
    stats_path = date_folders[0] / "stats.json"
    if stats_path.exists():
        with open(stats_path, "r") as f:
            return json.load(f)
    
    raise HTTPException(status_code=404, detail="Stats file missing")

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)