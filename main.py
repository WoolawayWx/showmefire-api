from fastapi import FastAPI  # , WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from contextlib import asynccontextmanager
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from synoptic import fetch_synoptic_data, get_station_data
from timeseries import fetchtimeseriesdata, get_timeseries_data
from fastapi.middleware.cors import CORSMiddleware
# from broadcast import add_client, remove_client, broadcast_update, connected_clients
import os
import logging
import json
from fastapi.staticfiles import StaticFiles
from starlette.responses import Response

IS_PRODUCTION = os.getenv("ENVIRONMENT", "development").lower() == "production"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("Starting scheduler...")
    scheduler = AsyncIOScheduler()
    # Run synoptic at :00, :05, :10, etc.
    scheduler.add_job(fetch_synoptic_data, 'interval', minutes=5, id='fetch_synoptic')
    # Run timeseries at :02, :07, :12, etc. (2 minutes offset)
    scheduler.add_job(fetchtimeseriesdata, 'interval', minutes=5, seconds=60, id='fetch_timeseries')
    scheduler.start()
    logger.info("Scheduler started")
    
    await fetch_synoptic_data()
    await fetchtimeseriesdata()
    
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

class NoCacheStaticFiles(StaticFiles):
    async def get_response(self, path, scope):
        response: Response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "public, max-age=0, must-revalidate"
        # This forces browser to check with server each time
        # but only downloads if file changed
        return response

# Use this instead of the default StaticFiles
app.mount("/images", NoCacheStaticFiles(directory="images"), name="images")

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
    await broadcast_update("synoptic", data)
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

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)