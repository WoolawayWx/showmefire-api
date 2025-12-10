from fastapi import FastAPI
from contextlib import asynccontextmanager
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from synoptic import fetch_synoptic_data, get_station_data
from timeseries import fetchtimeseriesdata, get_timeseries_data
from fastapi.middleware.cors import CORSMiddleware
import os

IS_PRODUCTION = os.getenv("ENVIRONMENT", "development").lower() == "production"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    scheduler = AsyncIOScheduler()
    scheduler.add_job(fetch_synoptic_data, 'interval', minutes=5)
    scheduler.add_job(fetchtimeseriesdata, 'interval', minutes=5)
    scheduler.start()
    
    await fetch_synoptic_data()
    await fetchtimeseriesdata()
    
    yield
    
    scheduler.shutdown()

app = FastAPI(
    title="Show Me Fire Weather API",
    description="API for fetching synoptic weather data",
    lifespan=lifespan,
    docs_url=None if IS_PRODUCTION else "/docs",
    redoc_url=None if IS_PRODUCTION else "/redoc",
    openapi_url=None if IS_PRODUCTION else "/openapi.json"
)

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

@app.get('/stations/timeseries/refresh')
async def timeseries_refresh():
    await fetchtimeseriesdata()
    return get_timeseries_data()

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)