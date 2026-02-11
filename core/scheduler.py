import os
import logging
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pytz import timezone
from services.synoptic import fetch_synoptic_data, fetch_raws_stations_multi_state
from services.timeseries import fetchtimeseriesdata
from tools.nfgs_firedetect import main as firedetect
from tools.firedetections import main as fetch_advanced_fire_detections

logger = logging.getLogger(__name__)

# Global storage for RAWS stations
raws_station_data = {
    "stations": None,
    "last_updated": None,
    "error": None
}

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

def create_scheduler():
    central_tz = timezone('America/Chicago')
    return AsyncIOScheduler(timezone=central_tz)

def start_scheduler_jobs(scheduler: AsyncIOScheduler):
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
    
    # Run advanced fire detections every 5 minutes
    scheduler.add_job(
        fetch_advanced_fire_detections,
        'interval',
        minutes=5,
        id='fetch_advanced_fire_detections'
    )
    
    scheduler.start()
    logger.info("Scheduler started")

async def run_initial_fetches():
    await fetch_synoptic_data()
    await fetchtimeseriesdata()
    await fetch_and_store_raws_stations()
