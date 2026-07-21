import asyncio
import os
import logging
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pytz import timezone
from services.synoptic import fetch_synoptic_data, fetch_raws_stations_multi_state
from services.timeseries import fetchtimeseriesdata
from tools.nfgs_firedetect import main as firedetect
from tools.firedetections import main as fetch_advanced_fire_detections
from alerts.activemoalerts import run_active_mo_alerts
from services.afds import ingest_latest_afds
from services.archive_bundler import run_end_of_day_archive
from services.rtma_capture import cleanup_rtma_cache, fetch_rtma, latest_complete_hour
from services.mobile_push import check_push_receipts, purge_delivery_records
from core.config import AFD_POLL_MINUTES

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


async def fetch_and_store_afds():
    """Fetch new AFD products and persist them to the database."""
    try:
        await ingest_latest_afds()
    except Exception as e:
        logger.error("Error fetching/storing AFDs: %s", e, exc_info=True)


async def capture_latest_rtma():
    """Run Herbie/netCDF work off the API event loop."""
    try:
        await asyncio.to_thread(fetch_rtma, latest_complete_hour())
        try:
            await asyncio.to_thread(cleanup_rtma_cache)
        except Exception as cleanup_error:
            logger.error("RTMA capture succeeded but retention cleanup failed: %s", cleanup_error, exc_info=True)
    except Exception as e:
        logger.error("RTMA capture failed: %s", e, exc_info=True)

def create_scheduler():
    central_tz = timezone('America/Chicago')
    return AsyncIOScheduler(timezone=central_tz)

def start_scheduler_jobs(scheduler: AsyncIOScheduler):
    scheduler.add_job(fetch_synoptic_data, 'interval', minutes=5, id='fetch_synoptic')
    scheduler.add_job(fetchtimeseriesdata, 'interval', minutes=5, seconds=60, id='fetch_timeseries')
    scheduler.add_job(fetch_and_store_raws_stations, 'interval', minutes=5, id='fetch_raws_stations')
    scheduler.add_job(fetch_and_store_afds, 'interval', minutes=AFD_POLL_MINUTES, id='fetch_afds')
    scheduler.add_job(run_active_mo_alerts, 'interval', minutes=5, id='fetch_active_mo_alerts')
    scheduler.add_job(check_push_receipts, 'interval', minutes=15, id='check_mobile_push_receipts')
    scheduler.add_job(
        purge_delivery_records,
        'cron',
        hour=2,
        minute=30,
        id='purge_mobile_push_delivery_records',
    )
    
    scheduler.add_job(
        firedetect, 
        'cron', 
        minute='0,5,10,15,20,25,30,35,40,45,50,55',
        hour='10-22',
        id='fetch_fire_detections'
    )
    
    scheduler.add_job(
        fetch_advanced_fire_detections,
        'cron',
        minute='0,5,10,15,20,25,30,35,40,45,50,55',
        hour='10-22',
        id='fetch_advanced_fire_detections'
    )

    scheduler.add_job(
        capture_latest_rtma,
        'cron',
        minute=50,
        id='capture_rtma',
        max_instances=1,
        coalesce=True,
    )

    scheduler.add_job(
        run_end_of_day_archive,
        'cron',
        hour=23,
        minute=45,
        id='end_of_day_archive'
    )

    scheduler.start()
    logger.info("Scheduler started")

async def run_initial_fetches():
    await fetch_synoptic_data()
    await fetchtimeseriesdata()
    await fetch_and_store_raws_stations()
    await fetch_and_store_afds()
