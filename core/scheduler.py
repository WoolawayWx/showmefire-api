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
from services.v5_verification import verify_pending as verify_v5_shadow
from services.fire_ingest import ingest_detection_files
from core.database import expire_unmoderated_fire_reports, purge_fire_submission_pii, purge_fire_throttle_rows
from services.spatial_fm_uncertainty_cache import purge_stale as purge_spatial_fm_uncertainty_cache
from services.seasonal_fuel_state import update_daily_gdd

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


async def verify_v5_shadow_observations():
    """Attach mature observations without blocking the API event loop."""
    try:
        await asyncio.to_thread(verify_v5_shadow)
    except Exception as error:
        logger.error("V5 shadow verification failed: %s", error, exc_info=True)


async def ingest_fire_detections_job():
    """
    Backfill the fire_events store from the existing detection GeoJSON
    files. A separate job from the fetch jobs that write those files -
    a store failure here must never affect the file pipeline that
    /fires/satdet and the mobile app depend on.
    """
    try:
        await asyncio.to_thread(ingest_detection_files)
    except Exception as error:
        logger.error("Fire detection ingest failed: %s", error, exc_info=True)


async def purge_spatial_fm_uncertainty_cache_job():
    """Delete spatial FM uncertainty cache files older than the retention window."""
    try:
        removed = await asyncio.to_thread(purge_spatial_fm_uncertainty_cache)
        logger.info("Spatial FM uncertainty cache purge: removed=%s", removed)
    except Exception as error:
        logger.error("Spatial FM uncertainty cache purge failed: %s", error, exc_info=True)


async def purge_fire_report_pii_job():
    """Expire stale pending reports, then purge PII past the retention window."""
    try:
        expired = await asyncio.to_thread(expire_unmoderated_fire_reports)
        purged = await asyncio.to_thread(purge_fire_submission_pii)
        await asyncio.to_thread(purge_fire_throttle_rows)
        logger.info("Fire report PII purge: expired=%s purged=%s", expired, purged)
    except Exception as error:
        logger.error("Fire report PII purge failed: %s", error, exc_info=True)

async def update_seasonal_fuel_state_job():
    """Advance the GDD accumulator before end-of-day archiving removes today's raw_data JSON."""
    try:
        state = await asyncio.to_thread(update_daily_gdd)
        logger.info(
            "Seasonal fuel state updated: gdd_accum_since_mar1=%.1f last_updated_date=%s",
            state.get("gdd_accum_since_mar1", 0.0), state.get("last_updated_date"),
        )
    except Exception as error:
        logger.error("Seasonal fuel state update failed: %s", error, exc_info=True)


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
        update_seasonal_fuel_state_job,
        'cron',
        hour=23,
        minute=30,
        id='update_seasonal_fuel_state',
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

    scheduler.add_job(
        verify_v5_shadow_observations,
        'interval',
        hours=3,
        id='verify_v5_shadow',
        max_instances=1,
        coalesce=True,
    )

    scheduler.add_job(
        ingest_fire_detections_job,
        'cron',
        minute='3,8,13,18,23,28,33,38,43,48,53,58',
        hour='10-22',
        id='ingest_fire_detections',
        max_instances=1,
        coalesce=True,
    )

    scheduler.add_job(
        purge_fire_report_pii_job,
        'cron',
        hour=2,
        minute=45,
        id='purge_fire_report_pii',
    )

    scheduler.add_job(
        purge_spatial_fm_uncertainty_cache_job,
        'cron',
        hour=3,
        minute=15,
        id='purge_spatial_fm_uncertainty_cache',
    )

    scheduler.start()
    logger.info("Scheduler started")

async def run_initial_fetches():
    await fetch_synoptic_data()
    await fetchtimeseriesdata()
    await fetch_and_store_raws_stations()
    await fetch_and_store_afds()
