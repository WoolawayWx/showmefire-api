import asyncio
import os
import logging
from datetime import datetime
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pytz import timezone
from services.synoptic import fetch_synoptic_data, fetch_raws_stations_multi_state, get_station_data
from services.timeseries import fetchtimeseriesdata
from tools.nfgs_firedetect import main as firedetect
from tools.firedetections import main as fetch_advanced_fire_detections
from alerts.activemoalerts import run_active_mo_alerts
from services.afds import ingest_latest_afds
from services.archive_bundler import run_end_of_day_archive
from services.rtma_capture import cleanup_rtma_cache, fetch_rtma, latest_complete_hour, spread_rate_poll_minutes
from services.mobile_push import check_push_receipts, purge_delivery_records
from core.config import AFD_POLL_MINUTES
from services.v5_verification import verify_pending as verify_v5_shadow
from services.drift_monitor import run_drift_check
from services.fire_ingest import ingest_detection_files
from core.database import (
    expire_unmoderated_fire_reports, purge_fire_submission_pii, purge_fire_throttle_rows,
    purge_feedback_throttle_rows,
)
from services.spatial_fm_uncertainty_cache import purge_stale as purge_spatial_fm_uncertainty_cache
from services.seasonal_fuel_state import update_daily_gdd
from services.rtma_peak import generate_rtma_peak, run_rtma_peak_job
from services.spread_rate import run_spread_rate_job, run_spread_rate_pipeline
from routers.burn_bans import run_burn_ban_maintenance
from services.beta_products import BETA_ROOT, load_manifest, refresh_observation_products, save_manifest
from services.beta_verification import run_beta_verification
from services.forecast_jobs import trigger_beta_forecast

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


async def refresh_testbed_observations_job():
    """Build isolated beta observation products from the latest station fetch."""
    try:
        await asyncio.to_thread(
            refresh_observation_products,
            get_station_data(),
            raws_station_data,
        )
    except Exception as error:
        logger.error("Testbed observation refresh failed: %s", error, exc_info=True)


async def refresh_testbed_rtma_job():
    """Build an isolated continuous-score RTMA peak after the production run."""
    try:
        result = await asyncio.to_thread(
            generate_rtma_peak,
            None,
            output_root=BETA_ROOT,
            experimental=True,
        )
        manifest = load_manifest()
        manifest["rtma_updated_at"] = datetime.now().isoformat()
        manifest.setdefault("products", {})["rtma_peak"] = {
            "kind": "image",
            "path": f"images/{result['png']}",
            "generated_at": manifest["rtma_updated_at"],
        }
        save_manifest(manifest)
    except Exception as error:
        logger.error("Testbed RTMA refresh failed: %s", error, exc_info=True)


async def refresh_testbed_spread_rate_job():
    """Poll RTMA and publish spread-rate artifacts on a 15-minute cadence."""
    try:
        await run_spread_rate_job(raws_station_data if raws_station_data.get("stations") else None)
    except Exception as error:
        logger.error("Testbed spread-rate refresh failed: %s", error, exc_info=True)


async def rtma_spread_rate_pipeline_job():
    """Ensure latest RTMA is cached on the server, then refresh spread-rate."""
    try:
        await asyncio.to_thread(run_spread_rate_pipeline, raws_station_data if raws_station_data.get("stations") else None)
        try:
            await asyncio.to_thread(cleanup_rtma_cache)
        except Exception as cleanup_error:
            logger.error("Spread-rate RTMA retention cleanup failed: %s", cleanup_error, exc_info=True)
    except Exception as error:
        logger.error("RTMA/spread-rate pipeline failed: %s", error, exc_info=True)


async def run_scheduled_beta_forecast_job():
    """Regenerate the isolated Testbed forecast daily so verification has fresh evidence to score.

    Without this, the nightly verification job has nothing new to compare
    unless an admin happens to click "Run beta forecast" that same day.
    """
    try:
        job = await asyncio.to_thread(trigger_beta_forecast, "scheduler")
        logger.info("Scheduled beta forecast triggered: job_id=%s", job.get("job_id"))
    except RuntimeError as error:
        # A beta forecast triggered manually (or by a prior tick) is still running.
        logger.info("Scheduled beta forecast skipped: %s", error)
    except Exception as error:
        logger.error("Scheduled beta forecast trigger failed: %s", error, exc_info=True)


async def verify_latest_beta_forecast_job():
    """Score Testbed outcomes before nightly archiving moves source observations."""
    try:
        report = await asyncio.to_thread(run_beta_verification)
        logger.info(
            "Beta verification completed: date=%s records=%s status=%s",
            report.get("date"), report.get("record_count"), report.get("status"),
        )
    except RuntimeError as error:
        # A beta forecast is intentionally optional. Missing or not-yet-mature
        # evidence should remain visible without failing an operational job.
        logger.info("Beta verification skipped: %s", error)
    except Exception as error:
        logger.error("Beta verification failed: %s", error, exc_info=True)


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


async def run_drift_check_job():
    """Evaluate feature/prediction drift across shadow-tracked model types."""
    try:
        await asyncio.to_thread(run_drift_check)
    except Exception as error:
        logger.error("Drift check failed: %s", error, exc_info=True)


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

async def purge_feedback_throttle_job():
    """Delete feedback rate-limit rows past the retention window - same cadence/pattern as fire reports'."""
    try:
        purged = await asyncio.to_thread(purge_feedback_throttle_rows)
        logger.info("Feedback throttle purge: purged=%s", purged)
    except Exception as error:
        logger.error("Feedback throttle purge failed: %s", error, exc_info=True)


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


async def burn_ban_maintenance_job():
    """Expire ended burn bans, purge old PII/throttle rows, and refresh the static map."""
    try:
        result = await asyncio.to_thread(run_burn_ban_maintenance)
        logger.info("Burn-ban maintenance: %s", result)
    except Exception as error:
        logger.error("Burn-ban maintenance failed: %s", error, exc_info=True)


def create_scheduler():
    central_tz = timezone('America/Chicago')
    return AsyncIOScheduler(timezone=central_tz)

def start_scheduler_jobs(scheduler: AsyncIOScheduler):
    scheduler.add_job(fetch_synoptic_data, 'interval', minutes=5, id='fetch_synoptic')
    scheduler.add_job(fetchtimeseriesdata, 'interval', minutes=5, seconds=60, id='fetch_timeseries')
    scheduler.add_job(fetch_and_store_raws_stations, 'interval', minutes=5, id='fetch_raws_stations')
    scheduler.add_job(
        refresh_testbed_observations_job,
        'interval',
        minutes=5,
        seconds=30,
        id='refresh_testbed_observations',
        max_instances=1,
        coalesce=True,
    )
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

    spread_rate_poll = spread_rate_poll_minutes()

    scheduler.add_job(
        rtma_spread_rate_pipeline_job,
        'interval',
        minutes=spread_rate_poll,
        id='rtma_spread_rate_pipeline',
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
        run_scheduled_beta_forecast_job,
        'cron',
        hour=9,
        minute=0,
        id='run_scheduled_beta_forecast',
        max_instances=1,
        coalesce=True,
    )

    scheduler.add_job(
        verify_latest_beta_forecast_job,
        'cron',
        hour=23,
        minute=40,
        id='verify_latest_beta_forecast',
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
        run_rtma_peak_job,
        'cron',
        hour=22,
        minute=20,
        id='end_of_day_rtma_peak',
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        refresh_testbed_rtma_job,
        'cron',
        hour=22,
        minute=25,
        id='refresh_testbed_rtma',
        max_instances=1,
        coalesce=True,
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
        purge_feedback_throttle_job,
        'cron',
        hour=2,
        minute=50,
        id='purge_feedback_throttle',
    )

    scheduler.add_job(
        purge_spatial_fm_uncertainty_cache_job,
        'cron',
        hour=3,
        minute=15,
        id='purge_spatial_fm_uncertainty_cache',
    )

    scheduler.add_job(
        run_drift_check_job,
        'cron',
        hour=4,
        minute=0,
        id='drift_check',
        max_instances=1,
        coalesce=True,
    )

    scheduler.add_job(
        burn_ban_maintenance_job,
        'interval',
        hours=3,
        id='burn_ban_maintenance',
        max_instances=1,
        coalesce=True,
    )

    scheduler.start()
    logger.info("Scheduler started")

async def run_initial_fetches():
    await fetch_synoptic_data()
    await fetchtimeseriesdata()
    await fetch_and_store_raws_stations()
    await refresh_testbed_observations_job()
    await fetch_and_store_afds()
