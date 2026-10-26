import asyncio
import functools
import os
import logging
from datetime import datetime
from pathlib import Path
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pytz import timezone
from core.executors import get_process_pool, get_rtma_job_lock, run_in_process_pool_async
from services.synoptic import fetch_synoptic_data, fetch_raws_stations_multi_state, get_station_data, get_buffer_station_data, fetch_buffer_stations
from services.timeseries import fetchtimeseriesdata
from services.station_precip import fetch_station_precip
from services.station_et import fetch_station_et
from tools.ngfs_ogc_firedetect import main as firedetect
from tools.firedetections import main as fetch_advanced_fire_detections
from alerts.activemoalerts import run_active_mo_alerts
from services.afds import ingest_latest_afds
from services.archive_bundler import run_end_of_day_archive, run_ngfs_raw_archive
from services import map_archive
from services import day_archive, full_obs_archive
from services.hrrr_hourly import scheduler_job as hrrr_hourly_job
from services.rtma_capture import cleanup_rtma_cache, fetch_rtma, latest_complete_hour, spread_rate_poll_minutes
from services.mrms_capture import cleanup_mrms_cache, fetch_mrms, mrms_enabled
from services.mobile_push import check_push_receipts, purge_delivery_records
from core.config import AFD_POLL_MINUTES
from services.v5_verification import verify_pending as verify_v5_shadow
from services.v4_verification import verify_pending as verify_v4_shadow
from services.drift_monitor import run_drift_check
from services.fire_ingest import ingest_detection_files
from services.recurring_source_detector import run_recurring_source_scan
from core.database import (
    expire_unmoderated_fire_reports, purge_fire_submission_pii, purge_fire_throttle_rows,
    purge_feedback_throttle_rows,
)
from services.spatial_fm_uncertainty_cache import purge_stale as purge_spatial_fm_uncertainty_cache
from services.log_maintenance import purge_old_logs
from services import storage_cleanup
from services import job_runs
from services import rtma_analysis
from services.seasonal_fuel_state import update_daily_gdd
from services.rtma_peak import generate_rtma_peak, run_rtma_peak_job
from services.spread_rate import run_spread_rate_job, run_spread_rate_pipeline
from routers.burn_bans import run_burn_ban_maintenance
from services.beta_products import BETA_ROOT, load_manifest, refresh_observation_products, save_manifest
from services.beta_verification import run_beta_verification
from services.forecast_v1_job import prune_forecast_v1_hot_storage, run_forecast_v1_operational, run_forecast_v1_shadow
from scripts.monitor_model_rollout import monitor_all
from services.gis_vectors import publish_fire_detections, publish_weather_stations
from services.spc_graphics_watcher import refresh_spc_graphics_job, refresh_usdm_graphics_job
from scripts.publish_precipitation_graphics import render as render_precipitation_graphics, upload as upload_precipitation_graphics

logger = logging.getLogger(__name__)


async def _render_in_pool(func, *args):
    """Run a scheduled job's matplotlib/cartopy drawing in the shared process pool instead of a thread of the API
    process, where it shared the API's CPU and GIL with request handling (the hourly precipitation set is ~9 s of CPU)
    and grew the API's own memory. Holds the grid-job lock: the precipitation render peaks around 650 MB, about what a
    grid job takes, and two of those at once have OOM-killed a pool worker before (see get_rtma_job_lock)."""
    async with get_rtma_job_lock():
        return await run_in_process_pool_async(func, *args)

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
        # A transient fetch failure must not wipe out the last known-good
        # RAWS list - this global feeds the testbed beta products and the
        # production weather-station GIS layer, both of which join on
        # fuel-moisture by station id, so clearing it to [] here turned any
        # single flaky poll into a 0-station blackout everywhere until the
        # next successful 5-minute cycle.
        raws_station_data["error"] = str(e)
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


async def publish_gis_observations_job():
    """Publish the latest production station observations for WMS/WFS."""
    try:
        await asyncio.to_thread(publish_weather_stations, get_station_data(), raws_station_data)
    except Exception as error:
        logger.error("GIS station publication failed: %s", error, exc_info=True)


async def record_station_history_job():
    """Log the latest station fetch (raw + derived fields) into
    maps/station_history.py's rolling table, so the current-conditions map
    catalog's "Today's Max/Min" and "N-hour Change" products have something
    to read. Reuses the already-cached get_station_data() feed - no extra
    Synoptic API call here."""
    try:
        from maps.station_history import prune_old, record_observations
        from services import derived_met
        from services.raws_calibration import fresh_station_fuel_moisture

        def _record():
            stations = get_station_data().get("stations") or []

            def _obs(name):
                return lambda s: s.get("observations", {}).get(name, {}).get("value")

            air_temp = _obs("air_temp")
            rh = _obs("relative_humidity")
            dewpoint = _obs("dew_point_temperature")
            wind_speed = _obs("wind_speed")

            def _feels_like(station):
                t, r, w = air_temp(station), rh(station), wind_speed(station)
                if t is None or r is None or w is None:
                    return None
                return derived_met.feels_like_f(t, r, w)

            def _wbgt(station):
                t, r, w = air_temp(station), rh(station), wind_speed(station)
                if t is None or r is None or w is None:
                    return None
                return derived_met.estimate_wbgt_f(t, r, w)

            fields = {
                # Backing the pressure, wind-gust and rainfall maps.
                "sea_level_pressure_mb": _obs("sea_level_pressure"),
                "wind_gust_mph": _obs("wind_gust"),
                "precip_since_midnight_in": _obs("precip_accum_since_local_midnight"),
                "air_temp_f": air_temp,
                "relative_humidity_pct": rh,
                "dew_point_temperature_f": dewpoint,
                "wind_speed_mph": wind_speed,
                "feels_like_f": _feels_like,
                "wbgt_f": _wbgt,
                # RAWS only; calibrates today's "what happened" fuel moisture (services/rtma_analysis.py).
                "fuel_moisture_pct": fresh_station_fuel_moisture,
            }
            record_observations(stations, fields)
            # Neighbor-state stations: pressure only, to anchor the 3-hour change maps at the border.
            buffer_stations = get_buffer_station_data().get("stations") or []
            record_observations(buffer_stations, {"sea_level_pressure_mb": _obs("sea_level_pressure")})
            prune_old()

        await asyncio.to_thread(_record)
    except Exception as error:
        logger.error("Station history recording failed: %s", error, exc_info=True)


async def refresh_testbed_rtma_job():
    """Build an isolated continuous-score RTMA peak after the production run."""
    try:
        async with get_rtma_job_lock():
            result = await run_in_process_pool_async(
                functools.partial(generate_rtma_peak, None, output_root=BETA_ROOT, experimental=True),
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
        async with get_rtma_job_lock():
            await run_in_process_pool_async(
                run_spread_rate_pipeline,
                raws_station_data if raws_station_data.get("stations") else None,
            )
        try:
            await asyncio.to_thread(cleanup_rtma_cache)
        except Exception as cleanup_error:
            logger.error("Spread-rate RTMA retention cleanup failed: %s", cleanup_error, exc_info=True)
    except Exception as error:
        logger.error("RTMA/spread-rate pipeline failed: %s", error, exc_info=True)


async def publish_hourly_precipitation_graphics_job():
    """Hourly refresh of the short-range accumulations (1/3/6/12/24-hour),
    which NOAA updates every hour; the multi-day totals stay on the twice-daily
    publish_precipitation_graphics_job."""
    try:
        from maps.precipitation_graphics import HOURLY_STEMS

        # Render everything before uploading, so a source or render error never replaces a published image.
        outputs = await _render_in_pool(render_precipitation_graphics, HOURLY_STEMS)
        published = await asyncio.to_thread(upload_precipitation_graphics, outputs)
        logger.info("Hourly precipitation graphics published: %s", published)
    except Exception as error:
        logger.error("Hourly precipitation graphics publish failed: %s", error, exc_info=True)


async def publish_precipitation_graphics_job():
    """Render and publish the six NOAA QPE precipitation graphics to R2.

    Previously a standalone `0 6,18 * * *` (CRON_TZ=America/Chicago) entry in
    /etc/cron.d - moved in-app so a missed firing (e.g. container restart
    around 18:00) shows up in the app's own logs/job state instead of being
    silently skipped with no record anywhere.
    """
    try:
        outputs = await _render_in_pool(render_precipitation_graphics)
        published = await asyncio.to_thread(upload_precipitation_graphics, outputs)
        logger.info("Precipitation graphics published: %s", published)
    except Exception as error:
        logger.error("Precipitation graphics publish failed: %s", error, exc_info=True)


async def run_forecast_v1_shadow_job():
    try:
        runner = run_forecast_v1_shadow if os.getenv("SMF_FORECAST_V1_SOURCE_MODE", "herbie").lower() == "staged" else run_forecast_v1_operational
        result = await asyncio.to_thread(runner)
        logger.info("Forecast-v1 shadow run completed: %s", result)
    except Exception as error:
        logger.error("Forecast-v1 shadow run failed: %s", error, exc_info=True)


async def prune_forecast_v1_hot_storage_job():
    try:
        result = await asyncio.to_thread(prune_forecast_v1_hot_storage)
        logger.info("Forecast-v1 retention completed: %s", result)
    except Exception as error:
        logger.error("Forecast-v1 retention failed: %s", error, exc_info=True)


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


async def capture_latest_mrms():
    """Cache the latest complete MRMS QPE when explicitly enabled."""
    if not mrms_enabled():
        return
    try:
        await asyncio.to_thread(fetch_mrms)
        await asyncio.to_thread(cleanup_mrms_cache)
    except Exception as error:
        logger.error("MRMS capture failed: %s", error, exc_info=True)


async def verify_v5_shadow_observations():
    """Attach mature observations without blocking the API event loop."""
    try:
        await asyncio.to_thread(verify_v5_shadow)
    except Exception as error:
        logger.error("V5 shadow verification failed: %s", error, exc_info=True)


async def verify_v4_shadow_observations():
    """Attach mature observations without blocking the API event loop.
    v4_shadow.py::attach_observations() existed since V4 shipped but had
    no caller at all until services/v4_verification.py was written - this
    is the first thing that actually invokes it."""
    try:
        await asyncio.to_thread(verify_v4_shadow)
    except Exception as error:
        logger.error("V4 shadow verification failed: %s", error, exc_info=True)


async def run_drift_check_job():
    """Evaluate feature/prediction drift across shadow-tracked model types."""
    try:
        await asyncio.to_thread(run_drift_check)
    except Exception as error:
        logger.error("Drift check failed: %s", error, exc_info=True)


async def recurring_source_detector_job():
    """Flag candidate recurring non-fire detection sources (mills, flares,
    kilns...) for admin review - never auto-confirms/suppresses anything."""
    try:
        await asyncio.to_thread(run_recurring_source_scan)
    except Exception as error:
        logger.error("Recurring source detector failed: %s", error, exc_info=True)


async def run_post_promotion_monitor_job():
    """Seven-day post-promotion guardrail: auto-rollback on live metric regression.

    Runs after validateForecast.sh's cron (04:30 UTC, i.e. 22:30-23:30 Central
    depending on DST) has written reports/validation_history.json, so today's
    live metrics are available to compare against the replaced model's recorded
    performance.
    """
    try:
        results = await asyncio.to_thread(monitor_all)
        for model_type, result in results.items():
            if result.get("action") == "rollback":
                logger.warning(
                    "Post-promotion monitor rolled back %s to %s: %s",
                    model_type, result.get("version"), result.get("reason"),
                )
            else:
                logger.info("Post-promotion monitor (%s): %s", model_type, result)
    except Exception as error:
        logger.error("Post-promotion rollout monitor failed: %s", error, exc_info=True)


async def ingest_fire_detections_job():
    """
    Backfill the fire_events store from the existing detection GeoJSON
    files. A separate job from the fetch jobs that write those files -
    a store failure here must never affect the file pipeline that
    /fires/satdet and the mobile app depend on.
    """
    try:
        await asyncio.to_thread(ingest_detection_files)
        # Give incidents that predate shape-based joining their join area.
        from core.database import backfill_incident_areas
        await asyncio.to_thread(backfill_incident_areas)
        # Fold side-by-side incidents of one fire together before anything
        # downstream (scoring, graphics, shapes) works on them.
        # Off until the dry-run report (scripts/merge_fire_incidents.py
        # --dry-run) has been reviewed: set FIRE_INCIDENT_MERGE_ENABLED=1.
        if os.getenv("FIRE_INCIDENT_MERGE_ENABLED", "").strip().lower() in {"1", "true", "yes"}:
            from services.incident_merger import merge_touching_incidents
            await asyncio.to_thread(merge_touching_incidents, since_hours=72)
        # Per-detection ML confidence (0-100%, detection-pattern features
        # only) - distinct from and complementary to the incident-cluster
        # confidence below, which scores a group of detections together.
        from services.detection_confidence import refresh_detection_confidence
        await asyncio.to_thread(refresh_detection_confidence)
        await asyncio.to_thread(publish_fire_detections)
        from services.fire_incident_graphics import refresh_incident_graphics
        await _render_in_pool(refresh_incident_graphics)
        from services.incident_shape_extractor import refresh_incident_shapes
        await asyncio.to_thread(refresh_incident_shapes)
        from services.fire_confidence import refresh_confidence_shapes
        await asyncio.to_thread(refresh_confidence_shapes)
        try:
            from services.fire_confidence import list_feedback_candidate_incidents
            from services.mobile_push import notify_feedback_candidates
            candidates = await asyncio.to_thread(list_feedback_candidate_incidents)
            await asyncio.to_thread(notify_feedback_candidates, candidates)
        except Exception as push_error:
            logger.error("Fire detection push notify failed: %s", push_error, exc_info=True)
    except Exception as error:
        logger.error("Fire detection ingest failed: %s", error, exc_info=True)


async def retrain_detection_confidence_job():
    """Nightly gated retrain for the per-detection confidence model (see
    detection-confidence-model/retrain.py) - runs standalone by direct file
    path, same pattern services/detection_confidence.py already uses to load
    the resulting model, to avoid the sys.path module-name collision with
    other model-training packages under api/ documented there."""
    try:
        import importlib.util

        module_path = Path(__file__).resolve().parent.parent / "detection-confidence-model" / "retrain.py"
        spec = importlib.util.spec_from_file_location("detection_confidence_retrain", module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = await asyncio.to_thread(module.retrain_and_gate)
        logger.info("Detection-confidence retrain: %s", result)
    except Exception as error:
        logger.error("Detection-confidence retrain failed: %s", error, exc_info=True)


async def purge_spatial_fm_uncertainty_cache_job():
    """Delete spatial FM uncertainty cache files older than the retention window."""
    try:
        removed = await asyncio.to_thread(purge_spatial_fm_uncertainty_cache)
        logger.info("Spatial FM uncertainty cache purge: removed=%s", removed)
    except Exception as error:
        logger.error("Spatial FM uncertainty cache purge failed: %s", error, exc_info=True)


async def purge_old_logs_job():
    """Rotate log files over 10 MB, delete ones untouched for 30 days, and prune old job-run history."""
    try:
        result = await asyncio.to_thread(purge_old_logs)
        pruned = await asyncio.to_thread(job_runs.prune_runs)
        logger.info("Log maintenance: rotated=%d removed=%d job_runs_pruned=%d",
                    len(result["rotated"]), len(result["removed"]), pruned)
    except Exception as error:
        logger.error("Log maintenance failed: %s", error, exc_info=True)


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


async def refresh_burn_ban_static_map_job():
    """Refresh the public burn-ban PNG and GIS publication every morning.

    Moderation and expiry changes continue to regenerate the map immediately;
    this scheduled run intentionally republishes even unchanged data so the
    static map's published timestamp remains current.
    """
    try:
        from services.burn_ban_map import generate_burn_ban_map

        result = await _render_in_pool(generate_burn_ban_map)
        logger.info("Daily burn-ban static-map refresh: %s", result)
    except Exception as error:
        logger.error("Daily burn-ban static-map refresh failed: %s", error, exc_info=True)


async def refresh_rtma_analysis_job():
    """Rebuild today's "what happened" RTMA analysis day for the data viewer, finish and publish yesterday's."""
    try:
        result = await asyncio.to_thread(rtma_analysis.refresh)
        logger.info("RTMA analysis days: built=%s published=%s pruned=%d",
                    result["built"], result["published"] or "none", len(result["pruned"]))
    except Exception as error:
        logger.error("RTMA analysis refresh failed: %s", error, exc_info=True)


def create_scheduler():
    central_tz = timezone('America/Chicago')
    # Create the process pool up front so its startup is logged with the scheduler's.
    get_process_pool()
    return AsyncIOScheduler(timezone=central_tz)

# Curated allowlist + human descriptions for
# routers/model_admin.py's GET /schedule endpoint (read-only introspection
# of the live scheduler). Only jobs relevant to model scoring/forecast
# generation are listed here - unrelated jobs (log purges, burn-ban
# maintenance, raw data pulls, etc.) are deliberately excluded, not just
# hidden client-side. Keep this in sync when adding/removing a
# models/forecast job below - job ids are hand-matched against add_job()'s
# own `id=` argument, there is no naming convention enforced automatically.
MODEL_RELEVANT_JOBS = {
    "verify_latest_beta_forecast": {
        "category": "verification",
        "description": "Nightly stable-vs-beta fuel_moisture verification against real observations "
                       "(services/beta_verification.py), feeding the Beta Operations Scorecard.",
    },
    "verify_v5_shadow": {
        "category": "shadow_verification",
        "description": "Attaches real observations to pending V5 shadow predictions and scores them "
                       "(services/v5_verification.py + shadow_observation_scoring.py).",
    },
    "verify_v4_shadow": {
        "category": "shadow_verification",
        "description": "Attaches real observations to pending V4 shadow predictions and scores them "
                       "(services/v4_verification.py + shadow_observation_scoring.py).",
    },
    "rtma_spread_rate_pipeline": {
        "category": "feature_pipeline",
        "description": "Builds the live RTMA-derived spread-rate features fire_weather_ml_shadow.py "
                       "scores against.",
    },
    "run_forecast_v1_shadow": {
        "category": "forecast_generation",
        "description": "Polls and scores the forecast_v1 NWP-blend pipeline (only registered when "
                       "SMF_FORECAST_V1_ENABLED=true).",
    },
    "prune_forecast_v1_hot_storage": {
        "category": "maintenance",
        "description": "Prunes forecast_v1's hot-storage retention window (only registered when "
                       "SMF_FORECAST_V1_ENABLED=true).",
    },
    "drift_check": {
        "category": "monitoring",
        "description": "Nightly feature/prediction drift check across active model types (services/drift_monitor.py).",
    },
    "post_promotion_monitor": {
        "category": "monitoring",
        "description": "Post-promotion rollout monitor across registered model families.",
    },
    "update_seasonal_fuel_state": {
        "category": "feature_pipeline",
        "description": "Updates the daily GDD/seasonal fuel-state accumulators several models depend on.",
    },
}


def start_scheduler_jobs(scheduler: AsyncIOScheduler):
    scheduler.add_job(fetch_synoptic_data, 'interval', minutes=5, id='fetch_synoptic')
    scheduler.add_job(fetchtimeseriesdata, 'interval', minutes=5, seconds=60, id='fetch_timeseries')
    scheduler.add_job(fetch_and_store_raws_stations, 'interval', minutes=5, id='fetch_raws_stations')
    scheduler.add_job(fetch_buffer_stations, 'interval', minutes=5, seconds=30, id='fetch_buffer_stations')
    scheduler.add_job(fetch_station_precip, 'interval', minutes=10, seconds=90, id='fetch_station_precip')
    scheduler.add_job(fetch_station_et, 'interval', minutes=15, seconds=45, id='fetch_station_et')
    scheduler.add_job(
        refresh_testbed_observations_job,
        'interval',
        minutes=5,
        seconds=30,
        id='refresh_testbed_observations',
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        publish_gis_observations_job,
        'interval',
        minutes=5,
        seconds=45,
        id='publish_gis_observations',
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        record_station_history_job,
        'interval',
        minutes=5,
        seconds=20,
        id='record_station_history',
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(fetch_and_store_afds, 'interval', minutes=AFD_POLL_MINUTES, id='fetch_afds')
    scheduler.add_job(
        run_active_mo_alerts,
        'interval',
        minutes=max(1, int(os.getenv("SMF_ALERT_POLL_MINUTES", "1"))),
        id='fetch_active_mo_alerts',
        max_instances=1,
        coalesce=True,
    )
    if os.getenv("SMF_GRAPHICS_SPC_AUTO_REFRESH", "true").lower() == "true":
        scheduler.add_job(
            refresh_spc_graphics_job,
            'interval',
            minutes=max(1, int(os.getenv("SMF_GRAPHICS_SPC_POLL_MINUTES", "2"))),
            id='refresh_spc_department_graphics',
            max_instances=1,
            coalesce=True,
        )
    # The Drought Monitor releases Thursdays ~8:30 AM ET; poll every 30 min from
    # 9:00 AM to 5:30 PM ET to catch late releases. Unchanged data re-renders nothing.
    scheduler.add_job(
        refresh_usdm_graphics_job,
        'cron',
        day_of_week='thu', hour='9-17', minute='0,30', timezone='America/New_York',
        id='refresh_usdm_department_graphics',
        max_instances=1,
        coalesce=True,
    )
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
        # GOES/NGFS scans continuously (geostationary), unlike the
        # polar-orbiting VIIRS/MODIS passes fetch_advanced_fire_detections
        # is limited to below - no daylight/hour restriction needed here.
        minute='0,10,20,30,40,50',
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
    # After the 15-minute RTMA capture has had time to pick up the hour (RTMA posts ~30-50 minutes after it).
    scheduler.add_job(
        refresh_rtma_analysis_job,
        'cron',
        minute=55,
        id='refresh_rtma_analysis',
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        capture_latest_mrms,
        'interval',
        minutes=15,
        id='capture_latest_mrms',
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
        publish_hourly_precipitation_graphics_job,
        'cron',
        minute=12,  # NOAA posts the hourly QPE a few minutes past the hour
        id='publish_hourly_precipitation_graphics',
        max_instances=1,
        coalesce=True,
    )
    scheduler.add_job(
        publish_precipitation_graphics_job,
        'cron',
        hour='6,18',
        minute=0,
        id='publish_precipitation_graphics',
        max_instances=1,
        coalesce=True,
    )

    if os.getenv("SMF_FORECAST_V1_ENABLED", "false").lower() == "true":
        # Poll after the configured cycle-age gate. Completed run IDs are
        # idempotently skipped; incomplete NOAA feeds are retried on the next
        # tick without replacing the current public pointer.
        scheduler.add_job(
            run_forecast_v1_shadow_job,
            "interval",
            minutes=max(15, int(os.getenv("SMF_FORECAST_V1_POLL_MINUTES", "30"))),
            id="run_forecast_v1_shadow",
            max_instances=1,
            coalesce=True,
        )
        scheduler.add_job(
            prune_forecast_v1_hot_storage_job,
            "cron",
            hour=3,
            minute=20,
            id="prune_forecast_v1_hot_storage",
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
        'interval',
        minutes=15,
        id='end_of_day_archive',
        max_instances=1,
        coalesce=True
    )

    scheduler.add_job(
        run_ngfs_raw_archive,
        'cron',
        hour=3,
        minute=15,
        id='archive_stale_ngfs_detections',
        max_instances=1,
        coalesce=True
    )

    # Zip each finished day's 15-minute CDN map copies, prune them after 14
    # days and tell staff on Discord when zips are ready (services/map_archive.py).
    scheduler.add_job(
        map_archive.run_scheduled,
        'cron',
        hour=2,
        minute=45,
        id='daily_map_archive',
        max_instances=1,
        coalesce=True
    )

    # All-network full-day observations for the finished Central day (01:00), then the
    # day bundle (03:30, after the map zips at 02:45): everything needed to reconstruct
    # a day goes to R2; reminders and expiry run in the same pass (services/day_archive.py).
    scheduler.add_job(
        full_obs_archive.run_scheduled,
        'cron',
        hour=1,
        minute=0,
        id='full_observation_capture',
        max_instances=1,
        coalesce=True
    )
    # Hour-by-hour HRRR product from every hourly HRRR run, built one forecast hour at a time (services/hrrr_hourly/
    # progressive.py): each tick adds whatever hours HRRR has published since the last one (f00 comes out about 51 minutes
    # after the cycle, f18 at about 1h14m) and publishes them, so the viewer fills in as HRRR does. A tick with nothing new
    # is a cheap no-op; the work itself runs in a subprocess.
    scheduler.add_job(
        hrrr_hourly_job.run_scheduled,
        'cron',
        minute='*/3',
        id='hrrr_hourly_forecast',
        max_instances=1,
        coalesce=True
    )
    scheduler.add_job(
        day_archive.run_scheduled,
        'cron',
        hour=3,
        minute=30,
        id='daily_day_archive',
        max_instances=1,
        coalesce=True
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
        verify_v4_shadow_observations,
        'interval',
        hours=3,
        id='verify_v4_shadow',
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
        retrain_detection_confidence_job,
        'cron',
        hour=3,
        minute=0,
        id='retrain_detection_confidence',
    )

    scheduler.add_job(
        purge_spatial_fm_uncertainty_cache_job,
        'cron',
        hour=3,
        minute=15,
        id='purge_spatial_fm_uncertainty_cache',
    )

    scheduler.add_job(
        purge_old_logs_job,
        'cron',
        hour=3,
        minute=45,
        id='purge_old_logs',
    )

    # Safety net for disk space: finished forecast_v1 downloads, stale staging folders, low-disk alert.
    scheduler.add_job(
        storage_cleanup.run_scheduled,
        'cron',
        minute=35,
        id='storage_cleanup',
        max_instances=1,
        coalesce=True,
    )

    # Moves the legacy zips to R2 (verified, then deleted locally) while the folder is mounted; a no-op once it is empty.
    scheduler.add_job(
        storage_cleanup.run_legacy_archive_scheduled,
        'cron',
        minute=5,
        id='legacy_archive',
        max_instances=1,
        coalesce=True,
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
        recurring_source_detector_job,
        'cron',
        hour=4,
        minute=20,
        id='recurring_source_detector',
        max_instances=1,
        coalesce=True,
    )

    scheduler.add_job(
        run_post_promotion_monitor_job,
        'cron',
        hour=0,
        minute=30,
        id='post_promotion_monitor',
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

    # Scheduler timezone is America/Chicago, including daylight-saving time.
    # This refreshes the static PNG/GIS timestamp even when no ban changed.
    scheduler.add_job(
        refresh_burn_ban_static_map_job,
        'cron',
        hour=7,
        minute=0,
        id='refresh_burn_ban_static_map_daily',
        max_instances=1,
        coalesce=True,
    )

    # Record every run for the admin Logs page (services/job_runs.py). Must stay after the last add_job.
    tracked = job_runs.track_scheduler_jobs(scheduler)
    scheduler.start()
    logger.info("Scheduler started with %d jobs", tracked)

async def run_initial_fetches():
    await fetch_synoptic_data()
    await fetchtimeseriesdata()
    await fetch_and_store_raws_stations()
    await refresh_testbed_observations_job()
    await fetch_and_store_afds()
