"""Which process runs which scheduled job.

`show-me-fire-api` used to run every APScheduler job inside the web process, so every deploy killed whatever long job
(forecast_v1, the hourly HRRR build, archives...) happened to be running. Jobs are now split into two groups:

  api    quick jobs, and jobs that keep in-memory state the web routes serve (station/timeseries/RAWS feeds, AFDs, alerts,
         drift state, pushes, purges, fire-detection ingest, WildCAD and NIFC listings, the watchdog). They stay in the API.
  heavy  long-running or CPU/disk/R2-heavy jobs (forecast_v1, HRRR hourly, RTMA, graphics, archiving, verification,
         retraining). They run in the scheduler worker container (`python -m core.scheduler_worker`), which an API deploy
         does not touch.

`SMF_SCHEDULER_GROUPS` picks the groups a process runs: `api,heavy` (the default, so dev, tests and an unsplit production
behave exactly as before), `api`, or `heavy`. JOB_GROUPS is the single source of truth: a test fails when a registered job
has no group, so a new job can never silently stop running.
"""
from __future__ import annotations

import logging
import os
from typing import Iterable

logger = logging.getLogger(__name__)

API, HEAVY = "api", "heavy"
ALL_GROUPS = (API, HEAVY)

HEAVY_JOBS = {
    # forecast_v1 and the HRRR / RTMA / MRMS pipelines
    "run_forecast_v1_shadow", "prune_forecast_v1_hot_storage", "forecast_v1_admin_requests",
    "hrrr_hourly_forecast", "rtma_spread_rate_pipeline", "refresh_rtma_analysis", "refresh_testbed_rtma",
    "end_of_day_rtma_peak", "capture_latest_mrms", "update_seasonal_fuel_state",
    # graphics that render in the process pool and upload to R2
    "publish_hourly_precipitation_graphics", "publish_precipitation_graphics", "refresh_burn_ban_static_map_daily",
    # archiving and exports
    "full_observation_capture", "end_of_day_archive", "daily_map_archive", "daily_day_archive", "legacy_archive",
    "archive_stale_ngfs_detections", "publish_detection_training_export", "storage_cleanup",
    # verification, monitoring and retraining
    "verify_latest_beta_forecast", "verify_v5_shadow", "verify_v4_shadow", "fire_danger_tracking",
    "post_promotion_monitor", "retrain_detection_confidence",
}

API_JOBS = {
    # live feeds whose results the API serves from memory, and the jobs that read those feeds
    "fetch_synoptic", "fetch_timeseries", "fetch_raws_stations", "fetch_buffer_stations", "fetch_station_precip",
    "fetch_station_et", "refresh_testbed_observations", "publish_gis_observations", "record_station_history",
    "fetch_afds", "fetch_active_mo_alerts", "refresh_spc_department_graphics", "refresh_usdm_department_graphics",
    # pushes, sandbox, admin auth and purges
    "deliver_scheduled_forecasts", "check_mobile_push_receipts", "check_staff_push_receipts", "purge_staff_push_records",
    "reset_idle_reviewer_sandbox", "purge_admin_auth_records", "purge_not_a_fire_incidents",
    "purge_mobile_push_delivery_records", "purge_fire_report_pii", "purge_feedback_throttle",
    "purge_spatial_fm_uncertainty_cache", "purge_old_logs",
    # fire detections, incidents and the dispatch feeds
    "fetch_fire_detections", "fetch_advanced_fire_detections", "ingest_fire_detections", "official_incidents",
    "wildcad_incidents", "wildcad_resources", "wildcad_health", "recurring_source_detector", "burn_ban_maintenance",
    "ground_cover_shadow",
    # fills in-process state that routers read (drift_monitor._state); alerts if the heavy worker dies
    "drift_check", "forecast_watchdog", "scheduler_worker_watch",
}

JOB_GROUPS: dict[str, str] = {**{job: API for job in API_JOBS}, **{job: HEAVY for job in HEAVY_JOBS}}

# Jobs registered only when an env flag is on (so the full list needs those flags forced on).
CONDITIONAL_FLAGS = {
    "SMF_FORECAST_V1_ENABLED": "true",
    "SMF_GRAPHICS_SPC_AUTO_REFRESH": "true",
    "SMF_DETECTION_R2_PUBLISH": "true",
}


def groups_from_env(value: str | None = None) -> set[str]:
    """`SMF_SCHEDULER_GROUPS` ("api", "heavy" or "api,heavy"; default both). Unknown names are ignored with a warning."""
    raw = os.getenv("SMF_SCHEDULER_GROUPS", "api,heavy") if value is None else value
    groups = {part.strip().lower() for part in raw.split(",") if part.strip()}
    unknown = groups - set(ALL_GROUPS)
    if unknown:
        logger.warning("Ignoring unknown scheduler groups: %s", ", ".join(sorted(unknown)))
    return (groups & set(ALL_GROUPS)) or set(ALL_GROUPS)


class GroupFilteredScheduler:
    """Wraps a scheduler so `add_job` only registers jobs in the enabled groups, leaving every add_job call untouched."""

    def __init__(self, scheduler, groups: Iterable[str]):
        self._scheduler = scheduler
        self.groups = set(groups)
        self.registered: list[str] = []
        self.skipped: list[str] = []
        self.unclassified: list[str] = []

    def add_job(self, func, *args, id=None, **kwargs):
        group = JOB_GROUPS.get(id)
        if group is None:
            self.unclassified.append(str(id))
            group = API   # run it somewhere rather than nowhere; the unit test turns this into a failure
            logger.error("Scheduled job %r has no group in core/job_groups.py; running it in the api group", id)
        if group not in self.groups:
            self.skipped.append(str(id))
            return None
        self.registered.append(str(id))
        return self._scheduler.add_job(func, *args, id=id, **kwargs)

    def __getattr__(self, name):
        return getattr(self._scheduler, name)


def group_of(job_id: str) -> str:
    return JOB_GROUPS.get(job_id, API)
