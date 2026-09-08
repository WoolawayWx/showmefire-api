"""Opt-in scheduler bridge for the isolated 72-hour forecast pipeline."""
from __future__ import annotations

import gc
import json
import logging
import os
import threading
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from core.database import get_db_path
from services.synoptic import get_station_data
from forecast_v1.acquisition import acquire_cycle, latest_publishable_12z
from forecast_v1.adapters import ADAPTERS
from forecast_v1.contracts import run_id_for_cycle
from forecast_v1.engine import blend_sources
from forecast_v1.pipeline import (
    archive_source_cube,
    extract_source_member_rows,
    initialize_fuel_moisture,
    publish_run,
    regrid_to_public,
    summarize_ensemble_for_public,
)
from forecast_v1.r2_store import ForecastR2Store
from forecast_v1.repository import ensure_schema, prune_hot_storage, transaction
from core.config import FORECAST_V1_DIR


JOB_STATE_PATH = FORECAST_V1_DIR / "admin-job.json"
logger = logging.getLogger(__name__)
_job_lock = threading.Lock()
_execution_lock = threading.Lock()
PHASE_LABELS = {
    "queued": "Waiting to start",
    "preparing": "Preparing run",
    "acquiring_sources": "Downloading forecast sources",
    "archiving_sources": "Archiving source data",
    "station_extraction": "Extracting station forecasts",
    "regridding": "Building the 3 km grid",
    "blending": "Blending forecast models",
    "fuel_initialization": "Initializing fuel moisture",
    "forecast_cube": "Calculating fire weather",
    "hourly_rasters": "Writing hourly map layers",
    "daily_products": "Rendering daily maps",
    "station_products": "Writing station products",
    "uploading": "Uploading forecast assets",
    "finalizing": "Finalizing publication",
    "completed": "Run completed",
    "failed": "Run failed",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _read_job() -> dict:
    try:
        return json.loads(JOB_STATE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"status": "idle"}


def _write_job(job: dict) -> None:
    JOB_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = JOB_STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, JOB_STATE_PATH)


def _report_progress(phase: str, message: str, percent: int, **details) -> dict:
    """Atomically publish worker progress for scheduled and manual runs."""
    job = _read_job()
    now = _now()
    event = {
        "time": now, "phase": phase, "label": PHASE_LABELS.get(phase, phase.replace("_", " ").title()),
        "message": message, "percent": max(0, min(100, int(percent))),
    }
    events = list(job.get("events") or [])
    if not events or any(events[-1].get(key) != event.get(key) for key in ("phase", "message", "percent")):
        events.append(event)
    job.update(
        status="running", phase=phase, phase_label=event["label"], message=message,
        progress_percent=event["percent"], progress=details, updated_at=now, events=events[-30:],
    )
    job.setdefault("started_at", now)
    _write_job(job)
    return job


def _finish_job(status: str, *, result: dict | None = None, error: BaseException | None = None) -> dict:
    job = _read_job()
    now = _now()
    phase = "completed" if status == "completed" else "failed"
    message = "Forecast products are ready" if status == "completed" else str(error or "Forecast run failed")
    event = {"time": now, "phase": phase, "label": PHASE_LABELS[phase], "message": message, "percent": 100 if status == "completed" else int(job.get("progress_percent") or 0)}
    events = list(job.get("events") or [])
    events.append(event)
    job.update(
        status=status, phase=phase, phase_label=event["label"], message=message,
        progress_percent=event["percent"], updated_at=now, finished_at=now, events=events[-30:],
    )
    if result is not None:
        job["result"] = result
    if error is not None:
        job.update(error=str(error), error_type=type(error).__name__)
    _write_job(job)
    return job


def _publication_progress(update: dict) -> None:
    fraction = max(0.0, min(1.0, float(update.get("fraction", 0))))
    details = {key: update[key] for key in ("current", "total", "item") if key in update}
    _report_progress(update["phase"], update["message"], 78 + round(20 * fraction), **details)


def _acquisition_progress(update: dict) -> None:
    source_index = max(1, int(update.get("source_index", 1)))
    source_total = max(1, int(update.get("source_total", 1)))
    member_total = max(1, int(update.get("total", 1)))
    member_completed = int(update.get("completed", 0))
    source_fraction = member_completed / member_total
    if update.get("event") in {"source_completed", "source_skipped", "source_failed"}:
        source_fraction = 1.0
    fraction = ((source_index - 1) + source_fraction) / source_total
    model = str(update.get("model", "source")).upper()
    event = update.get("event")
    if event == "member_completed":
        message = f"Downloading {model} member {update.get('member')} ({member_completed}/{member_total})"
    elif event == "source_completed":
        message = f"{model} download complete"
    elif event == "source_skipped":
        message = f"Skipped {model}: {update.get('reason', 'unavailable')}"
    elif event == "source_failed":
        message = f"{model} unavailable: {update.get('reason', 'download failed')}"
    else:
        message = f"Starting {model} download"
    _report_progress(
        "acquiring_sources", message, 5 + round(45 * fraction), model=model,
        current=member_completed, total=member_total, sourceCurrent=source_index, sourceTotal=source_total,
    )


def _run_with_memory_limit(func, *args):
    """Apply a reversible address-space ceiling inside the forecast worker."""
    try:
        import resource
    except ImportError:  # pragma: no cover - production workers are Linux
        return func(*args)
    limit_gb = float(os.getenv("SMF_FORECAST_V1_MEMORY_LIMIT_GB", "8"))
    if limit_gb <= 0:
        raise ValueError("SMF_FORECAST_V1_MEMORY_LIMIT_GB must be greater than zero")
    requested = int(limit_gb * 1024**3)
    previous = resource.getrlimit(resource.RLIMIT_AS)
    hard_limit = previous[1]
    effective = requested if hard_limit == resource.RLIM_INFINITY else min(requested, hard_limit)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (effective, hard_limit))
    except (OSError, ValueError):  # macOS does not enforce RLIMIT_AS reliably
        logger.warning("Forecast worker memory ceiling is unsupported on this platform")
        return func(*args)
    try:
        return func(*args)
    finally:
        resource.setrlimit(resource.RLIMIT_AS, previous)


def _run_forecast_v1_staged_impl(make_public: bool | None = None) -> dict:
    _report_progress("preparing", "Reading the staged forecast inputs", 3)
    source_root = Path(os.getenv("SMF_FORECAST_V1_SOURCE_DIR", "data/forecast-v1-input"))
    publish_root = FORECAST_V1_DIR
    cycle_file = source_root / "cycle.json"
    stations_file = source_root / "stations.json"
    if not cycle_file.is_file() or not stations_file.is_file():
        raise RuntimeError("forecast-v1 staged cycle.json and stations.json are required")
    cycle_payload = json.loads(cycle_file.read_text(encoding="utf-8"))
    cycle = datetime.fromisoformat(cycle_payload["cycleTime"].replace("Z", "+00:00")).astimezone(timezone.utc)
    if cycle.hour != 12:
        raise RuntimeError("forecast-v1 scheduler will publish only a 12Z staged cycle")
    stations = json.loads(stations_file.read_text(encoding="utf-8"))
    native_cubes = {}
    r2 = ForecastR2Store()
    staged_models = ("hrrr", "rrfs", "refs", "gefs")
    for model_index, model in enumerate(staged_models, 1):
        path = source_root / f"{model}.nc"
        if not path.is_file():
            continue
        _report_progress("archiving_sources", f"Validating and archiving {model.upper()}", 10 + round(40 * model_index / len(staged_models)), model=model.upper(), current=model_index, total=len(staged_models))
        native_cubes[model] = archive_source_cube(ADAPTERS[model]().open(path, cycle), publish_root, r2)
    run_id = run_id_for_cycle(cycle)
    _report_progress("station_extraction", "Extracting full-member forecasts at station locations", 55, current=0, total=len(stations))
    source_member_rows = extract_source_member_rows(native_cubes, stations, run_id)
    cubes = {}
    models = list(native_cubes)
    for model_index, model in enumerate(models, 1):
        _report_progress("regridding", f"Regridding {model.upper()} to the public 3 km grid", 58 + round(16 * model_index / len(models)), model=model.upper(), current=model_index, total=len(models))
        native = native_cubes.pop(model)
        cubes[model] = regrid_to_public(summarize_ensemble_for_public(native))
    gc.collect()
    manifest = publish_run(
        cubes, stations, initial_fuel_moisture=float(cycle_payload.get("fallbackFuelMoisture", 12.0)),
        publish_root=publish_root,
        make_public=os.getenv("SMF_FORECAST_V1_PUBLIC", "false").lower() == "true" if make_public is None else make_public,
        r2_store=r2,
        source_member_rows=source_member_rows,
        progress_callback=_publication_progress,
    )
    return {"run_id": manifest["runId"], "warnings": manifest["warnings"]}


def _exclusive_run(func, *args) -> dict:
    """Run `func(*args)` on the shared process pool while holding the
    single-run lock for the full duration (matching the old in-process
    semantics) - so the memory/CPU cost of Herbie downloads + regridding
    lands on an isolated worker process instead of the API server itself.
    """
    if not _execution_lock.acquire(blocking=False):
        raise RuntimeError("A scheduled or manually requested forecast-v1 run is already active")
    try:
        from core.executors import run_in_process_pool

        job = _read_job()
        if job.get("status") not in {"queued", "running"}:
            now = _now()
            job = {
                "job_id": uuid.uuid4().hex, "status": "running", "requested_by": "scheduler",
                "requested_at": now, "started_at": now,
                "source_mode": os.getenv("SMF_FORECAST_V1_SOURCE_MODE", "herbie"), "publication": "shadow",
            }
            _write_job(job)
        _report_progress("preparing", "Starting isolated forecast worker", 1)
        job_timeout = int(os.getenv("SMF_FORECAST_V1_JOB_TIMEOUT_SECONDS", "14400"))
        try:
            result = run_in_process_pool(_run_with_memory_limit, func, *args, timeout=job_timeout)
        except Exception as error:
            _finish_job("failed", error=error)
            raise
        _finish_job("completed", result=result)
        return result
    finally:
        _execution_lock.release()


def run_forecast_v1_shadow(make_public: bool | None = None) -> dict:
    return _exclusive_run(_run_forecast_v1_staged_impl, make_public)


def _station_inputs(raw_stations: list[dict] | None) -> tuple[list[dict], list[dict]]:
    """Convert a live Synoptic snapshot to public metadata and fuel observations."""
    raw = raw_stations or []
    if not raw:
        raise RuntimeError("live RAWS/ASOS/AWOS station metadata is unavailable")
    stations: list[dict] = []
    observations: list[dict] = []
    for item in raw:
        station_id = item.get("stid") or item.get("id")
        if not station_id or item.get("latitude") is None or item.get("longitude") is None:
            continue
        network = str(item.get("network") or "other").upper()
        station = {
            "station_id": str(station_id), "name": item.get("name") or str(station_id),
            "latitude": float(item["latitude"]), "longitude": float(item["longitude"]),
            "network_type": network if network in {"RAWS", "ASOS", "AWOS"} else "other",
            "elevation_m": float(item["elevation"]) * 0.3048 if item.get("elevation") is not None else None,
            "timezone": item.get("timezone") or "America/Chicago", "is_active": item.get("status") != "INACTIVE",
            "sensor_capabilities": item.get("sensors") or {},
        }
        stations.append(station)
        if network != "RAWS":
            continue
        fuel = (item.get("observations") or {}).get("fuel_moisture") or {}
        if fuel.get("value") is not None and fuel.get("time"):
            observations.append({
                "station_id": str(station_id), "latitude": station["latitude"], "longitude": station["longitude"],
                "network": "RAWS", "observed_at_utc": fuel["time"], "fuel_moisture": fuel["value"],
                "qc_state": "fail" if item.get("qc_flagged") else "pass",
            })
    if not stations:
        raise RuntimeError("no usable live stations are available")
    return stations, observations


def _run_forecast_v1_operational_impl(
    now: datetime | None = None, make_public: bool | None = None, raw_stations: list[dict] | None = None,
) -> dict:
    """Acquire the latest mature 12Z cycle with Herbie and publish one beta run."""
    _report_progress("preparing", "Selecting the latest mature 12Z forecast cycle", 2)
    minimum_age = int(os.getenv("SMF_FORECAST_V1_MIN_CYCLE_AGE_HOURS", "6"))
    cycle = latest_publishable_12z(now, minimum_age_hours=minimum_age)
    run_id = run_id_for_cycle(cycle)
    database = get_db_path()
    ensure_schema(database)
    with transaction(database) as connection:
        existing = connection.execute(
            "SELECT status FROM forecast_runs WHERE run_id=? AND status IN ('complete','superseded')", (run_id,)
        ).fetchone()
    if existing:
        return {"run_id": run_id, "status": "already_complete", "warnings": []}

    publish_root = FORECAST_V1_DIR
    cache_root = publish_root / "download-cache" / cycle.strftime("%Y%m%d%H")
    _report_progress("acquiring_sources", f"Starting source acquisition for {cycle:%Y-%m-%d %H}Z", 5, cycleTime=cycle.isoformat().replace("+00:00", "Z"))
    result = acquire_cycle(cycle, cache_root, progress_callback=_acquisition_progress)
    r2 = ForecastR2Store()
    stations, observations = _station_inputs(raw_stations)
    native_cubes = {}
    source_items = list(result.cubes.items())
    for model_index, (model, source) in enumerate(source_items, 1):
        _report_progress("archiving_sources", f"Packing and archiving {model.upper()}", 51 + round(10 * model_index / len(source_items)), model=model.upper(), current=model_index, total=len(source_items))
        flags = tuple(sorted(set(source.quality_flags).union(result.warnings)))
        native_cubes[model] = archive_source_cube(replace(source, quality_flags=flags), publish_root, r2)
    result.cubes.clear()
    _report_progress("station_extraction", "Extracting full-member forecasts at station locations", 63, current=0, total=len(stations))
    source_member_rows = extract_source_member_rows(native_cubes, stations, run_id)
    cubes = {}
    models = list(native_cubes)
    for model_index, model in enumerate(models, 1):
        _report_progress("regridding", f"Regridding {model.upper()} to the public 3 km grid", 64 + round(9 * model_index / len(models)), model=model.upper(), current=model_index, total=len(models))
        native = native_cubes.pop(model)
        cubes[model] = regrid_to_public(summarize_ensemble_for_public(native))
    gc.collect()
    _report_progress("blending", "Blending available models across all 73 hours", 75)
    atmosphere = blend_sources(cubes)
    _report_progress("fuel_initialization", "Initializing fuel moisture from recent station observations", 77)
    initialization = initialize_fuel_moisture(atmosphere, cycle, observations)
    manifest = publish_run(
        cubes, stations, initial_fuel_moisture=initialization, publish_root=publish_root,
        make_public=os.getenv("SMF_FORECAST_V1_PUBLIC", "false").lower() == "true" if make_public is None else make_public,
        r2_store=r2,
        source_member_rows=source_member_rows,
        progress_callback=_publication_progress,
    )
    return {"run_id": manifest["runId"], "status": "complete", "warnings": manifest["warnings"]}


def run_forecast_v1_operational(now: datetime | None = None, make_public: bool | None = None) -> dict:
    # Snapshot the live station cache here, in the caller's process, and hand
    # it to the worker as plain data - the worker is a long-lived forked
    # process pool member and would otherwise only ever see synoptic.py's
    # in-memory cache as it was at pool startup.
    raw_stations = get_station_data().get("stations") or []
    return _exclusive_run(_run_forecast_v1_operational_impl, now, make_public, raw_stations)


def prune_forecast_v1_hot_storage() -> dict:
    r2 = ForecastR2Store()
    return prune_hot_storage(FORECAST_V1_DIR, archive_verified=r2.configured)


def _run_admin_forecast(job: dict) -> None:
    job.update(status="running", phase="preparing", phase_label=PHASE_LABELS["preparing"], message="Starting forecast worker", progress_percent=0, started_at=_now(), updated_at=_now(), events=[])
    _write_job(job)
    try:
        runner = run_forecast_v1_shadow if os.getenv("SMF_FORECAST_V1_SOURCE_MODE", "herbie").lower() == "staged" else run_forecast_v1_operational
        # Browser controls are deliberately shadow-only. Public promotion
        # remains an environment/deployment decision after verification.
        runner(make_public=False)
    except Exception as error:
        current = _read_job()
        if current.get("status") != "failed":
            _finish_job("failed", error=error)


def trigger_forecast_v1(requested_by: str) -> dict:
    """Start one admin-requested shadow run without blocking the API worker."""
    with _job_lock:
        if _execution_lock.locked():
            raise RuntimeError("A scheduled forecast-v1 run is already active")
        current = _read_job()
        if current.get("status") in {"queued", "running"}:
            timestamp = current.get("started_at") or current.get("requested_at")
            try:
                started = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00")).astimezone(timezone.utc)
                active = (datetime.now(timezone.utc) - started).total_seconds() < int(os.getenv("SMF_FORECAST_V1_JOB_TIMEOUT_SECONDS", "14400"))
            except (TypeError, ValueError):
                active = False
            if active:
                raise RuntimeError("A forecast-v1 run is already active")
        job = {
            "job_id": uuid.uuid4().hex, "status": "queued", "requested_by": requested_by,
            "requested_at": _now(), "source_mode": os.getenv("SMF_FORECAST_V1_SOURCE_MODE", "herbie"),
            "publication": "shadow", "phase": "queued", "phase_label": PHASE_LABELS["queued"],
            "message": "Waiting for an isolated forecast worker", "progress_percent": 0, "events": [],
        }
        _write_job(job)
        threading.Thread(target=_run_admin_forecast, args=(job,), daemon=True, name="forecast-v1-admin").start()
        return job


def get_forecast_v1_job_status() -> dict:
    job = _read_job()
    job["execution_active"] = _execution_lock.locked()
    if job.get("status") not in {"queued", "running"} and job["execution_active"]:
        job["status"] = "scheduled_running"
    started = job.get("started_at") or job.get("requested_at")
    try:
        start_time = datetime.fromisoformat(str(started).replace("Z", "+00:00")).astimezone(timezone.utc)
        end_time = datetime.now(timezone.utc) if job.get("status") in {"queued", "running", "scheduled_running"} else datetime.fromisoformat(str(job.get("finished_at")).replace("Z", "+00:00")).astimezone(timezone.utc)
        job["elapsed_seconds"] = max(0, int((end_time - start_time).total_seconds()))
    except (TypeError, ValueError):
        job["elapsed_seconds"] = None
    job["memory_limit_gb"] = float(os.getenv("SMF_FORECAST_V1_MEMORY_LIMIT_GB", "8"))
    return job
