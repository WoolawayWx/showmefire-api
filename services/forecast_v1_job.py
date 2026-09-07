"""Opt-in scheduler bridge for the isolated 72-hour forecast pipeline."""
from __future__ import annotations

import json
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
from forecast_v1.pipeline import archive_source_cube, initialize_fuel_moisture, publish_run, regrid_to_public
from forecast_v1.r2_store import ForecastR2Store
from forecast_v1.repository import ensure_schema, prune_hot_storage, transaction
from core.config import FORECAST_V1_DIR


JOB_STATE_PATH = FORECAST_V1_DIR / "admin-job.json"
_job_lock = threading.Lock()
_execution_lock = threading.Lock()


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


def _run_forecast_v1_staged_impl(make_public: bool | None = None) -> dict:
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
    cubes = {}
    r2 = ForecastR2Store()
    for model in ("hrrr", "rrfs", "refs", "gefs"):
        path = source_root / f"{model}.nc"
        if not path.is_file():
            continue
        native = archive_source_cube(ADAPTERS[model]().open(path, cycle), publish_root, r2)
        cubes[model] = regrid_to_public(native)
    stations = json.loads(stations_file.read_text(encoding="utf-8"))
    manifest = publish_run(
        cubes, stations, initial_fuel_moisture=float(cycle_payload.get("fallbackFuelMoisture", 12.0)),
        publish_root=publish_root,
        make_public=os.getenv("SMF_FORECAST_V1_PUBLIC", "false").lower() == "true" if make_public is None else make_public,
        r2_store=r2,
    )
    return {"run_id": manifest["runId"], "warnings": manifest["warnings"]}


def _exclusive_run(callback) -> dict:
    if not _execution_lock.acquire(blocking=False):
        raise RuntimeError("A scheduled or manually requested forecast-v1 run is already active")
    try:
        return callback()
    finally:
        _execution_lock.release()


def run_forecast_v1_shadow(make_public: bool | None = None) -> dict:
    return _exclusive_run(lambda: _run_forecast_v1_staged_impl(make_public))


def _station_inputs() -> tuple[list[dict], list[dict]]:
    """Convert the live Synoptic snapshot to public metadata and fuel observations."""
    raw = get_station_data().get("stations") or []
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


def _run_forecast_v1_operational_impl(now: datetime | None = None, make_public: bool | None = None) -> dict:
    """Acquire the latest mature 12Z cycle with Herbie and publish one beta run."""
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
    result = acquire_cycle(cycle, cache_root)
    r2 = ForecastR2Store()
    cubes = {}
    for model, source in result.cubes.items():
        flags = tuple(sorted(set(source.quality_flags).union(result.warnings)))
        native = archive_source_cube(replace(source, quality_flags=flags), publish_root, r2)
        cubes[model] = regrid_to_public(native)
    stations, observations = _station_inputs()
    atmosphere = blend_sources(cubes)
    initialization = initialize_fuel_moisture(atmosphere, cycle, observations)
    manifest = publish_run(
        cubes, stations, initial_fuel_moisture=initialization, publish_root=publish_root,
        make_public=os.getenv("SMF_FORECAST_V1_PUBLIC", "false").lower() == "true" if make_public is None else make_public,
        r2_store=r2,
    )
    return {"run_id": manifest["runId"], "status": "complete", "warnings": manifest["warnings"]}


def run_forecast_v1_operational(now: datetime | None = None, make_public: bool | None = None) -> dict:
    return _exclusive_run(lambda: _run_forecast_v1_operational_impl(now, make_public))


def prune_forecast_v1_hot_storage() -> dict:
    r2 = ForecastR2Store()
    return prune_hot_storage(FORECAST_V1_DIR, archive_verified=r2.configured)


def _run_admin_forecast(job: dict) -> None:
    job.update(status="running", started_at=_now())
    _write_job(job)
    try:
        runner = run_forecast_v1_shadow if os.getenv("SMF_FORECAST_V1_SOURCE_MODE", "herbie").lower() == "staged" else run_forecast_v1_operational
        # Browser controls are deliberately shadow-only. Public promotion
        # remains an environment/deployment decision after verification.
        job["result"] = runner(make_public=False)
        job["status"] = "completed"
    except Exception as error:
        job.update(status="failed", error=str(error), error_type=type(error).__name__)
    job["finished_at"] = _now()
    _write_job(job)


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
            "publication": "shadow",
        }
        _write_job(job)
        threading.Thread(target=_run_admin_forecast, args=(job,), daemon=True, name="forecast-v1-admin").start()
        return job


def get_forecast_v1_job_status() -> dict:
    job = _read_job()
    job["execution_active"] = _execution_lock.locked()
    if job.get("status") not in {"queued", "running"} and job["execution_active"]:
        job["status"] = "scheduled_running"
    return job
