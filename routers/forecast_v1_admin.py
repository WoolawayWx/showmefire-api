"""Authenticated operations console for the 72-hour forecast pipeline."""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, status

from core.config import FORECAST_V1_DIR
from core.security import verify_token
from forecast_v1.acquisition import latest_publishable_12z
from forecast_v1.contracts import run_id_for_cycle, utc_rfc3339
from forecast_v1.repository import ensure_schema, transaction
from services.forecast_v1_job import (
    get_forecast_v1_job_status,
    prune_forecast_v1_hot_storage,
    trigger_forecast_v1,
)


router = APIRouter(prefix="/api/admin/forecast-v1", tags=["forecast-v1-admin"])


def _require_admin(token: Optional[str] = None) -> str:
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return email


def _decode_json(value, fallback):
    try:
        return json.loads(value) if value else fallback
    except (TypeError, json.JSONDecodeError):
        return fallback


def _asset_url(object_key: str | None) -> str | None:
    if not object_key:
        return None
    return "/forecast-v1-assets/" + object_key.removeprefix("forecast-v1/")


def _schedule_status() -> dict:
    now = datetime.now(timezone.utc)
    minimum_age = max(1, int(os.getenv("SMF_FORECAST_V1_MIN_CYCLE_AGE_HOURS", "6")))
    eligible_cycle = latest_publishable_12z(now, minimum_age_hours=minimum_age)
    eligible_at = eligible_cycle + timedelta(hours=minimum_age)
    next_cycle = now.replace(hour=12, minute=0, second=0, microsecond=0)
    if now >= next_cycle + timedelta(hours=minimum_age):
        next_cycle += timedelta(days=1)
    next_eligible = next_cycle + timedelta(hours=minimum_age)
    central = ZoneInfo("America/Chicago")
    return {
        "schedulerEnabled": os.getenv("run_sch", "false").lower() == "true",
        "forecastEnabled": os.getenv("SMF_FORECAST_V1_ENABLED", "false").lower() == "true",
        "sourceMode": os.getenv("SMF_FORECAST_V1_SOURCE_MODE", "herbie"),
        "publicationMode": "public" if os.getenv("SMF_FORECAST_V1_PUBLIC", "false").lower() == "true" else "shadow",
        "pollIntervalMinutes": max(15, int(os.getenv("SMF_FORECAST_V1_POLL_MINUTES", "30"))),
        "minimumCycleAgeHours": minimum_age,
        "memoryLimitGb": float(os.getenv("SMF_FORECAST_V1_MEMORY_LIMIT_GB", "8")),
        "eligibleCycle": utc_rfc3339(eligible_cycle),
        "eligibleAt": utc_rfc3339(eligible_at),
        "eligibleAtCentral": eligible_at.astimezone(central).isoformat(timespec="minutes"),
        "nextCycleEligibleAt": utc_rfc3339(next_eligible),
        "nextCycleEligibleAtCentral": next_eligible.astimezone(central).isoformat(timespec="minutes"),
        "policy": "HRRR through hour 48; RRFS preferred for 49-72; GEFS ensemble-mean fallback when RRFS is unavailable",
    }


@router.get("/job")
def forecast_v1_admin_job(token: Optional[str] = None):
    """Lightweight endpoint for frequent progress polling during a run."""
    _require_admin(token)
    return get_forecast_v1_job_status()


@router.get("/status")
def forecast_v1_admin_status(token: Optional[str] = None):
    _require_admin(token)
    ensure_schema()
    with transaction() as connection:
        run_rows = connection.execute(
            "SELECT * FROM forecast_runs ORDER BY cycle_time_utc DESC LIMIT 12"
        ).fetchall()
        latest = next((row for row in run_rows if row["status"] in {"complete", "superseded"}), None)
        graphics = []
        stations = []
        if latest:
            graphics = connection.execute(
                "SELECT variable,aggregation,object_key,checksum,byte_size FROM forecast_assets "
                "WHERE run_id=? AND kind='graphic' AND aggregation LIKE '%-png' AND status='ready' "
                "ORDER BY aggregation,variable", (latest["run_id"],),
            ).fetchall()
            stations = connection.execute(
                "SELECT station_id,name,network_type FROM forecast_station_metadata WHERE is_active=1 ORDER BY network_type,name"
            ).fetchall()
        storage = connection.execute(
            "SELECT kind,COUNT(*) AS asset_count,COALESCE(SUM(byte_size),0) AS bytes FROM forecast_assets "
            "WHERE status='ready' GROUP BY kind ORDER BY kind"
        ).fetchall()
    runs = [{
        "runId": row["run_id"], "cycleTime": row["cycle_time_utc"], "issuedAt": row["issued_at_utc"],
        "completedAt": row["completed_at_utc"], "status": row["status"], "public": bool(row["is_public"]),
        "horizonHours": row["horizon_hours"], "sources": _decode_json(row["source_cycles_json"], {}),
        "warnings": _decode_json(row["warnings_json"], []),
    } for row in run_rows]
    return {
        "schedule": _schedule_status(), "job": get_forecast_v1_job_status(), "runs": runs,
        "latestRunId": latest["run_id"] if latest else None,
        "graphics": [{
            "variable": row["variable"], "aggregation": row["aggregation"],
            "day": int(row["aggregation"].split("-")[1]), "url": _asset_url(row["object_key"]),
            "checksum": row["checksum"], "byteSize": row["byte_size"],
        } for row in graphics],
        "stations": [dict(row) for row in stations],
        "storage": [dict(row) for row in storage],
        "paths": {"root": str(Path(FORECAST_V1_DIR)), "jobState": str(Path(FORECAST_V1_DIR) / "admin-job.json")},
    }


@router.post("/run", status_code=status.HTTP_202_ACCEPTED)
def run_forecast_v1_admin(token: Optional[str] = None):
    email = _require_admin(token)
    try:
        return trigger_forecast_v1(email)
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.post("/retention")
def run_forecast_v1_retention(token: Optional[str] = None):
    _require_admin(token)
    return {"status": "completed", "result": prune_forecast_v1_hot_storage()}
