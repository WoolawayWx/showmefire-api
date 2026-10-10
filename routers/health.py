"""
Health/readiness checks for the Show Me Fire Weather API.

GET /health              - liveness probe (no dependency checks, near-zero cost)
GET /health?verbose=true - readiness probe (alias for /health/ready)
GET /health/ready         - readiness probe, runs every registered check

CONTRIBUTOR CHECKLIST - update this file when you add:
  - a new router with a hard external dependency (DB table, paid API, file dir)
      -> add a check_*() function and append it to CHECKS below
  - a new APScheduler job in core/scheduler.py
      -> no action needed; check_scheduler already counts jobs generically.
         Only add a dedicated check if the job's failure should independently
         flip overall status (e.g. a job whose absence is a production incident).
  - a new required external data source (new NWP model, new paid feed)
      -> add a lightweight reachability/staleness check function, but keep it
         out of CHECKS unless it reliably runs in well under 250ms. Prefer a
         separate opt-in endpoint (e.g. /health/external) for anything
         network-bound, following the pattern in services/hrrr_readiness.py.
"""
from __future__ import annotations

import os
import shutil
import sqlite3
from datetime import datetime, timezone
from enum import Enum
from time import perf_counter
from typing import Callable

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from core.database import get_db_path
from core.config import (
    IMAGES_DIR,
    GIS_DIR,
    FORECAST_V1_DIR,
    PUBLIC_DIR,
    REPORTS_DIR,
)
from services.beta_products import BETA_ROOT
from services.synoptic import get_station_data
from services.timeseries import get_timeseries_data

router = APIRouter()

PROCESS_START = datetime.now(timezone.utc)


class CheckStatus(str, Enum):
    ok = "ok"
    degraded = "degraded"
    error = "error"


class CheckResult(BaseModel):
    name: str
    status: CheckStatus
    message: str
    latency_ms: float


def _timed(name: str, fn: Callable[[], tuple]) -> CheckResult:
    start = perf_counter()
    try:
        status, message = fn()
    except Exception as exc:
        status, message = CheckStatus.error, f"{type(exc).__name__}: {exc}"
    return CheckResult(
        name=name,
        status=status,
        message=message,
        latency_ms=round((perf_counter() - start) * 1000, 2),
    )


def check_database() -> CheckResult:
    def _run():
        conn = sqlite3.connect(get_db_path(), timeout=2)
        try:
            conn.execute("SELECT 1").fetchone()
        finally:
            conn.close()
        return CheckStatus.ok, "connected"

    return _timed("database", _run)


def check_scheduler(request: Request) -> CheckResult:
    def _run():
        scheduler = getattr(request.app.state, "scheduler", None)
        if scheduler is None:
            return CheckStatus.degraded, "scheduler disabled (run_sch=false)"
        if not scheduler.running:
            return CheckStatus.error, "scheduler instance present but not running"
        job_count = len(scheduler.get_jobs())
        return CheckStatus.ok, f"running, {job_count} jobs scheduled"

    return _timed("scheduler", _run)


def check_data_freshness() -> CheckResult:
    def _run():
        synoptic = get_station_data()
        timeseries = get_timeseries_data()
        problems = []
        if synoptic.get("error") or not synoptic.get("last_updated"):
            problems.append("synoptic")
        if timeseries.get("error") or not timeseries.get("last_updated"):
            problems.append("timeseries")
        if problems:
            return CheckStatus.degraded, f"stale/errored: {', '.join(problems)}"
        return CheckStatus.ok, "synoptic and timeseries data fresh"

    return _timed("data_freshness", _run)


def check_writable_dirs() -> CheckResult:
    def _run():
        bad = [
            str(d)
            for d in (IMAGES_DIR, GIS_DIR, FORECAST_V1_DIR, REPORTS_DIR, PUBLIC_DIR, BETA_ROOT)
            if not (d.exists() and os.access(d, os.W_OK))
        ]
        if bad:
            return CheckStatus.error, f"not writable: {', '.join(bad)}"
        return CheckStatus.ok, "all directories writable"

    return _timed("writable_dirs", _run)


def check_disk_space() -> CheckResult:
    def _run():
        usage = shutil.disk_usage(REPORTS_DIR)
        free_pct = usage.free / usage.total * 100
        if free_pct < 5:
            return CheckStatus.error, f"{free_pct:.1f}% free"
        if free_pct < 15:
            return CheckStatus.degraded, f"{free_pct:.1f}% free"
        return CheckStatus.ok, f"{free_pct:.1f}% free"

    return _timed("disk_space", _run)


# Extensibility point: append new check functions here as new subsystems are
# added. See the contributor checklist in this module's docstring.
CHECKS: list[Callable[[], CheckResult]] = [
    check_database,
    check_writable_dirs,
    check_disk_space,
    check_data_freshness,
]


def _uptime_seconds() -> float:
    return round((datetime.now(timezone.utc) - PROCESS_START).total_seconds(), 1)


def _run_readiness(request: Request) -> JSONResponse:
    results = [check() for check in CHECKS] + [check_scheduler(request)]

    if any(r.status == CheckStatus.error for r in results):
        overall = "unhealthy"
        status_code = 503
    elif any(r.status == CheckStatus.degraded for r in results):
        overall = "degraded"
        status_code = 200
    else:
        overall = "healthy"
        status_code = 200

    is_production = os.getenv("ENVIRONMENT", "development").lower() == "production"

    body = {
        "status": overall,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": _uptime_seconds(),
        "environment": "production" if is_production else "development",
        "checks": [r.model_dump() for r in results],
    }
    return JSONResponse(content=body, status_code=status_code)


@router.get("/health")
def health_liveness(request: Request, verbose: bool = False):
    """Fast liveness probe - no dependency checks. For LB/uptime pings.

    Pass ?verbose=true for a full readiness check (same as /health/ready).
    """
    if verbose:
        return _run_readiness(request)
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": _uptime_seconds(),
    }


@router.get("/health/ready")
def health_readiness(request: Request):
    """Readiness probe - runs every registered check (DB, scheduler, disk, data freshness)."""
    return _run_readiness(request)
