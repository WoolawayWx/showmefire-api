"""Job runner for the secondary 9z HRRR forecast run.

Runs the same operational pipeline as the 12z forecast (DailyForecast.py +
forecast_ai.py), but pinned to the 9z HRRR cycle via FORECAST_CYCLE_HOUR, with
every generated image/status-key suffixed so it can never collide with the
12z run's files, and with no Discord/mobile notification step. Modeled on the
Testbed job runner (services/forecast_jobs.py), but runs the real operational
model instead of the shadow/testbed model, and lets the real run write to the
shared database (station_forecasts + forecasts tables already isolate rows by
run_time/cycle - see core/database.py).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from core.config import DATA_DIR


API_ROOT = Path(__file__).resolve().parent.parent
FORECAST_DIR = API_ROOT / "forecast"
DAILY_FORECAST_SCRIPT = FORECAST_DIR / "DailyForecast.py"
FORECAST_AI_SCRIPT = FORECAST_DIR / "forecast_ai.py"
COMPARE_SCRIPT = API_ROOT / "scripts" / "compare_09z_12z.py"

JOB_STATE_PATH = DATA_DIR / "forecast_09z_job.json"
_job_lock = threading.Lock()

STALE_JOB_GRACE_SECONDS = int(os.getenv("FORECAST_09Z_STALE_GRACE_SECONDS", "300"))
JOB_TIMEOUT_SECONDS = int(os.getenv("FORECAST_09Z_TIMEOUT_SECONDS", "3600"))

# Pinned onto every subprocess this job runs. The suffix is what actually
# keeps 09z artifacts from colliding with the 12z run's images/status key.
FORECAST_09Z_ENV = {
    "FORECAST_CYCLE_HOUR": "9",
    "FORECAST_IMAGE_SUFFIX": "_09z",
    "FORECAST_STATUS_KEY": "ForecastFireDanger09z",
    "CDN_TEST_PREFIX": "09z",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_job() -> dict | None:
    try:
        return json.loads(JOB_STATE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _write_job(job: dict) -> None:
    JOB_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = JOB_STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, JOB_STATE_PATH)


def _job_is_stale(job: dict, now: datetime | None = None) -> bool:
    if job.get("status") not in {"queued", "running"}:
        return False
    timestamp = job.get("started_at") or job.get("requested_at")
    if not timestamp:
        return True
    try:
        started = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return True
    age = ((now or datetime.now(timezone.utc)) - started).total_seconds()
    return age > JOB_TIMEOUT_SECONDS + STALE_JOB_GRACE_SECONDS


def _failure_excerpt(log_path: Path, line_limit: int = 20) -> str | None:
    try:
        lines = [line.strip() for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except OSError:
        return None
    return "\n".join(lines[-line_limit:]) or None


def _run_step(script: Path, log_path: Path, environment: dict) -> int:
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n=== Running {script.name} at {_now()} ===\n")
        log.flush()
        completed = subprocess.run(
            [sys.executable, str(script)],
            cwd=str(API_ROOT),
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=JOB_TIMEOUT_SECONDS,
            check=False,
        )
    return completed.returncode


def _run_09z_forecast(job: dict) -> None:
    log_path = DATA_DIR / "forecast_09z.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")  # start each job with a fresh log
    job["status"] = "running"
    job["started_at"] = _now()
    _write_job(job)

    environment = os.environ.copy()
    environment.update(FORECAST_09Z_ENV)

    try:
        return_code = _run_step(DAILY_FORECAST_SCRIPT, log_path, environment)
        if return_code == 0:
            return_code = _run_step(FORECAST_AI_SCRIPT, log_path, environment)
        if return_code == 0 and COMPARE_SCRIPT.exists():
            # Comparison is best-effort: a missing same-day 12z archive (12z
            # hasn't run yet) should not fail an otherwise-successful 9z job.
            _run_step(COMPARE_SCRIPT, log_path, environment)

        job["status"] = "completed" if return_code == 0 else "failed"
        job["return_code"] = return_code
        if return_code != 0:
            job["error"] = f"9z forecast pipeline exited with code {return_code}."
            job["error_detail"] = _failure_excerpt(log_path)
    except subprocess.TimeoutExpired:
        job["status"] = "failed"
        job["error"] = "9z forecast exceeded its timeout."
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)

    job["finished_at"] = _now()
    _write_job(job)


def trigger_09z_forecast(requested_by: str) -> dict:
    with _job_lock:
        existing = _read_job()
        stale_job_id = None
        if existing and existing.get("status") in {"queued", "running"} and not _job_is_stale(existing):
            raise RuntimeError("A 9z forecast is already running.")
        if existing and _job_is_stale(existing):
            stale_job_id = existing.get("job_id")
        job = {
            "job_id": uuid.uuid4().hex,
            "status": "queued",
            "requested_by": requested_by,
            "requested_at": _now(),
        }
        if stale_job_id:
            job["replaces_stale_job_id"] = stale_job_id
        _write_job(job)
        threading.Thread(target=_run_09z_forecast, args=(job,), daemon=True).start()
        return job


def get_09z_forecast_status() -> dict:
    return _read_job() or {"status": "idle"}
