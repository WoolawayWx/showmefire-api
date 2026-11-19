"""APScheduler entry point: run `python -m services.data_capture.capture --tick` in a low-priority subprocess.

Same pattern as services/hrrr_hourly/scheduler_job.py: the downloads and GRIB decoding run outside the API process, at the
lowest CPU and I/O priority, so capture can never slow the website. Off switch: SMF_CAPTURE_MODELS_ENABLED=false.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

API_ROOT = Path(__file__).resolve().parents[2]
TIMEOUT_SECONDS = 50 * 60
THREAD_LIMITS = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}


def enabled() -> bool:
    return os.getenv("SMF_CAPTURE_MODELS_ENABLED", "true").strip().lower() not in ("false", "0", "no", "off")


def _lower_priority() -> None:
    os.nice(19)
    try:
        subprocess.run(["ionice", "-c", "3", "-p", str(os.getpid())], check=False, capture_output=True)
    except OSError:
        pass


def _run() -> None:
    result = subprocess.run([sys.executable, "-m", "services.data_capture.capture", "--tick"], capture_output=True, text=True,
                            timeout=TIMEOUT_SECONDS, cwd=API_ROOT, preexec_fn=_lower_priority if os.name != "nt" else None,
                            env={**os.environ, **THREAD_LIMITS})
    if result.returncode != 0:
        logger.error("data capture failed (exit %s): %s", result.returncode, (result.stderr or "")[-1500:])
        return
    lines = (result.stdout or "").strip().splitlines()
    try:
        summary = json.loads(lines[-1]) if lines else {}
    except ValueError:
        summary = {"output": lines[-1:] if lines else []}
    if summary.get("captured") or summary.get("failed") or summary.get("skipped"):
        logger.info("data capture: captured=%s failed=%s skipped=%s", summary.get("captured"), summary.get("failed"), summary.get("skipped"))


async def run_scheduled() -> None:
    if not enabled():
        return
    try:
        await asyncio.to_thread(_run)
    except Exception:
        logger.exception("data capture scheduler tick failed")
