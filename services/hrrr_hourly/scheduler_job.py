"""APScheduler entry point: run scripts/hrrr_hourly_forecast.py --auto in a subprocess."""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

SCRIPT = Path(__file__).resolve().parent.parent.parent / "scripts" / "hrrr_hourly_forecast.py"
TIMEOUT_SECONDS = 3 * 3600


# Lowest priority of the three jobs (forecast, 15-minute maps, this): minimum CPU and I/O priority, and one
# numeric thread so it never takes more than a core from the forecast or the API.
THREAD_LIMITS = {name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                          "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}


def _lower_priority() -> None:
    os.nice(19)
    try:
        subprocess.run(["ionice", "-c", "3", "-p", str(os.getpid())], check=False, capture_output=True)
    except OSError:
        pass


def _run() -> str:
    result = subprocess.run([sys.executable, str(SCRIPT), "--auto"], capture_output=True, text=True,
                            timeout=TIMEOUT_SECONDS, cwd=SCRIPT.parent.parent, preexec_fn=_lower_priority,
                            env={**os.environ, **THREAD_LIMITS})
    if result.returncode != 0:
        logger.error("hrrr_hourly failed (exit %s): %s", result.returncode, (result.stderr or "")[-1500:])
    else:
        logger.info("hrrr_hourly: %s", (result.stdout or "").strip().splitlines()[-1:] or "ok")
    return result.stdout


async def run_scheduled() -> None:
    try:
        await asyncio.to_thread(_run)
    except Exception:
        logger.exception("hrrr_hourly scheduler tick failed")
