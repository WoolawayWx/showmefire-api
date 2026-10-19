#!/usr/bin/env python3
"""Wait until the configured HRRR cycle has every required forecast frame."""
from __future__ import annotations

import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from services.hrrr_readiness import check_hrrr_ready, cycle_run_time, not_before_utc


logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def main() -> int:
    cycle_hour = _env_int("FORECAST_CYCLE_HOUR", 9)
    start_hour = _env_int("FORECAST_START_HOUR", 5)
    end_hour = _env_int("FORECAST_END_HOUR", 18)
    timeout_seconds = _env_int("HRRR_READINESS_TIMEOUT_SECONDS", 7200)
    poll_seconds = _env_int("HRRR_READINESS_POLL_SECONDS", 300)
    forecast_hours = range(start_hour, end_hour + 1)
    # Optional earliest start, as a local (Central) HH:MM. The 12z forecast sets 07:30 so it never starts before its
    # morning fuel-moisture observations are in.
    not_before_local = os.getenv("FORECAST_NOT_BEFORE_LOCAL") or None
    started = time.monotonic()

    while True:
        now = datetime.now(timezone.utc)
        floor = not_before_utc(now, not_before_local)
        if floor is not None and now < floor:
            if time.monotonic() - started >= timeout_seconds:
                logger.error("Still before the %s local start floor after %d seconds; forecast was not started.",
                             not_before_local, timeout_seconds)
                return 1
            wait = max(1, int((floor - now).total_seconds()))
            logger.info("Not starting before %s local (%s UTC); waiting %d s.", not_before_local,
                        floor.strftime("%H:%M"), wait)
            time.sleep(min(poll_seconds, wait))
            continue
        run_time = cycle_run_time(now, cycle_hour)
        result = check_hrrr_ready(run_time, forecast_hours)
        if result.ready:
            logger.info(
                "HRRR %02dz f%02d-f%02d is complete; starting forecast.",
                cycle_hour,
                start_hour,
                end_hour,
            )
            return 0

        elapsed = time.monotonic() - started
        details = []
        if result.missing_files:
            details.append("missing files " + ", ".join(f"f{hour:02d}" for hour in result.missing_files))
        if result.missing_fields:
            details.append("missing fields " + ", ".join(result.missing_fields))
        if result.error:
            details.append("check error: " + result.error)
        logger.info("HRRR input is not complete yet (%s).", "; ".join(details) or "not published")

        if elapsed >= timeout_seconds:
            logger.error("HRRR readiness timed out after %d seconds; forecast was not started.", timeout_seconds)
            return 1
        time.sleep(min(poll_seconds, max(1, timeout_seconds - int(elapsed))))


if __name__ == "__main__":
    raise SystemExit(main())
