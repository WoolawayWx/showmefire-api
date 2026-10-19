#!/usr/bin/env python3
"""Compare the shared HRRR cache (assembled by the hourly prefetch) with the one DailyForecast downloaded itself.

Run from scripts/forecasts.sh right after DailyForecast, while FORECAST_USE_SHARED_HRRR is off: the prefetch writes
its file to the shadow directory, DailyForecast downloads its own, and this checks that the shared one would have
produced the same inputs (services/hrrr_shared_cache.compare_caches). It only reports. It always exits 0, so it can
never block or change a forecast.

Each run appends one JSON line to data/hrrr_shared_cache_compare.jsonl and logs a one-line verdict. Switch the
forecast over (FORECAST_USE_SHARED_HRRR=true) after the verdict has been "match" for a few days.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from services import hrrr_shared_cache as shared  # noqa: E402
from services.hrrr_readiness import cycle_run_time  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
LOG_PATH = PROJECT_DIR / "data" / "hrrr_shared_cache_compare.jsonl"


def main() -> int:
    if shared.use_shared():
        logger.info("FORECAST_USE_SHARED_HRRR is on: the forecast reads the shared file itself, nothing to compare.")
        return 0
    hours = shared.required_hours()
    run = cycle_run_time(datetime.now(timezone.utc), int(os.getenv("FORECAST_CYCLE_HOUR", "12")))
    name = shared.cache_name(run, hours)
    shadow = shared.shadow_cache_dir() / name
    candidates = sorted(shared.live_cache_dir().glob(name.replace(f"_{run:%H}z_", "_??z_")))
    if not shadow.is_file():
        logger.info("No shared cache %s (the prefetch did not produce one today); nothing to compare.", shadow)
        return 0
    if not candidates:
        logger.info("DailyForecast wrote no cache file for %s; nothing to compare.", name)
        return 0
    try:
        report = shared.compare_caches(shadow, candidates[-1])
    except Exception as error:  # noqa: BLE001 - reporting only
        logger.exception("Comparison failed: %s", error)
        report = {"ok": False, "problems": [f"comparison failed: {error}"], "shared": str(shadow),
                  "real": str(candidates[-1]), "variables": {}}
    report["checked_at"] = datetime.now(timezone.utc).isoformat()
    report["cycle"] = f"{run:%Y-%m-%d} {run:%H}Z"
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(report) + "\n")
    except OSError as error:
        logger.warning("Could not append to %s: %s", LOG_PATH, error)
    if report["ok"]:
        logger.info("Shared HRRR cache MATCHES DailyForecast's own for %s (%s)", report["cycle"],
                    ", ".join(f"{k} {v['max_abs_diff']:g}" for k, v in sorted(report["variables"].items())))
    else:
        logger.warning("Shared HRRR cache DIFFERS from DailyForecast's own for %s: %s", report["cycle"],
                       "; ".join(report["problems"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
