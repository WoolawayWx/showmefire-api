#!/usr/bin/env python3
"""Build the hour-by-hour HRRR product for an HRRR cycle (see services/hrrr_hourly).

    scripts/hrrr_hourly_forecast.py --auto            scheduler tick: newest hourly cycle that is ready and not built
    scripts/hrrr_hourly_forecast.py --cycle 20261016  build that day's 12z cycle (add --force to rebuild)
    scripts/hrrr_hourly_forecast.py --cycle 2026101614  build the 14z cycle of that day
    scripts/hrrr_hourly_forecast.py --publish 20261016  upload an already built 12z cycle to the CDN (no rebuild)
    scripts/hrrr_hourly_forecast.py --publish 2026101614  the same for the 14z cycle
    scripts/hrrr_hourly_forecast.py --prefetch        process each hour of today's 12z run as HRRR publishes it
                                                      (cron ~12:45Z); the later --auto build then only writes the store
Writes only under data/hrrr_hourly and images/hrrr_hourly (SMF_HRRR_HOURLY_DIR / SMF_HRRR_HOURLY_IMAGES).
"""
import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if "--prefetch" in sys.argv:
    # Lowest priority, like the scheduler-run build (services/hrrr_hourly/scheduler_job.py): one numeric thread. Must be set
    # before numpy loads, so before the imports below.
    for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(_name, "1")

from dotenv import load_dotenv  # noqa: E402

from services.hrrr_hourly import CYCLE_HOUR, cycle_id, job, publish, root_dir  # noqa: E402


def _parse_cycle(text: str) -> datetime:
    """YYYYMMDD means that day's 12z run; YYYYMMDDHH names the hour."""
    if len(text) == 10:
        return datetime.strptime(text, "%Y%m%d%H").replace(tzinfo=timezone.utc)
    return datetime.strptime(text, "%Y%m%d").replace(hour=CYCLE_HOUR, tzinfo=timezone.utc)


def _prefetch(cycle_time, hours) -> int:
    """Process the 12z hours as HRRR publishes them; as soon as the hours the daily forecast reads are saved, build the
    forecast's cache file from them (services/hrrr_shared_cache.py: a shadow copy unless FORECAST_USE_SHARED_HRRR)."""
    from services import hrrr_shared_cache
    from services.hrrr_hourly import fetch

    cycle_time = cycle_time or fetch.latest_cycle(datetime.now(timezone.utc), CYCLE_HOUR)
    needed = hrrr_shared_cache.required_hours()
    order = [*needed, *(hour for hour in hours if hour not in needed)]   # the forecast's hours first
    state = {"built": False}

    def build_shared(lead, done):
        if state["built"] or not set(needed) <= set(done):
            return
        state["built"] = True
        try:
            info = hrrr_shared_cache.build_shared_cache(cycle_time, root_dir())
            logging.info("shared HRRR cache: %s", info)
        except hrrr_shared_cache.SharedCacheUnavailable as error:
            logging.warning("shared HRRR cache not written: %s", error)

    result = job.prefetch_cycle(cycle_time=cycle_time, hours=order, on_hour_done=build_shared)
    print(json.dumps(result))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--auto", action="store_true", help="scheduler mode: skip quietly unless there is work")
    parser.add_argument("--cycle", help="YYYYMMDD (the 12z run) or YYYYMMDDHH of the run to build")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--prefetch", action="store_true",
                        help="process each hour of the 12z run as it is published, ahead of the build (cron ~12:45Z)")
    parser.add_argument("--publish", help="YYYYMMDD or YYYYMMDDHH of an already built run to upload to the CDN, then exit")
    parser.add_argument("--hours", default="0-18", help="forecast hours, e.g. 0-18 or 4-15")
    args = parser.parse_args(argv)
    load_dotenv()  # R2 credentials for publishing the run to the CDN (the scheduler's subprocess may not inherit them)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.publish:
        cycle = cycle_id(_parse_cycle(args.publish))
        result = publish.try_publish_cycle(cycle)
        print(json.dumps({"cycle": cycle, "published": result}))
        return 0 if result else 1

    low, _, high = args.hours.partition("-")
    hours = tuple(range(int(low), int(high or low) + 1))
    cycle_time = None
    if args.cycle:
        cycle_time = _parse_cycle(args.cycle)
    elif not (args.auto or args.prefetch):
        parser.error("pass --auto, --prefetch or --cycle YYYYMMDD")
    if args.prefetch:
        if hasattr(os, "nice"):
            os.nice(19)
        return _prefetch(cycle_time, hours)
    result = job.run_auto(force=args.force or bool(args.cycle), cycle_time=cycle_time, hours=hours)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
