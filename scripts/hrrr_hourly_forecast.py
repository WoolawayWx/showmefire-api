#!/usr/bin/env python3
"""Build the hour-by-hour HRRR product for a 12z cycle (see services/hrrr_hourly).

    scripts/hrrr_hourly_forecast.py --auto            scheduler tick: newest 12z cycle if ready and not built
    scripts/hrrr_hourly_forecast.py --cycle 20261016  build that day's 12z cycle (add --force to rebuild)
Writes only under data/hrrr_hourly and images/hrrr_hourly (SMF_HRRR_HOURLY_DIR / SMF_HRRR_HOURLY_IMAGES).
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services.hrrr_hourly import CYCLE_HOUR, job  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--auto", action="store_true", help="scheduler mode: skip quietly unless there is work")
    parser.add_argument("--cycle", help="YYYYMMDD of the 12z run to build")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--hours", default="0-18", help="forecast hours, e.g. 0-18 or 4-15")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    low, _, high = args.hours.partition("-")
    hours = tuple(range(int(low), int(high or low) + 1))
    cycle_time = None
    if args.cycle:
        cycle_time = datetime.strptime(args.cycle, "%Y%m%d").replace(hour=CYCLE_HOUR, tzinfo=timezone.utc)
    elif not args.auto:
        parser.error("pass --auto or --cycle YYYYMMDD")
    result = job.run_auto(force=args.force or bool(args.cycle), cycle_time=cycle_time, hours=hours)
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
