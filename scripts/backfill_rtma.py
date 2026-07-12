#!/usr/bin/env python3
"""Resumable RTMA backfill over the rolling one-year data window."""
from __future__ import annotations

import argparse
import logging
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from services.archive_bundler import OUTPUT_DIR, ROOT_DIR
from services.rtma_capture import fetch_rtma


def archived_hours(day: datetime) -> set[int]:
    path = OUTPUT_DIR / f"{day:%Y%m%d}.zip"
    if not path.exists():
        return set()
    prefix = f"rtma/rtma_{day:%Y%m%d}_"
    with zipfile.ZipFile(path) as zf:
        return {int(Path(n).stem.split("_")[-1][:-1]) for n in zf.namelist() if n.startswith(prefix) and n.endswith("z.nc")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", help="YYYY-MM-DD; defaults to one year ago")
    parser.add_argument("--end", help="YYYY-MM-DD; defaults to today UTC")
    parser.add_argument("--hours", default="0-23", help="0-23 or comma-separated UTC hours")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    today = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc) if args.start else today - timedelta(days=365)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc) if args.end else today
    hours = range(24) if args.hours == "0-23" else [int(v) for v in args.hours.split(",")]
    print(f"RTMA backfill {start.date()} through {end.date()}")
    if args.dry_run:
        return
    failures = []
    day = start
    while day <= end:
        complete = archived_hours(day)
        for hour in hours:
            target = ROOT_DIR / "cache" / "rtma" / f"rtma_{day:%Y%m%d}_{hour:02d}z.nc"
            if hour in complete or target.exists():
                continue
            run_dt = day.replace(hour=hour)
            for attempt in range(3):
                try:
                    fetch_rtma(run_dt)
                    break
                except Exception as exc:
                    if attempt == 2:
                        failures.append((run_dt.isoformat(), str(exc)))
                        logging.error("%s unavailable/failed: %s", run_dt, exc)
                    else:
                        time.sleep(2 ** attempt)
        day += timedelta(days=1)
    if failures:
        print(f"{len(failures)} RTMA hour(s) failed")
        raise SystemExit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
