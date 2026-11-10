"""Build the fire escape shadow maps for one day (services/fire_escape).

    python scripts/run_fire_escape.py                         # today (America/Chicago), from the daily forecast's HRRR cache
    python scripts/run_fire_escape.py --date 2026-03-14 --hrrr path/to/hrrr_20260314_12z_f04-15.nc --out some/dir

Today's maps are published to the Testbed (behind its acknowledgement); a run for another date is only written
to the output folder unless --publish is given.

Does nothing unless SMF_FIRE_ESCAPE_ENABLED is on or --force is given. Never exits non-zero for a day it
cannot do: it prints the reason, so a cron line after the daily forecast cannot break that forecast.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

API_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(API_ROOT))

# cron starts jobs with a near-empty environment: read the API's own .env before anything looks at a setting.
try:
    from dotenv import load_dotenv
    load_dotenv(API_ROOT / ".env")
except ImportError:
    pass

from services import fire_escape
from services.fire_escape import job


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--date", type=date.fromisoformat, default=None, help="local valid date, YYYY-MM-DD")
    parser.add_argument("--hrrr", type=Path, default=None, help="cached HRRR file (default: found in the HRRR cache)")
    parser.add_argument("--bundle", type=Path, default=None, help="release folder (default: SMF_FIRE_ESCAPE_BUNDLE)")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default: the fire-escape evidence folder)")
    parser.add_argument("--no-maps", action="store_true")
    parser.add_argument("--publish", action="store_true", help="publish to the Testbed even for a date given with --date")
    parser.add_argument("--force", action="store_true", help="run even when SMF_FIRE_ESCAPE_ENABLED is off")
    args = parser.parse_args()
    if not (args.force or fire_escape.enabled()):
        print(json.dumps({"status": "skipped", "reason": "SMF_FIRE_ESCAPE_ENABLED is off"}))
        return 0
    day = args.date or datetime.now(ZoneInfo("America/Chicago")).date()
    publish = args.publish or args.date is None
    print(json.dumps(job.run(day, args.hrrr, args.bundle, args.out, render_maps=not args.no_maps, publish=publish), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
