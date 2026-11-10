"""Draw the one-sheet comparison of every county-scoring fire danger model for a day (services/model_overview.py).

    python scripts/run_model_overview.py                  # today (America/Chicago); published to the Testbed
    python scripts/run_model_overview.py --date 2026-03-14 --out some/dir

Run it after the day's shadow models have written their evidence (the daily forecast, then the 12z ensemble run).
Today's sheet is published to the Testbed; a sheet for another date is only written to --out unless --publish.
Never exits non-zero for a day it cannot draw.
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

from services import fire_escape, model_overview


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--date", type=date.fromisoformat, default=None, help="local valid date, YYYY-MM-DD")
    parser.add_argument("--out", type=Path, default=None, help="output folder (default: the shadow-evidence folder)")
    parser.add_argument("--publish", action="store_true", help="publish to the Testbed even for a date given with --date")
    args = parser.parse_args()
    today = datetime.now(ZoneInfo("America/Chicago")).date()
    day = args.date or today
    out = args.out or fire_escape.evidence_dir().parent / "model-overview"
    print(json.dumps(model_overview.run(day, out, publish_to_testbed=args.publish or args.date is None), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
