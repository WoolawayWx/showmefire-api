"""
Backfill VIIRS/MODIS detections older than the live feed from NASA FIRMS' standard-processing archive.

Needs MODIS_KEY (the FIRMS MAP_KEY the live poller already uses). Re-runnable: detections already in fire_events are
updated in place, never duplicated. NGFS has no archive, so it cannot be backfilled.

Usage:
    python scripts/backfill_firms_history.py --start 2025-12-01 --end 2026-08-08 --dry-run
    python scripts/backfill_firms_history.py --start 2025-12-01 --end 2026-08-08
    python scripts/backfill_firms_history.py --start 2026-08-01 --products VIIRS_SNPP_SP MODIS_SP
"""
import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.database import init_database
from services.firms_backfill import ARCHIVE_PRODUCTS, DEFAULT_PRODUCTS, NRT_PRODUCTS, FirmsError, backfill


def main():
    parser = argparse.ArgumentParser(description="Backfill fire_events from the FIRMS archive.")
    parser.add_argument("--start", required=True, type=date.fromisoformat, help="first day, YYYY-MM-DD")
    parser.add_argument("--end", type=date.fromisoformat, default=None, help="last day, YYYY-MM-DD (default: today)")
    parser.add_argument("--products", nargs="+", default=list(DEFAULT_PRODUCTS), choices=list(ARCHIVE_PRODUCTS) + list(NRT_PRODUCTS))
    parser.add_argument("--dry-run", action="store_true", help="fetch and count, write nothing")
    args = parser.parse_args()

    end = args.end or datetime.now(timezone.utc).date()
    if not args.dry_run:
        init_database()
    try:
        print(json.dumps(backfill(args.start, end, products=args.products, dry_run=args.dry_run), indent=2, default=str))
    except FirmsError as exc:
        sys.exit(f"FIRMS backfill failed: {exc}")


if __name__ == "__main__":
    main()
