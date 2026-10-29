"""
Fetch VIIRS/MODIS detections older than the live feed from NASA FIRMS' standard-processing archive into a CSV.

Writes NOTHING to the database. Compare the CSV with NERIS (model-training: `python -m detection_dataset.overlap`),
then add only the overlapping rows to the DB with scripts/ingest_firms_history_csv.py.

Needs MODIS_KEY (the FIRMS MAP_KEY the live poller already uses). NGFS has no archive, so it cannot be backfilled.

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

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")  # MODIS_KEY lives here; a one-off `docker exec` doesn't load it

from services.firms_backfill import ARCHIVE_PRODUCTS, DEFAULT_PRODUCTS, NRT_PRODUCTS, FirmsError, backfill


def main():
    parser = argparse.ArgumentParser(description="Fetch FIRMS archive detections into a CSV (no database writes).")
    parser.add_argument("--start", required=True, type=date.fromisoformat, help="first day, YYYY-MM-DD")
    parser.add_argument("--end", type=date.fromisoformat, default=None, help="last day, YYYY-MM-DD (default: today)")
    parser.add_argument("--products", nargs="+", default=list(DEFAULT_PRODUCTS), choices=list(ARCHIVE_PRODUCTS) + list(NRT_PRODUCTS))
    parser.add_argument("--dry-run", action="store_true", help="fetch and count, write no file")
    args = parser.parse_args()

    end = args.end or datetime.now(timezone.utc).date()

    def progress(product, done, total, kept):
        print(f"  {product}: window {done}/{total}, {kept} Missouri detections so far", file=sys.stderr, flush=True)

    try:
        print(json.dumps(backfill(args.start, end, products=args.products, dry_run=args.dry_run, progress=progress), indent=2, default=str))
    except FirmsError as exc:
        sys.exit(f"FIRMS backfill failed: {exc}")


if __name__ == "__main__":
    main()
