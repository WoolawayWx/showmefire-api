"""
One-shot ingest of USFS FPA-FOD Missouri wildfire records into the
fire_events store as verification_tier='official_source_confirmed'.

See services/fpa_fod_ingest.py for the full rationale and the date-math
subtleties. This is a live network call to a public USDA Forest Service
ArcGIS REST endpoint - no credentials, no bulk download.

Usage:
    python scripts/ingest_fpa_fod.py --dry-run
    python scripts/ingest_fpa_fod.py --since-year 2011 --until-year 2020
    python scripts/ingest_fpa_fod.py                      # all years, 1992-2020
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.database import init_database
from services.fpa_fod_ingest import ingest_fpa_fod


def main():
    parser = argparse.ArgumentParser(description="Ingest USFS FPA-FOD Missouri wildfire records.")
    parser.add_argument("--since-year", type=int, default=None, help="Earliest fire_year to include (inclusive).")
    parser.add_argument("--until-year", type=int, default=None, help="Latest fire_year to include (inclusive).")
    parser.add_argument("--dry-run", action="store_true", help="Fetch and map but do not write to the database.")
    args = parser.parse_args()

    init_database()
    result = ingest_fpa_fod(since_year=args.since_year, until_year=args.until_year, dry_run=args.dry_run)
    print({k: v for k, v in result.items() if k != "errors"})
    if result["errors"]:
        print(f"{len(result['errors'])} errors (showing first 10):")
        for error in result["errors"][:10]:
            print(f"  {error}")


if __name__ == "__main__":
    main()
