"""
Add detections from a FIRMS history CSV to fire_events: typically the NERIS-overlap subset that
model-training's `python -m detection_dataset.overlap` writes. Detections only by default (no incidents), so old
history never reaches the review queue or the map. Re-runnable: a detection already stored is updated, not duplicated.

Back up showmefire.db first.

Usage:
    python scripts/ingest_firms_history_csv.py --csv firms_neris_overlap.csv --dry-run
    python scripts/ingest_firms_history_csv.py --csv firms_neris_overlap.csv
    python scripts/ingest_firms_history_csv.py --csv firms_neris_overlap.csv --with-incidents
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.database import init_database
from services.firms_backfill import ingest_csv


def main():
    parser = argparse.ArgumentParser(description="Add detections from a FIRMS history CSV to fire_events.")
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true", help="count the rows, write nothing")
    parser.add_argument("--with-incidents", action="store_true", help="also cluster them into incidents (default: detections only)")
    args = parser.parse_args()
    if not args.csv.exists():
        sys.exit(f"{args.csv} not found")
    if not args.dry_run:
        init_database()
    print(json.dumps(ingest_csv(args.csv, cluster_incidents=args.with_incidents, dry_run=args.dry_run), indent=2, default=str))


if __name__ == "__main__":
    main()
