"""
One-shot backfill/re-ingest of the existing file-based fire detections
into the unified fire_events store.

Usage:
    python scripts/backfill_fire_detections.py --dry-run
    python scripts/backfill_fire_detections.py
    python scripts/backfill_fire_detections.py --satdet-path archive/satfiredetection_2026-08-01.geojson
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.database import init_database
from services.fire_ingest import NGFS_PATH, SATDET_PATH, ingest_detection_files


def main():
    parser = argparse.ArgumentParser(description="Backfill fire_events from existing detection GeoJSON files.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be ingested without writing.")
    parser.add_argument("--satdet-path", type=Path, default=SATDET_PATH, help="Override satfiredetection.geojson path.")
    parser.add_argument("--ngfs-path", type=Path, default=NGFS_PATH, help="Override missouri_fires.geojson path.")
    args = parser.parse_args()

    init_database()
    result = ingest_detection_files(
        paths={"satdet": args.satdet_path, "ngfs": args.ngfs_path},
        dry_run=args.dry_run,
    )
    print(result)


if __name__ == "__main__":
    main()
