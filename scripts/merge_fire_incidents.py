"""
One-off / manual pass: merge fire incidents that are really one fire (see
services/incident_merger.py). Run --dry-run first and review the report.

Usage:
    python scripts/merge_fire_incidents.py --dry-run
    python scripts/merge_fire_incidents.py --since-hours 168
    python scripts/merge_fire_incidents.py
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.database import init_database
from services.incident_merger import MERGE_GAP_KM, MAX_MERGED_EXTENT_KM, merge_touching_incidents


def main():
    parser = argparse.ArgumentParser(description="Merge touching fire incidents.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would merge without writing.")
    parser.add_argument("--since-hours", type=float, default=None, help="Only consider incidents active in the last N hours.")
    parser.add_argument("--gap-km", type=float, default=MERGE_GAP_KM)
    parser.add_argument("--max-extent-km", type=float, default=MAX_MERGED_EXTENT_KM)
    parser.add_argument("--json", action="store_true", help="Print the full report as JSON.")
    args = parser.parse_args()

    init_database()
    result = merge_touching_incidents(
        dry_run=args.dry_run, since_hours=args.since_hours,
        gap_km=args.gap_km, max_extent_km=args.max_extent_km,
    )
    if args.json:
        print(json.dumps(result, indent=2))
        return

    verb = "Would merge" if args.dry_run else "Merged"
    print(f"Considered {result['considered']} active incidents (gap {args.gap_km} km, max extent {args.max_extent_km} km).")
    print(f"{verb} {result['merged_away']} incidents into {len(result['groups'])} survivors.")
    for group in sorted(result["groups"], key=lambda g: -g["detections"])[:15]:
        print(f"  #{group['survivor_id']} <- {group['merged_ids']}  ({group['detections']} detections, {group['extent_km']} km)")
    if result["oversized"]:
        print(f"Skipped {len(result['oversized'])} oversized groups (review manually):")
        for group in result["oversized"]:
            print(f"  #{group['survivor_id']} + {group['merged_ids']}  ({group['detections']} detections, {group['extent_km']} km)")


if __name__ == "__main__":
    main()
