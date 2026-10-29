#!/usr/bin/env python3
"""Compare satellite detections with independent confirmed fires (see services/detection_validation.py).

  python scripts/compare_detections_to_confirmed.py [--radius-km 3] [--before-hours 6] [--after-hours 24] [--csv FILE]

Read-only. Prints how far back the data goes, then how many confirmed fires had a detection nearby, by fire size,
and lists the ones the satellites missed. --csv writes one row per confirmed fire.
"""
import argparse
import csv
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.database import get_db_path  # noqa: E402
from services.detection_validation import compare  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--radius-km", type=float, default=3.0)
    parser.add_argument("--before-hours", type=float, default=6.0, help="how long before the reported time a detection still counts")
    parser.add_argument("--after-hours", type=float, default=24.0, help="how long after the reported time a detection still counts")
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    with sqlite3.connect(f"file:{get_db_path()}?mode=ro", uri=True) as conn:
        result = compare(conn, args.radius_km, args.before_hours, args.after_hours)

    print("DATA COVERAGE (non-deleted events)")
    for row in result["coverage"]["sources"]:
        print(f"  {row['source']:<16} {row['events']:>7}  {str(row['first'])[:10]} .. {str(row['last'])[:10]}"
              f"   approved {row['approved'] or 0}, rejected {row['rejected'] or 0}")
    print(f"  satellite detections: {str(result['coverage']['satellite_first'])[:10]} .. {str(result['coverage']['satellite_last'])[:10]}")

    s = result["settings"]
    print(f"\nCONFIRMED FIRES vs DETECTIONS (within {s['radius_km']:g} km, {s['before_hours']:g} h before to {s['after_hours']:g} h after the report)")
    if not result["reference_fires"]:
        print("  No approved reports, leads or official records to compare against.")
        return
    print(f"  {result['detected']} of {result['reference_fires']} confirmed fires had a detection ({result['detection_rate']:.0%})")
    if result["median_lead_hours"] is not None:
        print(f"  median time from the report to the first detection: {result['median_lead_hours']:+.1f} h (negative = detected first)")
    for label, entry in sorted(result["by_size"].items()):
        print(f"  {label:<16} {entry['detected']}/{entry['fires']} detected")
    print("  detected by:", ", ".join(f"{k} {v}" for k, v in sorted(result["detected_by_source"].items())) or "-")
    missed = [m for m in result["matches"] if not m["detected"]]
    if missed:
        print("\nMISSED")
        for m in missed:
            print(f"  #{m['id']} {m['source']} {str(m['occurred_at'])[:16]} {m.get('county_name') or ''} acres={m.get('acres')}")

    if args.csv:
        with args.csv.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "source", "occurred_at", "county", "acres", "detected", "first_lead_hours", "nearest_km", "detections"])
            for m in result["matches"]:
                writer.writerow([m["id"], m["source"], m["occurred_at"], m.get("county_name"), m.get("acres"), m["detected"],
                                 m["first_lead_hours"], m["nearest_km"], len(m["detections"])])
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
