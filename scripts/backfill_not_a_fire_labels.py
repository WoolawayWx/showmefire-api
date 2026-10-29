#!/usr/bin/env python3
"""Give the "not a fire" label to detections of incidents that were marked not a fire but never got it (detections
that joined during the 7-day hold, and incidents deleted before the label was kept), so they count as negative
examples for the detection model. Statuses are not changed. Safe to re-run.

  python scripts/backfill_not_a_fire_labels.py [--dry-run]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.database import backfill_not_a_fire_labels  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    count = backfill_not_a_fire_labels(dry_run=args.dry_run)
    print(f"{count} detection(s) {'would be' if args.dry_run else 'were'} labelled not_a_fire")


if __name__ == "__main__":
    main()
