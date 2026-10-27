#!/usr/bin/env python3
"""Attach public fire reports that are in no incident yet to one (joining a nearby detection cluster, or starting
a new one), the way create_fire_report() now does when a report arrives. Reports keep their status, so pending
ones stay pending. Safe to re-run: reports that already have an incident are skipped.

  python scripts/backfill_report_incidents.py [--dry-run]
"""
import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.database import correlate_report_with_incident, get_db_path  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    with sqlite3.connect(get_db_path()) as conn:
        ids = [row[0] for row in conn.execute(
            "SELECT id FROM fire_events WHERE source = 'user_submission' AND incident_id IS NULL "
            "AND status IN ('pending', 'approved') ORDER BY id")]
    for event_id in ids:
        if args.dry_run:
            print(f"[dry-run] report #{event_id} would be attached to an incident")
        else:
            print(f"report #{event_id} -> incident {correlate_report_with_incident(event_id)}")
    print(f"{len(ids)} report(s) {'found' if args.dry_run else 'processed'}")


if __name__ == "__main__":
    main()
