"""
One-shot backfill: retroactively group already-ingested satellite-sourced
fire_events rows (source in modis/viirs/ngfs) into fire_incidents.

Only touches rows where incident_id IS NULL, so it's safe to re-run - a
second run reports 0 newly assigned.

Usage:
    python scripts/backfill_fire_incidents.py --dry-run
    python scripts/backfill_fire_incidents.py
"""
import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.database import (
    find_or_create_incident_for_detection,
    get_db_path,
    init_database,
    list_unclustered_satellite_events,
)

BATCH_COMMIT_SIZE = 500


def main():
    parser = argparse.ArgumentParser(description="Backfill fire_incidents for existing satellite fire_events rows.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be assigned without writing.")
    args = parser.parse_args()

    init_database()
    rows = list_unclustered_satellite_events()
    if args.dry_run:
        print(f"Would assign {len(rows)} detections to incidents.")
        return

    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    assigned = 0
    try:
        for row in rows:
            incident_id = find_or_create_incident_for_detection(
                cursor, row["latitude"], row["longitude"], row["occurred_at"],
                row["county_fips"], row["county_name"],
            )
            cursor.execute('UPDATE fire_events SET incident_id = ? WHERE id = ?', (incident_id, row["id"]))
            assigned += 1
            if assigned % BATCH_COMMIT_SIZE == 0:
                conn.commit()
        conn.commit()
    finally:
        conn.close()

    print(f"Assigned {assigned} detections to incidents.")


if __name__ == "__main__":
    main()
