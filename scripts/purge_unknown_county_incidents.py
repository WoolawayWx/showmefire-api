"""
One-off cleanup: incidents with no county.

Backfills county on an incident whose centroid does resolve to a Missouri
county; soft-deletes (audit trail kept) any whose centroid still doesn't -
those are out-of-state clusters that got in via the loose bbox before
services/fire_ingest.py started dropping county-less detections.

Dry run by default.
    python scripts/purge_unknown_county_incidents.py            # report only
    python scripts/purge_unknown_county_incidents.py --apply
"""
import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.database import delete_fire_incident_and_members, get_db_path
from services.county_lookup import county_for_point


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, public_slug, centroid_latitude, centroid_longitude, detection_count "
        "FROM fire_incidents WHERE county_fips IS NULL OR county_fips = ''"
    ).fetchall()

    backfill, purge = [], []
    for r in rows:
        fips, name = county_for_point(r["centroid_latitude"], r["centroid_longitude"])
        (backfill if fips else purge).append((r, fips, name))

    print(f"{len(rows)} county-less incidents: {len(backfill)} backfillable, {len(purge)} to delete")
    for r, _, _ in purge:
        print(f"  delete #{r['id']} {r['public_slug']} ({r['centroid_latitude']:.3f}, {r['centroid_longitude']:.3f}) x{r['detection_count']}")

    if not args.apply:
        print("dry run - pass --apply to make changes")
        return

    for r, fips, name in backfill:
        conn.execute("UPDATE fire_incidents SET county_fips = ?, county_name = ? WHERE id = ?", (fips, name, r["id"]))
    conn.commit()
    conn.close()
    for r, _, _ in purge:
        delete_fire_incident_and_members(r["id"], actor="system", reason="no Missouri county (out-of-state cleanup)")
    print("done")


if __name__ == "__main__":
    main()
