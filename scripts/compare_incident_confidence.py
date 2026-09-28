"""
Compare the legacy incident confidence (fire_confidence.py) with the v2
rules-based score (incident_confidence.py) over recent incidents, so the
switch can be judged on real data before FIRE_INCIDENT_CONFIDENCE_VERSION=v2
is set.

Usage:
    python scripts/compare_incident_confidence.py
    python scripts/compare_incident_confidence.py --days 7 --show 25
"""
import argparse
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.database import init_database, list_fire_incident_members, list_fire_incidents
from services.fire_confidence import _features, _score
from services.incident_confidence import score_incident


def main():
    parser = argparse.ArgumentParser(description="Compare legacy vs v2 incident confidence.")
    parser.add_argument("--days", type=float, default=7)
    parser.add_argument("--show", type=int, default=20, help="How many incidents to print in detail (biggest first).")
    args = parser.parse_args()

    init_database()
    since = (datetime.now(timezone.utc) - timedelta(days=args.days)).strftime("%Y-%m-%dT%H:%M:%SZ")
    incidents = list_fire_incidents(since=since, limit=500)
    rows = []
    for incident in incidents:
        members = list_fire_incident_members(incident["id"])
        old_score, old_label, _ = _score(_features(incident, members))
        new = score_incident(incident, members)
        best_detection = max((m["detection_confidence_pct"] for m in members if m.get("detection_confidence_pct") is not None), default=None)
        rows.append((incident, old_score, old_label, best_detection, new))

    print(f"{len(rows)} incidents active in the last {args.days:g} days\n")
    print("label      legacy   v2")
    old_counts = Counter(r[2] for r in rows)
    new_counts = Counter(r[4]["label"] for r in rows)
    for label in ("high", "moderate", "low"):
        print(f"{label:<10} {old_counts[label]:>6} {new_counts[label]:>4}")
    changed = sum(1 for r in rows if r[2] != r[4]["label"])
    print(f"\nLabel changed on {changed} of {len(rows)} incidents.\n")

    print(f"{'id':>6} {'dets':>5}  legacy        max-det  v2")
    for incident, old_score, old_label, best, new in sorted(rows, key=lambda r: -(r[0].get("detection_count") or 0))[: args.show]:
        best_text = f"{best:.0f}%" if best is not None else "-"
        print(f"{incident['id']:>6} {incident.get('detection_count') or 0:>5}  {old_label:<8} {old_score*100:>3.0f}   {best_text:>6}  {new['label']:<8} {new['score']:>3}  {'; '.join(new['reasons'][:2])}")


if __name__ == "__main__":
    main()
