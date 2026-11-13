"""Print the ground-cover shadow-scoring review (plan section 8 step 3).

    python scripts/review_ground_cover_shadow.py [--days 30] [--top 20] [--json]

Read-only. Check the 20 largest changes by hand against satellite imagery and what staff decided before deciding anything.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.ground_cover_shadow import review  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--json", action="store_true", help="print the raw JSON")
    args = parser.parse_args()
    report = review(days=args.days, top=args.top)
    if args.json:
        print(json.dumps(report, indent=2))
        return
    change = report["score_change"]
    print(f"{report['incidents']} incidents over {report['days_of_data']} days "
          f"(ready for review: {report['ready_for_review']}; rule: {report['readiness_rule']})")
    print(f"score change: {change['unchanged']} unchanged, {change['lowered']} lowered; "
          f"mean {change['mean']}, median {change['median']}, range {change['min']}..{change['max']}")
    print(f"label flips: {report['label_flip_count']} {report['label_flips']}")
    print(f"sampled over: {report['by_basis']}")
    print(f"\nLargest changes (top {args.top}):")
    for row in report["largest_changes"]:
        print(f"  #{row['incident_id']} {row['public_slug'] or '':14s} {row['live_score']:>3} {row['live_label']:8s}"
              f"-> {row['shadow_score']:>3} {row['shadow_label']:8s} ({row['delta']:+d})  {row['basis']:9s} {row['stand_in_land_cover']}")


if __name__ == "__main__":
    main()
