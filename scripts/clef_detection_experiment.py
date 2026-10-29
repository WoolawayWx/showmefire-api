#!/usr/bin/env python3
"""Does Clef agree with staff about satellite fire incidents? An offline experiment: nothing is stored or acted on.

  python scripts/clef_detection_experiment.py [--limit 60] [--csv FILE]

Takes incidents staff already decided (review_decision approved/rejected), shows Clef each one's satellite facts
(never the decision), and prints how its "is this a real fire?" probability lines up with the staff decision.
Reads the database read-only; needs CLOUDFLARE_AI_API_KEY + an account id (not CLEF_ENABLED, which gates the live app).
Each call costs a few hundred tokens, so a run of 60 incidents is a fraction of a cent.
"""
import argparse
import csv
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai.clef import ClefClient, clean_answers  # noqa: E402
from core.database import get_db_path  # noqa: E402

QUESTIONS = {
    "real_fire": {
        "type": "noul",
        "instructions": "Do these satellite heat detections most likely come from a real vegetation fire, as opposed to an "
                        "industrial heat source, a false detection, or something else that is not a fire? A planned or "
                        "agricultural burn counts as a real fire.",
        "criteria": {"true": "A real fire (wild, prescribed or agricultural)", "false": "Not a fire"},
    },
}


def incidents(conn: sqlite3.Connection, limit: int) -> list[dict]:
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT id, review_decision, county_name, first_detected_at, last_detected_at, detection_count
           FROM fire_incidents WHERE review_decision IN ('approved', 'rejected') AND merged_into_id IS NULL
           ORDER BY id DESC LIMIT ?""", (limit,)).fetchall()
    result = []
    for row in rows:
        members = conn.execute(
            """SELECT source, satellite, confidence, frp, daynight, land_cover, weather_danger_category, occurred_at
               FROM fire_events WHERE incident_id = ? AND source != 'user_submission' ORDER BY occurred_at LIMIT 12""",
            (row["id"],)).fetchall()
        if members:
            result.append({"id": row["id"], "decision": row["review_decision"], "row": dict(row),
                           "members": [dict(m) for m in members]})
    return result


def auc(pairs: list[tuple[float, int]]) -> float | None:
    positives = [p for p, y in pairs if y == 1]
    negatives = [p for p, y in pairs if y == 0]
    if not positives or not negatives:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    client = ClefClient()
    if not client.configured:
        sys.exit("CLOUDFLARE_AI_API_KEY and CLOUDFLARE_ACCOUNT_ID (or R2_ACCOUNT_ID) must be set")

    with sqlite3.connect(f"file:{get_db_path()}?mode=ro", uri=True) as conn:
        items = incidents(conn, args.limit)
    print(f"{len(items)} decided incidents (approved = fire, rejected = not a fire)")

    scored = []
    for item in items:
        state = {
            "county": item["row"]["county_name"],
            "first_detected": item["row"]["first_detected_at"],
            "last_detected": item["row"]["last_detected_at"],
            "detections": item["row"]["detection_count"],
            "sample_detections": item["members"],
        }
        try:
            answers = clean_answers(client.run(state, QUESTIONS)["answers"], QUESTIONS)
        except Exception as exc:  # noqa: BLE001 - keep going; report how many failed
            print(f"  incident {item['id']}: failed ({exc})")
            continue
        if "real_fire" in answers:
            scored.append({"id": item["id"], "decision": item["decision"], "p_fire": answers["real_fire"]["noul"]})

    if not scored:
        sys.exit("no incidents were scored")
    pairs = [(row["p_fire"], 1 if row["decision"] == "approved" else 0) for row in scored]
    agree = sum((p >= 0.5) == bool(y) for p, y in pairs)
    mean = lambda values: sum(values) / len(values) if values else float("nan")  # noqa: E731
    print(f"scored {len(scored)}; agrees with staff at 0.5: {agree}/{len(scored)} ({agree / len(scored):.0%})")
    print(f"AUC {auc(pairs) if auc(pairs) is not None else 'n/a (needs both classes)'}")
    print(f"mean P(fire): approved {mean([p for p, y in pairs if y]):.2f}, rejected {mean([p for p, y in pairs if not y]):.2f}")
    print("compare with the existing scorer's number on the same incidents before drawing conclusions.")
    if args.csv:
        with args.csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "decision", "p_fire"])
            writer.writeheader()
            writer.writerows(scored)


if __name__ == "__main__":
    main()
