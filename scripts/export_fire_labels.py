"""
Write a checksummed CSV + manifest of label-eligible fire events for the
training repo to pull into its own SMF_DATA_ROOT.

model-training/paths.py is explicit that its DB_PATH is "this repo's own
independent training DB - never the server's", so this script hands the
training side a file, not a database connection.

Usage:
    python scripts/export_fire_labels.py --min-tier official_source_confirmed
    python scripts/export_fire_labels.py --min-tier admin_reviewed --since 2026-01-01T00:00:00Z
"""
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import DATA_DIR
from core.database import export_fire_labels
from core.fire_events import TIER_RANK, VERIFICATION_TIERS

SCHEMA_VERSION = "fire-labels-v1"
FIRE_LABELS_DIR = DATA_DIR / "fire_labels"

CSV_COLUMNS = [
    "event_id", "source", "verification_tier", "label_weight",
    "latitude", "longitude", "county_fips",
    "occurred_at", "occurred_at_precision", "occurred_at_tz_offset_minutes",
    "cause_category", "official_source_system",
    "acres", "acres_is_estimate", "fuel_types",
    "frp", "confidence", "satellite",
    "label_revision", "revised_at",
]


def _label_weight(tier: str, admin_reviewed_weight: float) -> float:
    return {"official_source_confirmed": 1.0, "admin_reviewed": admin_reviewed_weight}.get(tier, 0.0)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export(min_tier: str, since: str, until: str, admin_reviewed_weight: float) -> Path:
    if min_tier not in VERIFICATION_TIERS:
        raise ValueError(f"min_tier must be one of {VERIFICATION_TIERS}")

    rows = export_fire_labels(min_tier=min_tier, since=since, until=until)
    rows_by_tier = {tier: 0 for tier in VERIFICATION_TIERS}
    for row in rows:
        rows_by_tier[row["verification_tier"]] += 1

    FIRE_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).strftime("%Y%m%d")
    csv_filename = f"fire_labels_{generated_at}.csv"
    csv_path = FIRE_LABELS_DIR / csv_filename

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            record = {col: row.get(col) for col in CSV_COLUMNS}
            record["label_weight"] = _label_weight(row["verification_tier"], admin_reviewed_weight)
            record["fuel_types"] = "|".join(row.get("fuel_types") or [])
            writer.writerow(record)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "min_tier": min_tier,
        "since": since,
        "until": until,
        "row_count": len(rows),
        "rows_by_tier": rows_by_tier,
        "cause_categories": _histogram(rows, "cause_category"),
        "csv_filename": csv_filename,
        "csv_sha256": _sha256_file(csv_path),
        "weights": {"official_source_confirmed": 1.0, "admin_reviewed": admin_reviewed_weight, "unverified": 0.0},
    }
    manifest_path = FIRE_LABELS_DIR / "fire_labels_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return csv_path


def _histogram(rows, key: str) -> dict:
    hist: dict = {}
    for row in rows:
        value = row.get(key) or "unknown"
        hist[value] = hist.get(value, 0) + 1
    return hist


def main():
    parser = argparse.ArgumentParser(description="Export label-eligible fire events for model training.")
    parser.add_argument("--min-tier", default="admin_reviewed", choices=VERIFICATION_TIERS)
    parser.add_argument("--since", default=None)
    parser.add_argument("--until", default=None)
    parser.add_argument("--admin-reviewed-weight", type=float, default=0.3)
    args = parser.parse_args()

    csv_path = export(args.min_tier, args.since, args.until, args.admin_reviewed_weight)
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
