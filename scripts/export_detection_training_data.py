"""
Write checksummed CSVs + a manifest of every satellite/NGFS detection, the
incidents they belong to, and the raw public/admin feedback on those
incidents, for model-training/detection_dataset to pull into SMF_DATA_ROOT.

Read-only: opens the SQLite DB with mode=ro. Like export_fire_labels.py this
hands the training side files, not a DB connection.

Usage:
    python scripts/export_detection_training_data.py
    python scripts/export_detection_training_data.py --since 2025-12-01T00:00:00Z
"""
import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import DATA_DIR
from core.database import get_db_path

SCHEMA_VERSION = "detection-training-v1"
OUT_DIR = DATA_DIR / "fire_labels"
DETECTION_SOURCES = ("viirs", "modis", "ngfs", "goes")

DETECTION_COLUMNS = [
    "id", "source", "external_id", "satellite", "occurred_at", "latitude", "longitude",
    "county_fips", "frp", "confidence", "bright_t7", "bright_t13", "pixel_area", "quality_flag",
    "solar_zenith_angle", "satellite_zenith_angle", "daynight", "land_cover",
    "detection_confidence_pct", "footprint_geojson", "recurring_source_id", "exclusion_zone_id",
    "fuel_model_fbfm40", "canopy_cover_pct", "weather_danger_category", "weather_danger_prob",
    "incident_id", "verification_tier", "cause_category", "status", "moderated_at", "first_seen_at",
]
INCIDENT_COLUMNS = [
    "id", "centroid_latitude", "centroid_longitude", "first_detected_at", "last_detected_at",
    "detection_count", "county_fips", "status", "merged_into_id", "review_decision", "review_label",
    "review_decided_at", "created_at",
]
FEEDBACK_COLUMNS = [
    "id", "incident_id", "classification", "status", "reviewed_by", "reviewed_at", "created_at",
    "submission_source",
]
REPORT_COLUMNS = [
    "id", "latitude", "longitude", "county_fips", "occurred_at", "status", "verification_tier",
    "cause_category", "reported_fire_status", "reporter_relationship", "acres", "moderated_at",
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _present(conn: sqlite3.Connection, table: str, wanted: list) -> list:
    """Older DB copies lack newer columns - export only what exists."""
    have = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    return [c for c in wanted if c in have]


def _dump(conn, name: str, table: str, wanted: list, where: str = "", params: tuple = ()) -> dict:
    cols = _present(conn, table, wanted)
    rows = conn.execute(f"SELECT {', '.join(cols)} FROM {table} {where}", params).fetchall()
    path = OUT_DIR / name
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(wanted)  # fixed header; absent columns come back blank
        index = {c: i for i, c in enumerate(cols)}
        for row in rows:
            writer.writerow([row[index[c]] if c in index else "" for c in wanted])
    return {"filename": name, "rows": len(rows), "sha256": _sha256_file(path), "missing_columns": [c for c in wanted if c not in cols]}


def _range(conn, where: str, params: tuple) -> dict:
    out = {}
    for source, lo, hi, n in conn.execute(
        f"SELECT source, MIN(occurred_at), MAX(occurred_at), COUNT(*) FROM fire_events {where} GROUP BY source", params
    ):
        out[source] = {"min": lo, "max": hi, "count": n}
    return out


def export(since: str | None, db_path: Path | None = None) -> Path:
    db_path = db_path or get_db_path()
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        marks = ",".join("?" for _ in DETECTION_SOURCES)
        where = f"WHERE source IN ({marks})"
        params: tuple = DETECTION_SOURCES
        if since:
            where += " AND occurred_at >= ?"
            params = params + (since,)

        files = {
            "detections": _dump(conn, "detections.csv", "fire_events", DETECTION_COLUMNS, where, params),
            "incidents": _dump(conn, "incidents.csv", "fire_incidents", INCIDENT_COLUMNS),
            "feedback": _dump(conn, "feedback.csv", "fire_incident_feedback", FEEDBACK_COLUMNS),
            "user_reports": _dump(conn, "user_reports.csv", "fire_events", REPORT_COLUMNS, "WHERE source = 'user_submission'"),
        }
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "since": since,
            "files": files,
            "detection_ranges": _range(conn, where, params),
        }
    finally:
        conn.close()
    manifest_path = OUT_DIR / "detection_training_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def main():
    parser = argparse.ArgumentParser(description="Export detections, incidents and feedback for detection-confidence training.")
    parser.add_argument("--since", default=None)
    args = parser.parse_args()
    print(f"Wrote {export(args.since)}")


if __name__ == "__main__":
    main()
