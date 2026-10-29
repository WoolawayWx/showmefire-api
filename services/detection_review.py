"""Reviewed vegetation-fire outcomes for satellite detections: sampling, outcomes, stats, and how the shadow models do.

Reviewers never see any confidence score (v1 or candidate), to avoid anchoring. Outcomes are append-only: a correction or
a second opinion is a new row, never an update. Two sampling streams are kept apart on purpose:
  random   - uniform over ALL incoming detections (masked and low-scoring included); the only stream that supports an
             unbiased estimate of performance, and 1-in-5 of it is a protected subset never used for training.
  enriched - deliberately picked buckets (masked, extreme, unusual) to get varied examples cheaply; its class balance is
             artificial, so metrics from it say nothing about the live stream.
"""
from __future__ import annotations

import hashlib
import json
import random
import sqlite3
from pathlib import Path
from typing import Any, Optional

from core import database

TARGET_VERSION = "veg-v1"
SAMPLING_VERSION = "random-v1"
SATELLITE_SOURCES = ("viirs", "modis", "ngfs")
OUTCOMES = ("vegetation_fire", "not_vegetation_fire", "cannot_determine")
REASONS = ("", "routine_industrial_heat", "reflection_or_glint", "structure_only_fire", "other_non_vegetation",
           "insufficient_evidence", "conflicting_evidence")
STREAMS = ("random", "enriched")
ENRICHED_BUCKETS = ("masked", "extreme_signal", "low_native_confidence", "mid_range_v1")
PROTECTED_ONE_IN = 5
PILOT_TARGET = 100
MIN_FOR_METRICS = 30


class ReviewError(ValueError):
    pass


def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or database.get_db_path(), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def is_protected(event_id: int) -> bool:
    """Stable 1-in-5 holdout flag. A hash, not Python's randomised hash(), so it is the same on every machine and run."""
    digest = hashlib.md5(f"veg-protected-{int(event_id)}".encode()).hexdigest()
    return int(digest[:8], 16) % PROTECTED_ONE_IN == 0


def _population_n(conn) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM fire_events WHERE source IN ('viirs','modis','ngfs')").fetchone()[0])


_BUCKET_SQL = {
    "masked": "(e.exclusion_zone_id IS NOT NULL OR e.recurring_source_id IS NOT NULL)",
    "extreme_signal": "(e.frp >= 50 OR e.bright_t7 >= 360)",
    "low_native_confidence": "(lower(CAST(e.confidence AS TEXT)) IN ('l','low') OR (CAST(e.confidence AS REAL) > 0 AND CAST(e.confidence AS REAL) < 30))",
    "mid_range_v1": "(e.detection_confidence_pct BETWEEN 40 AND 50)",
}


def _pick(conn, where: str, params: tuple = ()) -> Optional[int]:
    row = conn.execute(
        f'''SELECT e.id FROM fire_events e
            WHERE e.source IN ('viirs','modis','ngfs') AND {where}
              AND NOT EXISTS (SELECT 1 FROM detection_review_outcome o WHERE o.event_id = e.id)
            ORDER BY RANDOM() LIMIT 1''', params).fetchone()
    return int(row[0]) if row else None


def next_detection(stream: str = "random", reviewer: str = "", db_path: Optional[Path] = None) -> Optional[dict]:
    """One unreviewed detection to review, with the selection recorded (stream, bucket, population size)."""
    if stream not in STREAMS:
        raise ReviewError(f"stream must be one of {STREAMS}")
    conn = _connect(db_path)
    try:
        bucket, event_id = "", None
        if stream == "random":
            event_id = _pick(conn, "1=1")
        else:
            order = list(ENRICHED_BUCKETS)
            random.shuffle(order)
            for name in order:
                event_id = _pick(conn, _BUCKET_SQL[name])
                if event_id:
                    bucket = name
                    break
        if not event_id:
            return None
        already = conn.execute("SELECT 1 FROM detection_review_selection WHERE event_id = ? AND stream = ?", (event_id, stream)).fetchone()
        if not already:
            conn.execute(
                '''INSERT INTO detection_review_selection (event_id, stream, bucket, sampling_version, population_n, protected)
                   VALUES (?, ?, ?, ?, ?, ?)''',
                (event_id, stream, bucket, SAMPLING_VERSION, _population_n(conn), int(stream == "random" and is_protected(event_id))))
            conn.commit()
        return review_payload(conn, event_id)
    finally:
        conn.close()


def second_review(reviewer: str, db_path: Optional[Path] = None) -> Optional[dict]:
    """A detection another reviewer has already judged and nobody has double-checked (for agreement tracking)."""
    conn = _connect(db_path)
    try:
        row = conn.execute(
            '''SELECT o.event_id FROM detection_review_outcome o
               GROUP BY o.event_id
               HAVING COUNT(*) = 1 AND MAX(o.reviewer) != ?
               ORDER BY RANDOM() LIMIT 1''', (reviewer,)).fetchone()
        return review_payload(conn, int(row[0])) if row else None
    finally:
        conn.close()


def review_payload(conn, event_id: int) -> dict:
    """What the reviewer sees. No confidence scores of any kind; the sensor's own native confidence is raw evidence."""
    e = conn.execute(
        '''SELECT id, source, satellite, occurred_at, latitude, longitude, frp, confidence, bright_t7, bright_t13, pixel_area,
                  daynight, land_cover, footprint_geojson, exclusion_zone_id, recurring_source_id, county_name
           FROM fire_events WHERE id = ?''', (event_id,)).fetchone()
    if not e:
        raise ReviewError("detection not found")
    e = dict(e)
    lat, lon = float(e["latitude"]), float(e["longitude"])
    dlat, dlon = 0.03, 0.04  # ~3 km box for nearby observation history
    nearby = conn.execute(
        '''SELECT id, source, satellite, occurred_at, frp FROM fire_events
           WHERE source IN ('viirs','modis','ngfs') AND id != ? AND latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ?
             AND occurred_at BETWEEN datetime(?, '-7 days') AND datetime(?, '+1 day')
           ORDER BY occurred_at DESC LIMIT 30''',
        (event_id, lat - dlat, lat + dlat, lon - dlon, lon + dlon, e["occurred_at"], e["occurred_at"])).fetchall()
    mask = {}
    if e["exclusion_zone_id"]:
        z = conn.execute("SELECT name, notes FROM fire_exclusion_zones WHERE id = ?", (e["exclusion_zone_id"],)).fetchone()
        mask["exclusion_zone"] = dict(z) if z else {"id": e["exclusion_zone_id"]}
    if e["recurring_source_id"]:
        r = conn.execute("SELECT label, source_type, detection_count, distinct_day_count FROM recurring_fire_sources WHERE id = ?",
                         (e["recurring_source_id"],)).fetchone()
        mask["recurring_source"] = dict(r) if r else {"id": e["recurring_source_id"]}
    prior = conn.execute("SELECT outcome, reviewer FROM detection_review_outcome WHERE event_id = ? ORDER BY id", (event_id,)).fetchall()
    footprint = None
    if e["footprint_geojson"]:
        try:
            footprint = json.loads(e["footprint_geojson"])
        except ValueError:
            footprint = None
    return {
        "detection": {k: e[k] for k in ("id", "source", "satellite", "occurred_at", "latitude", "longitude", "frp", "confidence",
                                          "bright_t7", "bright_t13", "pixel_area", "daynight", "land_cover", "county_name")},
        "footprint": footprint,
        "mask_context": mask,
        "observation_history": [dict(r) for r in nearby],
        "imagery_links": {
            "google_satellite": f"https://www.google.com/maps/@{lat:.5f},{lon:.5f},17z/data=!3m1!1e3",
            "sentinel_eo_browser": f"https://apps.sentinel-hub.com/eo-browser/?zoom=15&lat={lat:.5f}&lng={lon:.5f}",
            "nasa_worldview": f"https://worldview.earthdata.nasa.gov/?v={lon - 0.2:.3f},{lat - 0.15:.3f},{lon + 0.2:.3f},{lat + 0.15:.3f}",
        },
        "already_reviewed_by": sorted({r["reviewer"] for r in prior}),
    }


def submit_outcome(event_id: int, outcome: str, reviewer: str, reason: str = "", evidence: str = "",
                   db_path: Optional[Path] = None) -> dict:
    if outcome not in OUTCOMES:
        raise ReviewError(f"outcome must be one of {OUTCOMES}")
    if reason not in REASONS:
        raise ReviewError(f"reason must be one of {REASONS}")
    if not reviewer:
        raise ReviewError("reviewer required")
    conn = _connect(db_path)
    try:
        if not conn.execute("SELECT 1 FROM fire_events WHERE id = ? AND source IN ('viirs','modis','ngfs')", (event_id,)).fetchone():
            raise ReviewError("detection not found")
        prior = [dict(r) for r in conn.execute(
            "SELECT outcome, reviewer, adjudication_status FROM detection_review_outcome WHERE event_id = ? ORDER BY id", (event_id,))]
        status = "single"
        others = [p for p in prior if p["reviewer"] != reviewer]
        if others:
            latest_other = others[-1]
            if prior and prior[-1]["adjudication_status"] == "conflict":
                status = "resolved"  # a third opinion settles an open conflict
            else:
                status = "agree" if latest_other["outcome"] == outcome else "conflict"
        conn.execute(
            '''INSERT INTO detection_review_outcome (event_id, target_version, outcome, reason, evidence, reviewer, adjudication_status)
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (event_id, TARGET_VERSION, outcome, reason, (evidence or "")[:2000], reviewer, status))
        conn.commit()
        return {"event_id": event_id, "outcome": outcome, "adjudication_status": status}
    finally:
        conn.close()


def final_outcomes(conn) -> dict[int, int]:
    """event_id -> 1/0 for detections with a settled binary outcome. Cannot-determine and unresolved conflicts are left
    out (and counted in stats(), never silently dropped from the reporting)."""
    out: dict[int, int] = {}
    latest: dict[int, sqlite3.Row] = {}
    for r in conn.execute("SELECT * FROM detection_review_outcome ORDER BY id"):
        latest[int(r["event_id"])] = r
    for eid, r in latest.items():
        if r["adjudication_status"] == "conflict" or r["outcome"] == "cannot_determine":
            continue
        out[eid] = 1 if r["outcome"] == "vegetation_fire" else 0
    return out


def stats(db_path: Optional[Path] = None) -> dict:
    conn = _connect(db_path)
    try:
        latest: dict[int, sqlite3.Row] = {}
        for r in conn.execute("SELECT * FROM detection_review_outcome ORDER BY id"):
            latest[int(r["event_id"])] = r
        by_outcome: dict[str, int] = {}
        conflicts = 0
        for r in latest.values():
            by_outcome[r["outcome"]] = by_outcome.get(r["outcome"], 0) + 1
            conflicts += int(r["adjudication_status"] == "conflict")
        double = conn.execute("SELECT COUNT(*) FROM (SELECT event_id FROM detection_review_outcome GROUP BY event_id HAVING COUNT(DISTINCT reviewer) > 1)").fetchone()[0]
        selections = {f"{r['stream']}:{r['bucket'] or '-'}": r["n"] for r in conn.execute(
            "SELECT stream, bucket, COUNT(*) n FROM detection_review_selection GROUP BY stream, bucket")}
        by_sensor = {r["source"]: r["n"] for r in conn.execute(
            '''SELECT e.source, COUNT(DISTINCT o.event_id) n FROM detection_review_outcome o JOIN fire_events e ON e.id = o.event_id GROUP BY e.source''')}
        decided = len(latest)
        undecidable = by_outcome.get("cannot_determine", 0)
        return {
            "reviewed": decided, "by_outcome": by_outcome, "by_sensor": by_sensor, "selections": selections,
            "cannot_determine_rate": round(undecidable / decided, 3) if decided else None,
            "double_reviewed": int(double), "open_conflicts": conflicts,
            "pilot": {"target": PILOT_TARGET, "reviewed": decided},
            "target_version": TARGET_VERSION,
        }
    finally:
        conn.close()


# ---------------------------------------------------------------- how the shadow models are doing

def _metrics(y: list[int], p: list[float]) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
    out: dict[str, Any] = {"n": len(y), "positives": int(sum(y))}
    if len(y) < MIN_FOR_METRICS or len(set(y)) < 2:
        out["note"] = f"too few reviewed detections to conclude (need >= {MIN_FOR_METRICS} with both outcomes)"
        return out
    out.update({"auc": round(float(roc_auc_score(y, p)), 4), "average_precision": round(float(average_precision_score(y, p)), 4),
                "brier": round(float(brier_score_loss(y, p)), 4)})
    return out


def shadow_summary(db_path: Optional[Path] = None) -> dict:
    """AUC / average precision / Brier of v1 and each candidate on reviewed RANDOM-stream detections (the unbiased
    sample), plus the same on the protected subset alone, and a prevalence-only Brier baseline. v1's number is a raw
    score, not a calibrated probability, so its Brier is shown but not comparable until it is calibrated; AUC and
    average precision are the fair comparison. Informational only: nothing acts on this."""
    conn = _connect(db_path)
    try:
        outcomes = final_outcomes(conn)
        random_sel = {int(r["event_id"]): bool(r["protected"]) for r in conn.execute(
            "SELECT event_id, MAX(protected) protected FROM detection_review_selection WHERE stream = 'random' GROUP BY event_id")}
        scores: dict[tuple, dict[int, float]] = {}
        for r in conn.execute("SELECT event_id, model_name, model_version, score FROM detection_score_history WHERE mode = 'initial' ORDER BY id"):
            scores.setdefault((r["model_name"], r["model_version"]), {})[int(r["event_id"])] = float(r["score"])
        pool = {eid: y for eid, y in outcomes.items() if eid in random_sel}
        report: dict[str, Any] = {
            "reviewed_random_stream": len(pool), "protected_subset": sum(1 for e in pool if random_sel[e]),
            "prevalence": round(sum(pool.values()) / len(pool), 4) if pool else None, "models": {},
            "note": "random-stream reviews only; informational, no decisions are made from this",
        }
        if pool:
            prev = sum(pool.values()) / len(pool)
            report["prevalence_only_brier"] = round(sum((y - prev) ** 2 for y in pool.values()) / len(pool), 4)
        for (name, version), table in scores.items():
            ids = [e for e in pool if e in table]
            key = f"{name}@{version}"
            report["models"][key] = {
                "random_stream": _metrics([pool[e] for e in ids], [table[e] for e in ids]),
                "protected_only": _metrics([pool[e] for e in ids if random_sel[e]], [table[e] for e in ids if random_sel[e]]),
                "coverage": f"{len(ids)} of {len(pool)} reviewed detections have a score",
            }
        return report
    finally:
        conn.close()
