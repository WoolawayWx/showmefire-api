"""Shadow-score incident confidence with a stand-in land-cover string. Never changes a live score.

Why: services/incident_confidence.py takes its cropland and water fractions from the NOAA NGFS `land_cover` string on the
incident's detections. Incidents seen only by VIIRS, MODIS, reports or dispatch have no such string, so those penalties
never apply to them (about 41% of incidents when this was written). The 480 m ground cover product (services/ground_cover.py)
can supply the same fractions for every incident. Before anyone relies on that, this job records what the score WOULD be:

  live_score    the v2 scorer on the incident exactly as stored
  shadow_score  the same scorer on a copy of the members with a stand-in `land_cover` string from our NLCD mix

Both use the v2 scorer so the only difference is the land cover; `live_is_v2` records whether v2 is what the site shows.
Rows go to `incident_confidence_shadow` only. Open question this answers: the cropland penalty was designed around NGFS's
coarse pixels, and whether it suits 375 m VIIRS pixels is not known from the code - hence a shadow comparison first.

Review with `python scripts/review_ground_cover_shadow.py` after about two weeks and at least ~100 incidents. Turning the
stand-in string on for live scoring would be a separate, flagged change made only after that review.
Disable the job with SMF_GROUND_COVER_SHADOW=0.
"""
from __future__ import annotations

import copy
import logging
import os
import sqlite3
import statistics
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

ACTIVE_HOURS = 24          # incidents with a detection in this window are scored
SCAN_LIMIT = 500
REVIEW_MIN_DAYS = 14
REVIEW_MIN_INCIDENTS = 100


def enabled() -> bool:
    return os.getenv("SMF_GROUND_COVER_SHADOW", "1").strip().lower() not in {"0", "false", "no", "off"}


def stand_in_land_cover(ground_cover: list[dict]) -> str:
    """NOAA-style string ('Cropland:40,Trees:55,...') from the product's NOAA-named classes. 'Other' is left out."""
    parts = [f"{g['group']}:{g['pct']}" for g in ground_cover or [] if g.get("group") != "Other" and g.get("pct", 0) > 0]
    return ",".join(parts)


def _fraction(land_cover: str, *keywords: str) -> float:
    from services.incident_confidence import _land_cover_fraction

    return _land_cover_fraction(land_cover, *keywords)


def score_pair(incident: dict, members: list[dict], stand_in: str) -> dict[str, Any]:
    """Score once as stored and once with `stand_in` on every member. Works on copies; inputs are not modified."""
    from services.incident_confidence import score_incident

    live = score_incident(copy.deepcopy(incident), copy.deepcopy(members))
    shadow_members = copy.deepcopy(members)
    for member in shadow_members:
        member["land_cover"] = stand_in
    shadow = score_incident(copy.deepcopy(incident), shadow_members)
    return {"live": live, "shadow": shadow}


def _connect() -> sqlite3.Connection:
    from core.database import get_db_path

    conn = sqlite3.connect(get_db_path(), timeout=15)
    conn.row_factory = sqlite3.Row
    return conn


def _already_recorded(conn: sqlite3.Connection, incident_id: int, live: int, shadow: int, stand_in: str) -> bool:
    row = conn.execute(
        "SELECT live_score, shadow_score, stand_in_land_cover FROM incident_confidence_shadow "
        "WHERE incident_id = ? ORDER BY id DESC LIMIT 1", (incident_id,)).fetchone()
    return bool(row) and (row["live_score"], row["shadow_score"], row["stand_in_land_cover"]) == (live, shadow, stand_in)


def run_shadow_scan(now: Optional[datetime] = None) -> dict[str, int]:
    """Score recent incidents that have no NOAA land-cover string and record the comparison. Returns a job summary."""
    from core.database import list_fire_incident_members, list_fire_incidents
    from services import ground_cover
    from services.incident_confidence import use_v2

    summary = {"considered": 0, "has_noaa_string": 0, "no_ground_cover": 0, "unchanged": 0, "recorded": 0}
    if not enabled():
        return summary
    now = now or datetime.now(timezone.utc)
    since = (now - timedelta(hours=ACTIVE_HOURS)).strftime("%Y-%m-%dT%H:%M:%S")
    incidents = list_fire_incidents(since=since, limit=SCAN_LIMIT)
    conn = _connect()
    try:
        for incident in incidents:
            summary["considered"] += 1
            members = list_fire_incident_members(incident["id"])
            if any(m.get("land_cover") for m in members):
                summary["has_noaa_string"] += 1      # NOAA's own mix already drives the live penalty
                continue
            sample = ground_cover.sample(
                incident.get("centroid_latitude"), incident.get("centroid_longitude"),
                footprints=[m.get("footprint_geojson") for m in members], outline=incident.get("shape_geojson"))
            stand_in = stand_in_land_cover(sample["ground_cover"]) if sample else ""
            if not stand_in:
                summary["no_ground_cover"] += 1
                continue
            result = score_pair(incident, members, stand_in)
            live, shadow = result["live"], result["shadow"]
            if _already_recorded(conn, incident["id"], live["score"], shadow["score"], stand_in):
                summary["unchanged"] += 1
                continue
            conn.execute(
                "INSERT INTO incident_confidence_shadow (incident_id, scored_at, live_score, live_label, shadow_score, "
                "shadow_label, delta, stand_in_land_cover, cropland_pct, water_pct, basis, cells, bundle_version, "
                "member_count, live_is_v2) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (incident["id"], now.isoformat(), live["score"], live["label"], shadow["score"], shadow["label"],
                 shadow["score"] - live["score"], stand_in,
                 round(100 * _fraction(stand_in, "Cropland", "Agricult"), 1), round(100 * _fraction(stand_in, "Water"), 1),
                 sample["basis"], sample["cells"], str(sample.get("bundle_version") or ""), len(members),
                 int(use_v2())))
            summary["recorded"] += 1
        conn.commit()
    finally:
        conn.close()
    logger.info("ground_cover_shadow: %s", summary)
    return summary


def review(days: int = 30, top: int = 20, now: Optional[datetime] = None) -> dict[str, Any]:
    """Summary of the shadow comparison for the review step: uses each incident's most recent row in the window."""
    now = now or datetime.now(timezone.utc)
    cutoff = (now - timedelta(days=days)).isoformat()
    conn = _connect()
    try:
        rows = [dict(r) for r in conn.execute(
            "SELECT s.*, i.public_slug FROM incident_confidence_shadow s "
            "LEFT JOIN fire_incidents i ON i.id = s.incident_id "
            "WHERE s.scored_at >= ? AND s.id IN (SELECT MAX(id) FROM incident_confidence_shadow "
            "WHERE scored_at >= ? GROUP BY incident_id)", (cutoff, cutoff))]
        first = conn.execute("SELECT MIN(scored_at) AS t FROM incident_confidence_shadow").fetchone()["t"]
    finally:
        conn.close()
    deltas = [r["delta"] for r in rows]
    flips: dict[str, int] = {}
    for r in rows:
        if r["live_label"] != r["shadow_label"]:
            key = f"{r['live_label']} -> {r['shadow_label']}"
            flips[key] = flips.get(key, 0) + 1
    span_days = 0.0
    if first:
        parsed = datetime.fromisoformat(first)
        span_days = (now - (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc))).total_seconds() / 86400
    by_basis: dict[str, int] = {}
    for r in rows:
        by_basis[r["basis"]] = by_basis.get(r["basis"], 0) + 1
    return {
        "incidents": len(rows),
        "days_of_data": round(span_days, 1),
        "ready_for_review": span_days >= REVIEW_MIN_DAYS and len(rows) >= REVIEW_MIN_INCIDENTS,
        "readiness_rule": f"at least {REVIEW_MIN_DAYS} days and {REVIEW_MIN_INCIDENTS} incidents",
        "score_change": {
            "unchanged": sum(1 for d in deltas if d == 0), "lowered": sum(1 for d in deltas if d < 0),
            "mean": round(statistics.fmean(deltas), 2) if deltas else None,
            "median": statistics.median(deltas) if deltas else None,
            "min": min(deltas) if deltas else None, "max": max(deltas) if deltas else None,
        },
        "label_flips": flips,
        "label_flip_count": sum(flips.values()),
        "by_basis": by_basis,
        "largest_changes": [
            {k: r[k] for k in ("incident_id", "public_slug", "live_score", "live_label", "shadow_score", "shadow_label",
                               "delta", "stand_in_land_cover", "basis")}
            for r in sorted(rows, key=lambda r: abs(r["delta"]), reverse=True)[:top]],
    }
