"""How well do the satellite detections line up with fires we know were real?

The reference fires are independent of the satellites: events a person or an official source reported and that a
reviewer approved (public reports, ingested leads, official FPA-FOD records), not the satellite incidents we review
ourselves. For each one we look for a satellite detection (any status, pending included) within `radius_km` and from
`before_hours` before to `after_hours` after the reported time, and report how often one exists, how big the fires
were that were missed, and how long after the report the first detection came.

This is a recall check ("of the fires people reported, how many did the satellites see?"). It cannot measure false
detections: a detection with no reference fire nearby is not necessarily false, since reports and FPA-FOD cover only
some fires (FPA-FOD leaves out prescribed and agricultural burns).

  python scripts/compare_detections_to_confirmed.py [--radius-km 3] [--before-hours 6] [--after-hours 24] [--csv FILE]
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta
from statistics import median
from typing import Any, Dict, List, Optional

from core.geo import degree_box, haversine_km

SATELLITE_SOURCES = ("modis", "viirs", "ngfs")
ACRE_BINS = (("under 1 acre", 0, 1), ("1-10 acres", 1, 10), ("10-100 acres", 10, 100), ("100+ acres", 100, float("inf")))


def _when(value: str) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def coverage(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Date range and counts of what is in the database, so it is clear what any comparison can cover."""
    sources = [dict(zip(("source", "events", "first", "last", "approved", "rejected"), row)) for row in conn.execute(
        """SELECT source, COUNT(*), MIN(occurred_at), MAX(occurred_at),
                  SUM(status = 'approved'), SUM(status = 'rejected')
           FROM fire_events WHERE status != 'deleted' GROUP BY source ORDER BY COUNT(*) DESC""")]
    satellite = [s for s in sources if s["source"] in SATELLITE_SOURCES]
    return {
        "sources": sources,
        "satellite_first": min((s["first"] for s in satellite), default=None),
        "satellite_last": max((s["last"] for s in satellite), default=None),
    }


def reference_fires(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Approved events that did not come from a satellite."""
    marks = ",".join("?" for _ in SATELLITE_SOURCES)
    conn.row_factory = sqlite3.Row
    return [dict(row) for row in conn.execute(
        f"""SELECT id, source, occurred_at, occurred_at_precision, latitude, longitude, acres, county_name
            FROM fire_events WHERE status = 'approved' AND source NOT IN ({marks}) ORDER BY occurred_at""",
        SATELLITE_SOURCES)]


def match(conn: sqlite3.Connection, reference: Dict[str, Any], radius_km: float, before_hours: float,
          after_hours: float) -> Dict[str, Any]:
    """The detections around one reference fire. A day-precision time (no time of day known) widens the window by
    12 hours each way."""
    slack = 12 if reference.get("occurred_at_precision") == "day" else 0
    when = _when(reference["occurred_at"])
    start, end = when - timedelta(hours=before_hours + slack), when + timedelta(hours=after_hours + slack)
    min_lat, max_lat, min_lon, max_lon = degree_box(reference["latitude"], reference["longitude"], radius_km)
    marks = ",".join("?" for _ in SATELLITE_SOURCES)
    rows = conn.execute(
        f"""SELECT id, source, occurred_at, latitude, longitude, status FROM fire_events
            WHERE source IN ({marks}) AND status != 'deleted'
              AND latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ?
              AND occurred_at BETWEEN ? AND ?""",
        (*SATELLITE_SOURCES, min_lat, max_lat, min_lon, max_lon, start.isoformat().replace("+00:00", "Z"),
         end.isoformat().replace("+00:00", "Z"))).fetchall()
    found = []
    for row in rows:
        distance = haversine_km(reference["latitude"], reference["longitude"], row[3], row[4])
        if distance <= radius_km:
            found.append({"id": row[0], "source": row[1], "distance_km": distance,
                          "hours_after_report": (_when(row[2]) - when).total_seconds() / 3600, "status": row[5]})
    found.sort(key=lambda item: item["hours_after_report"])
    return {**reference, "detections": found, "detected": bool(found),
            "first_lead_hours": found[0]["hours_after_report"] if found else None,
            "nearest_km": min((item["distance_km"] for item in found), default=None)}


def _bin(acres: Optional[float]) -> str:
    if acres is None:
        return "size not given"
    return next((label for label, low, high in ACRE_BINS if low <= acres < high), "size not given")


def compare(conn: sqlite3.Connection, radius_km: float = 3.0, before_hours: float = 6.0,
            after_hours: float = 24.0) -> Dict[str, Any]:
    matches = [match(conn, ref, radius_km, before_hours, after_hours) for ref in reference_fires(conn)]
    by_size: Dict[str, Dict[str, int]] = {}
    for item in matches:
        entry = by_size.setdefault(_bin(item.get("acres")), {"fires": 0, "detected": 0})
        entry["fires"] += 1
        entry["detected"] += item["detected"]
    leads = [m["first_lead_hours"] for m in matches if m["detected"]]
    by_source: Dict[str, int] = {}
    for item in matches:
        for source in {d["source"] for d in item["detections"]}:
            by_source[source] = by_source.get(source, 0) + 1
    return {
        "settings": {"radius_km": radius_km, "before_hours": before_hours, "after_hours": after_hours},
        "coverage": coverage(conn),
        "reference_fires": len(matches),
        "detected": sum(m["detected"] for m in matches),
        "detection_rate": (sum(m["detected"] for m in matches) / len(matches)) if matches else None,
        "median_lead_hours": median(leads) if leads else None,
        "by_size": by_size,
        "detected_by_source": by_source,
        "matches": matches,
    }
