"""Area view for admin review: every incident near a seed incident, with each
member report labelled `{incident_id}_{NN}`.

Clicking one report used to show only its own incident. A real fire often
spans several incident rows (and has reports not yet attached to any), so this
gathers the whole neighbourhood in one list:

  * related incidents - near in space and time, chosen by a small logistic
    scorer (`relatedness`) rather than a single radius cut-off
  * unassigned reports - pending/approved reports with no incident yet that
    sit inside the area
  * sub-incident labels - `{incident_id}_{NNN}` where NNN is the member's
    1-based position within its incident in ingest order (event id). Computed
    on read, never stored, so a merge or split (which changes incident_id)
    can't leave a stale or colliding label behind.

The scorer's weights are hand-set (admins edit them via get/save_settings), not learned: there are no labelled
"same fire / different fire" pairs yet. Admin merge/split decisions are the
natural source for them later; the feature vector is returned with every
score (`features`) so that data is ready to collect. Read-only: nothing here
changes an incident.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
from datetime import datetime, timedelta, timezone

from core.database import _parse_occurred_at, get_db_path

MAX_RELATED = 40
MAX_PAIR_POINTS = 300  # cap per side for the nearest-point search

# logit = bias - distance*km - gap*hours + county (same county) + overlap (join areas touch)
# Admin-editable at runtime (area_review_settings table); env vars only seed
# the defaults. (default, min, max, label)
SETTING_SPECS: dict[str, tuple[float, float, float, str]] = {
    "radius_km": (float(os.getenv("FIRE_AREA_REVIEW_RADIUS_KM", "15.0")), 1.0, 50.0, "Search radius (km)"),
    "window_hours": (float(os.getenv("FIRE_AREA_REVIEW_WINDOW_HOURS", "72.0")), 1.0, 720.0, "Time window (hours)"),
    "threshold": (float(os.getenv("FIRE_AREA_REVIEW_THRESHOLD", "0.5")), 0.05, 0.99, "Relatedness cut-off (0-1)"),
    "bias": (2.0, -10.0, 10.0, "Base score"),
    "w_distance": (0.35, 0.0, 5.0, "Penalty per km apart"),
    "w_gap": (0.02, 0.0, 1.0, "Penalty per hour of time gap"),
    "w_county": (0.5, 0.0, 5.0, "Bonus for same county"),
    "w_overlap": (2.0, 0.0, 10.0, "Bonus when outlines touch"),
}


def _settings_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path(), timeout=15)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS area_review_settings ("
        "key TEXT PRIMARY KEY, value REAL NOT NULL, updated_by TEXT, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
    )
    return conn


def get_settings() -> dict[str, float]:
    values = {key: spec[0] for key, spec in SETTING_SPECS.items()}
    conn = _settings_conn()
    try:
        for key, value in conn.execute("SELECT key, value FROM area_review_settings"):
            if key in values:
                values[key] = value
    finally:
        conn.close()
    return values


def save_settings(updates: dict, updated_by: str | None = None) -> dict[str, float]:
    """Validate and store overrides; raises ValueError on an unknown key or out-of-range value."""
    clean: dict[str, float] = {}
    for key, raw in updates.items():
        if key not in SETTING_SPECS:
            raise ValueError(f"Unknown setting: {key}")
        _, low, high, label = SETTING_SPECS[key]
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{label} must be a number")
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{label} must be between {low:g} and {high:g}")
        clean[key] = value
    conn = _settings_conn()
    try:
        for key, value in clean.items():
            conn.execute(
                "INSERT INTO area_review_settings (key, value, updated_by) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_by = excluded.updated_by, "
                "updated_at = CURRENT_TIMESTAMP",
                (key, value, updated_by),
            )
        conn.commit()
    finally:
        conn.close()
    return get_settings()


def reset_settings() -> dict[str, float]:
    conn = _settings_conn()
    try:
        conn.execute("DELETE FROM area_review_settings")
        conn.commit()
    finally:
        conn.close()
    return get_settings()


def sub_incident_label(incident_id: int, seq: int) -> str:
    return f"{incident_id}_{seq:03d}"


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def _when(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = _parse_occurred_at(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _gap_hours(a: dict, b: dict) -> float:
    a0, a1 = _when(a.get("first_detected_at")), _when(a.get("last_detected_at"))
    b0, b1 = _when(b.get("first_detected_at")), _when(b.get("last_detected_at"))
    if not all((a0, a1, b0, b1)):
        return 0.0  # can't tell - don't penalise
    return max(0.0, (max(a0, b0) - min(a1, b1)).total_seconds() / 3600.0)


def _nearest_km(points_a: list[tuple], points_b: list[tuple]) -> float:
    if not points_a or not points_b:
        return float("inf")
    points_a, points_b = points_a[:MAX_PAIR_POINTS], points_b[:MAX_PAIR_POINTS]
    return min(_haversine_km(a[0], a[1], b[0], b[1]) for a in points_a for b in points_b)


def _areas_touch(a_geojson: str | None, b_geojson: str | None) -> bool:
    if not a_geojson or not b_geojson:
        return False
    try:
        from shapely.geometry import shape
        return bool(shape(json.loads(a_geojson)).intersects(shape(json.loads(b_geojson))))
    except Exception:
        return False


def relatedness(seed: dict, other: dict, seed_points: list[tuple], other_points: list[tuple], settings: dict | None = None) -> dict:
    """Probability-like 0..1 that `other` is part of the same fire area as
    `seed`, with the features that produced it."""
    cfg = settings or get_settings()
    distance = _nearest_km(seed_points, other_points)
    if not math.isfinite(distance):
        distance = _haversine_km(
            seed["centroid_latitude"], seed["centroid_longitude"],
            other["centroid_latitude"], other["centroid_longitude"],
        )
    gap = _gap_hours(seed, other)
    same_county = bool(seed.get("county_fips")) and seed.get("county_fips") == other.get("county_fips")
    overlap = _areas_touch(seed.get("join_area_geojson"), other.get("join_area_geojson"))
    logit = (cfg["bias"] - cfg["w_distance"] * distance - cfg["w_gap"] * gap
             + (cfg["w_county"] if same_county else 0.0) + (cfg["w_overlap"] if overlap else 0.0))
    return {
        "score": round(1.0 / (1.0 + math.exp(-logit)), 3),
        "features": {
            "nearest_member_km": round(distance, 2),
            "time_gap_hours": round(gap, 1),
            "same_county": same_county,
            "join_areas_touch": overlap,
        },
    }


def _members(cursor: sqlite3.Cursor, incident_id: int) -> list[dict]:
    rows = cursor.execute(
        """SELECT id, source, status, verification_tier, latitude, longitude, occurred_at,
                  satellite, confidence, frp, acres, description, reporter_org, address_text
           FROM fire_events
           WHERE incident_id = ? AND status NOT IN ('deleted')
           ORDER BY id ASC""",
        (incident_id,),
    ).fetchall()
    members = []
    for seq, row in enumerate(rows, start=1):
        member = dict(row)
        member["seq"] = seq
        member["label"] = sub_incident_label(incident_id, seq)
        member["incident_id"] = incident_id
        members.append(member)
    return members


def _incident_summary(row: sqlite3.Row, members: list[dict]) -> dict:
    sources: dict[str, int] = {}
    for member in members:
        sources[member["source"]] = sources.get(member["source"], 0) + 1
    return {
        "incident_id": row["id"],
        "public_slug": row["public_slug"],
        "status": row["status"],
        "review_decision": row["review_decision"],
        "county_name": row["county_name"],
        "centroid_latitude": row["centroid_latitude"],
        "centroid_longitude": row["centroid_longitude"],
        "first_detected_at": row["first_detected_at"],
        "last_detected_at": row["last_detected_at"],
        "member_count": len(members),
        "pending_count": sum(1 for m in members if m["status"] == "pending"),
        "sources": sources,
        "members": members,
    }


def resolve_seed_incident(event_id: int) -> int | None:
    """The incident a report belongs to, following merges; None if unattached."""
    conn = sqlite3.connect(get_db_path(), timeout=15)
    try:
        row = conn.execute("SELECT incident_id FROM fire_events WHERE id = ?", (event_id,)).fetchone()
        incident_id = row[0] if row else None
        hops = 0
        while incident_id is not None and hops < 10:
            inc = conn.execute("SELECT status, merged_into_id FROM fire_incidents WHERE id = ?", (incident_id,)).fetchone()
            if inc and inc[0] == "merged" and inc[1]:
                incident_id, hops = inc[1], hops + 1
            else:
                break
        return incident_id
    finally:
        conn.close()


def build_area(seed_incident_id: int, radius_km: float | None = None, window_hours: float | None = None) -> dict | None:
    cfg = get_settings()
    radius_km = cfg["radius_km"] if radius_km is None else radius_km
    window_hours = cfg["window_hours"] if window_hours is None else window_hours
    conn = sqlite3.connect(get_db_path(), timeout=15)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        seed = cursor.execute("SELECT * FROM fire_incidents WHERE id = ? AND status != 'deleted'", (seed_incident_id,)).fetchone()
        if not seed:
            return None
        seed = dict(seed)
        lat, lon = seed["centroid_latitude"], seed["centroid_longitude"]
        dlat = radius_km / 110.574
        dlon = radius_km / (111.320 * max(0.05, math.cos(math.radians(lat))))
        seed_members = _members(cursor, seed_incident_id)
        seed_points = [(m["latitude"], m["longitude"]) for m in seed_members]

        candidates = cursor.execute(
            """SELECT * FROM fire_incidents
               WHERE id != ? AND status NOT IN ('merged', 'deleted')
                 AND centroid_latitude BETWEEN ? AND ? AND centroid_longitude BETWEEN ? AND ?""",
            (seed_incident_id, lat - dlat, lat + dlat, lon - dlon, lon + dlon),
        ).fetchall()

        pair_labels = _labels_for(cursor, seed_incident_id)
        related = []
        for row in candidates:
            other = dict(row)
            if _gap_hours(seed, other) > window_hours:
                continue
            members = _members(cursor, other["id"])
            if not members:
                continue
            result = relatedness(seed, other, seed_points, [(m["latitude"], m["longitude"]) for m in members], cfg)
            if result["score"] < cfg["threshold"]:
                continue
            entry = _incident_summary(row, members)
            entry["relatedness"] = result
            entry["pair_label"] = pair_labels.get(row["id"])
            related.append(entry)
        related.sort(key=lambda e: e["relatedness"]["score"], reverse=True)
        related = related[:MAX_RELATED]

        # Reports with no incident yet (e.g. a fresh public submission), inside the area and window.
        since = (datetime.now(timezone.utc) - timedelta(hours=window_hours * 2)).strftime("%Y-%m-%dT%H:%M:%SZ")
        loose = cursor.execute(
            """SELECT id, source, status, verification_tier, latitude, longitude, occurred_at,
                      satellite, confidence, frp, acres, description, reporter_org, address_text
               FROM fire_events
               WHERE incident_id IS NULL AND status IN ('pending', 'approved')
                 AND latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ?
                 AND occurred_at >= ?
               ORDER BY occurred_at DESC LIMIT 100""",
            (lat - dlat, lat + dlat, lon - dlon, lon + dlon, since),
        ).fetchall()
        unassigned = []
        for row in loose:
            event = dict(row)
            event["label"] = None
            event["distance_km"] = round(_haversine_km(lat, lon, event["latitude"], event["longitude"]), 2)
            if event["distance_km"] <= radius_km:
                unassigned.append(event)

        seed_entry = _incident_summary(
            cursor.execute("SELECT * FROM fire_incidents WHERE id = ?", (seed_incident_id,)).fetchone(), seed_members
        )
        incidents = [seed_entry] + related
        all_members = [m for entry in incidents for m in entry["members"]]
        sources: dict[str, int] = {}
        for member in all_members:
            sources[member["source"]] = sources.get(member["source"], 0) + 1
        times = [e[k] for e in incidents for k in ("first_detected_at", "last_detected_at") if e.get(k)]
        return {
            "seed_incident_id": seed_incident_id,
            "radius_km": radius_km,
            "window_hours": window_hours,
            "incidents": incidents,
            "unassigned_reports": unassigned,
            "summary": {
                "incident_count": len(incidents),
                "member_count": len(all_members),
                "pending_count": sum(1 for m in all_members if m["status"] == "pending") + sum(1 for r in unassigned if r["status"] == "pending"),
                "unassigned_count": len(unassigned),
                "sources": sources,
                "first_detected_at": min(times) if times else None,
                "last_detected_at": max(times) if times else None,
            },
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Admin "same fire / different fire" labels, and fitting the weights to them.
#
# Every label stores the feature vector the scorer saw at that moment, so the
# weights can be re-fitted later without re-deriving anything. Only incidents
# the scorer already surfaced can be labelled, so the fit learns to sharpen
# the boundary of what it shows - it can't learn about fires it never offered.
# ---------------------------------------------------------------------------

PAIR_LABELS = ("same", "different")
MIN_FIT_LABELS = 20
MIN_FIT_PER_CLASS = 5
FIT_FEATURES = ("nearest_member_km", "time_gap_hours", "same_county", "join_areas_touch")
# setting key for each feature, and its sign in the logit (penalty = -1)
_FEATURE_WEIGHTS = {
    "nearest_member_km": ("w_distance", -1.0),
    "time_gap_hours": ("w_gap", -1.0),
    "same_county": ("w_county", 1.0),
    "join_areas_touch": ("w_overlap", 1.0),
}


_PAIRS_DDL = (
    "CREATE TABLE IF NOT EXISTS area_review_pair_labels ("
    "low_incident_id INTEGER NOT NULL, high_incident_id INTEGER NOT NULL, label TEXT NOT NULL, "
    "score REAL, features_json TEXT NOT NULL, decided_by TEXT, "
    "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, "
    "PRIMARY KEY (low_incident_id, high_incident_id))"
)


def _pairs_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path(), timeout=15)
    conn.execute(_PAIRS_DDL)
    return conn


def _labels_for(cursor: sqlite3.Cursor, incident_id: int) -> dict[int, str]:
    cursor.execute(_PAIRS_DDL)
    labels: dict[int, str] = {}
    for low, high, label in cursor.execute(
        "SELECT low_incident_id, high_incident_id, label FROM area_review_pair_labels "
        "WHERE low_incident_id = ? OR high_incident_id = ?", (incident_id, incident_id),
    ):
        labels[high if low == incident_id else low] = label
    return labels


def _pair_relatedness(a_id: int, b_id: int) -> dict | None:
    conn = sqlite3.connect(get_db_path(), timeout=15)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        a = cursor.execute("SELECT * FROM fire_incidents WHERE id = ? AND status != 'deleted'", (a_id,)).fetchone()
        b = cursor.execute("SELECT * FROM fire_incidents WHERE id = ? AND status != 'deleted'", (b_id,)).fetchone()
        if not a or not b:
            return None
        pts = lambda inc_id: [(m["latitude"], m["longitude"]) for m in _members(cursor, inc_id)]
        return relatedness(dict(a), dict(b), pts(a_id), pts(b_id))
    finally:
        conn.close()


def record_pair_label(incident_id: int, other_incident_id: int, label: str, decided_by: str | None = None) -> dict | None:
    """Store (or overwrite) the admin's verdict on a pair; label 'clear' removes it.
    Returns None if either incident doesn't exist. Raises ValueError on a bad label."""
    if incident_id == other_incident_id:
        raise ValueError("An incident can't be compared with itself")
    if label not in PAIR_LABELS + ("clear",):
        raise ValueError(f"label must be one of: {', '.join(PAIR_LABELS + ('clear',))}")
    low, high = sorted((incident_id, other_incident_id))
    result = _pair_relatedness(low, high)
    if result is None:
        return None
    conn = _pairs_conn()
    try:
        if label == "clear":
            conn.execute("DELETE FROM area_review_pair_labels WHERE low_incident_id = ? AND high_incident_id = ?", (low, high))
        else:
            conn.execute(
                "INSERT INTO area_review_pair_labels (low_incident_id, high_incident_id, label, score, features_json, decided_by) "
                "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(low_incident_id, high_incident_id) DO UPDATE SET "
                "label = excluded.label, score = excluded.score, features_json = excluded.features_json, "
                "decided_by = excluded.decided_by, updated_at = CURRENT_TIMESTAMP",
                (low, high, label, result["score"], json.dumps(result["features"]), decided_by),
            )
        conn.commit()
    finally:
        conn.close()
    return {"incident_id": incident_id, "other_incident_id": other_incident_id,
            "label": None if label == "clear" else label, "relatedness": result}


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def _vector(features: dict) -> list[float]:
    # signed feature values, so the logit is bias + sum(weight_i * x_i) with weight_i >= 0
    return [
        -min(float(features.get("nearest_member_km", 0.0)), 100.0),
        -min(float(features.get("time_gap_hours", 0.0)), 720.0),
        1.0 if features.get("same_county") else 0.0,
        1.0 if features.get("join_areas_touch") else 0.0,
    ]


def _accuracy(rows: list[tuple[list[float], int]], bias: float, weights: list[float], threshold: float) -> float:
    if not rows:
        return 0.0
    hits = sum(
        1 for x, y in rows
        if (_sigmoid(bias + sum(w * v for w, v in zip(weights, x))) >= threshold) == bool(y)
    )
    return round(hits / len(rows), 3)


def fit_weights() -> dict:
    """Fit bias and the four weights to the admin labels (logistic regression,
    batch gradient descent, weights kept inside their allowed ranges and
    pulled gently toward the current settings so a small sample can't swing
    them wildly). Returns a suggestion only - nothing is saved."""
    conn = _pairs_conn()
    try:
        raw = conn.execute("SELECT features_json, label FROM area_review_pair_labels").fetchall()
    finally:
        conn.close()
    rows = [(_vector(json.loads(f)), 1 if label == "same" else 0) for f, label in raw]
    same, different = sum(y for _, y in rows), sum(1 - y for _, y in rows)
    result = {"labels": len(rows), "same": same, "different": different,
              "needed": f"at least {MIN_FIT_LABELS} labels, {MIN_FIT_PER_CLASS} of each kind", "ready": False}
    if len(rows) < MIN_FIT_LABELS or same < MIN_FIT_PER_CLASS or different < MIN_FIT_PER_CLASS:
        return result

    cfg = get_settings()
    prior = [cfg["bias"]] + [cfg[_FEATURE_WEIGHTS[f][0]] for f in FIT_FEATURES]
    bounds = [(SETTING_SPECS["bias"][1], SETTING_SPECS["bias"][2])] + [
        (SETTING_SPECS[_FEATURE_WEIGHTS[f][0]][1], SETTING_SPECS[_FEATURE_WEIGHTS[f][0]][2]) for f in FIT_FEATURES
    ]
    params = list(prior)
    ridge, rate, n = 0.05, 0.05, len(rows)
    for _ in range(3000):
        grads = [0.0] * 5
        for x, y in rows:
            err = _sigmoid(params[0] + sum(w * v for w, v in zip(params[1:], x))) - y
            grads[0] += err
            for i, v in enumerate(x, start=1):
                grads[i] += err * v
        for i in range(5):
            grads[i] = grads[i] / n + ridge * (params[i] - prior[i])
            low, high = bounds[i]
            params[i] = min(high, max(low, params[i] - rate * grads[i]))

    suggested = {"bias": round(params[0], 3)}
    for feature, value in zip(FIT_FEATURES, params[1:]):
        suggested[_FEATURE_WEIGHTS[feature][0]] = round(value, 3)
    threshold = cfg["threshold"]
    result.update({
        "ready": True,
        "suggested": suggested,
        "accuracy_current": _accuracy(rows, prior[0], prior[1:], threshold),
        "accuracy_suggested": _accuracy(rows, params[0], params[1:], threshold),
    })
    return result
