"""Is this fire probably a cropland (field / crop-residue) burn?

A narrow second opinion that sits BESIDE the incident confidence score (services/incident_confidence.py), never inside
it: the score answers "is this a real fire?", this answers "given a fire, is it a field burn?". A cropland burn is still a
real fire, so nothing here lowers the confidence score.

How it works
  * Features describe the incident, not the field: the share of cultivated crops around it (NLCD, from the 480 m ground
    cover product, else the FBFM cropland share), how long it burned, the time of day it started, the season and its
    heat output. Satellite positions are only good to a few hundred metres (and GOES to ~2 km), so no field boundary is
    ever looked for - the crop share is an area average over the footprint, and the model learns how much to trust it.
  * A hand-set rules prior gives a sensible answer on day one. A logistic model is then shifted away from the prior one
    answer at a time (online SGD, pulled back toward the prior so a few answers cannot swing it). It runs on the server,
    inside the answer request: nothing trains elsewhere.
  * Every answer is stored with the features it was judged on, so the model can be rebuilt from scratch
    (`rebuild_from_labels`), rolled back to a snapshot (`rollback`), and audited. Before each update the model's own
    prediction is recorded, so its real-world accuracy (`status`) is measured on answers it had not yet learned from.
  * If the recent precision of its confident calls falls below MIN_PRECISION the prompt falls back to the rules prior.

Answers: staff outweigh the public, and one visitor's weight is capped per incident (one row per incident and visitor).
"""
from __future__ import annotations

import json
import logging
import math
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

FEATURES = ("crop", "pasture_grass", "forest", "short_burn", "multi_day", "afternoon", "season", "heat")
# Hand-set prior (logistic). Tuned so that a mostly-cropland footprint that burned for hours on an afternoon in a
# field-burning month scores ~0.95, a half-cropland one ~0.6, and a mostly-pasture/forest one stays low.
PRIOR_BIAS = -5.5
PRIOR_WEIGHTS = {
    "crop": 7.0, "pasture_grass": -1.0, "forest": -3.0, "short_burn": 1.0, "multi_day": -1.5,
    "afternoon": 0.8, "season": 0.6, "heat": 0.3,
}
# Months when crop residue is burned in the Bootheel/SE Missouri: spring cleanup (Mar-Apr), wheat stubble (Jun-Jul),
# and fall harvest residue (Sep-Nov). Only a starting point - the model learns how much it matters.
BURN_MONTHS = {3, 4, 6, 7, 9, 10, 11}

LEARNING_RATE = 0.15
PULL_TO_PRIOR = 0.02            # L2 pull of the learned shift back toward zero
STAFF_WEIGHT, PUBLIC_WEIGHT = 3.0, 1.0
PROMPT_THRESHOLD = float(os.getenv("CROPLAND_PROMPT_THRESHOLD", "0.75"))
MIN_PRECISION, PRECISION_WINDOW, PRECISION_MIN_CALLS = 0.5, 50, 10
SNAPSHOT_EVERY, SNAPSHOT_KEEP = 10, 30
CROPLAND_BURN_TYPE = "cropland"

_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS cropland_burn_labels (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        incident_id INTEGER NOT NULL,
        answer INTEGER NOT NULL,                 -- 1 = it is a cropland burn, 0 = it is not
        source TEXT NOT NULL,                    -- 'public' | 'staff'
        voter_key TEXT NOT NULL DEFAULT '',      -- hashed visitor bucket (or staff account); one answer per incident
        weight REAL NOT NULL,
        features_json TEXT NOT NULL,
        likelihood_before REAL,                  -- what the prompt was showing (the model's call) just before this answer
        method_before TEXT,
        created_at TEXT NOT NULL,
        UNIQUE (incident_id, voter_key)
    )""",
    "CREATE INDEX IF NOT EXISTS idx_cropland_labels_incident ON cropland_burn_labels(incident_id)",
    """CREATE TABLE IF NOT EXISTS cropland_burn_model (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        kind TEXT NOT NULL,                      -- 'current' | 'snapshot'
        delta_json TEXT NOT NULL,                -- learned shift away from the prior, per feature plus 'bias'
        n_updates INTEGER NOT NULL,
        created_at TEXT NOT NULL
    )""",
    "CREATE INDEX IF NOT EXISTS idx_cropland_model_kind ON cropland_burn_model(kind, id DESC)",
)


def _db() -> sqlite3.Connection:
    from core.database import get_db_path

    conn = sqlite3.connect(get_db_path(), timeout=15)
    conn.row_factory = sqlite3.Row
    for statement in _SCHEMA:
        conn.execute(statement)
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def _parse_time(value: Any) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _group_pct(groups: Any, names: Iterable[str]) -> float:
    wanted = set(names)
    return sum(float(g.get("pct") or 0) for g in groups or [] if g.get("group") in wanted) / 100.0


# ------------------------------------------------------------------------------------------------- features
def extract_features(incident: dict, members: list[dict], cover: Optional[dict] = None) -> dict[str, float]:
    """The model's inputs (each 0-1), from an incident and its member detections. `cover` is an
    incident_fuel.fuel_context() result; it is looked up when omitted. Absent evidence contributes 0, never raises."""
    if cover is None:
        try:
            from services.incident_fuel import fuel_context

            cover = fuel_context(
                incident.get("centroid_latitude"), incident.get("centroid_longitude"),
                footprints=[m.get("footprint_geojson") for m in members], outline=incident.get("shape_geojson"),
            )
        except Exception:  # noqa: BLE001 - enrichment only
            cover = None
    cover = cover or {}
    ground = cover.get("ground_cover")
    nlcd = cover.get("nlcd_groups")
    if nlcd:  # 480 m product: real NLCD classes
        crop = _group_pct(nlcd, ["Cultivated crops"])
        pasture_grass = _group_pct(nlcd, ["Pasture/hay", "Grassland"]) + _group_pct(nlcd, ["Shrub"]) * 0.5
        forest = _group_pct(nlcd, ["Forest", "Wetlands"])
    elif ground:
        crop = _group_pct(ground, ["Cropland"])
        pasture_grass = _group_pct(ground, ["Grass/Herbs", "Shrubs"])
        forest = _group_pct(ground, ["Trees"])
    else:  # coarse 3 km bundle: only the FBFM cropland class is a crop signal
        groups = cover.get("groups")
        crop = _group_pct(groups, ["Cropland"])
        pasture_grass = _group_pct(groups, ["Grass", "Grass and shrub", "Shrub"])
        forest = _group_pct(groups, ["Timber with understory", "Timber litter"])

    times = sorted(t for t in (_parse_time(m.get("occurred_at")) for m in members) if t is not None)
    if not times:
        first = _parse_time(incident.get("first_detected_at"))
        last = _parse_time(incident.get("last_detected_at"))
        times = [t for t in (first, last) if t is not None]
    duration_h = (times[-1] - times[0]).total_seconds() / 3600 if len(times) > 1 else 0.0
    # Local (Central) clock time of the first detection; a fixed UTC-6 offset is close enough for "afternoon".
    afternoon = 0.0
    month = None
    if times:
        local = times[0].astimezone(timezone.utc).timestamp() - 6 * 3600
        local_dt = datetime.fromtimestamp(local, timezone.utc)
        afternoon = 1.0 if 11 <= local_dt.hour <= 18 else 0.0
        month = local_dt.month
    peak_frp = max((float(m.get("frp") or 0.0) for m in members), default=0.0)
    return {
        "crop": round(min(1.0, crop), 3),
        "pasture_grass": round(min(1.0, pasture_grass), 3),
        "forest": round(min(1.0, forest), 3),
        "short_burn": 1.0 if duration_h <= 12 else 0.0,
        "multi_day": 1.0 if duration_h > 72 else 0.0,
        "afternoon": afternoon,
        "season": 1.0 if month in BURN_MONTHS else 0.0,
        "heat": round(min(1.0, math.log1p(peak_frp) / math.log1p(100)), 3),
    }


# ------------------------------------------------------------------------------------------------- the model
def _load_delta(conn: sqlite3.Connection) -> tuple[dict[str, float], int]:
    row = conn.execute("SELECT delta_json, n_updates FROM cropland_burn_model WHERE kind = 'current' ORDER BY id DESC LIMIT 1").fetchone()
    if not row:
        return {}, 0
    try:
        return {k: float(v) for k, v in json.loads(row["delta_json"]).items()}, int(row["n_updates"])
    except (ValueError, TypeError):
        return {}, 0


def _store_delta(conn: sqlite3.Connection, delta: dict[str, float], n_updates: int) -> None:
    payload = json.dumps({k: round(v, 6) for k, v in delta.items()})
    conn.execute("DELETE FROM cropland_burn_model WHERE kind = 'current'")
    conn.execute("INSERT INTO cropland_burn_model (kind, delta_json, n_updates, created_at) VALUES ('current', ?, ?, ?)",
                 (payload, n_updates, _now()))
    if n_updates % SNAPSHOT_EVERY == 0:
        conn.execute("INSERT INTO cropland_burn_model (kind, delta_json, n_updates, created_at) VALUES ('snapshot', ?, ?, ?)",
                     (payload, n_updates, _now()))
        conn.execute("DELETE FROM cropland_burn_model WHERE kind = 'snapshot' AND id NOT IN "
                     "(SELECT id FROM cropland_burn_model WHERE kind = 'snapshot' ORDER BY id DESC LIMIT ?)", (SNAPSHOT_KEEP,))


def _predict(features: dict[str, float], delta: dict[str, float]) -> float:
    z = PRIOR_BIAS + delta.get("bias", 0.0)
    for name in FEATURES:
        z += (PRIOR_WEIGHTS[name] + delta.get(name, 0.0)) * float(features.get(name, 0.0))
    return _sigmoid(z)


def _learn(delta: dict[str, float], features: dict[str, float], answer: int, weight: float) -> dict[str, float]:
    error = _predict(features, delta) - answer          # gradient of log loss with respect to z
    updated = dict(delta)
    for name in ("bias",) + FEATURES:
        x = 1.0 if name == "bias" else float(features.get(name, 0.0))
        step = LEARNING_RATE * weight * error * x + PULL_TO_PRIOR * delta.get(name, 0.0)
        updated[name] = delta.get(name, 0.0) - step
    return updated


def _recent_precision(conn: sqlite3.Connection) -> tuple[Optional[float], int]:
    """Precision of the model's confident calls (>= PROMPT_THRESHOLD) on the latest answers, judged on what it predicted
    BEFORE learning from each one. (None, n) until there are PRECISION_MIN_CALLS such calls."""
    rows = conn.execute(
        "SELECT answer, likelihood_before FROM cropland_burn_labels WHERE likelihood_before IS NOT NULL "
        "ORDER BY id DESC LIMIT ?", (PRECISION_WINDOW,)).fetchall()
    calls = [r["answer"] for r in rows if r["likelihood_before"] >= PROMPT_THRESHOLD]
    if len(calls) < PRECISION_MIN_CALLS:
        return None, len(calls)
    return sum(calls) / len(calls), len(calls)


def _use_learned(conn: sqlite3.Connection) -> bool:
    precision, _ = _recent_precision(conn)
    return precision is None or precision >= MIN_PRECISION


def predict(features: dict[str, float], conn: Optional[sqlite3.Connection] = None) -> tuple[float, str]:
    """(likelihood 0-1, method) where method is 'rules' (prior only) or 'learned' (prior + answers so far)."""
    own = conn is None
    conn = conn or _db()
    try:
        delta, n_updates = _load_delta(conn)
        if n_updates and _use_learned(conn):
            return _predict(features, delta), "learned"
        return _predict(features, {}), "rules"
    finally:
        if own:
            conn.close()


# ------------------------------------------------------------------------------------------------- serving
def prompt_enabled() -> bool:
    return os.getenv("CROPLAND_BURN_PROMPT", "true").strip().lower() in ("1", "true", "yes", "on")


def assess(incident: dict, members: list[dict], cover: Optional[dict] = None) -> dict[str, Any]:
    """What the incident read model returns as `cropland_burn`:
    {likelihood, label, method, show_prompt, crop_share_pct, answered}. Never raises."""
    try:
        features = extract_features(incident, members, cover)
        conn = _db()
        try:
            likelihood, method = predict(features, conn)
            answered = conn.execute("SELECT answer FROM cropland_burn_labels WHERE incident_id = ? ORDER BY id DESC LIMIT 1",
                                    (incident["id"],)).fetchone()
        finally:
            conn.close()
        label = "likely" if likelihood >= PROMPT_THRESHOLD else "possible" if likelihood >= 0.4 else "unlikely"
        return {
            "likelihood": round(likelihood, 3), "label": label, "method": method,
            "crop_share_pct": round(features["crop"] * 100),
            "answered": None if answered is None else ("yes" if answered["answer"] else "no"),
            "show_prompt": bool(prompt_enabled() and label == "likely" and answered is None),
        }
    except Exception:  # noqa: BLE001 - an extra on the incident page must never break it
        logger.info("cropland burn assessment failed", exc_info=True)
        return {"likelihood": None, "label": "unknown", "method": "unavailable", "crop_share_pct": None,
                "answered": None, "show_prompt": False}


def record_answer(incident: dict, members: list[dict], answer: bool, source: str, voter_key: str) -> dict[str, Any]:
    """Store one yes/no and learn from it immediately. A visitor's repeat answer on the same incident is ignored
    (returns recorded=False). Returns {recorded, likelihood_before, n_updates}."""
    weight = STAFF_WEIGHT if source == "staff" else PUBLIC_WEIGHT
    features = extract_features(incident, members)
    conn = _db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        delta, n_updates = _load_delta(conn)
        # What the prompt was showing (rules or learned) just before this answer; also the figure precision is judged on.
        before, method = predict(features, conn)
        try:
            conn.execute(
                """INSERT INTO cropland_burn_labels
                   (incident_id, answer, source, voter_key, weight, features_json, likelihood_before, method_before,
                    created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (incident["id"], 1 if answer else 0, source, voter_key or "", weight, json.dumps(features),
                 round(before, 4), method, _now()))
        except sqlite3.IntegrityError:
            conn.rollback()
            return {"recorded": False, "likelihood_before": round(before, 3), "n_updates": n_updates}
        delta = _learn(delta, features, 1 if answer else 0, weight)
        _store_delta(conn, delta, n_updates + 1)
        conn.commit()
        return {"recorded": True, "likelihood_before": round(before, 3), "n_updates": n_updates + 1}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ------------------------------------------------------------------------------------------------- maintenance
def rebuild_from_labels() -> dict[str, Any]:
    """Throw the learned shift away and replay every stored answer in order. Use after changing the learning rate,
    the features' meaning, or to drop a bad batch (delete its label rows first)."""
    conn = _db()
    try:
        conn.execute("BEGIN IMMEDIATE")
        delta: dict[str, float] = {}
        n = 0
        for row in conn.execute("SELECT answer, weight, features_json FROM cropland_burn_labels ORDER BY id").fetchall():
            delta = _learn(delta, json.loads(row["features_json"]), int(row["answer"]), float(row["weight"]))
            n += 1
        _store_delta(conn, delta, n)
        conn.commit()
        return {"replayed": n}
    finally:
        conn.close()


def rollback(steps: int = 1) -> Optional[int]:
    """Restore the model to its `steps`-th most recent snapshot (1 = latest). Returns the restored update count, or None
    when there is no such snapshot. The label rows are untouched; `rebuild_from_labels()` undoes a rollback."""
    conn = _db()
    try:
        row = conn.execute("SELECT delta_json, n_updates FROM cropland_burn_model WHERE kind = 'snapshot' "
                           "ORDER BY id DESC LIMIT 1 OFFSET ?", (max(0, steps - 1),)).fetchone()
        if not row:
            return None
        conn.execute("DELETE FROM cropland_burn_model WHERE kind = 'current'")
        conn.execute("INSERT INTO cropland_burn_model (kind, delta_json, n_updates, created_at) VALUES ('current', ?, ?, ?)",
                     (row["delta_json"], row["n_updates"], _now()))
        conn.commit()
        return int(row["n_updates"])
    finally:
        conn.close()


def status() -> dict[str, Any]:
    """Model health for staff: answers learned, accuracy of its calls before learning from each answer, which method
    is serving, and the learned shift away from the prior."""
    conn = _db()
    try:
        delta, n_updates = _load_delta(conn)
        precision, calls = _recent_precision(conn)
        totals = conn.execute("SELECT COUNT(*) n, SUM(answer) yes FROM cropland_burn_labels").fetchone()
        return {
            "answers": totals["n"], "yes": totals["yes"] or 0, "updates": n_updates,
            "recent_precision": None if precision is None else round(precision, 3), "recent_confident_calls": calls,
            "serving": "learned" if n_updates and _use_learned(conn) else "rules",
            "prompt_threshold": PROMPT_THRESHOLD, "prompt_enabled": prompt_enabled(),
            "learned_shift": {k: round(v, 3) for k, v in delta.items()},
        }
    finally:
        conn.close()
