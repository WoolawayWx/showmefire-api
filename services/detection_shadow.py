"""Shadow scoring for the vegetation-fire confidence candidate (v2).

Records, for every satellite detection, the features known at scoring time and what v1 (and v2, when a trained bundle
exists) said, in append-only tables. It NEVER writes detection_confidence_pct and changes nothing user-facing: this is
for measuring how the candidate does against reviewed outcomes, not for making decisions.

Off unless SMF_DETECTION_SHADOW=true. Every entry point swallows errors so ingest is never affected.
"""
from __future__ import annotations

import importlib.util
import json
import logging
import math
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from core import database
from services.detection_bundle import BUNDLE_SCHEMA, score_bundle  # noqa: F401 (re-exported for callers/tests)

logger = logging.getLogger(__name__)

SATELLITE_SOURCES = ("viirs", "modis", "ngfs")
V1_NAME, V1_VERSION = "v1_heuristic", "formula"
INITIAL_DELAY_MIN = 15      # an "initial" score is what was knowable this long after the detection arrived
UPDATED_AFTER_H = 6.0       # an "updated" score is recomputed once, after this long, with everything that arrived since
UPDATED_LOOKAHEAD_H = 3.0
HISTORY_DAYS = 366
DEFAULT_BUNDLE = Path(__file__).resolve().parent.parent / "detection-confidence-model" / "models" / "v2" / "veg_conf_model.json"

_FEATURES_PATH = Path(__file__).resolve().parent / "detection_features_v2.py"
_features_module = None
_bundle_cache: dict[str, Any] = {}


def shadow_enabled() -> bool:
    return os.getenv("SMF_DETECTION_SHADOW", "false").strip().lower() in {"1", "true", "yes"}


def features_module():
    """detection_features_v2, loaded by path so training can load the very same file."""
    global _features_module
    if _features_module is None:
        spec = importlib.util.spec_from_file_location("detection_features_v2", _FEATURES_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _features_module = module
    return _features_module


# ---------------------------------------------------------------- bundle (plain JSON, scored in pure python)

def bundle_path() -> Path:
    return Path(os.getenv("SMF_DETECTION_V2_MODEL", str(DEFAULT_BUNDLE)))


def load_bundle(path: Optional[Path] = None) -> Optional[dict]:
    """The trained v2 bundle, or None when there isn't one (or it doesn't match this feature schema)."""
    path = Path(path or bundle_path())
    try:
        stamp = path.stat().st_mtime
    except OSError:
        return None
    cached = _bundle_cache.get(str(path))
    if cached and cached[0] == stamp:
        return cached[1]
    try:
        bundle = json.loads(path.read_text(encoding="utf-8"))
        if bundle.get("schema") != BUNDLE_SCHEMA or bundle.get("feature_schema") != features_module().FEATURE_SCHEMA:
            logger.warning("detection v2 bundle %s ignored: schema mismatch", path)
            bundle = None
    except Exception as exc:
        logger.warning("detection v2 bundle %s unreadable: %s", path, exc)
        bundle = None
    _bundle_cache[str(path)] = (stamp, bundle)
    return bundle



# ---------------------------------------------------------------- data access

def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or database.get_db_path(), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _targets(conn, mode: str, now: datetime, limit: int) -> pd.DataFrame:
    base = '''SELECT e.id, e.source, e.occurred_at, e.first_seen_at, e.latitude, e.longitude, e.frp, e.confidence,
                     e.bright_t7, e.bright_t13, e.pixel_area, e.solar_zenith_angle, e.satellite_zenith_angle,
                     e.daynight, e.land_cover, e.exclusion_zone_id, e.recurring_source_id, e.detection_confidence_pct,
                     e.quality_flag, e.satellite, e.fuel_model_fbfm40, e.canopy_cover_pct, e.county_fips
              FROM fire_events e
              WHERE e.source IN ('viirs','modis','ngfs') AND e.latitude IS NOT NULL AND e.longitude IS NOT NULL '''
    if mode == "initial":
        sql = base + '''AND NOT EXISTS (SELECT 1 FROM detection_feature_snapshot s WHERE s.event_id = e.id AND s.mode = 'initial')
                        ORDER BY e.id DESC LIMIT ?'''
        rows = conn.execute(sql, (limit,)).fetchall()
    else:
        cutoff = _iso(now - timedelta(hours=UPDATED_AFTER_H))
        sql = base + '''AND e.first_seen_at <= ?
                        AND EXISTS (SELECT 1 FROM detection_feature_snapshot s WHERE s.event_id = e.id AND s.mode = 'initial')
                        AND NOT EXISTS (SELECT 1 FROM detection_feature_snapshot s WHERE s.event_id = e.id AND s.mode = 'updated')
                        ORDER BY e.id DESC LIMIT ?'''
        rows = conn.execute(sql, (cutoff, limit)).fetchall()
    return pd.DataFrame([dict(r) for r in rows])


def _history(conn, earliest_iso: str) -> pd.DataFrame:
    rows = conn.execute(
        '''SELECT id, source, occurred_at, first_seen_at, latitude, longitude, frp FROM fire_events
           WHERE source IN ('viirs','modis','ngfs') AND occurred_at >= ? AND latitude IS NOT NULL''', (earliest_iso,)).fetchall()
    return pd.DataFrame([dict(r) for r in rows], columns=["id", "source", "occurred_at", "first_seen_at", "latitude", "longitude", "frp"])


def _recurring(conn) -> pd.DataFrame:
    rows = conn.execute('''SELECT id, latitude, longitude, detection_count, reviewed_at, created_at
                           FROM recurring_fire_sources WHERE status = 'confirmed' ''').fetchall()
    return pd.DataFrame([dict(r) for r in rows], columns=["id", "latitude", "longitude", "detection_count", "reviewed_at", "created_at"])


def _v1_score(row: dict) -> Optional[float]:
    """v1 as the live system computes it. Masked detections are never scored live, so compute it here (nothing written)."""
    stored = row.get("detection_confidence_pct")
    if stored is not None and not (isinstance(stored, float) and math.isnan(stored)):
        return float(stored) / 100.0
    try:
        from services.detection_confidence import score_detection
        return score_detection(row) / 100.0
    except Exception:
        logger.debug("v1 shadow score unavailable for event %s", row.get("id"), exc_info=True)
        return None


# ---------------------------------------------------------------- the pass

def _score_set(conn, targets: pd.DataFrame, history: pd.DataFrame, recurring: pd.DataFrame, mode: str, now: datetime,
               bundle: Optional[dict]) -> dict:
    fm = features_module()
    if targets.empty:
        return {"snapshots": 0, "v1": 0, "v2": 0}
    arrival = pd.to_datetime(targets["first_seen_at"], utc=True, errors="coerce")
    if mode == "initial":
        as_of = (arrival + pd.Timedelta(minutes=INITIAL_DELAY_MIN)).clip(upper=pd.Timestamp(now))
        as_of = as_of.fillna(pd.Timestamp(now))
        lookahead = 0.0
    else:
        as_of = pd.Series(pd.Timestamp(now), index=targets.index)
        lookahead = UPDATED_LOOKAHEAD_H
    tf = targets.assign(as_of=as_of)
    X, avail = fm.build_features(tf, history, recurring, lookahead_h=lookahead)

    zone_ids = {r["id"]: dict(r) for r in conn.execute("SELECT id, created_at, updated_at, status FROM fire_exclusion_zones").fetchall()}
    rec_rows = {int(r["id"]): r for r in recurring.to_dict("records")} if len(recurring) else {}
    counts = {"snapshots": 0, "v1": 0, "v2": 0}
    scored_at = _iso(now)
    for pos, (idx, row) in enumerate(tf.iterrows()):
        feats = {k: (None if pd.isna(v) else float(v)) for k, v in X.loc[idx].items()}
        mask = {"exclusion_zone_id": None if pd.isna(row["exclusion_zone_id"]) else int(row["exclusion_zone_id"]),
                "recurring_source_id": None if pd.isna(row["recurring_source_id"]) else int(row["recurring_source_id"])}
        if mask["exclusion_zone_id"] in zone_ids:
            mask["exclusion_zone"] = zone_ids[mask["exclusion_zone_id"]]
        if mask["recurring_source_id"] in rec_rows:
            r = rec_rows[mask["recurring_source_id"]]
            mask["recurring_source"] = {"reviewed_at": r.get("reviewed_at"), "created_at": r.get("created_at")}
        availability = {"as_of": _iso(as_of.loc[idx].to_pydatetime()), "arrival_known": int(avail.loc[idx, "arrival_known"]),
                        "lookahead_h": lookahead}
        cur = conn.execute(
            '''INSERT INTO detection_feature_snapshot (event_id, mode, schema_version, features_json, availability_json, mask_json, scored_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (int(row["id"]), mode, fm.FEATURE_SCHEMA, json.dumps(feats), json.dumps(availability), json.dumps(mask, default=str), scored_at))
        snapshot_id = cur.lastrowid
        counts["snapshots"] += 1
        if mode == "initial":
            v1 = _v1_score(row.to_dict())
            if v1 is not None:
                conn.execute('''INSERT INTO detection_score_history (event_id, model_name, model_version, mode, score, calibrated, snapshot_id, scored_at)
                                VALUES (?, ?, ?, 'initial', ?, 0, ?, ?)''', (int(row["id"]), V1_NAME, V1_VERSION, v1, snapshot_id, scored_at))
                counts["v1"] += 1
        # v2 only for sensors the bundle was trained on; never a silent score for anything else
        if bundle and str(row["source"]).lower() in set(bundle.get("supported_sources", [])):
            p, calibrated = score_bundle(bundle, feats)
            conn.execute('''INSERT INTO detection_score_history (event_id, model_name, model_version, mode, score, calibrated, snapshot_id, scored_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                         (int(row["id"]), bundle.get("model_name", "veg_conf_v2"), str(bundle.get("version", "")), mode, p, int(calibrated), snapshot_id, scored_at))
            counts["v2"] += 1
    return counts


def run_shadow_pass(limit: int = 2000, now: Optional[datetime] = None, db_path: Optional[Path] = None) -> dict:
    """Snapshot + score new detections (initial) and rescore ones old enough (updated). Returns counts."""
    now = now or datetime.now(timezone.utc)
    conn = _connect(db_path)
    try:
        bundle = load_bundle()
        out: dict = {"bundle": bool(bundle)}
        sets = {mode: _targets(conn, mode, now, limit) for mode in ("initial", "updated")}
        occurred = pd.concat([pd.to_datetime(t["occurred_at"], utc=True, errors="coerce") for t in sets.values() if not t.empty],
                             ignore_index=True) if any(not t.empty for t in sets.values()) else pd.Series(dtype="datetime64[ns, UTC]")
        if occurred.empty:
            return {**out, "initial": {"snapshots": 0, "v1": 0, "v2": 0}, "updated": {"snapshots": 0, "v1": 0, "v2": 0}}
        history = _history(conn, _iso((occurred.min() - pd.Timedelta(days=HISTORY_DAYS)).to_pydatetime()))
        recurring = _recurring(conn)
        for mode, targets in sets.items():
            out[mode] = _score_set(conn, targets, history, recurring, mode, now, bundle)
            conn.commit()
        return out
    finally:
        conn.close()


def shadow_job() -> dict:
    """Scheduler entry point. Never raises."""
    if not shadow_enabled():
        return {"enabled": False}
    try:
        result = run_shadow_pass()
        logger.info("detection shadow pass: %s", result)
        return result
    except Exception as exc:
        logger.warning("detection shadow pass failed: %s", exc, exc_info=True)
        return {"error": str(exc)}
