"""Failure-isolated stable/beta comparison for fuel-moisture models."""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xgboost as xgb

from core.fire_danger import calculate_fire_danger, meters_per_second_to_knots
from models.features import validate_feature_contract
from models.versioning import get_model_entry, load_active_model_path, update_beta_metadata

logger = logging.getLogger(__name__)
LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "model_shadow.jsonl"
ADVISORY_MODEL_PATH = Path(__file__).resolve().parent.parent / "fire-danger-model" / "models" / "fire_danger_model.json"
ADVISORY_META_PATH = ADVISORY_MODEL_PATH.with_name("fire_danger_model_meta.json")
MAX_CONSECUTIVE_FAILURES = int(os.getenv("MODEL_SHADOW_MAX_FAILURES", "3"))
_state = {"enabled": True, "consecutive_failures": 0, "last_error": None, "last_run": None}


def diagnostics():
    return dict(_state)


def _predict(path, frame, metadata):
    model = xgb.Booster(); model.load_model(str(path))
    columns = metadata.get("feature_columns") or model.feature_names
    columns = validate_feature_contract(frame, {**metadata, "feature_columns": list(columns or [])})
    return model.predict(xgb.DMatrix(frame[columns], feature_names=columns))


def run_shadow(frame, stable_predictions):
    """Log beta comparisons and always return the caller's stable predictions."""
    if not _state["enabled"]:
        return stable_predictions
    entry = get_model_entry("fuel_moisture")
    beta = entry.get("beta")
    if not beta:
        return stable_predictions
    started = time.perf_counter()
    try:
        beta_path = load_active_model_path("fuel_moisture", "beta")
        beta_pred = _predict(beta_path, frame, beta.get("metadata") or {})
        stable_pred = np.asarray(stable_predictions)
        wind = meters_per_second_to_knots(frame["wind_speed_ms"].to_numpy())
        stable_category = [calculate_fire_danger(fm, rh, ws) for fm, rh, ws in
                           zip(stable_pred, frame["rel_humidity"], wind)]
        beta_category = [calculate_fire_danger(fm, rh, ws) for fm, rh, ws in
                         zip(beta_pred, frame["rel_humidity"], wind)]
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stable_version": (entry.get("stable") or {}).get("version"),
            "beta_version": beta.get("version"), "count": len(frame),
            "mean_absolute_difference": float(np.mean(np.abs(beta_pred - stable_pred))),
            "category_disagreements": int(np.sum(np.asarray(beta_category) != np.asarray(stable_category))),
            "elevated_samples": int(sum(value is not None and value >= 2 for value in stable_category)),
            "unavailable_categories": int(sum(value is None for value in stable_category)),
            "feature_out_of_range": int(frame.get("feature_out_of_range", False).sum())
                if "feature_out_of_range" in frame else 0,
            "max_feature_age_minutes": float(frame["match_age_minutes"].max())
                if "match_age_minutes" in frame else None,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2), "failed": False,
        }
        # Advisory direct-danger output is recorded but never returned to callers.
        try:
            advisory_meta = json.loads(ADVISORY_META_PATH.read_text(encoding="utf-8"))
            stride = max(1, len(frame) // 5000)
            advisory_frame = frame.iloc[::stride]
            advisory_pred = _predict(ADVISORY_MODEL_PATH, advisory_frame, advisory_meta)
            thresholds = advisory_meta.get("category_thresholds", [0.5, 1.5, 2.5, 3.5])
            advisory_category = np.clip(np.digitize(advisory_pred, thresholds), 0, 4)
            record["advisory_disagreements"] = int(
                np.sum(advisory_category != np.asarray(stable_category)[::stride])
            )
            record["advisory_sample_count"] = len(advisory_frame)
            record["advisory_version"] = advisory_meta.get("created_utc")
        except Exception as advisory_error:
            record["advisory_error"] = str(advisory_error)
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        _state.update(consecutive_failures=0, last_error=None, last_run=record["timestamp"])
    except Exception as exc:
        _state["consecutive_failures"] += 1; _state["last_error"] = str(exc)
        logger.exception("Beta shadow inference failed; stable output is unchanged")
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(),
                                     "failed": True, "error": str(exc)}) + "\n")
        if _state["consecutive_failures"] >= MAX_CONSECUTIVE_FAILURES:
            _state["enabled"] = False
    return stable_predictions


def evaluate_shadow_evidence(path=LOG_PATH, minimum_days=30, minimum_elevated_samples=1):
    records = []
    if Path(path).exists():
        records = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    successful = [record for record in records if not record.get("failed")]
    dates = {record["timestamp"][:10] for record in successful}
    failures = sum(bool(record.get("failed")) for record in records)
    elevated = sum(int(record.get("elevated_samples", 0)) for record in successful)
    evidence = {"days": len(dates), "runs": len(successful), "failures": failures,
                "elevated_samples": elevated,
                "passed": len(dates) >= minimum_days and failures == 0 and elevated >= minimum_elevated_samples}
    return evidence


def record_shadow_gate(path=LOG_PATH, minimum_days=30, minimum_elevated_samples=1):
    evidence = evaluate_shadow_evidence(path, minimum_days, minimum_elevated_samples)
    update_beta_metadata("fuel_moisture", {"shadow": evidence})
    return evidence
