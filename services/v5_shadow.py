"""Failure-isolated, immutable V5 prospective shadow evidence."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from core.fire_danger import RULE_SPEC_SHA256, calculate_fire_danger
from core.precipitation import PRECIPITATION_CONTRACT_SHA256, PRECIPITATION_CONTRACT_VERSION

BUNDLE_ENV = "SMF_V5_SHADOW_BUNDLE"
ENABLED_ENV = "V5_SHADOW_ENABLED"
EVIDENCE_ROOT = Path(os.getenv("SMF_V5_EVIDENCE_ROOT") or
                     (Path(os.getenv("DATA_DIR", "data")) / "model-shadow" / "v5"))
MAX_FAILURES = int(os.getenv("V5_SHADOW_MAX_FAILURES", "3"))
STATE_PATH = EVIDENCE_ROOT / "shadow-state.json"


def _configured(): return bool(os.getenv(BUNDLE_ENV, "").strip())
def _requested(): return os.getenv(ENABLED_ENV, "false").strip().lower() in {"1", "true", "yes", "on"}


def _initial_state():
    state = {"configured": _configured(), "enabled": _configured() and _requested(), "healthy": True,
             "consecutive_failures": 0, "last_error": None, "last_success": None,
             "runs": 0, "successful_runs": 0, "public_forecast_failures": 0,
             "fallback_rows": 0, "unavailable": 0, "latency_ms": None,
             "rows": 0, "bundle_checksum": None, "observation_verification_status": "pending"}
    try:
        if STATE_PATH.exists():
            stored = json.loads(STATE_PATH.read_text())
            for key in state: state[key] = stored.get(key, state[key])
    except Exception:
        pass
    state["configured"] = _configured()
    state["enabled"] = bool(state["enabled"] and _configured() and _requested())
    return state


_state = _initial_state()


def _persist_state():
    try:
        EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
        temporary = STATE_PATH.with_suffix(".tmp")
        temporary.write_text(json.dumps(_state, indent=2)); temporary.replace(STATE_PATH)
    except Exception:
        pass


def diagnostics():
    result = dict(_state)
    result["configured"] = _configured()
    result["enabled"] = bool(_state["enabled"] and _requested() and _configured())
    return result
def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_bundle(directory=None):
    directory = Path(directory or os.getenv(BUNDLE_ENV, ""))
    if not str(directory) or not directory.is_dir(): raise FileNotFoundError(f"{BUNDLE_ENV} is not a bundle directory")
    contract = json.loads((directory / "contract.json").read_text())
    if contract.get("rule_spec_sha256") != RULE_SPEC_SHA256: raise ValueError("V5 rule contract mismatch")
    if (contract.get("precipitation_contract_version") != PRECIPITATION_CONTRACT_VERSION or
            contract.get("precipitation_contract_sha256") != PRECIPITATION_CONTRACT_SHA256):
        raise ValueError("V5 precipitation contract mismatch")
    shadow = json.loads((directory / "shadow_bundle_manifest.json").read_text())
    if shadow.get("status") != "experimental_shadow_only" or shadow.get("registry_channel") is not None:
        raise ValueError("V5 bundle is not shadow-only")
    if shadow.get("rule_spec_sha256") != RULE_SPEC_SHA256 or shadow.get("precipitation_contract_sha256") != PRECIPITATION_CONTRACT_SHA256:
        raise ValueError("V5 shadow contract mismatch")
    for filename, digest in shadow.get("assets", {}).items():
        if _sha(directory / filename) != digest: raise ValueError(f"V5 shadow checksum mismatch: {filename}")
    direct = {"base_xgboost.json": "base_model_sha256", "specialist_xgboost.json": "specialist_model_sha256",
              "guard.json": "guard_sha256", "uncertainty.json": "uncertainty_sha256"}
    for filename, field in direct.items():
        if _sha(directory / filename) != contract.get(field): raise ValueError(f"V5 checksum mismatch: {filename}")
    contract["bundle_sha256"] = _sha(directory / "shadow_bundle_manifest.json")
    return contract


def record_predictions(run_id, row_keys, stable_fm, base_fm, v5_fm, intervals,
                       raw_correction, guard_weights, guard_reasons, regimes, rh, wind_kts,
                       bundle_dir=None, evidence_root=EVIDENCE_ROOT,
                       feature_freshness_minutes=None, latency_ms=None):
    """Write one pre-observation record. Failure never affects stable output."""
    if not _state["enabled"]: return False
    try:
        contract = validate_bundle(bundle_dir)
        stable_fm, base_fm, v5_fm = map(lambda value: np.asarray(value, float), (stable_fm, base_fm, v5_fm))
        intervals = np.asarray(intervals, float); weights = np.asarray(guard_weights, float)
        if intervals.shape != (len(v5_fm), 3) or np.any(np.diff(intervals, axis=1) < 0): raise ValueError("invalid V5 intervals")
        if not all(len(value) == len(v5_fm) for value in (stable_fm, base_fm, weights, rh, wind_kts, row_keys)):
            raise ValueError("V5 shadow row alignment mismatch")
        stable_category = [calculate_fire_danger(f, r, w) for f, r, w in zip(stable_fm, rh, wind_kts)]
        v5_category = [calculate_fire_danger(f, r, w) for f, r, w in zip(v5_fm, rh, wind_kts)]
        unavailable = sum(a is None or b is None for a, b in zip(stable_category, v5_category))
        record = {"run_id": str(run_id), "recorded_at": datetime.now(timezone.utc).isoformat(),
                  "observation_attached": False, "row_keys": list(map(str, row_keys)),
                  "stable_fm": stable_fm.tolist(), "rain_aware_base_fm": base_fm.tolist(), "v5_fm": v5_fm.tolist(),
                  "v5_p10_p50_p90": intervals.tolist(), "raw_correction": np.asarray(raw_correction, float).tolist(),
                  "guard_weights": weights.tolist(), "guard_reasons": list(map(str, guard_reasons)),
                  "regimes": list(map(str, regimes)), "fallback": (weights == 0).tolist(),
                  "stable_category": stable_category, "v5_category": v5_category,
                  "category_disagreements": sum(a != b for a, b in zip(stable_category, v5_category)),
                  "feature_freshness_minutes": feature_freshness_minutes, "latency_ms": latency_ms,
                  "unavailable": unavailable, "bundle_manifest_sha256": contract["manifest_sha256"]}
        evidence_root = Path(evidence_root); evidence_root.mkdir(parents=True, exist_ok=True)
        path = evidence_root / f"{run_id}.prediction.json"
        with path.open("x", encoding="utf-8") as stream: json.dump(record, stream, indent=2)
        _state.update(consecutive_failures=0, last_error=None, healthy=True,
                      last_success=datetime.now(timezone.utc).isoformat(), runs=_state["runs"] + 1,
                      successful_runs=_state.get("successful_runs", 0) + 1, latency_ms=latency_ms,
                      rows=_state.get("rows", 0) + len(v5_fm), bundle_checksum=contract["bundle_sha256"],
                      fallback_rows=_state["fallback_rows"] + int(np.sum(weights == 0)),
                      unavailable=_state["unavailable"] + unavailable)
        _persist_state()
        return True
    except Exception as error:
        _state["runs"] += 1; _state["consecutive_failures"] += 1; _state["last_error"] = str(error)
        _state["healthy"] = False
        if _state["consecutive_failures"] >= MAX_FAILURES: _state["enabled"] = False
        _persist_state()
        return False


def attach_observations(run_id, observations, evidence_root=EVIDENCE_ROOT):
    prediction = Path(evidence_root) / f"{run_id}.prediction.json"
    if not prediction.exists(): raise FileNotFoundError("prediction must exist before observation")
    path = Path(evidence_root) / f"{run_id}.observation.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump({"run_id": str(run_id), "attached_at": datetime.now(timezone.utc).isoformat(),
                   "observations": observations}, stream, indent=2)
    _state["observation_verification_status"] = "attached"
    _persist_state()
    return path


def score_and_record(run_id, rows, stable_fm, *, bundle_dir=None, evidence_root=EVIDENCE_ROOT):
    """Score station sequences and persist evidence; exceptions stay off the public path."""
    if not diagnostics()["enabled"]: return False
    try:
        from services.v5_scorer import score
        result = score(bundle_dir or os.getenv(BUNDLE_ENV), rows)
        frame = result["prepared"]
        row_keys = [f"{run_id}|{station}|{valid}" for station, valid in zip(frame.station_id, frame.valid_time)]
        return record_predictions(
            run_id, row_keys, stable_fm, result["base"], result["prediction"], result["intervals"],
            result["raw_correction"], result["guard_weights"], result["guard_reasons"], result["regimes"],
            frame.hrrr_rh.to_numpy(float), frame.hrrr_wind_ms.to_numpy(float) * 1.9438444924406,
            bundle_dir=bundle_dir, evidence_root=evidence_root,
            feature_freshness_minutes=float(frame.initial_age_hours.max()), latency_ms=result["latency_ms"])
    except Exception as error:
        _state["runs"] += 1; _state["consecutive_failures"] += 1; _state["last_error"] = str(error); _state["healthy"] = False
        if _state["consecutive_failures"] >= MAX_FAILURES: _state["enabled"] = False
        _persist_state()
        return False
