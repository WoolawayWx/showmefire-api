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
EVIDENCE_ROOT = Path(__file__).resolve().parent.parent / "logs" / "v5_shadow"
MAX_FAILURES = int(os.getenv("V5_SHADOW_MAX_FAILURES", "3"))
_state = {"enabled": True, "consecutive_failures": 0, "last_error": None,
          "runs": 0, "fallback_rows": 0, "unavailable": 0}


def diagnostics(): return dict(_state)
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
        _state.update(consecutive_failures=0, last_error=None, runs=_state["runs"] + 1,
                      fallback_rows=_state["fallback_rows"] + int(np.sum(weights == 0)),
                      unavailable=_state["unavailable"] + unavailable)
        return True
    except Exception as error:
        _state["consecutive_failures"] += 1; _state["last_error"] = str(error)
        if _state["consecutive_failures"] >= MAX_FAILURES: _state["enabled"] = False
        return False


def attach_observations(run_id, observations, evidence_root=EVIDENCE_ROOT):
    prediction = Path(evidence_root) / f"{run_id}.prediction.json"
    if not prediction.exists(): raise FileNotFoundError("prediction must exist before observation")
    path = Path(evidence_root) / f"{run_id}.observation.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump({"run_id": str(run_id), "attached_at": datetime.now(timezone.utc).isoformat(),
                   "observations": observations}, stream, indent=2)
    return path
