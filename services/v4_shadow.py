"""Failure-isolated, non-registry V4 prospective shadow evidence."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from core.fire_danger import RULE_SPEC_SHA256
from core.precipitation import PRECIPITATION_CONTRACT_SHA256, PRECIPITATION_CONTRACT_VERSION
from services import shadow_metrics

BUNDLE_ENV = "SMF_V4_SHADOW_BUNDLE"
ENABLED_ENV = "V4_SHADOW_ENABLED"  # v4 had no enable/disable switch at all before this
EVIDENCE_ROOT = Path(__file__).resolve().parent.parent / "logs" / "v4_shadow"
MAX_FAILURES = int(os.getenv("V4_SHADOW_MAX_FAILURES", "3"))
_state = {"enabled": True, "consecutive_failures": 0, "last_error": None,
          "runs": 0, "unavailable": 0, "model_version": None}


def _configured() -> bool:
    """Matches the other four shadow modules' _configured() shape exactly."""
    if os.getenv(BUNDLE_ENV, "").strip():
        return True
    try:
        from models.versioning import get_model_entry
        return bool(get_model_entry("v4").get("stable"))
    except Exception:
        return False


def _requested() -> bool:
    """DB setting wins when present (live, no-restart admin control via the
    website); the env var is only the deploy-time default/fallback. Default
    "true" (not "false" like the other four) preserves v4's pre-existing
    always-on-unless-failing behavior for anyone who never touches the new
    env var or DB row."""
    try:
        from core.database import get_shadow_model_setting
        setting = get_shadow_model_setting("v4")
        if setting is not None:
            return setting["enabled"]
    except Exception:
        pass
    return os.getenv(ENABLED_ENV, "true").strip().lower() in {"1", "true", "yes", "on"}


def diagnostics() -> dict:
    """v4 previously just returned a bare passthrough of _state, which never
    reflected configured/requested/DB-setting changes - real callers (the
    admin API, beta_operations) need a fresh recompute on every call, same
    as the other four shadow modules' diagnostics() already does. Does NOT
    change record_predictions()'s own guard (still the raw `_state["enabled"]`
    flag, unchanged) - v4 has no live production caller today (confirmed:
    only tests call record_predictions directly), so there's no existing
    call site to migrate onto this fresh gate; a future real caller should
    check diagnostics()["enabled"] first, mirroring v5_shadow.py's
    score_and_record precedent, rather than calling record_predictions
    directly."""
    _state["configured"] = _configured()
    _state["requested"] = _requested()
    auto_disabled = _state.get("consecutive_failures", 0) >= MAX_FAILURES
    _state["auto_disabled"] = auto_disabled
    _state["enabled"] = bool(_state["configured"] and _state["requested"] and not auto_disabled)
    try:
        from services.shadow_observation_scoring import rolling_accuracy_summary
        _state["observation_accuracy"] = rolling_accuracy_summary("v4", EVIDENCE_ROOT)
    except Exception:
        _state["observation_accuracy"] = None
    return dict(_state)


def _sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

# Every file validate_bundle() reads, by role - reused by
# models/shadow_bundles.py to know what to dual-write into the unified
# registry, and by the registry-fallback branch below to resolve the same
# roles from load_active_assets() instead of a fixed directory.
BUNDLE_ASSET_FILENAMES = {
    "contract": "contract.json", "shadow_manifest": "shadow_bundle_manifest.json",
    "base_model": "base_xgboost.json", "model": "guarded_gru.pt", "lead_guard": "lead_guard.json",
}


def _resolve_bundle_files(directory):
    """Resolve role -> file path, preferring an explicit directory /
    BUNDLE_ENV override, and falling back to the unified registry's
    `stable` channel once this family is migrated and nothing overrides
    it. Returns (files, version_hint)."""
    directory = Path(directory or os.getenv(BUNDLE_ENV, "")) if (directory or os.getenv(BUNDLE_ENV, "").strip()) else None
    if directory is not None:
        if not directory.is_dir(): raise FileNotFoundError(f"{BUNDLE_ENV} is not a bundle directory")
        files = {role: directory / filename for role, filename in BUNDLE_ASSET_FILENAMES.items()}
        version = None
        version_path = directory / "registered_version.json"
        if version_path.is_file():
            try:
                version = json.loads(version_path.read_text(encoding="utf-8")).get("version")
            except Exception:
                version = None
        return files, version

    from models.versioning import get_model_entry, load_active_assets
    resolved = load_active_assets("v4", channel="stable")
    files = {role: asset["path"] for role, asset in resolved.items()}
    stable = get_model_entry("v4").get("stable") or {}
    version = (stable.get("metadata") or {}).get("shadow_bundle_version") or stable.get("version")
    return files, version


def validate_bundle(directory=None):
    files, version = _resolve_bundle_files(directory)
    missing = [role for role in BUNDLE_ASSET_FILENAMES if role not in files]
    if missing: raise FileNotFoundError(f"V4 bundle missing asset(s): {missing}")
    contract = json.loads(Path(files["contract"]).read_text())
    if contract.get("rule_spec_sha256") != RULE_SPEC_SHA256: raise ValueError("V4 rule contract mismatch")
    if (contract.get("precipitation_contract_version") != PRECIPITATION_CONTRACT_VERSION or
            contract.get("precipitation_contract_sha256") != PRECIPITATION_CONTRACT_SHA256):
        raise ValueError("V4 precipitation contract mismatch")
    shadow = json.loads(Path(files["shadow_manifest"]).read_text())
    if shadow.get("status") != "experimental_shadow_only" or shadow.get("registry_channel") is not None:
        raise ValueError("V4 bundle is not shadow-only")
    if shadow.get("rule_spec_sha256") != RULE_SPEC_SHA256: raise ValueError("V4 shadow rule mismatch")
    if shadow.get("precipitation_contract_sha256") != PRECIPITATION_CONTRACT_SHA256:
        raise ValueError("V4 shadow precipitation mismatch")
    for filename, digest in shadow.get("assets", {}).items():
        matching = next((role for role, name in BUNDLE_ASSET_FILENAMES.items() if name == filename), None)
        path = Path(files[matching]) if matching else Path(files["contract"]).parent / filename
        if _sha(path) != digest: raise ValueError(f"V4 shadow checksum mismatch: {filename}")
    assets = {"base_model": "base_model_sha256", "model": "residual_model_sha256",
              "lead_guard": "lead_guard_sha256"}
    for role, field in assets.items():
        if _sha(Path(files[role])) != contract.get(field): raise ValueError(f"V4 checksum mismatch: {role}")
    contract["registered_version"] = version
    return contract


def record_predictions(run_id, row_keys, stable_fm, quantiles, rh, wind_kts,
                       category_probability, lead_weights, bundle_dir=None,
                       evidence_root=EVIDENCE_ROOT,
                       feature_freshness_minutes=None, fallback_used=False,
                       latency_ms=None):
    """Write one immutable pre-observation record; never affects stable output."""
    if not _state["enabled"]: return False
    try:
        contract = validate_bundle(bundle_dir); quantiles = np.asarray(quantiles, float)
        if quantiles.shape[-1] != 7 or np.any(np.diff(quantiles, axis=-1) < 0): raise ValueError("invalid V4 quantiles")
        stable_fm = np.asarray(stable_fm, float); rh = np.asarray(rh, float); wind_kts = np.asarray(wind_kts, float)
        stable_category = shadow_metrics.categories_for(stable_fm, rh, wind_kts)
        v4_category = shadow_metrics.categories_for(quantiles[:, 3], rh, wind_kts)
        p10_category = shadow_metrics.categories_for(quantiles[:, 1], rh, wind_kts)
        p90_category = shadow_metrics.categories_for(quantiles[:, 5], rh, wind_kts)
        disagreement = shadow_metrics.category_disagreement_summary(stable_category, v4_category)
        unavailable = disagreement["unavailable"]
        record = {"run_id": str(run_id), "recorded_at": datetime.now(timezone.utc).isoformat(),
                  "observation_attached": False, "row_keys": list(map(str, row_keys)),
                  "stable_fm": stable_fm.tolist(), "v4_quantiles": quantiles.tolist(),
                  "category_probability": np.asarray(category_probability, float).tolist(),
                  "lead_weights": np.asarray(lead_weights, float).tolist(),
                  "stable_category": stable_category, "v4_category": v4_category,
                  "p10_category": p10_category, "p90_category": p90_category,
                  "category_disagreements": disagreement["category_disagreements"],
                  "feature_freshness_minutes": feature_freshness_minutes,
                  "fallback_used": bool(fallback_used), "latency_ms": latency_ms,
                  "unavailable": unavailable, "bundle_manifest_sha256": contract["manifest_sha256"]}
        evidence_root = Path(evidence_root); evidence_root.mkdir(parents=True, exist_ok=True)
        path = evidence_root / f"{run_id}.prediction.json"
        with path.open("x", encoding="utf-8") as stream: json.dump(record, stream, indent=2)
        _state.update(consecutive_failures=0, last_error=None, runs=_state["runs"] + 1,
                      unavailable=_state["unavailable"] + unavailable,
                      model_version=contract.get("registered_version"))
        return True
    except Exception as error:
        _state["consecutive_failures"] += 1; _state["last_error"] = str(error)
        if _state["consecutive_failures"] >= MAX_FAILURES: _state["enabled"] = False
        return False


def attach_observations(run_id, observations, evidence_root=EVIDENCE_ROOT):
    """Append observations to a separate immutable file without rewriting predictions."""
    prediction = Path(evidence_root) / f"{run_id}.prediction.json"
    if not prediction.exists(): raise FileNotFoundError("prediction must exist before observation")
    path = Path(evidence_root) / f"{run_id}.observation.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump({"run_id": str(run_id), "attached_at": datetime.now(timezone.utc).isoformat(),
                   "observations": observations}, stream, indent=2)
    return path
