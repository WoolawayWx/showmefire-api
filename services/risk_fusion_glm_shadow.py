"""
Failure-isolated, immutable fire_risk_fusion Phase-B (learned GLM) shadow
evidence - mirrors services/v5_shadow.py's structure (kill switch,
persisted state, immutable evidence files) for the same reason: this
scores a real, trained model in shadow only, alongside the zero-label
Phase-A rule-Monte-Carlo shadow (services/risk_fusion_shadow.py), and
must never affect the public forecast path.

Phase A needs no fire-occurrence labels; this Phase B module scores the
actual fire_risk_fusion GLM (monthly-baseline + fast-weather residual -
model-training/risk_fusion/model_bundle.py) against live county-day
weather features. The GLM predicts expected fire count (lam) and
P(>=1 fire) per county-day - a different question from Phase A's "how bad
would conditions be" (danger category probabilities): this answers "how
likely is a fire to actually start here today."

Loaded from a raw bundle directory via SMF_RISK_FUSION_GLM_BUNDLE - the
same shadow-only pattern V4/V5 use (SMF_V4_SHADOW_BUNDLE/
SMF_V5_SHADOW_BUNDLE), NOT through the model registry. The registered
training-side beta candidate (model-training/risk_fusion/
register_risk_fusion_beta.py) is explicitly "not production eligible...
nothing here touches the api/ repo" - this module is the first thing that
does, and it does so in shadow only, exactly like V4/V5 did before their
own registry promotion. The api/pipelines/import_model.py "fire_risk_fusion"
asset-role set (glm/guard/calibration/parity_vector/evaluation) does not
match what model_bundle.py's GLM-only v1 actually produces (climatology/
residual/effort/county_reference/contract) - it appears written for a
future, richer version. Going through that pipeline for this bundle would
be forcing a shape mismatch, not activating anything real.

The GLM math (linear predictor / predict / predict_residual /
log-effort-scaled offset) is reimplemented here in pure numpy from the
bundle's serialized params/means/stds, rather than importing
model-training/risk_fusion/model_bundle.py or fit_glm.py directly - same
repo-independence reason as services/rule_uncertainty.py and
core/risk_fusion_features.py. Only the FAST_WEATHER_FEATURES the real
registered v1 GLM was actually fit with (rh_mean, rh_min_afternoon,
wind_kts_max, wind_kts_p90, vpd_kpa_max, precip_24h_mm, is_weekend) are
needed at score time - KBDI/GDD are NOT: model_bundle.fit() fits the
residual GLM with FAST_WEATHER_FEATURES only, never fit_glm.DEFAULT_FEATURES
(an earlier, unused draft feature list that still includes kbdi/
gdd_accum_since_mar1).

Every public function catches its own exceptions and never raises into
the caller - a bug here must never affect forecast generation.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

FEATURES_MODULE_PATH = Path(__file__).resolve().parent.parent / "core" / "risk_fusion_features.py"

BUNDLE_ENV = "SMF_RISK_FUSION_GLM_BUNDLE"
ENABLED_ENV = "RISK_FUSION_GLM_SHADOW_ENABLED"
EVIDENCE_ROOT = Path(os.getenv("SMF_RISK_FUSION_GLM_EVIDENCE_ROOT") or
                     (Path(os.getenv("DATA_DIR", "data")) / "model-shadow" / "risk-fusion-glm"))
MAX_FAILURES = int(os.getenv("RISK_FUSION_GLM_SHADOW_MAX_FAILURES", "3"))
STATE_PATH = EVIDENCE_ROOT / "shadow-state.json"

# Must match model-training/risk_fusion/fit_glm.py's FAST_WEATHER_FEATURES
# exactly - this is what the real registered v1 GLM's residual term was
# fit with (verified against training-data/models/versions/
# fire_risk_fusion_0.0.1-beta.2_residual.json's own feature_columns).
FAST_WEATHER_FEATURES = [
    "rh_mean", "rh_min_afternoon", "wind_kts_max", "wind_kts_p90",
    "vpd_kpa_max", "precip_24h_mm", "is_weekend",
]
MONTH_DUMMY_COLUMNS = [f"month_{m}" for m in range(2, 13)]  # January is the reference level

BUNDLE_ASSET_FILENAMES = {
    "contract": "contract.json",
    "climatology": "glm_climatology.json",
    "residual": "glm_residual.json",
    "effort": "effort.json",
    "county_reference": "county_reference.json",
}


def month_dummies(month: int) -> Dict[str, float]:
    """month_2..month_12 indicators for one calendar month (1-12). January is the implicit reference level."""
    return {f"month_{m}": 1.0 if month == m else 0.0 for m in range(2, 13)}


def _configured() -> bool:
    return bool(os.getenv(BUNDLE_ENV, "").strip())


def _requested() -> bool:
    return os.getenv(ENABLED_ENV, "false").strip().lower() in {"1", "true", "yes", "on"}


def _initial_state() -> dict:
    state = {
        "phase": "B",
        "configured": _configured(),
        "enabled": _configured() and _requested(),
        "healthy": True,
        "auto_disabled": False,
        "consecutive_failures": 0,
        "last_error": None,
        "last_success": None,
        "runs": 0,
        "successful_runs": 0,
        "counties_scored": 0,
        "county_days_recorded": 0,
        "bundle_checksum": None,
        "advisory_published": False,
        "public_path_unchanged": True,
    }
    try:
        if STATE_PATH.exists():
            stored = json.loads(STATE_PATH.read_text())
            for key in state:
                state[key] = stored.get(key, state[key])
    except Exception:
        pass
    state["configured"] = _configured()
    state["enabled"] = bool(state["configured"] and _requested() and not state.get("auto_disabled", False))
    return state


_state = _initial_state()


def _persist_state() -> None:
    try:
        EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
        temporary = STATE_PATH.with_suffix(".tmp")
        temporary.write_text(json.dumps(_state, indent=2))
        temporary.replace(STATE_PATH)
    except Exception:
        pass


def diagnostics() -> dict:
    """Mirrors v5_shadow.diagnostics()'s cron-process reread: forecasts run under cron in a separate process."""
    try:
        if STATE_PATH.exists():
            stored = json.loads(STATE_PATH.read_text())
            for key in _state:
                if key in stored:
                    _state[key] = stored[key]
    except Exception as error:
        _state["healthy"] = False
        _state["last_error"] = f"unable to read shadow state: {error}"
    _state["configured"] = _configured()
    _state["enabled"] = bool(_state["configured"] and _requested() and not _state.get("auto_disabled", False))
    return dict(_state)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bundle(directory: Optional[Path] = None) -> Dict:
    """Loads and validates a fire_risk_fusion GLM-only v1 bundle directory.

    Raises on any contract mismatch - a caller must never score with a
    bundle whose feature module has drifted from what
    core/risk_fusion_features.py actually computes today (see
    core/contract_mirrors.json's "risk_fusion_features" pair), or whose
    contract isn't the advisory-only GLM-only v1 shape this module knows
    how to score.
    """
    directory = Path(directory or os.getenv(BUNDLE_ENV, ""))
    if not str(directory) or not directory.is_dir():
        raise FileNotFoundError(f"{BUNDLE_ENV} is not a bundle directory")
    assets = {}
    for role, filename in BUNDLE_ASSET_FILENAMES.items():
        path = directory / filename
        if not path.exists():
            raise FileNotFoundError(f"risk_fusion GLM bundle missing asset: {filename}")
        assets[role] = json.loads(path.read_text(encoding="utf-8"))
    contract = assets["contract"]
    if contract.get("advisory_only") is not True:
        raise ValueError("risk_fusion GLM bundle is not advisory_only")
    if contract.get("model_family") != "glm":
        raise ValueError(
            f"risk_fusion GLM shadow only knows how to score model_family='glm', got {contract.get('model_family')!r}")
    expected_feature_hash = _sha256_file(FEATURES_MODULE_PATH)
    if contract.get("feature_module_sha256") != expected_feature_hash:
        raise ValueError(
            "risk_fusion GLM bundle feature_module_sha256 does not match core/risk_fusion_features.py "
            "- retrain or re-mirror before scoring")
    bundle_checksum = hashlib.sha256(
        "".join(_sha256_file(directory / filename) for filename in BUNDLE_ASSET_FILENAMES.values()).encode()
    ).hexdigest()
    return {**assets, "bundle_checksum": bundle_checksum}


def _standardized_dot(fit: Dict, feature_values: Dict[str, float]) -> float:
    """The fitted-terms contribution to log(lambda) for a serialized GLM fit (excludes the offset)."""
    total = float(fit["params"][0])  # constant term, matching statsmodels' add_constant ordering
    for index, column in enumerate(fit["feature_columns"]):
        value = float(feature_values[column])
        mean = float(fit["means"][column])
        std = float(fit["stds"][column]) or 1.0
        total += float(fit["params"][index + 1]) * (value - mean) / std
    return total


def _log_effort_scaled(county_fips: str, effort_bundle: Dict, county_reference: Dict) -> float:
    rate_row = next((row for row in effort_bundle["rate_table"] if row["county_fips"] == county_fips), None)
    if rate_row is None:
        raise KeyError(f"{county_fips} not present in the risk_fusion GLM bundle's reporting-rate table")
    burnable_area_km2 = float(county_reference[county_fips]["burnable_area_km2"])
    log_effort = float(np.log(burnable_area_km2) + np.log(float(rate_row["reporting_rate_shrunk"])))
    return float(effort_bundle["effort_exponent"]) * log_effort


def score_county_day(bundle: Dict, county_fips: str, calendar_row: Dict, weather_row: Dict) -> Dict:
    """Expected fire count (lam) and P(>=1 fire) for one county-day - same math as model_bundle.score()."""
    climatology_features = {column: float(calendar_row.get(column, 0.0)) for column in MONTH_DUMMY_COLUMNS}
    log_effort_scaled = _log_effort_scaled(county_fips, bundle["effort"], bundle["county_reference"])
    base_eta = _standardized_dot(bundle["climatology"], climatology_features) + log_effort_scaled
    residual_features = {**weather_row, "is_weekend": float(bool(calendar_row.get("is_weekend", False)))}
    lam = float(np.exp(_standardized_dot(bundle["residual"], residual_features) + base_eta))
    return {"lam": lam, "p_ge1_fire": float(1.0 - np.exp(-lam))}


def score_glm_for_forecast(
    run_id: str,
    valid_local_date: str,
    county_fips: List[str],
    calendar_rows: Dict[str, Dict],
    weather_rows: Dict[str, Dict],
    bundle_dir: Optional[Path] = None,
    evidence_root: Optional[Path] = None,
) -> bool:
    """Scores one forecast run's county-day GLM features and writes an immutable evidence file.

    calendar_rows/weather_rows are keyed by county_fips - callers (see
    services/risk_fusion_hook.py) build these once per run since month
    dummies/is_weekend are the same for every county on a given date but
    weather aggregates are county-specific. Never raises; returns False on
    any failure (including shadow being disabled).
    """
    if not diagnostics()["enabled"]:
        return False
    try:
        bundle = load_bundle(bundle_dir)
        scored = {
            fips: score_county_day(bundle, fips, calendar_rows[fips], weather_rows[fips])
            for fips in county_fips
        }
        record = {
            "run_id": str(run_id),
            "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "valid_local_date": valid_local_date,
            "bundle_checksum": bundle["bundle_checksum"],
            "county_fips": list(county_fips),
            "lam": [scored[fips]["lam"] for fips in county_fips],
            "p_ge1_fire": [scored[fips]["p_ge1_fire"] for fips in county_fips],
        }
        root = Path(evidence_root or EVIDENCE_ROOT)
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{run_id}.glm_score.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)

        n = len(county_fips)
        _state.update(
            consecutive_failures=0, last_error=None, healthy=True,
            last_success=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            runs=_state["runs"] + 1, successful_runs=_state.get("successful_runs", 0) + 1,
            counties_scored=n, county_days_recorded=_state.get("county_days_recorded", 0) + n,
            bundle_checksum=bundle["bundle_checksum"],
        )
        _persist_state()
        return True
    except Exception as error:
        _state["runs"] += 1
        _state["consecutive_failures"] += 1
        _state["last_error"] = str(error)
        _state["healthy"] = False
        if _state["consecutive_failures"] >= MAX_FAILURES:
            _state["enabled"] = False
            _state["auto_disabled"] = True
        _persist_state()
        return False


def record_skipped_run(reason: str) -> bool:
    """Makes an enabled-but-empty shadow attempt observable, mirroring risk_fusion_shadow.record_skipped_run."""
    if not diagnostics()["enabled"]:
        return False
    _state["runs"] += 1
    _state["consecutive_failures"] += 1
    _state["last_error"] = str(reason)
    _state["healthy"] = False
    if _state["consecutive_failures"] >= MAX_FAILURES:
        _state["enabled"] = False
        _state["auto_disabled"] = True
    _persist_state()
    return True
