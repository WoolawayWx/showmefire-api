"""
Failure-isolated shadow harness for fire_risk_fusion - Phase A only.

Phase A is the rule-derived half of the product: category probabilities
from Monte Carlo propagation of forecast uncertainty through the
untouched canonical rule (services/rule_uncertainty.py). It needs ZERO
fire-occurrence labels and no trained bundle, unlike V5's shadow harness
(services/v5_shadow.py), which this module otherwise mirrors structurally
(kill switches, persisted state, immutable evidence files).

Phase B - the learned ignition_* half, which DOES need a trained GLM/GBM
bundle - is blocked on label accumulation (see the plan's staging) and is
not implemented here. When it exists, it gets its own
validate_bundle()-style guard mirroring v5_shadow.validate_bundle,
including the parity-vector check described in the project plan; Phase A
below has no bundle to validate, only the rule-MC parity self-check.

Every public function catches its own exceptions and never raises into
the caller - a bug here must never affect forecast generation, matching
the comment already in DailyForecast.py about V5's failure isolation.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from core.fire_danger import RULE_SPEC_SHA256, RULE_SPEC_VERSION, calculate_fire_danger
from services import rule_uncertainty as ru

ENABLED_ENV = "RISK_FUSION_SHADOW_ENABLED"
EVIDENCE_ROOT = Path(os.getenv("SMF_RISK_FUSION_EVIDENCE_ROOT") or
                     (Path(os.getenv("DATA_DIR", "data")) / "model-shadow" / "risk-fusion"))
MAX_FAILURES = int(os.getenv("RISK_FUSION_SHADOW_MAX_FAILURES", "3"))
STATE_PATH = EVIDENCE_ROOT / "shadow-state.json"


def _requested() -> bool:
    return os.getenv(ENABLED_ENV, "false").strip().lower() in {"1", "true", "yes", "on"}


def _initial_state() -> dict:
    state = {
        "phase": "A",
        "enabled": _requested(),
        "healthy": True,
        "auto_disabled": False,
        "consecutive_failures": 0,
        "last_error": None,
        "last_success": None,
        "runs": 0,
        "successful_runs": 0,
        "counties_scored": 0,
        "county_days_recorded": 0,
        "rule_mc_draws": None,
        "rule_mc_parity_status": "not_yet_checked",
        "public_path_unchanged": True,
        "advisory_published": False,
        "rule_spec_version": RULE_SPEC_VERSION,
        "rule_spec_sha256": RULE_SPEC_SHA256,
    }
    try:
        if STATE_PATH.exists():
            stored = json.loads(STATE_PATH.read_text())
            for key in state:
                state[key] = stored.get(key, state[key])
    except Exception:
        pass
    state["enabled"] = bool(_requested() and not state.get("auto_disabled", False))
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
    """
    Mirrors v5_shadow.diagnostics()'s cron-process re-read: forecasts run
    under cron in a separate Python process from the long-running API
    process, so this reloads persisted counters on every call.
    """
    try:
        if STATE_PATH.exists():
            stored = json.loads(STATE_PATH.read_text())
            for key in _state:
                if key in stored:
                    _state[key] = stored[key]
    except Exception as error:
        _state["healthy"] = False
        _state["last_error"] = f"unable to read shadow state: {error}"
    _state["enabled"] = bool(_requested() and not _state.get("auto_disabled", False))
    return dict(_state)


def check_rule_mc_parity() -> bool:
    """
    Runs rule_uncertainty.check_parity against the REAL calculate_fire_danger
    and records the result. A failing parity check means the two
    implementations have drifted - shadow must refuse to run rather than
    publish probabilities computed from a rule that no longer matches the
    canonical one.
    """
    try:
        from core.fire_danger import RULE_SPEC
        ru.check_parity(calculate_fire_danger, RULE_SPEC["thresholds"])
        _state["rule_mc_parity_status"] = "ok"
        _persist_state()
        return True
    except Exception as error:
        _state["rule_mc_parity_status"] = f"failed: {error}"
        _state["healthy"] = False
        _persist_state()
        return False


def record_rule_mc(
    run_id: str,
    county_fips: list,
    valid_local_date: str,
    fm: np.ndarray,
    rh: np.ndarray,
    wind_kts: np.ndarray,
    fm_sigma: np.ndarray,
    rh_sigma: np.ndarray,
    wind_sigma_log: np.ndarray,
    thresholds: dict,
    n_draws: int = ru.DEFAULT_N_DRAWS,
    evidence_root: Optional[Path] = None,
) -> bool:
    """
    Runs the rule Monte Carlo for one forecast run's set of counties and
    writes an immutable {run_id}.rule_mc.json evidence file. Never raises;
    returns False on any failure (including shadow being disabled).
    """
    if not diagnostics()["enabled"]:
        return False
    try:
        if _state["rule_mc_parity_status"] != "ok" and not check_rule_mc_parity():
            raise RuntimeError(f"rule MC parity check failed: {_state['rule_mc_parity_status']}")

        n = len(county_fips)
        for name, arr in (("rh", rh), ("wind_kts", wind_kts), ("fm_sigma", fm_sigma),
                          ("rh_sigma", rh_sigma), ("wind_sigma_log", wind_sigma_log)):
            if len(arr) != n:
                raise ValueError(f"risk_fusion shadow row alignment mismatch: {name}")

        result = ru.sample_category_probabilities(
            fm=fm, rh=rh, wind_kts=wind_kts, fm_sigma=fm_sigma, rh_sigma=rh_sigma,
            wind_sigma_log=wind_sigma_log, thresholds=thresholds, n_draws=n_draws,
        )

        record = {
            "run_id": str(run_id),
            "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "valid_local_date": valid_local_date,
            "county_fips": list(county_fips),
            "n_draws": n_draws,
            "rule_spec_version": RULE_SPEC_VERSION,
            "rule_spec_sha256": RULE_SPEC_SHA256,
            "rule_category_deterministic": result["deterministic_category"].tolist(),
            "rule_category_modal": result["modal_category"].tolist(),
            "rule_category_modal_disagrees": result["modal_disagrees"].tolist(),
            "rule_category_stability": result["stability"].tolist(),
            "rule_probability_at_or_above_elevated": result["probability_at_or_above_elevated"].tolist(),
            "rule_probability_at_or_above_critical": result["probability_at_or_above_critical"].tolist(),
            "rule_probability_at_or_above_extreme": result["probability_at_or_above_extreme"].tolist(),
        }

        root = Path(evidence_root or EVIDENCE_ROOT)
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{run_id}.rule_mc.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)

        _state.update(
            consecutive_failures=0, last_error=None, healthy=True,
            last_success=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            runs=_state["runs"] + 1, successful_runs=_state.get("successful_runs", 0) + 1,
            counties_scored=n, county_days_recorded=_state.get("county_days_recorded", 0) + n,
            rule_mc_draws=n_draws,
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
    """Makes an enabled-but-empty shadow attempt observable, mirroring v5_shadow.record_skipped_run."""
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
