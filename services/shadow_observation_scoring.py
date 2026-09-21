"""Reads a {run_id}.prediction.json + its sibling {run_id}.observation.json
(written by v4_shadow.py/v5_shadow.py's record_predictions/
attach_observations) and computes real accuracy against the attached
observation - the step attach_observations() started but never finished.

No schema change to either file was needed: attach_observations()'s caller
(services/v5_verification.py::verify_pending, and the new
services/v4_verification.py mirroring it) already computes and persists,
per row_key, `target_fm` and an already-derived `actual_category` (via
core.fire_danger.calculate_fire_danger) in the observation record. The
matching prediction record already persists a point-estimate prediction
(v5_fm for v5, quantiles[:, 3] i.e. v4_category's own median column for
v4) and its own derived category, aligned by the same row_keys list. This
module just aligns the two by row_key and diffs values already sitting in
both files.

Writes results to a THIRD sibling file, {run_id}.scored.json, so
prediction/observation/scored stay separately immutable and re-scoring is
idempotent (just overwrite .scored.json, never touch the other two).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from services.shadow_metrics import continuous_error_summary


def _load(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def find_unscored_runs(evidence_root: Path) -> List[str]:
    """run_ids with both .prediction.json and .observation.json present but
    no .scored.json yet - the set a periodic scoring pass iterates."""
    evidence_root = Path(evidence_root)
    run_ids = []
    for observation_path in evidence_root.glob("*.observation.json"):
        run_id = observation_path.name[: -len(".observation.json")]
        if not (evidence_root / f"{run_id}.scored.json").exists():
            run_ids.append(run_id)
    return run_ids


def _aligned_observation_rows(observation_payload: dict, row_keys: List[str]) -> List[Optional[dict]]:
    """observations is a {row_key: {target_fm, actual_category, ...}}
    mapping (see services/v5_verification.py::verify_pending) - aligns to
    the prediction's own row_keys order via a straight dict lookup, no
    re-matching logic needed. A row_key with no matching observation
    (not yet mature, or no nearby station) is None."""
    observations = observation_payload.get("observations") or {}
    return [observations.get(key) for key in row_keys]


def _category_match_summary(predicted_category: List[Optional[int]], observed_rows: List[Optional[dict]]) -> dict:
    pairs = [
        (predicted, row["actual_category"])
        for predicted, row in zip(predicted_category, observed_rows)
        if row is not None and row.get("available")
    ]
    if not pairs:
        return {"n": 0, "match_rate": None}
    matches = sum(1 for predicted, actual in pairs if predicted == actual)
    return {"n": len(pairs), "match_rate": round(matches / len(pairs), 4)}


def score_v4_run(run_id: str, evidence_root: Path) -> Optional[dict]:
    """v4's prediction carries 7 quantiles per row - index 3 is the p50
    median, already treated as v4's headline point prediction/category
    elsewhere in v4_shadow.py (see v4_category's own computation)."""
    prediction = _load(Path(evidence_root) / f"{run_id}.prediction.json")
    observation = _load(Path(evidence_root) / f"{run_id}.observation.json")
    if prediction is None or observation is None:
        return None
    observed_rows = _aligned_observation_rows(observation, prediction["row_keys"])
    predicted_fm = [row[3] for row in prediction["v4_quantiles"]]
    observed_fm = [row["target_fm"] if row is not None else None for row in observed_rows]
    error = continuous_error_summary(
        [value for value in observed_fm if value is not None],
        [predicted_fm[i] for i, value in enumerate(observed_fm) if value is not None],
    )
    category_match = _category_match_summary(prediction.get("v4_category", []), observed_rows)
    return _finish_score(run_id, "v4", prediction, error, category_match)


def score_v5_run(run_id: str, evidence_root: Path) -> Optional[dict]:
    """v5's prediction carries a p10/p50/p90 interval per row - v5_fm is
    already the point (p50) prediction, no index lookup needed."""
    prediction = _load(Path(evidence_root) / f"{run_id}.prediction.json")
    observation = _load(Path(evidence_root) / f"{run_id}.observation.json")
    if prediction is None or observation is None:
        return None
    observed_rows = _aligned_observation_rows(observation, prediction["row_keys"])
    predicted_fm = prediction["v5_fm"]
    observed_fm = [row["target_fm"] if row is not None else None for row in observed_rows]
    error = continuous_error_summary(
        [value for value in observed_fm if value is not None],
        [predicted_fm[i] for i, value in enumerate(observed_fm) if value is not None],
    )
    category_match = _category_match_summary(prediction.get("v5_category", []), observed_rows)
    return _finish_score(run_id, "v5", prediction, error, category_match)


def _finish_score(run_id: str, family: str, prediction: dict, error: dict, category_match: dict) -> dict:
    return {
        "run_id": run_id,
        "family": family,
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "n_rows": len(prediction["row_keys"]),
        "n_matched": error["n"],
        "mae_vs_observation": error["mae"],
        "bias_vs_observation": error["bias"],
        "category_match": category_match,
        "prediction_manifest_sha256": prediction.get("bundle_manifest_sha256"),
    }


SCORERS = {"v4": score_v4_run, "v5": score_v5_run}


def score_pending_runs(family: str, evidence_root: Path) -> dict:
    """Scores every unscored run for one family, writes each result to its
    own {run_id}.scored.json, and returns a rollup summary for
    diagnostics()/beta_operations. Never raises - matches every other
    shadow module's failure-isolation convention."""
    scorer = SCORERS[family]
    evidence_root = Path(evidence_root)
    scored_count = 0
    errors = []
    for run_id in find_unscored_runs(evidence_root):
        try:
            result = scorer(run_id, evidence_root)
            if result is None:
                continue
            path = evidence_root / f"{run_id}.scored.json"
            with path.open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2)
            scored_count += 1
        except FileExistsError:
            continue  # already scored by a concurrent/prior pass - fine
        except Exception as error:
            errors.append({"run_id": run_id, "error": str(error)})
    return {"scored_count": scored_count, "errors": errors}


def rolling_accuracy_summary(family: str, evidence_root: Path, window: int = 30) -> dict:
    """Aggregates the most recent `window` .scored.json files into a single
    MAE/category-match figure for diagnostics()/the website card, mirroring
    beta_operations's own rolling-window convention."""
    evidence_root = Path(evidence_root)
    scored_files = sorted(evidence_root.glob("*.scored.json"), key=lambda p: p.stat().st_mtime)[-window:]
    records = [_load(path) for path in scored_files]
    maes = [r["mae_vs_observation"] for r in records if r and r.get("mae_vs_observation") is not None]
    match_rates = [r["category_match"]["match_rate"] for r in records
                   if r and (r.get("category_match") or {}).get("match_rate") is not None]
    return {
        "window": window,
        "scored_runs": len(records),
        "mean_mae_vs_observation": round(float(np.mean(maes)), 4) if maes else None,
        "mean_category_match_rate": round(float(np.mean(match_rates)), 4) if match_rates else None,
    }
