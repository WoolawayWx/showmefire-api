"""Population Stability Index (PSI) drift detection over existing shadow evidence.

Watches feature/prediction distribution shift *between* the day14/day30
shadow checkpoints (model-training/spatial/evaluate_v5_shadow.py) - the
shadow pipeline only checks accuracy AT those checkpoints, so a silently
degrading input (e.g. a station going stale) wouldn't otherwise be caught
until the next one. Reuses each model family's EXISTING evidence files
rather than logging anything new - no new instrumentation, just a new
read of data already being written:

  fuel_moisture   - api/logs/model_shadow.jsonl (services/model_shadow.py)
  v5              - SMF_V5_EVIDENCE_ROOT *.prediction.json/*.observation.json
                    (services/v5_shadow.py)
  risk_fusion_glm - SMF_RISK_FUSION_GLM_EVIDENCE_ROOT *.glm_score.json
                    (services/risk_fusion_glm_shadow.py)

All evidence lives under api/'s own data/logs directories, written by this
same server process - this module deliberately does NOT import
model-training/spatial/evaluate_v5_shadow.py's own V5 loader, even though
it reads the same file shape: model-training/ is a separate repo that is
not part of the production api/ Docker image (see api/Dockerfile's
`COPY . .`, scoped to the api/ build context only), so a runtime import
from there would work in local dev and fail in production. The V5 reader
below reimplements just enough of the same record shape (as written by
services/v5_shadow.py::record_predictions/attach_observations) to stay
self-contained.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DRIFT_PSI_WARN = float(os.getenv("DRIFT_PSI_WARN", "0.1"))
DRIFT_PSI_ALERT = float(os.getenv("DRIFT_PSI_ALERT", "0.25"))

_DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
_LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


def compute_psi(reference, current, bins: int = 10) -> float:
    """Population Stability Index between two 1-D numeric samples.

    PSI < 0.1 stable, 0.1-0.25 moderate, > 0.25 flagged (standard cutoffs -
    see DRIFT_PSI_WARN/DRIFT_PSI_ALERT). Bin edges come from the reference
    sample's quantiles so each reference bin starts with ~equal mass;
    fractions are epsilon-floored so an empty bin never produces a
    divide-by-zero / -inf contribution. Returns 0.0 (not an error) when
    there isn't enough reference data to form distinct bins - callers
    should treat that as "not enough evidence yet", not "no drift".
    """
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]
    if len(reference) < bins or len(current) == 0:
        return 0.0
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)
    ref_frac = np.maximum(ref_counts / max(len(reference), 1), 1e-6)
    cur_frac = np.maximum(cur_counts / max(len(current), 1), 1e-6)
    return float(np.sum((cur_frac - ref_frac) * np.log(cur_frac / ref_frac)))


def _split_reference_current(frame: pd.DataFrame, time_column: str, reference_window_days: int, current_window_days: int):
    times = pd.to_datetime(frame[time_column], utc=True, errors="coerce")
    frame = frame.assign(_drift_time=times).dropna(subset=["_drift_time"])
    if frame.empty:
        return frame, frame
    cutoff = frame["_drift_time"].max()
    current_start = cutoff - pd.Timedelta(days=current_window_days)
    reference_start = current_start - pd.Timedelta(days=reference_window_days)
    reference = frame[(frame["_drift_time"] >= reference_start) & (frame["_drift_time"] < current_start)]
    current = frame[frame["_drift_time"] >= current_start]
    return reference, current


def _flags(feature_psi: Dict[str, float], prediction_psi: Optional[float]) -> List[str]:
    flags = [name for name, value in feature_psi.items() if value > DRIFT_PSI_ALERT]
    if prediction_psi is not None and prediction_psi > DRIFT_PSI_ALERT:
        flags.append("prediction")
    return flags


def _empty_report() -> Dict:
    return {"features": {}, "prediction_psi": None, "flags": [], "support": {"reference": 0, "current": 0}}


def _evaluate_fuel_moisture(evidence_root, reference_window_days, current_window_days) -> Dict:
    path = Path(evidence_root) if evidence_root else _LOGS_DIR / "model_shadow.jsonl"
    if not path.exists():
        return _empty_report()
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    successful = [record for record in records if not record.get("failed")]
    frame = pd.DataFrame(successful)
    if frame.empty or "timestamp" not in frame.columns:
        return _empty_report()
    reference, current = _split_reference_current(frame, "timestamp", reference_window_days, current_window_days)
    prediction_psi = None
    if not reference.empty and not current.empty and "mean_absolute_difference" in frame.columns:
        prediction_psi = compute_psi(reference["mean_absolute_difference"], current["mean_absolute_difference"])
    return {"features": {}, "prediction_psi": prediction_psi, "flags": _flags({}, prediction_psi),
            "support": {"reference": int(len(reference)), "current": int(len(current))}}


def _load_v5_evidence(root: Path) -> pd.DataFrame:
    """Minimal, self-contained reader for V5 prediction/observation pairs - see module docstring for why this
    reimplements evaluate_v5_shadow.py's loader instead of importing it."""
    rows = []
    if not root.exists():
        return pd.DataFrame(rows)
    for prediction_path in sorted(root.glob("*.prediction.json")):
        try:
            prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        run_id = str(prediction.get("run_id"))
        observation_path = prediction_path.with_name(f"{run_id}.observation.json")
        if not observation_path.exists():
            continue
        try:
            observation = json.loads(observation_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        observed = observation.get("observations", {})
        if isinstance(observed, list):
            observed = {str(item["row_key"]): item for item in observed}
        v5_fm = prediction.get("v5_fm") or []
        for index, row_key in enumerate(prediction.get("row_keys", [])):
            target = observed.get(str(row_key))
            if not target or not target.get("available", True) or index >= len(v5_fm):
                continue
            if target.get("target_fm") is None:
                continue
            rows.append({
                "candidate_fm": v5_fm[index],
                "match_age_minutes": target.get("match_age_minutes"),
                "observation_time": target.get("observation_time"),
            })
    return pd.DataFrame(rows)


def _evaluate_v5(evidence_root, reference_window_days, current_window_days) -> Dict:
    root = Path(evidence_root or os.getenv("SMF_V5_EVIDENCE_ROOT") or (_DATA_DIR / "model-shadow" / "v5"))
    frame = _load_v5_evidence(root)
    if frame.empty or "observation_time" not in frame.columns:
        return _empty_report()
    reference, current = _split_reference_current(frame, "observation_time", reference_window_days, current_window_days)
    features = {}
    if not reference.empty and not current.empty and "match_age_minutes" in frame.columns:
        features["match_age_minutes"] = compute_psi(reference["match_age_minutes"], current["match_age_minutes"])
    prediction_psi = (compute_psi(reference["candidate_fm"], current["candidate_fm"])
                      if not reference.empty and not current.empty else None)
    return {"features": features, "prediction_psi": prediction_psi, "flags": _flags(features, prediction_psi),
            "support": {"reference": int(len(reference)), "current": int(len(current))}}


def _evaluate_risk_fusion_glm(evidence_root, reference_window_days, current_window_days) -> Dict:
    root = Path(evidence_root or os.getenv("SMF_RISK_FUSION_GLM_EVIDENCE_ROOT")
                or (_DATA_DIR / "model-shadow" / "risk-fusion-glm"))
    rows = []
    if root.exists():
        for path in sorted(root.glob("*.glm_score.json")):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            for lam in record.get("lam", []):
                rows.append({"recorded_at": record.get("recorded_at"), "lam": lam})
    frame = pd.DataFrame(rows)
    if frame.empty or "recorded_at" not in frame.columns:
        return _empty_report()
    reference, current = _split_reference_current(frame, "recorded_at", reference_window_days, current_window_days)
    prediction_psi = (compute_psi(reference["lam"], current["lam"])
                      if not reference.empty and not current.empty else None)
    return {"features": {}, "prediction_psi": prediction_psi, "flags": _flags({}, prediction_psi),
            "support": {"reference": int(len(reference)), "current": int(len(current))}}


_EVALUATORS = {
    "fuel_moisture": _evaluate_fuel_moisture,
    "v5": _evaluate_v5,
    "risk_fusion_glm": _evaluate_risk_fusion_glm,
}


def evaluate_drift(model_type: str, evidence_root=None, reference_window_days: int = 30,
                    current_window_days: int = 7) -> Dict:
    """Feature/prediction PSI report for one model family's existing shadow evidence.

    Returns {model_type, generated_at, features, prediction_psi, flags,
    support}. Raises ValueError for an unknown model_type - services/
    drift_monitor.py isolates that per model type so one family's failure
    never blocks the others.
    """
    evaluator = _EVALUATORS.get(model_type)
    if evaluator is None:
        raise ValueError(f"Unknown model_type for drift evaluation: {model_type!r}")
    report = evaluator(evidence_root, reference_window_days, current_window_days)
    return {"model_type": model_type, "generated_at": datetime.now(timezone.utc).isoformat(), **report}


def write_drift_report(report: Dict, drift_dir, run_id: str) -> Path:
    """Write-once persistence (mirrors services/v5_shadow.py::record_predictions' 'x' mode) -
    a drift history becomes an append-only audit trail, same invariant tested for
    prediction logs in api/tests/test_v5_safety.py."""
    drift_dir = Path(drift_dir)
    drift_dir.mkdir(parents=True, exist_ok=True)
    path = drift_dir / f"{report['model_type']}_{run_id}.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
    return path
