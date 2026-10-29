"""Gated retrain for the per-detection confidence model:

    python retrain.py

Trains a candidate model (train.py's logic), evaluates it by incident-grouped
cross-validation (evaluate.py), and only overwrites the currently-registered
model (models/detection_confidence_model.json) if the candidate beats the
satellite's own confidence on held-out detections and is not meaningfully
worse than the registered model (compared only when both were measured the
same way) - the same train -> evaluate -> gate -> register discipline
model-training/ensemble_fire_danger/retrain.py uses, scaled down to this
package's single small model instead of that pipeline's versioned registry.

Intended to run on a schedule (see core/scheduler.py) rather than only
manually - CPU-only XGBoost, seconds of runtime even at a few thousand
rows, no GPU and no external billing.
"""
import json
import logging
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
API_DIR = BASE_DIR.parent
sys.path.append(str(API_DIR))

from config import LATEST_MODEL_META_PATH, LATEST_MODEL_PATH, MIN_PER_CLASS, MIN_TRAINING_SAMPLES, MODELS_DIR  # noqa: E402
from evaluate import evaluate_rows  # noqa: E402
from features import FEATURE_NAMES  # noqa: E402
from train import fit_booster, load_training_rows, training_set  # noqa: E402

logger = logging.getLogger(__name__)

# A candidate model is registered even if its AUC is slightly lower than the
# currently-registered one, since a small labeled set makes AUC noisy - only
# reject a candidate that's meaningfully worse.
AUC_REGRESSION_TOLERANCE = 0.03


def _previous_auc() -> float | None:
    """The registered model's AUC, but only if it was measured the same (grouped) way as a new candidate's: older
    registrations used a random split that overstated AUC, so they are not comparable."""
    if not LATEST_MODEL_META_PATH.exists():
        return None
    try:
        evaluation = json.loads(LATEST_MODEL_META_PATH.read_text()).get("eval", {})
        return evaluation.get("auc") if evaluation.get("method") == "grouped_cv" else None
    except Exception:
        return None


def retrain_and_gate() -> dict:
    rows = load_training_rows()
    evaluation = evaluate_rows(rows)

    samples, labels, _groups = training_set(rows)
    fires, not_fires = sum(labels), len(labels) - sum(labels)
    if len(samples) < MIN_TRAINING_SAMPLES or min(fires, not_fires) < MIN_PER_CLASS:
        return {
            "trained": False,
            "registered": False,
            "reason": f"only {len(samples)} labeled detections ({fires} fire, {not_fires} not a fire); need >= "
                      f"{MIN_TRAINING_SAMPLES} with at least {MIN_PER_CLASS} of each",
            "eval": evaluation,
        }

    booster = fit_booster(samples, labels)

    candidate_auc = evaluation.get("auc") if evaluation.get("evaluated") else None
    baseline_auc = evaluation.get("baseline_auc") if evaluation.get("evaluated") else None
    if candidate_auc is not None and baseline_auc is not None and candidate_auc < baseline_auc:
        return {
            "trained": True,
            "registered": False,
            "reason": f"candidate AUC {candidate_auc:.3f} does not beat the satellite's own confidence ({baseline_auc:.3f}) "
                      "on held-out incidents, so the built-in scoring stays",
            "eval": evaluation,
        }

    previous_auc = _previous_auc()
    if previous_auc is not None and candidate_auc is not None and candidate_auc < previous_auc - AUC_REGRESSION_TOLERANCE:
        return {
            "trained": True,
            "registered": False,
            "reason": f"candidate AUC {candidate_auc:.3f} regresses vs registered {previous_auc:.3f} by more than {AUC_REGRESSION_TOLERANCE}",
            "eval": evaluation,
        }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(LATEST_MODEL_PATH))
    LATEST_MODEL_META_PATH.write_text(
        json.dumps(
            {
                "feature_names": FEATURE_NAMES,
                "training_samples": len(samples),
                "positive_samples": sum(labels),
                "eval": evaluation,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"trained": True, "registered": True, "training_samples": len(samples), "eval": evaluation}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(retrain_and_gate(), indent=2))
