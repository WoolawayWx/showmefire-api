"""Gated retrain for the per-detection confidence model:

    python retrain.py

Trains a candidate model (train.py's logic), evaluates it on a held-out
split (evaluate.py), and only overwrites the currently-registered model
(models/detection_confidence_model.json) if the candidate's AUC is not
worse than the currently-registered model's recorded AUC (or none is
registered yet) - the same train -> evaluate -> gate -> register discipline
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

from config import LATEST_MODEL_META_PATH, LATEST_MODEL_PATH, MIN_TRAINING_SAMPLES, MODELS_DIR, RANDOM_STATE  # noqa: E402
from evaluate import evaluate_rows  # noqa: E402
from features import FEATURE_NAMES, build_feature_vector  # noqa: E402
from train import FALSE_POSITIVE_CAUSES, load_training_rows  # noqa: E402

logger = logging.getLogger(__name__)

# A candidate model is registered even if its AUC is slightly lower than the
# currently-registered one, since a small labeled set makes AUC noisy - only
# reject a candidate that's meaningfully worse.
AUC_REGRESSION_TOLERANCE = 0.03


def _previous_auc() -> float | None:
    if not LATEST_MODEL_META_PATH.exists():
        return None
    try:
        meta = json.loads(LATEST_MODEL_META_PATH.read_text())
        return meta.get("eval", {}).get("auc")
    except Exception:
        return None


def retrain_and_gate() -> dict:
    rows = load_training_rows()
    evaluation = evaluate_rows(rows)

    samples = [build_feature_vector(row) for row in rows]
    labels = [0 if row.get("cause_category") in FALSE_POSITIVE_CAUSES else 1 for row in rows]
    if len(samples) < MIN_TRAINING_SAMPLES or len(set(labels)) < 2:
        return {
            "trained": False,
            "registered": False,
            "reason": f"only {len(samples)} labeled real-detection samples (need >= {MIN_TRAINING_SAMPLES} with both classes present)",
            "eval": evaluation,
        }

    import xgboost as xgb

    dtrain = xgb.DMatrix(samples, label=labels, feature_names=FEATURE_NAMES, missing=float("nan"))
    booster = xgb.train(
        {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3, "eta": 0.1, "seed": RANDOM_STATE},
        dtrain,
        num_boost_round=100,
    )

    previous_auc = _previous_auc()
    candidate_auc = evaluation.get("auc") if evaluation.get("evaluated") else None
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
