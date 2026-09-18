"""Offline trainer for the per-detection confidence model. Run standalone:

    python train.py

Not imported by the live API process - services/detection_confidence.py
loads the resulting models/detection_confidence_model.json directly (same
pattern services/model_shadow.py already uses for fire-danger-model),
avoiding any sys.path collision between this package's modules and other
model-training packages under api/.
"""
import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
API_DIR = BASE_DIR.parent
sys.path.append(str(API_DIR))

from config import LATEST_MODEL_META_PATH, LATEST_MODEL_PATH, MIN_TRAINING_SAMPLES, MODELS_DIR, RANDOM_STATE
from features import FEATURE_NAMES, build_feature_vector

FALSE_POSITIVE_CAUSES = {"prescribed", "agricultural", "debris_burn"}


def load_training_rows():
    from core.database import list_labeled_detection_events
    return list_labeled_detection_events()


def train_and_save() -> dict:
    import xgboost as xgb

    rows = load_training_rows()
    samples, labels = [], []
    for row in rows:
        samples.append(build_feature_vector(row))
        labels.append(0 if row.get("cause_category") in FALSE_POSITIVE_CAUSES else 1)

    if len(samples) < MIN_TRAINING_SAMPLES or len(set(labels)) < 2:
        return {
            "trained": False,
            "reason": f"only {len(samples)} labeled samples (need >= {MIN_TRAINING_SAMPLES} with both classes present)",
        }

    dtrain = xgb.DMatrix(samples, label=labels, feature_names=FEATURE_NAMES, missing=float("nan"))
    booster = xgb.train(
        {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 3,
            "eta": 0.1,
            "seed": RANDOM_STATE,
        },
        dtrain,
        num_boost_round=100,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(LATEST_MODEL_PATH))
    LATEST_MODEL_META_PATH.write_text(
        json.dumps(
            {
                "feature_names": FEATURE_NAMES,
                "training_samples": len(samples),
                "positive_samples": sum(labels),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"trained": True, "training_samples": len(samples), "positive_samples": sum(labels)}


if __name__ == "__main__":
    print(train_and_save())
