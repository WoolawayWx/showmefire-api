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

from config import LATEST_MODEL_META_PATH, LATEST_MODEL_PATH, MIN_PER_CLASS, MIN_TRAINING_SAMPLES, MODELS_DIR, RANDOM_STATE
from features import FEATURE_NAMES, build_feature_vector, label_for

XGB_PARAMS = {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3, "eta": 0.1, "seed": RANDOM_STATE}
BOOST_ROUNDS = 100


def load_training_rows():
    from core.database import list_labeled_detection_events
    return list_labeled_detection_events()


def training_set(rows):
    """(feature vectors, labels, groups) for the rows that have a usable label (features.label_for). The group is
    the incident, so detections of the same fire stay together when the data is split for evaluation."""
    samples, labels, groups = [], [], []
    for row in rows:
        label = label_for(row)
        if label is None:
            continue
        samples.append(build_feature_vector(row))
        labels.append(label)
        groups.append(row.get("incident_id") or f"event-{row.get('id')}")
    return samples, labels, groups


def fit_booster(samples, labels):
    import xgboost as xgb

    dtrain = xgb.DMatrix(samples, label=labels, feature_names=FEATURE_NAMES, missing=float("nan"))
    return xgb.train(XGB_PARAMS, dtrain, num_boost_round=BOOST_ROUNDS)


def train_and_save() -> dict:
    samples, labels, _groups = training_set(load_training_rows())

    if len(samples) < MIN_TRAINING_SAMPLES or min(sum(labels), len(labels) - sum(labels)) < MIN_PER_CLASS:
        return {
            "trained": False,
            "reason": f"only {len(samples)} labeled samples ({sum(labels)} fire, {len(labels) - sum(labels)} not a fire); "
                      f"need >= {MIN_TRAINING_SAMPLES} with at least {MIN_PER_CLASS} of each",
        }

    booster = fit_booster(samples, labels)

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
