"""Held-out evaluation for the per-detection confidence model. Run
standalone or imported by retrain.py - never imported by the live API
process (see train.py's module docstring for why)."""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
API_DIR = BASE_DIR.parent
sys.path.append(str(API_DIR))

from config import RANDOM_STATE
from features import FEATURE_NAMES, build_feature_vector

FALSE_POSITIVE_CAUSES = {"prescribed", "agricultural", "debris_burn", "not_a_fire"}


def evaluate_rows(rows: list) -> dict:
    """Trains on an 80% split and reports AUC/logloss on the held-out 20%,
    using the same features/labels train.py's full-data model uses. Returns
    {"evaluated": False, "reason": ...} if there isn't enough data to split."""
    import numpy as np
    import xgboost as xgb
    from sklearn.metrics import log_loss, roc_auc_score
    from sklearn.model_selection import train_test_split

    samples = [build_feature_vector(row) for row in rows]
    labels = [0 if row.get("cause_category") in FALSE_POSITIVE_CAUSES else 1 for row in rows]

    if len(samples) < 10 or len(set(labels)) < 2:
        return {"evaluated": False, "reason": f"only {len(samples)} samples, need >=10 with both classes"}

    x_train, x_test, y_train, y_test = train_test_split(
        samples, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels,
    )
    dtrain = xgb.DMatrix(x_train, label=y_train, feature_names=FEATURE_NAMES, missing=float("nan"))
    dtest = xgb.DMatrix(x_test, label=y_test, feature_names=FEATURE_NAMES, missing=float("nan"))
    booster = xgb.train(
        {"objective": "binary:logistic", "eval_metric": "logloss", "max_depth": 3, "eta": 0.1, "seed": RANDOM_STATE},
        dtrain,
        num_boost_round=100,
    )
    predicted = booster.predict(dtest)
    if len(set(y_test)) < 2:
        auc = None
    else:
        auc = float(roc_auc_score(y_test, predicted))
    return {
        "evaluated": True,
        "test_samples": len(x_test),
        "auc": auc,
        "logloss": float(log_loss(y_test, np.clip(predicted, 1e-6, 1 - 1e-6))),
    }


def evaluate_current() -> dict:
    from core.database import list_labeled_detection_events

    return evaluate_rows(list_labeled_detection_events())


if __name__ == "__main__":
    import json
    print(json.dumps(evaluate_current(), indent=2))
