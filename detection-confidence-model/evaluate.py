"""Held-out evaluation for the per-detection confidence model. Run
standalone or imported by retrain.py - never imported by the live API
process (see train.py's module docstring for why).

Detections of the same fire are near-duplicates, so a random split would let the model "recognise" test rows from
their training twins. This splits by incident (so a fire is entirely in train or entirely in test) with stratified
k-fold, scores every row once while it is held out, and reports:

  auc / pr_auc / brier / logloss   how well the held-out scores separate fire from not-a-fire, and how honest the
                                   percentages are (brier and logloss are lower-is-better)
  baseline_auc                     the same AUC using only the satellite's own confidence. A model that cannot beat
                                   this adds nothing, so retrain.py will not register it
  auc_lift                         auc - baseline_auc
"""
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
API_DIR = BASE_DIR.parent
sys.path.append(str(API_DIR))

from config import MIN_PER_CLASS, RANDOM_STATE
from features import FEATURE_NAMES, RAW_CONFIDENCE_INDEX
from train import BOOST_ROUNDS, XGB_PARAMS, training_set

FOLDS = 5
MIN_EVALUATION_SAMPLES = 10


def _xgboost_fit_predict(train_x, train_y, test_x):
    import xgboost as xgb

    booster = xgb.train(
        XGB_PARAMS, xgb.DMatrix(train_x, label=train_y, feature_names=FEATURE_NAMES, missing=float("nan")),
        num_boost_round=BOOST_ROUNDS,
    )
    return booster.predict(xgb.DMatrix(test_x, feature_names=FEATURE_NAMES, missing=float("nan")))


def evaluate_rows(rows: list, fit_predict=None, folds: int = FOLDS) -> dict:
    """Grouped, stratified cross-validation. `fit_predict(train_x, train_y, test_x) -> probabilities` defaults to the
    production XGBoost settings; tests pass a stand-in. Returns {"evaluated": False, "reason": ...} when there is
    too little labelled data of either class to say anything."""
    import numpy as np
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    samples, labels, groups = training_set(rows)
    positives, negatives = sum(labels), len(labels) - sum(labels)
    if len(samples) < MIN_EVALUATION_SAMPLES or min(positives, negatives) < 3:
        return {"evaluated": False,
                "reason": f"only {len(samples)} labeled samples ({positives} fire, {negatives} not a fire); need >= "
                          f"{MIN_EVALUATION_SAMPLES} with at least 3 of each"}
    splits = min(folds, positives, negatives, len(set(groups)))
    if splits < 2:
        return {"evaluated": False, "reason": "too few incidents of each class to split by incident"}

    fit_predict = fit_predict or _xgboost_fit_predict
    x, y, g = np.array(samples, dtype=float), np.array(labels), np.array([str(item) for item in groups])
    held_out = np.full(len(y), np.nan)
    for train_index, test_index in StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=RANDOM_STATE).split(x, y, g):
        if len(set(y[train_index])) < 2:
            continue
        held_out[test_index] = fit_predict(x[train_index].tolist(), y[train_index].tolist(), x[test_index].tolist())
    scored = ~np.isnan(held_out)
    if not scored.any() or len(set(y[scored])) < 2:
        return {"evaluated": False, "reason": "no fold had both classes to train and test on"}

    truth, predicted = y[scored], np.clip(held_out[scored], 1e-6, 1 - 1e-6)
    auc = float(roc_auc_score(truth, predicted))
    baseline = x[scored][:, RAW_CONFIDENCE_INDEX]
    baseline_auc = float(roc_auc_score(truth, baseline)) if len(set(baseline)) > 1 else None
    return {
        "evaluated": True,
        "method": "grouped_cv",
        "folds": splits,
        "samples": int(scored.sum()),
        "positives": int(truth.sum()),
        "negatives": int(len(truth) - truth.sum()),
        "groups": len(set(g[scored])),
        "auc": auc,
        "pr_auc": float(average_precision_score(truth, predicted)),
        "brier": float(brier_score_loss(truth, predicted)),
        "logloss": float(log_loss(truth, predicted)),
        "baseline_auc": baseline_auc,
        "auc_lift": None if baseline_auc is None else auc - baseline_auc,
    }


def evaluate_current() -> dict:
    from core.database import list_labeled_detection_events

    return evaluate_rows(list_labeled_detection_events())


if __name__ == "__main__":
    import json
    print(json.dumps(evaluate_current(), indent=2))
