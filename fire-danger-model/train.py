import argparse
import json
import shutil

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, f1_score

from config import (
    ALL_CATEGORY_IDS,
    LATEST_MODEL_META_PATH,
    LATEST_MODEL_PATH,
    MODELS_DIR,
    SPLIT_META_PATH,
    TARGET_CATEGORY_COL,
    TARGET_SCORE_COL,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
)
from utils import (
    calibrate_category_thresholds,
    category_support,
    ensure_dirs,
    map_score_to_category,
    utc_timestamp,
    write_json,
)


def train_model(n_estimators=300, learning_rate=0.05, max_depth=5):
    with open(SPLIT_META_PATH, "r", encoding="utf-8") as f:
        split_meta = json.load(f)

    feature_cols = split_meta["feature_columns"]

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET_SCORE_COL]
    y_train_cat = train_df[TARGET_CATEGORY_COL].astype(int)

    X_test = test_df[feature_cols]
    y_test = test_df[TARGET_SCORE_COL]
    y_test_cat = test_df[TARGET_CATEGORY_COL].astype(int)

    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    category_thresholds = calibrate_category_thresholds(pred_train, y_train_cat.values)

    pred_train_cat = map_score_to_category(pred_train, thresholds=category_thresholds)
    pred_test_cat = map_score_to_category(pred_test, thresholds=category_thresholds)

    metrics = {
        "train": {
            "mae": float(mean_absolute_error(y_train, pred_train)),
            "rmse": float(np.sqrt(mean_squared_error(y_train, pred_train))),
            "r2": float(r2_score(y_train, pred_train)),
            "macro_f1": float(
                f1_score(
                    y_train_cat,
                    pred_train_cat,
                    labels=ALL_CATEGORY_IDS,
                    average="macro",
                    zero_division=0,
                )
            ),
            "weighted_f1": float(
                f1_score(
                    y_train_cat,
                    pred_train_cat,
                    labels=ALL_CATEGORY_IDS,
                    average="weighted",
                    zero_division=0,
                )
            ),
        },
        "test": {
            "mae": float(mean_absolute_error(y_test, pred_test)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred_test))),
            "r2": float(r2_score(y_test, pred_test)),
            "macro_f1": float(
                f1_score(
                    y_test_cat,
                    pred_test_cat,
                    labels=ALL_CATEGORY_IDS,
                    average="macro",
                    zero_division=0,
                )
            ),
            "weighted_f1": float(
                f1_score(
                    y_test_cat,
                    pred_test_cat,
                    labels=ALL_CATEGORY_IDS,
                    average="weighted",
                    zero_division=0,
                )
            ),
        },
    }

    ensure_dirs(MODELS_DIR)
    ts = utc_timestamp()
    versioned_model_path = MODELS_DIR / f"fire_danger_model_{ts}.json"
    model.save_model(versioned_model_path)
    shutil.copy2(versioned_model_path, LATEST_MODEL_PATH)

    model_meta = {
        "model_type": "xgboost_regressor",
        "created_utc": ts,
        "versioned_model_path": str(versioned_model_path),
        "latest_model_path": str(LATEST_MODEL_PATH),
        "feature_columns": feature_cols,
        "target_score_col": TARGET_SCORE_COL,
        "target_category_col": TARGET_CATEGORY_COL,
        "category_thresholds": category_thresholds,
        "class_distribution": {
            "train": category_support(y_train_cat),
            "test": category_support(y_test_cat),
        },
        "training_metrics": metrics,
        "split_metadata": split_meta,
        "parameters": {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "reg:squarederror",
        },
    }
    write_json(LATEST_MODEL_META_PATH, model_meta)

    print("Standalone fire danger model training complete")
    print(f"  saved: {LATEST_MODEL_PATH}")
    print(f"  category thresholds: {category_thresholds}")
    print(f"  test macro_f1: {metrics['test']['macro_f1']:.4f}")
    print(f"  test mae:      {metrics['test']['mae']:.4f}")



def main():
    parser = argparse.ArgumentParser(description="Train standalone fire danger model.")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=5)
    args = parser.parse_args()

    train_model(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
    )


if __name__ == "__main__":
    main()
