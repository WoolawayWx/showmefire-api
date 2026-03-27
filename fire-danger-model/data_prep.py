import argparse
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    ALL_CATEGORY_IDS,
    FEATURE_CANDIDATES,
    OUTPUT_DATA_DIR,
    RANDOM_STATE,
    SOURCE_DATA_PATH,
    SPLIT_META_PATH,
    TARGET_CATEGORY_COL,
    TARGET_SCORE_COL,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    TRAIN_FRACTION,
)
from utils import calculate_fire_danger_category, category_support, ensure_dirs


def prepare_dataset(source_csv: str):
    source_path = SOURCE_DATA_PATH if source_csv is None else source_csv
    df = pd.read_csv(source_path)

    required_for_label = ["target_fm", "rel_humidity", "wind_speed_ms"]
    missing = [c for c in required_for_label if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for label creation: {missing}")

    df = df.copy()
    df["wind_kts"] = df["wind_speed_ms"] * 1.94384
    df[TARGET_CATEGORY_COL] = [
        calculate_fire_danger_category(fm, rh, wk)
        for fm, rh, wk in zip(df["target_fm"], df["rel_humidity"], df["wind_kts"])
    ]
    df[TARGET_SCORE_COL] = df[TARGET_CATEGORY_COL].astype(float)

    feature_cols = [c for c in FEATURE_CANDIDATES if c in df.columns]
    if not feature_cols:
        raise ValueError("No candidate feature columns found in source data.")

    model_df = df[feature_cols + [TARGET_SCORE_COL, TARGET_CATEGORY_COL]].dropna().copy()

    split_strategy = "random"
    if "obs_time" in df.columns:
        timestamps = pd.to_datetime(df.loc[model_df.index, "obs_time"], errors="coerce")
        if timestamps.notna().sum() > 0:
            model_df["_obs_time"] = timestamps
            model_df = model_df.sort_values("_obs_time")
            cutoff = int(len(model_df) * TRAIN_FRACTION)
            train_df = model_df.iloc[:cutoff].drop(columns=["_obs_time"])
            test_df = model_df.iloc[cutoff:].drop(columns=["_obs_time"])
            split_strategy = "time"
        else:
            train_df, test_df = train_test_split(
                model_df,
                train_size=TRAIN_FRACTION,
                random_state=RANDOM_STATE,
                shuffle=True,
            )
    else:
        train_df, test_df = train_test_split(
            model_df,
            train_size=TRAIN_FRACTION,
            random_state=RANDOM_STATE,
            shuffle=True,
        )

    ensure_dirs(OUTPUT_DATA_DIR)
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    metadata = {
        "source": str(source_path),
        "split_strategy": split_strategy,
        "random_state": RANDOM_STATE,
        "train_fraction": TRAIN_FRACTION,
        "n_rows_total": int(len(model_df)),
        "n_rows_train": int(len(train_df)),
        "n_rows_test": int(len(test_df)),
        "feature_columns": feature_cols,
        "target_score_col": TARGET_SCORE_COL,
        "target_category_col": TARGET_CATEGORY_COL,
        "all_category_ids": ALL_CATEGORY_IDS,
        "class_distribution": {
            "full": category_support(model_df[TARGET_CATEGORY_COL]),
            "train": category_support(train_df[TARGET_CATEGORY_COL]),
            "test": category_support(test_df[TARGET_CATEGORY_COL]),
        },
    }

    with open(SPLIT_META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Prepared standalone fire danger datasets")
    print(f"  train: {TRAIN_DATA_PATH}")
    print(f"  test:  {TEST_DATA_PATH}")
    print(f"  split strategy: {split_strategy}")
    print(f"  features ({len(feature_cols)}): {feature_cols}")
    print(f"  class distribution (test): {metadata['class_distribution']['test']}")


def main():
    parser = argparse.ArgumentParser(description="Prepare standalone fire danger model data.")
    parser.add_argument(
        "--source-csv",
        default=None,
        help="Optional source csv path (defaults to api/data/final_training_data.csv).",
    )
    args = parser.parse_args()
    prepare_dataset(args.source_csv)


if __name__ == "__main__":
    main()
