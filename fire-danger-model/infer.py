import argparse
import json
from pathlib import Path

import pandas as pd
import xgboost as xgb

from config import LATEST_MODEL_META_PATH, LATEST_MODEL_PATH
from utils import map_score_to_category


def load_input_df(input_file):
    input_path = Path(input_file)
    suffix = input_path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(input_path)
    if suffix in (".json", ".jsonl"):
        with input_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            return pd.DataFrame([payload])
        raise ValueError("Unsupported JSON payload format.")

    raise ValueError("Input must be .csv or .json")


def infer(input_file, output_file=None, model_path=None, model_meta_path=None):
    model_path = LATEST_MODEL_PATH if model_path is None else model_path
    model_meta_path = LATEST_MODEL_META_PATH if model_meta_path is None else model_meta_path

    with open(model_meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_cols = meta["feature_columns"]
    category_thresholds = meta.get("category_thresholds")
    df = load_input_df(input_file)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    X = df[feature_cols]

    booster = xgb.Booster()
    booster.load_model(model_path)
    dmat = xgb.DMatrix(X)
    pred_score = booster.predict(dmat)
    pred_category = map_score_to_category(pred_score, thresholds=category_thresholds)

    out_df = df.copy()
    out_df["predicted_fire_danger_score"] = pred_score
    out_df["predicted_fire_danger_category"] = pred_category

    if output_file:
        out_path = Path(output_file)
        if out_path.suffix.lower() == ".json":
            out_path.write_text(out_df.to_json(orient="records", indent=2), encoding="utf-8")
        else:
            out_df.to_csv(out_path, index=False)
        print(f"Saved predictions to {out_path}")
    else:
        print(out_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Run standalone fire danger model inference.")
    parser.add_argument("--input-file", required=True, help="CSV/JSON with model feature columns.")
    parser.add_argument("--output-file", default=None, help="Optional output path (.csv or .json).")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--model-meta-path", default=None)
    args = parser.parse_args()

    infer(
        input_file=args.input_file,
        output_file=args.output_file,
        model_path=args.model_path,
        model_meta_path=args.model_meta_path,
    )


if __name__ == "__main__":
    main()
