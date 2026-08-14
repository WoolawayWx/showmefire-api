"""Train an out-of-time validated fuel-moisture beta candidate."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.fire_danger import RULE_SPEC_VERSION, calculate_fire_danger, meters_per_second_to_knots
from core.precipitation import PRECIPITATION_CONTRACT_SHA256, PRECIPITATION_CONTRACT_VERSION
from models.features import DEFAULT_FEATURES, FEATURE_SCHEMA_VERSION, LEGACY_FEATURES, feature_ranges
from models.fm_uncertainty import fit_uncertainty
from models.versioning import load_active_model_path, register_trained_model


def _metrics(actual, predicted):
    error = np.asarray(predicted) - np.asarray(actual)
    return {"mae": float(np.mean(np.abs(error))), "bias": float(np.mean(error)),
            "rmse": float(np.sqrt(np.mean(error ** 2))), "count": int(len(error))}


def _danger_metrics(frame, actual_fm, predicted_fm):
    wind = meters_per_second_to_knots(frame["wind_speed_ms"].to_numpy())
    actual = np.array([calculate_fire_danger(fm, rh, ws) for fm, rh, ws in
                       zip(actual_fm, frame["rel_humidity"], wind)])
    predicted = np.array([calculate_fire_danger(fm, rh, ws) for fm, rh, ws in
                          zip(predicted_fm, frame["rel_humidity"], wind)])
    elevated = actual >= 2
    false_negative_rate = float(np.mean(predicted[elevated] < 2)) if elevated.any() else None
    return {"high_impact_support": int(elevated.sum()), "high_impact_false_negative_rate": false_negative_rate,
            "errors_over_one_category": int((np.abs(predicted - actual) > 1).sum())}


def _rolling_date_folds(frame, count=3):
    dates = pd.Series(frame["obs_time"].dt.floor("D").unique()).sort_values().to_numpy()
    if len(dates) < count + 1:
        return []
    blocks = np.array_split(dates, count + 1)
    folds = []
    row_dates = frame["obs_time"].dt.floor("D")
    for index in range(1, len(blocks)):
        training_dates = np.concatenate(blocks[:index])
        validation_dates = blocks[index]
        train_idx = np.flatnonzero(row_dates.isin(training_dates).to_numpy())
        validation_idx = np.flatnonzero(row_dates.isin(validation_dates).to_numpy())
        if len(train_idx) and len(validation_idx):
            folds.append((train_idx, validation_idx))
    return folds


def train_fuel_moisture_model(channel="beta", bump="patch"):
    if channel != "beta":
        raise ValueError("New fuel-moisture artifacts must be registered as beta candidates")
    df = pd.read_csv("data/final_training_data.csv")
    df["obs_time"] = pd.to_datetime(df["obs_time"], errors="coerce", utc=True)
    df = df.dropna(subset=["obs_time", "target_fm"]).sort_values(["obs_time", "station_id"])
    features = [name for name in DEFAULT_FEATURES if name in df.columns]
    if not set(LEGACY_FEATURES[:10]).issubset(features):
        raise ValueError("Training data is missing required legacy feature columns")
    model_df = df.dropna(subset=features + ["target_fm"]).copy()
    unique_dates = pd.Series(model_df["obs_time"].dt.floor("D").unique()).sort_values().to_list()
    if len(unique_dates) < 2:
        raise ValueError("At least two distinct dates are required for an out-of-time holdout")
    cutoff_date = unique_dates[max(1, int(len(unique_dates) * 0.8))]
    train = model_df[model_df["obs_time"].dt.floor("D") < cutoff_date]
    holdout = model_df[model_df["obs_time"].dt.floor("D") >= cutoff_date]
    if train.empty or holdout.empty:
        raise ValueError("At least two time-ordered rows are required for train/holdout data")

    params = dict(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.9,
                  colsample_bytree=0.9, objective="reg:squarederror", random_state=42)
    folds = []
    if len(train) >= 20:
        for fold, (train_idx, val_idx) in enumerate(_rolling_date_folds(train), 1):
            candidate = xgb.XGBRegressor(**params).fit(train.iloc[train_idx][features], train.iloc[train_idx]["target_fm"])
            prediction = candidate.predict(train.iloc[val_idx][features])
            folds.append({"fold": fold, **_metrics(train.iloc[val_idx]["target_fm"], prediction),
                          "start": train.iloc[val_idx]["obs_time"].min().isoformat(),
                          "end": train.iloc[val_idx]["obs_time"].max().isoformat()})

    model = xgb.XGBRegressor(**params).fit(train[features], train["target_fm"])
    prediction = model.predict(holdout[features])
    candidate_metrics = _metrics(holdout["target_fm"], prediction)
    baselines = {
        "emc": _metrics(holdout["target_fm"], holdout["emc_baseline"]),
        "climatology": _metrics(holdout["target_fm"], np.repeat(train["target_fm"].mean(), len(holdout))),
    }
    persistence = holdout.groupby("station_id")["target_fm"].shift(1).fillna(train["target_fm"].iloc[-1])
    baselines["persistence"] = _metrics(holdout["target_fm"], persistence)
    danger = _danger_metrics(holdout, holdout["target_fm"].to_numpy(), prediction)
    stable_metrics = None
    stable_danger = None
    try:
        stable = xgb.Booster(); stable.load_model(str(load_active_model_path("fuel_moisture")))
        stable_features = stable.feature_names or features
        stable_prediction = stable.predict(xgb.DMatrix(holdout[stable_features], feature_names=stable_features))
        stable_metrics = _metrics(holdout["target_fm"], stable_prediction)
        stable_danger = _danger_metrics(holdout, holdout["target_fm"].to_numpy(), stable_prediction)
        baselines["current_stable"] = stable_metrics
    except (FileNotFoundError, KeyError, ValueError, xgb.XGBoostError):
        pass

    if any(name.startswith("precip_") or name == "hours_since_rain" for name in features):
        model.get_booster().set_attr(precipitation_contract_version=PRECIPITATION_CONTRACT_VERSION,
                                     precipitation_contract_sha256=PRECIPITATION_CONTRACT_SHA256)
    # Regime = calendar month: same out-of-time holdout used for promotion
    # metrics above, bucketed by "month" (already a feature column) since
    # this flat training frame has no rain-based regime columns to reuse
    # from spatial/v5_guard.py's scheme.
    uncertainty = fit_uncertainty(holdout["target_fm"], prediction, holdout["month"])
    stamp = f"{datetime.now():%Y%m%d_%H%M%S}"
    scratch = Path("models") / f".scratch_fuel_moisture_{stamp}.json"
    scratch_uncertainty = Path("models") / f".scratch_fuel_moisture_{stamp}_uncertainty.json"
    scratch.parent.mkdir(exist_ok=True); model.save_model(scratch)
    scratch_uncertainty.write_text(json.dumps(uncertainty, indent=2))
    match_meta_path = Path("data/training_set_mo_meta.json")
    match_meta = json.loads(match_meta_path.read_text()) if match_meta_path.exists() else {}
    support = {str(key): int(value) for key, value in pd.Series(
        [calculate_fire_danger(fm, rh, meters_per_second_to_knots(ws)) for fm, rh, ws in
         zip(holdout["target_fm"], holdout["rel_humidity"], holdout["wind_speed_ms"])]
    ).value_counts().to_dict().items()}
    metadata = {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_columns": features,
        "feature_ranges": feature_ranges(train, features),
        "imputation_policy": {"hours_since_rain_without_history": 24.0,
                              "other_missing_features": "reject_row"},
        "clipping_policy": {"temp_c": [-60, 60], "rel_humidity": [0, 100],
                            "wind_speed_ms": [0, 75], "precip_mm": [0, 500]},
        "max_feature_age_minutes": 60,
        "rule_spec_version": RULE_SPEC_VERSION,
        "precipitation_contract_version": PRECIPITATION_CONTRACT_VERSION,
        "precipitation_contract_sha256": PRECIPITATION_CONTRACT_SHA256,
        "training_window": {"start": train["obs_time"].min().isoformat(), "end": train["obs_time"].max().isoformat(),
                            "holdout_start": holdout["obs_time"].min().isoformat(), "holdout_end": holdout["obs_time"].max().isoformat()},
        "data_match_policy": match_meta.get("data_match_policy", {"direction": "nearest_prior", "tolerance_minutes": 60}),
        "validation_folds": folds,
        "class_support": support,
        "baselines": baselines,
        "danger_metrics": danger, "stable_danger_metrics": stable_danger,
        "promotion_gates": {
            "holdout_mae_not_worse_than_emc": candidate_metrics["mae"] <= baselines["emc"]["mae"],
            "holdout_bias_not_worse_than_emc": abs(candidate_metrics["bias"]) <= abs(baselines["emc"]["bias"]),
            "holdout_mae_not_worse_than_stable": stable_metrics is not None and candidate_metrics["mae"] <= stable_metrics["mae"],
            "holdout_bias_not_worse_than_stable": stable_metrics is not None and abs(candidate_metrics["bias"]) <= abs(stable_metrics["bias"]),
            "high_impact_false_negatives_within_2pct": stable_danger is not None and (
                danger["high_impact_false_negative_rate"] is not None and
                stable_danger["high_impact_false_negative_rate"] is not None and
                danger["high_impact_false_negative_rate"] <= stable_danger["high_impact_false_negative_rate"] + 0.02
            ),
        },
        "shadow_required": True,
        "shadow": {"passed": False},
    }
    try:
        version = register_trained_model(
            "fuel_moisture", performance=candidate_metrics, bump=bump, channel=channel, metadata=metadata,
            assets={"model": scratch, "uncertainty": scratch_uncertainty},
        )
    finally:
        scratch.unlink(missing_ok=True)
        scratch_uncertainty.unlink(missing_ok=True)
    print(f"Registered fuel_moisture {version} in {channel}; holdout MAE={candidate_metrics['mae']:.3f}")
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the fuel moisture model")
    parser.add_argument("--channel", choices=["beta"], default="beta")
    parser.add_argument("--bump", choices=["major", "minor", "patch"], default="patch")
    args = parser.parse_args(); train_fuel_moisture_model(args.channel, args.bump)
