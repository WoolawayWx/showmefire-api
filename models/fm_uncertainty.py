"""Calibrated empirical-quantile uncertainty for the fuel-moisture XGBoost model.

Same shape as model-training/spatial/v5_guard.py's fit_uncertainty/intervals
(global + per-regime empirical error quantiles, target coverage, minimum-
support fallback to "global") - reused rather than reinvented so serving
code can lean on an identical lookup contract. Unlike V5, fuel-moisture
training and serving both happen inside this repo (api/pipelines/train_model.py
trains, api/forecast/DailyForecast.py serves), so there is no cross-repo
byte-identity constraint here and this module is imported directly by both.

Regime is calendar month (1-12): v5_guard.py's rain-based regimes
(summer_dry/post_rain/active_rain) depend on active_rain_indicator/
post_rain_3h_indicator columns that don't exist in the flat fuel-moisture
training frame - month is already a feature column here and is a
reasonable proxy for seasonal residual spread (green-up/dormant fuels).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def fit_uncertainty(actual, prediction, regime, target: float = 0.8, minimum_rows: int = 300) -> dict:
    """Empirical residual-quantile half-widths, global + per-regime.

    `actual`/`prediction` should come from held-out (never trained-on) rows -
    the out-of-time holdout set already used for promotion metrics in
    api/pipelines/train_model.py, matching v5_guard.py's out-of-fold
    discipline. `regime` values are stringified so the result is plain-JSON
    safe and looks up the same way regardless of the caller's dtype.
    """
    actual = np.asarray(actual, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    regime = np.asarray([str(value) for value in regime])
    if len(actual) != len(prediction) or len(actual) != len(regime):
        raise ValueError("actual, prediction, and regime must be the same length")

    errors = np.abs(actual - prediction)
    result = {
        "target_coverage": target,
        "global": float(np.quantile(errors, target, method="higher")),
        "global_support": int(len(errors)),
        "regimes": {},
    }
    frame = pd.DataFrame({"regime": regime, "error": errors})
    for value, group in frame.groupby("regime"):
        if len(group) >= minimum_rows:
            result["regimes"][value] = {
                "half_width": float(np.quantile(group["error"], target, method="higher")),
                "support": int(len(group)),
            }
    return result


def intervals(prediction, regime, uncertainty: dict) -> np.ndarray:
    """Column-stacked (lo, prediction, hi) - same shape as v5_guard.py::intervals.

    Falls back to the global half-width for any regime with insufficient
    training support (or one never seen during fitting).
    """
    prediction = np.asarray(prediction, dtype=float)
    regime = np.asarray([str(value) for value in np.broadcast_to(regime, prediction.shape)])
    widths = np.asarray([
        uncertainty.get("regimes", {}).get(value, {}).get("half_width", uncertainty["global"])
        for value in regime
    ])
    return np.column_stack((prediction - widths, prediction, prediction + widths))
