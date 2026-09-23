"""
Hourly XGBoost fuel-moisture grids for ensemble members.

Contract-mirror discipline (api/core/contract_mirrors.json, pair
"ensemble_fire_danger_fm_features"): byte-identical to
model-training/ensemble_fire_danger/fm_features.py; imports only
numpy/pandas/stdlib (the xgboost Booster is passed in).

build_fm_frame() reproduces the feature construction inside
api/forecast/DailyForecast.py::predict_fm_grid exactly (same rolling
windows, same precip-history lag, same SWE handling), and
predict_fm_hourly() reproduces DailyForecast.process_forecast_with_
observations' per-hour loop around it (history buffers appended AFTER each
hour's prediction, capped at 24). It deliberately does NOT call
services.model_shadow.run_shadow - the ensemble must never write fuel-
moisture shadow evidence 10-40 times per forecast run. The parity test in
api/tests/test_ensemble_fire_danger.py compares this against
predict_fm_grid on random grids whenever DailyForecast is importable.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

SNOW_THRESHOLD_IN = 0.04
SNOW_FM_FLOOR_PCT = 30.0
_MM_TO_IN = 0.03937


def build_fm_frame(temp_grid: np.ndarray, rh_grid: np.ndarray, ws_grid: np.ndarray, hour: int, month: int,
                   t_hist: Optional[List[np.ndarray]] = None, rh_hist: Optional[List[np.ndarray]] = None,
                   precip_hist: Optional[List[np.ndarray]] = None, swe_grid: Optional[np.ndarray] = None,
                   day_of_year: Optional[int] = None, snow_threshold_in: float = SNOW_THRESHOLD_IN) -> pd.DataFrame:
    shape = temp_grid.shape
    t_hist = t_hist or []
    rh_hist = rh_hist or []
    curr_t_stack = t_hist + [temp_grid]
    curr_rh_stack = rh_hist + [rh_grid]
    rh_flat = rh_grid.flatten()
    day_of_year = day_of_year or int((month - 1) * 365.25 / 12 + 15)

    df = pd.DataFrame({
        "temp_c": temp_grid.flatten(),
        "rel_humidity": rh_flat,
        "wind_speed_ms": ws_grid.flatten(),
        "hour": hour,
        "month": month,
        "emc_baseline": rh_flat / 5.0,
        "temp_mean_3h": np.mean(curr_t_stack[-3:], axis=0).flatten(),
        "rh_mean_3h": np.mean(curr_rh_stack[-3:], axis=0).flatten(),
        "temp_mean_6h": np.mean(curr_t_stack[-6:], axis=0).flatten(),
        "rh_mean_6h": np.mean(curr_rh_stack[-6:], axis=0).flatten(),
    })
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    df["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    zeros = np.zeros(shape).flatten()
    if precip_hist:
        stack = precip_hist
        precip_1h = stack[-1].flatten() if len(stack) >= 1 else zeros.copy()
        precip_3h = np.sum(stack[-3:], axis=0).flatten() if len(stack) >= 3 else zeros.copy()
        precip_6h = np.sum(stack[-6:], axis=0).flatten() if len(stack) >= 6 else zeros.copy()
        precip_24h = np.sum(stack[-24:], axis=0).flatten() if len(stack) >= 24 else zeros.copy()
        hours_since_rain = np.full(shape, 24.0).flatten()
        for h in range(len(stack)):
            rained = stack[-(h + 1)].flatten() > 0.1
            update = rained & (hours_since_rain > h)
            hours_since_rain[update] = h
    else:
        precip_1h, precip_3h, precip_6h, precip_24h = zeros.copy(), zeros.copy(), zeros.copy(), zeros.copy()
        hours_since_rain = np.full(shape, 24).flatten()

    if swe_grid is not None:
        swe_flat = swe_grid.flatten()
        has_snow = swe_flat * _MM_TO_IN > snow_threshold_in
        if np.any(has_snow):
            hours_since_rain[has_snow] = 0
            precip_24h = np.maximum(precip_24h, swe_flat)
            precip_6h = np.maximum(precip_6h, swe_flat)
            precip_3h = np.maximum(precip_3h, swe_flat)
            precip_1h = np.maximum(precip_1h, swe_flat)

    df["precip_1h"] = precip_1h
    df["precip_3h"] = precip_3h
    df["precip_6h"] = precip_6h
    df["precip_24h"] = precip_24h
    df["hours_since_rain"] = hours_since_rain
    return df


def predict_fm_hourly(booster, feature_names: Sequence[str], temp_c: np.ndarray, rh: np.ndarray, ws_ms: np.ndarray,
                      precip_mm: np.ndarray, valid_hours_utc: Sequence, swe_grid: Optional[np.ndarray] = None,
                      snow_threshold_in: float = SNOW_THRESHOLD_IN,
                      snow_fm_floor_pct: float = SNOW_FM_FLOOR_PCT) -> np.ndarray:
    """(hours, y, x) weather -> (hours, y, x) FM %. valid_hours_utc: pandas
    Timestamps (UTC) per hour - hour/month/day-of-year features use them,
    exactly as DailyForecast derives them from base_time + lead."""
    import xgboost as xgb

    t_hist: List[np.ndarray] = []
    rh_hist: List[np.ndarray] = []
    p_hist: List[np.ndarray] = []
    out = []
    shape = temp_c.shape[1:]
    for i, valid in enumerate(valid_hours_utc):
        valid = pd.Timestamp(valid)
        df = build_fm_frame(temp_c[i], rh[i], ws_ms[i], valid.hour, valid.month, t_hist, rh_hist, p_hist,
                            swe_grid=swe_grid, day_of_year=valid.dayofyear, snow_threshold_in=snow_threshold_in)
        preds = booster.predict(xgb.DMatrix(df[list(feature_names)])).reshape(shape)
        if snow_fm_floor_pct > 0 and swe_grid is not None:
            preds = np.where(swe_grid * _MM_TO_IN > snow_threshold_in, np.maximum(preds, snow_fm_floor_pct), preds)
        out.append(preds)
        t_hist.append(temp_c[i])
        rh_hist.append(rh[i])
        p_hist.append(np.nan_to_num(precip_mm[i]))
        if len(t_hist) > 24:
            t_hist.pop(0)
            rh_hist.pop(0)
            p_hist.pop(0)
    return np.stack(out)
