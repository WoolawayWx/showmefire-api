"""Live-derived explainability for the fuel-moisture XGBoost model.

Loads its OWN copy of the serving stable model rather than importing
forecast/DailyForecast.py - that module is a heavy, script-shaped file
(cartopy/geopandas/herbie imports, module-level logging/font setup) meant
to run standalone via forecasts.sh, not to be imported into the always-on
API server process. api/main.py needs GET /api/models/formulas to work
without paying that startup cost.

Uses XGBoost's own native contribution/importance output (get_score,
pred_contribs=True - TreeSHAP under the hood) rather than adding a `shap`
dependency - both are already part of the pinned xgboost package.
"""
from __future__ import annotations

import logging
from typing import Dict

import xgboost as xgb

from models.features import LEGACY_FEATURES
from models.versioning import load_active_model_path

logger = logging.getLogger(__name__)

# Loaded lazily/tolerantly: api/main.py imports this module at server
# startup (to serve GET /api/models/formulas), so a missing/invalid stable
# model must never prevent the whole process from starting - every other
# shadow/diagnostics module in this codebase (model_shadow.py, v5_shadow.py,
# risk_fusion_glm_shadow.py) degrades the same way instead of raising at
# import time.
try:
    _FM_MODEL = xgb.Booster()
    _FM_MODEL.load_model(str(load_active_model_path("fuel_moisture", auto_rollback=True)))
    # The exact features the currently-serving stable model expects - same
    # fallback convention as forecast/DailyForecast.py's FEATURES.
    FEATURES = list(_FM_MODEL.feature_names or LEGACY_FEATURES)
except Exception as error:
    logger.error(f"fm_explain could not load the stable fuel_moisture model: {error}")
    _FM_MODEL = None
    FEATURES = list(LEGACY_FEATURES)

_HUMAN_NAMES = {
    "temp_c": "Temperature (C)",
    "rel_humidity": "Relative Humidity (%)",
    "wind_speed_ms": "Wind Speed (m/s)",
    "hour": "Hour of Day",
    "month": "Month",
    "emc_baseline": "Equilibrium Moisture Baseline (RH / 5)",
    "temp_mean_3h": "3-Hour Mean Temperature",
    "rh_mean_3h": "3-Hour Mean Relative Humidity",
    "temp_mean_6h": "6-Hour Mean Temperature",
    "rh_mean_6h": "6-Hour Mean Relative Humidity",
    "precip_1h": "1-Hour Precipitation",
    "precip_3h": "3-Hour Precipitation",
    "precip_6h": "6-Hour Precipitation",
    "precip_24h": "24-Hour Precipitation",
    "hours_since_rain": "Hours Since Last Rain",
    "hour_sin": "Hour (cyclical sin)",
    "hour_cos": "Hour (cyclical cos)",
    "day_of_year_sin": "Day of Year (cyclical sin)",
    "day_of_year_cos": "Day of Year (cyclical cos)",
}


def global_importance() -> Dict[str, float]:
    """Gain-based feature importance for every serving feature (0.0 if XGBoost never split on it)."""
    if _FM_MODEL is None:
        raise RuntimeError("No stable fuel_moisture model is registered - nothing to explain")
    raw = _FM_MODEL.get_score(importance_type="gain")
    return {_HUMAN_NAMES.get(name, name): float(raw.get(name, 0.0)) for name in FEATURES}


def explain_prediction(feature_row: Dict[str, float]) -> Dict:
    """Per-prediction TreeSHAP contribution breakdown for a single feature row.

    `feature_row` must contain every column in FEATURES (extra keys are
    ignored). Returns {"prediction", "base_value", "contributions"} -
    contributions sum to prediction - base_value within floating-point
    tolerance, a property of TreeSHAP/pred_contribs (see
    api/tests/test_fm_explain.py).
    """
    if _FM_MODEL is None:
        raise RuntimeError("No stable fuel_moisture model is registered - nothing to explain")
    missing = [name for name in FEATURES if name not in feature_row]
    if missing:
        raise ValueError(f"feature_row is missing required columns: {missing}")
    row = [[float(feature_row[name]) for name in FEATURES]]
    dmat = xgb.DMatrix(row, feature_names=FEATURES)
    contribs = _FM_MODEL.predict(dmat, pred_contribs=True)[0]
    base_value = float(contribs[-1])
    contributions = {_HUMAN_NAMES.get(name, name): float(value) for name, value in zip(FEATURES, contribs[:-1])}
    prediction = base_value + sum(contributions.values())
    return {"prediction": prediction, "base_value": base_value, "contributions": contributions}
