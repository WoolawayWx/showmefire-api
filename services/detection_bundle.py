"""Pure-python scoring of the v2 logistic-regression bundle (JSON).

No numpy, sklearn or database imports on purpose: the training package loads this very file to prove the served score
equals the fitted model's score, and the API needs no pickle. A bundle holds the feature order, which features get
log1p, training medians for missing values, missing-indicator features, scaler stats, coefficients, the intercept, an
optional sigmoid calibrator over the model's logit, and the sensors it was trained on.
"""
from __future__ import annotations

import math

BUNDLE_SCHEMA = "veg-conf-model-v1"


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))


def _missing(value) -> bool:
    return value is None or (isinstance(value, float) and math.isnan(value))


def design_row(bundle: dict, features: dict) -> list[float]:
    """The standardised input vector the coefficients apply to: log1p where listed, training median for missing values,
    then one missing-indicator per feature that had missing values in training."""
    log1p = set(bundle.get("log1p", []))
    row: list[float] = []
    for name in bundle["features"]:
        raw = features.get(name)
        if _missing(raw):
            value = bundle["impute"][name]
        else:
            value = math.log1p(max(float(raw), 0.0)) if name in log1p else float(raw)
        row.append(value)
    for name in bundle.get("missing_indicators", []):
        row.append(1.0 if _missing(features.get(name)) else 0.0)
    return row


def logit(bundle: dict, features: dict) -> float:
    z = float(bundle["intercept"])
    for x, mu, sd, coef in zip(design_row(bundle, features), bundle["scaler"]["mean"], bundle["scaler"]["scale"], bundle["coef"]):
        z += coef * ((x - mu) / (sd or 1.0))
    return z


def score_bundle(bundle: dict, features: dict) -> tuple[float, bool]:
    """(probability, calibrated)."""
    z = logit(bundle, features)
    cal = bundle.get("calibrator")
    if cal:
        return _sigmoid(cal["a"] * z + cal["b"]), True
    return _sigmoid(z), False
