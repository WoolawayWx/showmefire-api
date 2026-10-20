"""Fire danger category (0 Low .. 4 Extreme) from HRRR fields alone: the canonical rule, vectorised.

The same rule the public forecast uses (core.fire_danger.calculate_fire_danger, thresholds from RULE_SPEC), but fed the
hourly product's own estimated 1-hour fuel moisture (3 + 0.25 x RH, see derive.py) instead of the RAWS-based fuel moisture
the daily forecast starts from. So it can differ from the official forecast, and is labelled an estimate wherever it is shown.
It leaves out the daily forecast's snow mask, smoothing and green-up adjustment.
"""
from __future__ import annotations

import numpy as np

from core.fire_danger import RULE_SPEC

MPH_TO_KT = 0.86897624190065


def fire_danger_category(fuel_moisture, rh, wind_kt) -> np.ndarray:
    """Category ids as float32, NaN where any input is missing. Same branch order as the scalar rule."""
    fm, rh, wind = (np.asarray(v, dtype="float64") for v in (fuel_moisture, rh, wind_kt))
    t = RULE_SPEC["thresholds"]
    valid = np.isfinite(fm) & np.isfinite(rh) & np.isfinite(wind)
    with np.errstate(invalid="ignore"):
        dry = fm < t["low_fm"]                      # at or above this fuel moisture it is Low whatever else holds
        moderate = dry & ((rh < t["moderate_rh"]) | (wind >= t["moderate_wind"]))
        elevated = (fm < t["elevated_fm"]) & (((rh < t["elevated_rh"]) & (wind >= t["elevated_wind"]))
                                             | ((rh < t["elevated_very_dry_rh"]) & (wind >= t["elevated_very_dry_wind"])))
        critical = (fm < t["elevated_fm"]) & (rh < t["critical_rh"]) & (wind >= t["critical_wind"])
        extreme = (fm < t["extreme_fm"]) & (rh < t["extreme_rh"]) & (wind >= t["extreme_wind"])
    category = np.zeros(fm.shape, dtype="float32")
    for level, condition in enumerate((moderate, elevated, critical, extreme), start=1):
        category[condition & dry] = level           # a later, stricter condition overrides an earlier one
    return np.where(valid, category, np.nan).astype("float32")
