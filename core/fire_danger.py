"""Canonical Show Me Fire danger categories and unit conversions.

This module is the single operational definition used by maps, forecasts,
training evaluation, and shadow comparisons.  Public category ids are stable.
"""
from __future__ import annotations

from enum import IntEnum
import hashlib
import json
import math
from pathlib import Path
from typing import Optional
from collections import Counter


RULE_SPEC_PATH = Path(__file__).with_name("fire_danger_rules.json")
RULE_SPEC = json.loads(RULE_SPEC_PATH.read_text(encoding="utf-8"))
RULE_SPEC_VERSION = RULE_SPEC["version"]
RULE_SPEC_SHA256 = hashlib.sha256(RULE_SPEC_PATH.read_bytes()).hexdigest()
MPS_TO_KNOTS = 1.9438444924406
MPH_TO_KNOTS = 0.86897624190065


class FireDangerCategory(IntEnum):
    LOW = 0
    MODERATE = 1
    ELEVATED = 2
    CRITICAL = 3
    EXTREME = 4


CATEGORY_LABELS = tuple(RULE_SPEC["categories"])
CATEGORY_IDS = tuple(int(category) for category in FireDangerCategory)
HIGH_IMPACT_CATEGORY_IDS = (
    int(FireDangerCategory.ELEVATED),
    int(FireDangerCategory.CRITICAL),
    int(FireDangerCategory.EXTREME),
)
_MISSING_COUNTERS = Counter()


def missing_input_diagnostics() -> dict:
    return dict(_MISSING_COUNTERS)


def reset_missing_input_diagnostics() -> None:
    _MISSING_COUNTERS.clear()


def meters_per_second_to_knots(value):
    return value * MPS_TO_KNOTS


def miles_per_hour_to_knots(value):
    return value * MPH_TO_KNOTS


def category_label(category: Optional[int]) -> Optional[str]:
    return None if category is None else CATEGORY_LABELS[int(category)]


def calculate_fire_danger(
    fuel_moisture: float,
    relative_humidity: float,
    wind_speed_knots: float,
    *,
    missing_category: Optional[int] = None,
) -> Optional[int]:
    """Return the authoritative public category id.

    Missing inputs are explicitly unavailable by default. Operational callers
    must choose a fallback rather than silently treating missing data as Low.
    """
    values = (fuel_moisture, relative_humidity, wind_speed_knots)
    try:
        missing = any(value is None or not math.isfinite(float(value)) for value in values)
    except (TypeError, ValueError):
        missing = True
    if missing:
        names = ("fuel_moisture", "relative_humidity", "wind_speed_knots")
        for name, value in zip(names, values):
            try:
                if value is None or not math.isfinite(float(value)): _MISSING_COUNTERS[name] += 1
            except (TypeError, ValueError):
                _MISSING_COUNTERS[name] += 1
        _MISSING_COUNTERS["unavailable_outputs"] += int(missing_category is None)
        _MISSING_COUNTERS["explicit_fallback_outputs"] += int(missing_category is not None)
        return missing_category

    fm, rh, wind = map(float, values)
    thresholds = RULE_SPEC["thresholds"]
    if fm >= thresholds["low_fm"]:
        return int(FireDangerCategory.LOW)
    if fm < thresholds["extreme_fm"] and rh < thresholds["extreme_rh"] and wind >= thresholds["extreme_wind"]:
        return int(FireDangerCategory.EXTREME)
    if fm < thresholds["elevated_fm"] and rh < thresholds["critical_rh"] and wind >= thresholds["critical_wind"]:
        return int(FireDangerCategory.CRITICAL)
    if fm < thresholds["elevated_fm"] and (
        (rh < thresholds["elevated_rh"] and wind >= thresholds["elevated_wind"])
        or (rh < thresholds["elevated_very_dry_rh"] and wind >= thresholds["elevated_very_dry_wind"])
    ):
        return int(FireDangerCategory.ELEVATED)
    if fm < thresholds["low_fm"] and (rh < thresholds["moderate_rh"] or wind >= thresholds["moderate_wind"]):
        return int(FireDangerCategory.MODERATE)
    return int(FireDangerCategory.LOW)
