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


# --- Seasonal live-fuel adjustment ------------------------------------------
#
# calculate_fire_danger above is the canonical, UNTOUCHED rule and must stay
# byte-identical to model-training/spatial/rule_contract.py and every
# contract-mirror consumer (services/rule_uncertainty.py's check_parity
# depends on this). The adjustment below is a deliberately separate, opt-in
# post-processing step: dead-fuel-stick dryness (the only fuel-moisture
# signal anywhere in this project) has no live/herbaceous-fuel term, so a
# hot, low-RH day in green-season reads exactly as dry as the same day would
# in cured-season.
#
# A flat "cap everything at Elevated" was tried first and rejected: on a
# real spring green-up date it moved every suppressed Critical cell into the
# SAME bucket as cells that were only ever marginally Elevated, inflating
# the reported Elevated footprint by 61% relative (3.72% -> 6.00% of the
# grid on 2026-04-15) even though nothing in that expanded area was
# genuinely borderline. This version instead demotes ONE tier (Elevated,
# Critical, or Extreme) at a time, and only for a cell that is merely
# marginal within its own tier - a cell deep into a tier's severity (close
# to also qualifying for the tier above) keeps its real category regardless
# of season, since a genuinely severe fm/rh/wind combination is real even
# during green-up. It still never raises a category, and callers who don't
# opt in (e.g. rule_uncertainty.py, training evaluation) are unaffected
# because calculate_fire_danger itself never changes.
#
# GDD-only for now, not KBDI: KBDI needs a verified per-county mean-annual-
# precipitation climate normal, and none exists yet anywhere in this repo
# (see model-training/risk_fusion/build_county_days.py's own documented
# gap - it uses an explicitly-labeled non-normal proxy for the same reason).
# kbdi_mm is accepted here so a real drought signal can override the
# green-season assumption once that data is sourced, without another
# signature change.
GREEN_SEASON_GDD_CEILING_C = 200.0  # below this, vegetation is presumed fully green/uncured
CURED_SEASON_GDD_FLOOR_C = 1200.0  # above this, curing is presumed complete; adjustment lifts
DROUGHT_OVERRIDE_KBDI_MM = 100.0  # confirmed drought stress overrides the green-season assumption
# Even at peak green-up (green_factor=1.0), only the more marginal half of
# a tier's severity range is eligible for demotion - mapping green_factor
# directly onto the fraction scale demoted an entire real spring tier
# (2026-04-15: all 2,715 base-Elevated cells, none survived), which is as
# distorting as the flat cap it replaced. This ceiling keeps the deepest
# half of every tier's severity intact regardless of season.
MAX_DEMOTION_FRACTION = 0.5


def _clip01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def _green_factor(gdd_accum_since_mar1: Optional[float], kbdi_mm: Optional[float] = None) -> float:
    """1.0 = deepest presumed green-up, 0.0 = curing presumed complete or no signal."""
    if gdd_accum_since_mar1 is None:
        return 0.0
    try:
        gdd = float(gdd_accum_since_mar1)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(gdd):
        return 0.0
    if kbdi_mm is not None:
        try:
            if math.isfinite(float(kbdi_mm)) and float(kbdi_mm) >= DROUGHT_OVERRIDE_KBDI_MM:
                return 0.0
        except (TypeError, ValueError):
            pass
    if gdd <= GREEN_SEASON_GDD_CEILING_C:
        return 1.0
    if gdd >= CURED_SEASON_GDD_FLOOR_C:
        return 0.0
    return (CURED_SEASON_GDD_FLOOR_C - gdd) / (CURED_SEASON_GDD_FLOOR_C - GREEN_SEASON_GDD_CEILING_C)


def _severity_fraction(category: int, fm: float, rh: float, wind_kts: float, thresholds: dict) -> float:
    """0.0 = a cell that only just crossed into `category`, 1.0 = deep enough to also qualify one tier higher.

    Mirrors the within-category gradient concept from the quarantined
    forecast/firedangermodel.py (rh/wind/fm factors averaged), but computed
    from the live canonical thresholds rather than a hand-tuned 0-100 score.
    """
    if category == int(FireDangerCategory.ELEVATED):
        wind_floor = min(thresholds["elevated_wind"], thresholds["elevated_very_dry_wind"])
        rh_component = _clip01((thresholds["elevated_rh"] - rh) / (thresholds["elevated_rh"] - thresholds["critical_rh"]))
        wind_component = _clip01((wind_kts - wind_floor) / (thresholds["critical_wind"] - wind_floor))
        fm_component = _clip01((thresholds["elevated_fm"] - fm) / thresholds["elevated_fm"])
    elif category == int(FireDangerCategory.CRITICAL):
        rh_component = _clip01((thresholds["critical_rh"] - rh) / (thresholds["critical_rh"] - thresholds["extreme_rh"]))
        wind_component = _clip01((wind_kts - thresholds["critical_wind"]) / (thresholds["extreme_wind"] - thresholds["critical_wind"]))
        fm_component = _clip01((thresholds["elevated_fm"] - fm) / (thresholds["elevated_fm"] - thresholds["extreme_fm"]))
    elif category == int(FireDangerCategory.EXTREME):
        # No defined tier above Extreme - use a fixed, generously-sized
        # "how much further past the threshold" reference instead of a
        # next-tier boundary.
        rh_component = _clip01((thresholds["extreme_rh"] - rh) / thresholds["extreme_rh"])
        wind_component = _clip01((wind_kts - thresholds["extreme_wind"]) / 15.0)
        fm_component = _clip01((thresholds["extreme_fm"] - fm) / thresholds["extreme_fm"])
    else:
        return 1.0  # Low/Moderate are never eligible for this adjustment.
    return (rh_component + wind_component + fm_component) / 3.0


def seasonal_dampening_adjustment(
    category: Optional[int],
    fm: float,
    relative_humidity: float,
    wind_speed_knots: float,
    gdd_accum_since_mar1: Optional[float],
    kbdi_mm: Optional[float] = None,
) -> Optional[int]:
    """Softens Elevated/Critical/Extreme by one tier for cells that are only marginal within their tier.

    Only demotes when BOTH conditions hold: green-up is presumed
    incomplete (see _green_factor) AND the cell is not deep into its
    category's own severity (see _severity_fraction) - a cell close to
    also qualifying for the tier above keeps its real category regardless
    of season, since that combination of fm/rh/wind is genuinely severe
    weather. Never raises a category, and returns the input unchanged for
    Low/Moderate or when category is None.

    The GDD thresholds and the severity-fraction weighting are a first-pass
    estimate, not a calibrated agronomic boundary - validate against real
    fire-occurrence outcomes (model-training/risk_fusion/labels.py's
    FPA-FOD panel) before trusting them operationally.
    """
    if category is None or category < int(FireDangerCategory.ELEVATED):
        return category
    green = _green_factor(gdd_accum_since_mar1, kbdi_mm)
    if green <= 0.0:
        return category
    try:
        fraction = _severity_fraction(category, float(fm), float(relative_humidity), float(wind_speed_knots), RULE_SPEC["thresholds"])
    except (TypeError, ValueError):
        return category
    return category - 1 if fraction < green * MAX_DEMOTION_FRACTION else category
