"""
County-day feature aggregation for the fire_risk_fusion model.

Contract-mirror discipline (see core/contract_mirrors.json): byte-identical
to model-training/risk_fusion/features.py. Imports ONLY numpy/pandas/
stdlib - never core.fire_danger, core.precipitation, or anything else
repo-specific, for the same reason rule_uncertainty.py doesn't: two repos,
one shared implementation, no import path between them.

This is why precipitation is NOT decoded here. core/precipitation.py (the
precipitation_contract.json mirror pair) is the one canonical place that
already solves "cumulative vs interval, difference-not-sum across leads" -
reimplementing it a third time here would be exactly the v5_scorer.py
class of duplication risk this project has already been burned by once.
Instead, every function below that needs precipitation takes an
already-decoded `precip_interval_mm` array as a plain argument; the
CALLER (which may freely import core.precipitation) decodes it first.

Everything else that would normally be `spatial.physics`/`spatial.v5_features`
imports is reimplemented locally for the same repo-independence reason -
each such function says so above its definition.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd

FEATURE_MODULE_SCHEMA = "risk-fusion-features-v1"

AFTERNOON_LEAD_HOURS = range(4, 10)   # f04-f09: noon-5pm CST, the leads risk-fusion features are built from
FULL_LEAD_HOURS = range(4, 16)        # f04-f15, matching model-training/spatial/hrrr_capture.DEFAULT_LEAD_HOURS
FIRE_WIND_REDUCTION_FACTOR = 0.8      # matches api/core/config.py's HRRR_FIRE_WIND_REDUCTION_FACTOR default
MPS_TO_KNOTS = 1.9438444924406
KBDI_SPINUP_DAYS = 90
GDD_BASE_TEMP_C = 10.0
GDD_SEASON_START = (3, 1)  # March 1


# --- Calendar features: pure stdlib/pandas, no reimplementation risk ---

def calendar_features(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """valid_doy_sin/cos, dow, is_weekend, is_holiday_us for each date."""
    from pandas.tseries.holiday import USFederalHolidayCalendar

    dates = pd.DatetimeIndex(dates)
    doy = dates.dayofyear.to_numpy(dtype="float64")
    days_in_year = np.where(dates.is_leap_year, 366.0, 365.0)
    angle = 2 * np.pi * doy / days_in_year

    holidays = set(USFederalHolidayCalendar().holidays(start=dates.min(), end=dates.max()))
    normalized = pd.DatetimeIndex(dates).normalize()

    dow = np.asarray(dates.dayofweek)
    return pd.DataFrame({
        "valid_doy_sin": np.sin(angle),
        "valid_doy_cos": np.cos(angle),
        "dow": dow,
        "is_weekend": dow >= 5,
        "is_holiday_us": np.asarray(normalized.isin(holidays)),
    }, index=dates)


# --- KBDI: reimplemented here (physics.py is spatial-only) ---

def keetch_byram_drought_index(
    daily_max_temp_c: np.ndarray,
    daily_precip_mm: np.ndarray,
    mean_annual_precip_mm: float,
    spinup_days: int = KBDI_SPINUP_DAYS,
) -> Dict[str, np.ndarray]:
    """
    Classic Keetch-Byram Drought Index (Keetch & Byram, 1968), computed
    from daily max temperature and daily precipitation. Needs
    `spinup_days` of antecedent daily data before day 0's value is
    trustworthy - the first `spinup_days` entries are marked invalid.

    mean_annual_precip_mm is a REQUIRED parameter, not a hardcoded
    per-county climate normal this module has no verified source for -
    callers must supply a real climatological value (e.g. from a NOAA
    normals product) or KBDI is not meaningful. Passing an unverified
    guess here would silently corrupt the one cheap drought proxy this
    project has - see the module docstring on precipitation for the same
    principle applied to a different risk.

    Returns {"kbdi": array, "valid": bool array}. KBDI units: mm of
    soil moisture deficit, bounded [0, 203.2] (0-8 inches, the standard
    KBDI range).
    """
    daily_max_temp_c = np.asarray(daily_max_temp_c, dtype="float64")
    daily_precip_mm = np.asarray(daily_precip_mm, dtype="float64")
    n = len(daily_max_temp_c)
    if len(daily_precip_mm) != n:
        raise ValueError("daily_max_temp_c and daily_precip_mm must be the same length")
    if mean_annual_precip_mm <= 0:
        raise ValueError("mean_annual_precip_mm must be positive")

    kbdi = np.zeros(n, dtype="float64")
    kbdi_prev = 0.0
    for i in range(n):
        temp_f = daily_max_temp_c[i] * 9.0 / 5.0 + 32.0
        precip_in = daily_precip_mm[i] / 25.4
        annual_precip_in = mean_annual_precip_mm / 25.4

        net_rain_in = max(0.0, precip_in - 0.2)
        kbdi_after_rain = max(0.0, kbdi_prev - net_rain_in * 100.0)

        if temp_f > 50.0:
            # Keetch & Byram (1968) potential-evapotranspiration term, in the
            # same 0.01-inch (hundredths) units as kbdi_after_rain - no
            # further unit conversion belongs in this line.
            et = (
                (800.0 - kbdi_after_rain) * (0.968 * np.exp(0.0486 * temp_f) - 8.30) * 1e-3
            ) / (1.0 + 10.88 * np.exp(-0.0441 * annual_precip_in))
            et = max(0.0, et)
        else:
            et = 0.0

        kbdi_today = np.clip(kbdi_after_rain + et, 0.0, 800.0)
        kbdi[i] = kbdi_today
        kbdi_prev = kbdi_today

    # Convert from the traditional 0-800 (hundredths of an inch) scale to mm.
    kbdi_mm = kbdi * 0.254
    valid = np.arange(n) >= spinup_days
    return {"kbdi": kbdi_mm, "valid": valid}


# --- Growing degree days: pure arithmetic ---

def growing_degree_days(daily_mean_temp_c: np.ndarray, dates: pd.DatetimeIndex,
                        base_temp_c: float = GDD_BASE_TEMP_C) -> np.ndarray:
    """
    Accumulated growing degree days since March 1 of the same year for
    each date, using daily mean temperature. Resets at each new season
    start (March 1); dates before March 1 in their year accumulate from
    the PRECEDING March 1 (i.e. late-winter dates carry the prior
    season's tail, which is intentional - GDD accumulation for fire risk
    cares about "how much has greened up so far this dormant-to-summer
    transition", not the calendar year boundary).
    """
    dates = pd.DatetimeIndex(dates)
    daily_mean_temp_c = np.asarray(daily_mean_temp_c, dtype="float64")
    if len(daily_mean_temp_c) != len(dates):
        raise ValueError("daily_mean_temp_c and dates must be the same length")

    daily_gdd = np.maximum(0.0, daily_mean_temp_c - base_temp_c)

    def _season_start(date: pd.Timestamp) -> pd.Timestamp:
        month, day = GDD_SEASON_START
        this_year_start = pd.Timestamp(year=date.year, month=month, day=day, tz=date.tz)
        return this_year_start if date >= this_year_start else pd.Timestamp(year=date.year - 1, month=month, day=day, tz=date.tz)

    result = np.zeros(len(dates), dtype="float64")
    order = np.argsort(dates.values)
    running_total = 0.0
    running_season_start = None
    for idx in order:
        season_start = _season_start(dates[idx])
        if running_season_start is None or season_start != running_season_start:
            running_total = 0.0
            running_season_start = season_start
        running_total += daily_gdd[idx]
        result[idx] = running_total
    return result


# --- HRRR cell -> county reduction ---

def reduce_cells_to_county(
    values_grid: np.ndarray,
    cell_to_fips: Dict[str, str],
    reducer=np.nanmean,
) -> Dict[str, float]:
    """
    values_grid: 2D array shaped like the HRRR grid (y, x).
    cell_to_fips: {"y,x": fips} from scripts/build_county_cells.py's county_cells.json.
    Returns {fips: reduced_value} using `reducer` over every cell mapped to that county.
    """
    by_county: Dict[str, list] = {}
    for key, fips in cell_to_fips.items():
        y, x = (int(part) for part in key.split(","))
        value = values_grid[y, x]
        by_county.setdefault(fips, []).append(value)
    return {fips: float(reducer(np.asarray(values))) for fips, values in by_county.items()}


def vapor_pressure_deficit_kpa(temp_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """Standard Magnus-approximation VPD, reimplemented locally (spatial.physics is spatial-only)."""
    temp_c = np.asarray(temp_c, dtype="float64")
    rh = np.asarray(rh, dtype="float64")
    saturation_kpa = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
    actual_kpa = saturation_kpa * np.clip(rh, 0.0, 100.0) / 100.0
    return np.maximum(0.0, saturation_kpa - actual_kpa)


def build_county_day_row(
    county_fips: str,
    afternoon_stats: Dict[str, float],
    full_day_stats: Dict[str, float],
    rule_features: Dict[str, float],
    calendar_row: Dict,
    static_row: Dict,
    kbdi_value: Optional[float],
    kbdi_valid: bool,
    gdd_value: Optional[float],
    gust_available: bool,
) -> Dict:
    """Assembles one county-day feature row from already-computed pieces. No aggregation logic here."""
    row = {
        "county_fips": county_fips,
        "fm_min_afternoon": afternoon_stats.get("fm_min"),
        "fm_mean": full_day_stats.get("fm_mean"),
        "rh_min_afternoon": afternoon_stats.get("rh_min"),
        "rh_mean": full_day_stats.get("rh_mean"),
        "temp_max_c": full_day_stats.get("temp_max"),
        "wind_kts_max": full_day_stats.get("wind_kts_max"),
        "wind_kts_p90": full_day_stats.get("wind_kts_p90"),
        "vpd_kpa_max": full_day_stats.get("vpd_kpa_max"),
        "precip_24h_mm": full_day_stats.get("precip_24h_mm"),
        "swe_mm_mean": full_day_stats.get("swe_mm_mean"),
        "snow_fraction": full_day_stats.get("snow_fraction"),
        "gust_available": gust_available,
        "gust_kts_max": full_day_stats.get("gust_kts_max") if gust_available else None,
        "kbdi": kbdi_value,
        "kbdi_valid": kbdi_valid,
        "gdd_accum_since_mar1": gdd_value,
        "burnable_area_km2": static_row.get("burnable_area_km2"),
        "burnable_fraction_source": static_row.get("burnable_fraction_source"),
        "region_id": static_row.get("region_id"),
        "region_method": static_row.get("region_method"),
    }
    row.update(calendar_row)
    row.update(rule_features)
    return row
