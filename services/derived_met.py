"""Pure meteorological derived-metric formulas - no I/O, numpy-array-friendly.

Backs the "current conditions" map catalog (api/maps/*.py): heat index, wind
chill, the NWS "feels like" composite, wet-bulb temperature, an estimated
WBGT, theta-e, and dewpoint depression. None of these existed anywhere in
this codebase before - every function here is a standalone, independently
testable formula with no file/network access, so it can be unit tested
without a database, a station feed, or a rendered map.

Two outputs are explicitly estimates rather than true measurements, because
Missouri's RAWS/ASOS/Mesonet/CWOP network (via Synoptic) has no sensor that
can measure them directly:
  - WBGT needs a wet globe thermometer; `estimate_wbgt_f` approximates it from
    temperature/humidity/wind(/solar) instead.
  - Theta-e needs station pressure, which not every station reports;
    `theta_e_k` accepts a pressure estimated from elevation
    (`standard_atmosphere_pressure_hpa`) as a fallback.
Any map/product built on these two must say "Estimated" in its title/legend/
status, the same convention services/spread_rate.py uses for "Experimental".
"""
from __future__ import annotations

import numpy as np

# --- Unit conversions -------------------------------------------------------


def c_to_f(temp_c):
    return np.asarray(temp_c, dtype=float) * 9.0 / 5.0 + 32.0


def f_to_c(temp_f):
    return (np.asarray(temp_f, dtype=float) - 32.0) * 5.0 / 9.0


def mph_to_ms(speed_mph):
    return np.asarray(speed_mph, dtype=float) * 0.44704


def ms_to_mph(speed_ms):
    return np.asarray(speed_ms, dtype=float) / 0.44704


# --- Heat index / wind chill / feels-like -----------------------------------

HEAT_INDEX_MIN_TEMP_F = 80.0
WIND_CHILL_MAX_TEMP_F = 50.0
WIND_CHILL_MIN_WIND_MPH = 3.0


def heat_index_f(temp_f, rh_pct):
    """NWS Rothfusz regression heat index, in degrees F.

    Only meaningful at/above 80F (NWS's documented valid range) - callers
    should gate on HEAT_INDEX_MIN_TEMP_F themselves (see feels_like_f) rather
    than trust this function's output below that threshold.
    """
    t = np.asarray(temp_f, dtype=float)
    r = np.asarray(rh_pct, dtype=float)

    simple = 0.5 * (t + 61.0 + (t - 68.0) * 1.2 + r * 0.094)
    hi = -42.379 + 2.04901523 * t + 10.14333127 * r - 0.22475541 * t * r \
        - 0.00683783 * t * t - 0.05481717 * r * r + 0.00122874 * t * t * r \
        + 0.00085282 * t * r * r - 0.00000199 * t * t * r * r

    # NWS adjustment terms for the regression's edge cases. Clip the sqrt
    # argument before the mask is applied - np.where evaluates both branches
    # eagerly, and outside 80-112F the radicand can go negative.
    low_rh_adjustment = ((13.0 - r) / 4.0) * np.sqrt(np.clip((17.0 - np.abs(t - 95.0)) / 17.0, 0.0, None))
    apply_low_rh = (r < 13.0) & (t >= 80.0) & (t <= 112.0)
    hi = np.where(apply_low_rh, hi - low_rh_adjustment, hi)

    high_rh_adjustment = ((r - 85.0) / 10.0) * ((87.0 - t) / 5.0)
    apply_high_rh = (r > 85.0) & (t >= 80.0) & (t <= 87.0)
    hi = np.where(apply_high_rh, hi + high_rh_adjustment, hi)

    result = np.where((t + r) < 160.0, simple, hi)
    return result if np.ndim(result) else float(result)


def wind_chill_f(temp_f, wind_mph):
    """NWS wind chill formula, in degrees F.

    Only meaningful at/below 50F with wind >= 3mph (NWS's documented valid
    range) - callers should gate on WIND_CHILL_MAX_TEMP_F/
    WIND_CHILL_MIN_WIND_MPH themselves (see feels_like_f).
    """
    t = np.asarray(temp_f, dtype=float)
    v = np.asarray(wind_mph, dtype=float)
    v_pow = np.power(np.clip(v, 0.0, None), 0.16)
    wc = 35.74 + 0.6215 * t - 35.75 * v_pow + 0.4275 * t * v_pow
    return wc if np.ndim(wc) else float(wc)


def feels_like_f(temp_f, rh_pct, wind_mph):
    """NWS-convention composite "feels like" temperature: heat index when
    hot, wind chill when cold, plain air temperature in between."""
    t = np.asarray(temp_f, dtype=float)
    r = np.asarray(rh_pct, dtype=float)
    v = np.asarray(wind_mph, dtype=float)

    hot = t >= HEAT_INDEX_MIN_TEMP_F
    cold = (t <= WIND_CHILL_MAX_TEMP_F) & (v >= WIND_CHILL_MIN_WIND_MPH)

    result = np.where(hot, heat_index_f(t, r), np.where(cold, wind_chill_f(t, v), t))
    return result if np.ndim(result) else float(result)


# --- Dewpoint-derived metrics ------------------------------------------------


def dewpoint_depression_f(temp_f, dewpoint_f):
    """Air temperature minus dewpoint, in degrees F - a dryness/lifting-
    condensation-level proxy. Larger values mean drier air."""
    t = np.asarray(temp_f, dtype=float)
    td = np.asarray(dewpoint_f, dtype=float)
    result = t - td
    return result if np.ndim(result) else float(result)


# Stull (2011), "Wet-Bulb Temperature from Relative Humidity and Air
# Temperature", J. Applied Meteorology and Climatology. Documented valid
# range: temperature -20 to 50C, RH 5 to 99%; degrades outside that range.
WET_BULB_VALID_TEMP_C = (-20.0, 50.0)
WET_BULB_VALID_RH_PCT = (5.0, 99.0)


def wet_bulb_f(temp_f, rh_pct):
    """Wet-bulb temperature via Stull's (2011) empirical approximation, in
    degrees F. Accuracy degrades outside WET_BULB_VALID_TEMP_C/
    WET_BULB_VALID_RH_PCT - callers working near those bounds should treat
    the result as approximate."""
    t_c = f_to_c(temp_f)
    r = np.asarray(rh_pct, dtype=float)

    tw_c = (
        t_c * np.arctan(0.151977 * np.sqrt(r + 8.313659))
        + np.arctan(t_c + r)
        - np.arctan(r - 1.676331)
        + 0.00391838 * np.power(r, 1.5) * np.arctan(0.023101 * r)
        - 4.686035
    )
    result = c_to_f(tw_c)
    return result if np.ndim(result) else float(result)


def saturation_vapor_pressure_hpa(temp_c):
    """Tetens' formula, hPa."""
    t = np.asarray(temp_c, dtype=float)
    return 6.1078 * np.power(10.0, (7.5 * t) / (t + 237.3))


# Kept as a module-level alias since theta_e_k/estimate_wbgt_f were written
# against the old private name.
_saturation_vapor_pressure_hpa = saturation_vapor_pressure_hpa


def specific_humidity_g_per_kg(dewpoint_f, pressure_hpa):
    """Specific humidity (g/kg) from dewpoint and pressure - used by
    maps/moisture_convergence.py's flux divergence calculation.
    `pressure_hpa` may come from standard_atmosphere_pressure_hpa() as a
    fallback when true station pressure is unavailable (same caveat as
    theta_e_k)."""
    dewpoint_c = f_to_c(dewpoint_f)
    p = np.asarray(pressure_hpa, dtype=float)
    vapor_pressure_hpa = saturation_vapor_pressure_hpa(dewpoint_c)
    mixing_ratio = 0.622 * vapor_pressure_hpa / np.clip(p - vapor_pressure_hpa, 1e-6, None)
    specific_humidity = mixing_ratio / (1.0 + mixing_ratio)
    result = specific_humidity * 1000.0
    return result if np.ndim(result) else float(result)


def estimate_wbgt_f(temp_f, rh_pct, wind_mph, solar_radiation=None):
    """Estimated (not measured) Wet Bulb Globe Temperature, in degrees F.

    Missouri's stations have no wet-globe thermometer, so this is an
    outdoor-estimate approximation (Australian Bureau of Meteorology style:
    WBGT ~= 0.567*T + 0.393*e + 3.94, where e is vapor pressure in hPa from
    actual RH/temperature) rather than a true measurement. `solar_radiation`
    (W/m^2) is accepted for future refinement but not currently used, since
    not every station network reports it reliably; wind is likewise accepted
    for interface symmetry with the other "feels like" functions but is not
    part of this simplified formula. ANY product built on this MUST be
    labeled "Estimated WBGT" wherever it's shown.
    """
    t_c = f_to_c(temp_f)
    r = np.asarray(rh_pct, dtype=float)
    vapor_pressure_hpa = (r / 100.0) * _saturation_vapor_pressure_hpa(t_c)
    wbgt_c = 0.567 * t_c + 0.393 * vapor_pressure_hpa + 3.94
    result = c_to_f(wbgt_c)
    return result if np.ndim(result) else float(result)


# --- Pressure / theta-e ------------------------------------------------------

SEA_LEVEL_PRESSURE_HPA = 1013.25
STANDARD_LAPSE_RATE_K_PER_M = 0.0065
STANDARD_SEA_LEVEL_TEMP_K = 288.15
GRAVITY_OVER_LAPSE_GAS_CONST = 5.25588  # g*M / (R*L) for the ICAO standard atmosphere


def standard_atmosphere_pressure_hpa(elevation_m):
    """ICAO standard-atmosphere pressure estimate at a given elevation, in
    hPa - used as a theta_e_k fallback when a station doesn't report its own
    pressure. This is a climatological estimate, not an observation; it
    ignores the actual synoptic pressure pattern, so theta-e computed from it
    carries extra error during strong highs/lows."""
    elevation = np.asarray(elevation_m, dtype=float)
    ratio = 1.0 - (STANDARD_LAPSE_RATE_K_PER_M * elevation) / STANDARD_SEA_LEVEL_TEMP_K
    pressure = SEA_LEVEL_PRESSURE_HPA * np.power(ratio, GRAVITY_OVER_LAPSE_GAS_CONST)
    return pressure if np.ndim(pressure) else float(pressure)


def theta_e_k(temp_c, dewpoint_c, pressure_hpa):
    """Equivalent potential temperature (theta-e), in Kelvin, via the Bolton
    (1980) approximation. `pressure_hpa` should be the station's own reported
    pressure when available; pass `standard_atmosphere_pressure_hpa(elevation)`
    as a fallback otherwise (see module docstring on the "Estimated" labeling
    requirement when that fallback is used)."""
    t_c = np.asarray(temp_c, dtype=float)
    td_c = np.asarray(dewpoint_c, dtype=float)
    p = np.asarray(pressure_hpa, dtype=float)

    t_k = t_c + 273.15
    vapor_pressure_hpa = _saturation_vapor_pressure_hpa(td_c)
    # Bolton's correction terms use mixing ratio in g/kg, not kg/kg.
    mixing_ratio_gkg = 1000.0 * 0.622 * vapor_pressure_hpa / np.clip(p - vapor_pressure_hpa, 1e-6, None)

    # LCL temperature (Bolton 1980, eq. 15).
    t_lcl = 1.0 / (1.0 / (td_c + 273.15 - 56.0) + np.log(t_k / (td_c + 273.15)) / 800.0) + 56.0

    theta_dl = t_k * np.power(1000.0 / p, 0.2854 * (1.0 - 0.28e-3 * mixing_ratio_gkg)) \
        * np.power(t_k / t_lcl, 0.28e-3 * mixing_ratio_gkg)
    theta_e = theta_dl * np.exp(
        (3.376 / t_lcl - 0.00254) * mixing_ratio_gkg * (1.0 + 0.81e-3 * mixing_ratio_gkg)
    )
    return theta_e if np.ndim(theta_e) else float(theta_e)
