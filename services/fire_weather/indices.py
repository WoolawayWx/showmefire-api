"""Fire-weather indices computed from model temperature/moisture/wind profiles.

Every function works on whole grids at once: profile arrays are shaped
(level, ...) with levels ordered bottom-up and NaN where a level is missing
(e.g. a pressure level that lies below the ground). Surface inputs are shaped
like one level. Heights are metres above ground level (AGL), temperatures
deg C, pressures hPa, winds m/s, unless a name says otherwise.
"""
from __future__ import annotations

import numpy as np

M_TO_FT = 3.28084
MS_TO_KT = 1.943844
MS_TO_MPH = 2.236936
KAPPA = 0.2857  # R_d / c_p, Poisson exponent


def potential_temperature_k(temp_c, pressure_hpa):
    """Poisson's equation, theta = T (1000 / p) ** (R_d / c_p)."""
    return (np.asarray(temp_c) + 273.15) * (1000.0 / np.asarray(pressure_hpa)) ** KAPPA


def saturation_vapor_pressure_hpa(temp_c):
    """Bolton (1980, Mon. Wea. Rev. 108, 1046-1053) eq. 10:
    e_s = 6.112 exp(17.67 T / (T + 243.5)), T in deg C, e_s in hPa."""
    temp_c = np.asarray(temp_c, dtype=float)
    return 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5))


def vapor_pressure_deficit_hpa(temp_c, dewpoint_c):
    """VPD = e_s(T) - e(Td), with e(Td) = e_s(Td) (Bolton 1980)."""
    return saturation_vapor_pressure_hpa(temp_c) - saturation_vapor_pressure_hpa(dewpoint_c)


def interpolate_to_height(heights: np.ndarray, values: np.ndarray, target) -> np.ndarray:
    """Linear interpolation of a bottom-up profile to height `target` (scalar or grid).
    NaN where the column does not span the target."""
    target = np.broadcast_to(np.asarray(target, dtype=float), heights.shape[1:])
    result = np.full(heights.shape[1:], np.nan)
    for k in range(heights.shape[0] - 1):
        z0, z1, v0, v1 = heights[k], heights[k + 1], values[k], values[k + 1]
        ok = np.isfinite(z0) & np.isfinite(z1) & np.isfinite(v0) & np.isfinite(v1) & (z1 > z0)
        inside = ok & np.isnan(result) & (target >= z0) & (target <= z1)
        weight = np.divide(target - z0, z1 - z0, out=np.zeros_like(target), where=ok)
        result = np.where(inside, v0 + weight * (v1 - v0), result)
    return result


def mixing_height_holzworth(heights: np.ndarray, theta: np.ndarray, theta_surface: np.ndarray,
                            theta_excess: float = 0.0) -> np.ndarray:
    """Holzworth (1967, J. Appl. Meteor. 6, 1039-1044) parcel method: lift the
    surface parcel dry-adiabatically (constant potential temperature) and take
    the height where it meets the environmental profile, i.e. where the
    profile's potential temperature first exceeds the parcel's. Returns m AGL;
    0 where the air just above the surface is already warmer (surface
    inversion); the column top where the parcel never meets the profile.

    `theta_excess` (K) gives the parcel a small temperature excess, the usual
    parcel-method allowance for noise in high-resolution soundings (Seibert et
    al. 2000, Atmos. Environ. 34, 1001-1027); 0 for smooth model profiles."""
    theta_surface = np.asarray(theta_surface, dtype=float) + theta_excess
    result = np.full(theta_surface.shape, np.nan)
    previous_z = np.zeros(theta_surface.shape)
    previous_theta = np.asarray(theta_surface, dtype=float)
    for k in range(heights.shape[0]):
        z, th = heights[k], theta[k]
        ok = np.isfinite(z) & np.isfinite(th)
        crossing = ok & np.isnan(result) & (th > theta_surface)
        span = th - previous_theta
        weight = np.divide(theta_surface - previous_theta, span, out=np.zeros_like(span), where=span > 0)
        result = np.where(crossing, previous_z + np.clip(weight, 0, 1) * (z - previous_z), result)
        previous_z = np.where(ok, z, previous_z)
        previous_theta = np.where(ok, th, previous_theta)
    return np.where(np.isnan(result), previous_z, result)


def transport_wind(heights: np.ndarray, u: np.ndarray, v: np.ndarray, mixing_height: np.ndarray,
                   bottom: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Transport wind: the average wind through the mixed layer ("average wind
    speed and direction from surface to top of mixed layer", NWS CLE fire
    weather forecast format). Height-weighted (trapezoidal) mean of the u and v
    components from `bottom` (the 10 m wind) to the mixing height; a mixed layer
    shallower than the lowest wind level gets that level's wind.
    Returns the mean (u, v) vector in the input wind's frame."""
    top = np.maximum(np.asarray(mixing_height, dtype=float), bottom)
    sum_u = np.zeros(top.shape)
    sum_v = np.zeros(top.shape)
    depth = np.zeros(top.shape)
    for k in range(heights.shape[0] - 1):
        z0, z1 = heights[k], heights[k + 1]
        ok = (np.isfinite(z0) & np.isfinite(z1) & np.isfinite(u[k]) & np.isfinite(u[k + 1])
              & np.isfinite(v[k]) & np.isfinite(v[k + 1]) & (z1 > z0))
        lo = np.maximum(z0, bottom)
        hi = np.minimum(z1, top)
        seg = np.where(ok & (hi > lo), hi - lo, 0.0)
        span = np.where(ok, z1 - z0, 1.0)
        # Trapezoid over the clipped segment, using the linear profile's values at its ends.
        u_lo = u[k] + (lo - z0) / span * (u[k + 1] - u[k])
        u_hi = u[k] + (hi - z0) / span * (u[k + 1] - u[k])
        v_lo = v[k] + (lo - z0) / span * (v[k + 1] - v[k])
        v_hi = v[k] + (hi - z0) / span * (v[k + 1] - v[k])
        sum_u += np.where(seg > 0, 0.5 * (u_lo + u_hi) * seg, 0.0)
        sum_v += np.where(seg > 0, 0.5 * (v_lo + v_hi) * seg, 0.0)
        depth += seg
    lowest_u = _first_valid(u)
    lowest_v = _first_valid(v)
    mean_u = np.where(depth > 0, np.divide(sum_u, depth, out=np.zeros_like(sum_u), where=depth > 0), lowest_u)
    mean_v = np.where(depth > 0, np.divide(sum_v, depth, out=np.zeros_like(sum_v), where=depth > 0), lowest_v)
    return mean_u, mean_v


def _first_valid(values: np.ndarray) -> np.ndarray:
    """Lowest finite value in each column."""
    result = np.full(values.shape[1:], np.nan)
    for k in range(values.shape[0]):
        result = np.where(np.isnan(result) & np.isfinite(values[k]), values[k], result)
    return result


def wind_speed(u, v):
    return np.hypot(u, v)


def wind_direction_from(u, v):
    """Meteorological direction (degrees the wind blows FROM), earth-relative u/v."""
    return (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0


def ventilation_rate_kt_ft(transport_speed_kt, mixing_height_ft):
    """Ventilation Rate = Transport Wind (kts) x Mixing Height (ft), knot-feet
    (NWS Springfield, https://www.weather.gov/sgf/firewx_smoke_dispersal)."""
    return np.asarray(transport_speed_kt) * np.asarray(mixing_height_ft)


# NWS Springfield smoke-dispersal categories, ventilation rate in knot-feet x 1000.
VENTILATION_CATEGORIES = [
    ("Poor", 0.0, "#d7301f"),
    ("Fair", 40.0, "#fc8d59"),
    ("Good", 60.0, "#fee08b"),
    ("Very Good", 100.0, "#91cf60"),
    ("Excellent", 150.0, "#1a9850"),
]


def ventilation_category(rate_kt_ft) -> np.ndarray:
    """Index into VENTILATION_CATEGORIES (0 Poor ... 4 Excellent); -1 where missing."""
    thousands = np.asarray(rate_kt_ft, dtype=float) / 1000.0
    edges = [low for _name, low, _color in VENTILATION_CATEGORIES[1:]]
    index = np.digitize(np.nan_to_num(thousands, nan=0.0), edges)
    return np.where(np.isfinite(thousands), index, -1)


def surface_inversion(heights: np.ndarray, temp_c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Surface-based inversion: from the lowest level (2 m), follow the profile
    upward while temperature keeps increasing with height (AMS Glossary,
    "inversion"). Returns (strength deg C = warmest - surface, top m AGL);
    0 / 0 where temperature does not rise above the surface level."""
    surface = temp_c[0]
    warmest = surface.copy()
    top = np.zeros(surface.shape)
    alive = np.isfinite(surface)
    previous = surface.copy()
    for k in range(1, heights.shape[0]):
        t, z = temp_c[k], heights[k]
        ok = np.isfinite(t) & np.isfinite(z)
        rising = alive & ok & (t > previous)
        warmest = np.where(rising, t, warmest)
        top = np.where(rising, z, top)
        alive = alive & (~ok | rising)  # a missing level does not end the walk
        previous = np.where(ok, t, previous)
    return warmest - surface, top


def lapse_rate_c_per_km(heights: np.ndarray, temp_c: np.ndarray, depth_m: float = 1000.0) -> np.ndarray:
    """Lapse rate -dT/dz over the lowest `depth_m` (AMS Glossary, "lapse rate"),
    deg C per km: (T_surface - T_at_depth) / depth. Negative = inversion."""
    t_top = interpolate_to_height(heights, temp_c, depth_m)
    return (temp_c[0] - t_top) / ((depth_m - heights[0]) / 1000.0)


def haines_low(t950_c, t850_c, td850_c) -> np.ndarray:
    """Low-elevation Haines Index (Haines 1988, Natl. Wea. Digest 13(2), 23-27;
    table as in NWS WR Technical Attachment 00-08):
    stability = T950 - T850: <=3 C -> 1, 4-7 -> 2, >=8 -> 3;
    moisture = 850 mb dewpoint depression: <=5 C -> 1, 6-9 -> 2, >=10 -> 3.
    The published bins are whole degrees, so values are rounded first."""
    stability = np.rint(np.asarray(t950_c) - np.asarray(t850_c))
    moisture = np.rint(np.asarray(t850_c) - np.asarray(td850_c))
    a = np.where(stability <= 3, 1, np.where(stability <= 7, 2, 3))
    b = np.where(moisture <= 5, 1, np.where(moisture <= 9, 2, 3))
    total = (a + b).astype(float)
    return np.where(np.isfinite(stability) & np.isfinite(moisture), total, np.nan)


def hot_dry_windy(vpd_hpa_levels: np.ndarray, wind_ms_levels: np.ndarray) -> np.ndarray:
    """Hot-Dry-Windy Index (Srock et al. 2018, Atmosphere 9, 279): the maximum
    vapor pressure deficit times the maximum wind speed over the surface and the
    lowest ~50 hPa above it. Callers pass only the levels inside that layer
    (NaN elsewhere). Units hPa m/s."""
    vpd = np.nanmax(np.where(np.isfinite(vpd_hpa_levels), vpd_hpa_levels, -np.inf), axis=0)
    wind = np.nanmax(np.where(np.isfinite(wind_ms_levels), wind_ms_levels, -np.inf), axis=0)
    hdw = np.maximum(vpd, 0.0) * np.maximum(wind, 0.0)
    return np.where(np.isfinite(hdw) & (vpd > -np.inf) & (wind > -np.inf), hdw, np.nan)


def earth_relative_winds(u_grid, v_grid, lon_deg, central_longitude: float, standard_parallel: float):
    """Rotate Lambert-conformal grid-relative winds (HRRR) to earth-relative:
    alpha = sin(standard parallel) * (lon - central lon);
    u_e = cos(a) u + sin(a) v, v_e = -sin(a) u + cos(a) v."""
    angle = np.sin(np.radians(standard_parallel)) * np.radians(((np.asarray(lon_deg) - central_longitude + 180) % 360) - 180)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return cos_a * u_grid + sin_a * v_grid, -sin_a * u_grid + cos_a * v_grid
