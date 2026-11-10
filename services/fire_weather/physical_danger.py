"""Physical Fire Danger Index (PFDI): fire danger as the fire behaviour the weather would produce.

EXPERIMENTAL - NOT FOR OPERATIONS. Nothing here is fitted to fire records and nothing has been
verified against them yet.

The other danger models in this repo combine weather thresholds (the rule), weighted ramps (FWSI)
or percentile ranks (the hybrid's relative tiers). This one has no thresholds on weather at all.
It asks one physical question: if a fire started now in open cured grass on flat ground, how
intense would its head be? The answer is Byram's fireline intensity, and the danger level is read
from the flame length on the fire-line handling chart, so a level means the same thing every day
of the year ("too intense for hand crews"), not "unusual for this dataset".

The chain, each link a published relation, with no tuned weights:

1. Sun on the fuel. Clear-sky irradiance from solar elevation (Haurwitz 1945), reduced for cloud
   (Kasten & Czeplak 1980).
2. Fuel-level air. Sunlit fine fuel is warmer and its boundary-layer air drier than the 2 m
   shelter values; wind ventilates the difference away (Byram & Jemison 1943, in the form used by
   Rothermel et al. 1986).
3. Dead fuel moisture. Equilibrium moisture content at the fuel-level temperature and humidity
   (Simard 1968), then the NFDRS no-memory estimates of 1-h and 10-h moisture (Fosberg & Deeming
   1971); measurable rain saturates both.
4. Live fuel. Herbaceous moisture sets how much of the grass load is cured (Scott & Burgan 2005).
5. Wind. 10 m wind to the 20 ft reference height by the neutral log profile. Fire does not burn
   in the mean wind: turbulence is described by the gust (peak = mean + 3 sigma), and the index
   uses the wind exceeded 10 % of the time.
6. Fire behaviour. Rothermel (1972) spread, Byram (1959) intensity and flame length, in a
   reference fuel bed (services/fire_weather_hybrid/behavior.py, the pyretechnics port), without
   Rothermel's wind-speed limit (Andrews et al. 2013; see `head_fire`).
7. Danger level from flame length (Andrews & Rothermel 1982): the scale the hybrid already uses.

What it deliberately leaves out, and so what it cannot see:
- Fuel moisture has no memory here. An hour after rain stops the fuel reads as dry as the air
  says; a stateful caller should pass its own `fm1_pct` / `fm10_pct`.
- Real fuels and terrain. It is a weather index in ONE reference fuel on flat ground, so it does
  not know a cell is forest, cropland, water or snow-covered.
- Night-time radiative cooling and dew.
- More wind is not always more danger here, on purpose: wind also strips the solar warming from
  the fuel, so with fuel near its moisture of extinction a breeze can lower the index. With fuel
  moisture held fixed the index never falls as wind rises.
- The cured fraction comes from a month table of assumed Missouri live moisture unless the caller
  passes `live_herb_pct`.

All functions work on whole grids, propagate NaN (a missing input is never treated as benign)
and do no I/O. Units: deg C, percent, m/s, W/m2, mm, unless a name says otherwise.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np

try:
    from services.fire_weather_hybrid import behavior, risk
except ImportError:  # model-training carries this file as a byte-identical mirror (fire_escape/physical_danger.py)
    from fire_weather_hybrid import behavior, risk

VERSION = "pfdi-1.0.0"

# --- constants, each with its source -----------------------------------------------------------------

REFERENCE_FUEL = 102              # Scott & Burgan GR2: low-load dry-climate grass, dynamic, extinction 15 %
VON_KARMAN = 0.4
ROUGHNESS_LENGTH_M = 0.03         # open grassland (WMO class), as in the hybrid's physics predictor
TWENTY_FT_M = 6.096
WIND_10M_TO_20FT = float(np.log(TWENTY_FT_M / ROUGHNESS_LENGTH_M) / np.log(10.0 / ROUGHNESS_LENGTH_M))  # 0.915
# Neutral surface layer: sigma_u = 2.5 u* (Panofsky & Dutton 1984) and U = (u*/k) ln(z/z0),
# so sigma_u / U = 2.5 k / ln(z/z0) = 0.17 at 10 m. Used only where no gust is given.
NEUTRAL_TURBULENCE_INTENSITY = float(2.5 * VON_KARMAN / np.log(10.0 / ROUGHNESS_LENGTH_M))
GUST_PEAK_FACTOR = 3.0            # gust = mean + 3 sigma_u (3 s gust in 10 min; Wieringa 1973)
DESIGN_WIND_Z = 1.2816            # standard normal 90th percentile: the wind exceeded 10 % of the time

WM2_PER_LANGLEY_MIN = 697.8       # 1 cal cm-2 min-1 in W m-2
RAIN_WET_MM = 0.25                # 0.01 in in the hour: measurable rain
RAIN_SATURATED_PCT = 35.0         # NFDRS: 1-h and 10-h moisture while it is raining
FM1_PER_EMC = 1.03                # Fosberg & Deeming (1971)
FM10_PER_EMC = 1.28
FM_MIN_PCT, FM_MAX_PCT = 1.0, 40.0

BYRAM_FLAME_M = 0.07747042253266703   # Byram (1959): flame length (m) = 0.0775 I^0.46, I in kW/m; the port's constant
_PROBE_WINDS_MS = (0.25, 0.5)         # 20 ft winds used to read Rothermel's wind power law off the port (see head_fire)
M_TO_FT = 3.28084
M_MIN_TO_CH_H = 60.0 / 20.1168
MS_TO_MPH = 2.236936
# Flame length (ft) at the edges of the five levels, and where the 0-100 index reaches each multiple
# of 20. The first four are the handling-chart limits; 15 ft only anchors the top of the index.
FLAME_LENGTH_ANCHORS_FT = (0.0, *risk.FLAME_LENGTH_SCALE["upper_bounds_ft"], 15.0)
INDEX_ANCHORS = (0.0, 20.0, 40.0, 60.0, 80.0, 100.0)
LEVELS = tuple(risk.FLAME_LENGTH_SCALE["levels"])
_MID_MONTH_DAY = (15.5, 45.0, 74.5, 105.0, 135.5, 166.0, 196.5, 227.5, 258.0, 288.5, 319.0, 349.5)


# --- 1. sun -----------------------------------------------------------------------------------------

def clear_sky_irradiance_wm2(solar_elevation_deg) -> np.ndarray:
    """Haurwitz (1945) clear-sky global horizontal irradiance: 1098 sin(h) exp(-0.057 / sin(h)).
    Zero with the sun at or below the horizon."""
    sin_h = np.sin(np.radians(np.asarray(solar_elevation_deg, dtype=float)))
    up = sin_h > 0
    safe = np.where(up, sin_h, 1.0)
    return np.where(up, 1098.0 * safe * np.exp(-0.057 / safe), np.where(np.isnan(sin_h), np.nan, 0.0))


def cloud_transmission(cloud_cover_pct) -> np.ndarray:
    """Kasten & Czeplak (1980): fraction of clear-sky irradiance under cloud fraction N, 1 - 0.75 N^3.4."""
    fraction = np.clip(np.asarray(cloud_cover_pct, dtype=float) / 100.0, 0.0, 1.0)
    return 1.0 - 0.75 * fraction ** 3.4


def solar_irradiance_wm2(lat_deg, lon_deg, when_utc: datetime, cloud_cover_pct) -> np.ndarray:
    """Global horizontal irradiance at the ground for a time and a cloud cover."""
    from services import derived_met      # API only: the training mirror supplies its own irradiance

    elevation = derived_met.solar_elevation_deg(lat_deg, lon_deg, when_utc)
    return clear_sky_irradiance_wm2(elevation) * cloud_transmission(cloud_cover_pct)


# --- 2-3. dead fuel moisture ------------------------------------------------------------------------

def fuel_level_conditions(temp_c, rh_pct, solar_wm2, fuel_wind_ms):
    """Temperature (deg C) and RH (%) of the air against sunlit fine fuel.

    Byram & Jemison (1943) as used by Rothermel et al. (1986, INT-359):
        T_f = T_a + I / (0.015 U + 0.026),   H_f = H_a exp(-0.033 (T_f - T_a))
    with T in deg F, I in cal cm-2 min-1 and U the wind at fuel level in mph."""
    irradiance = np.maximum(np.asarray(solar_wm2, dtype=float), 0.0) / WM2_PER_LANGLEY_MIN
    wind_mph = np.maximum(np.asarray(fuel_wind_ms, dtype=float), 0.0) * MS_TO_MPH
    warming_f = irradiance / (0.015 * wind_mph + 0.026)
    return np.asarray(temp_c, dtype=float) + warming_f / 1.8, np.asarray(rh_pct, dtype=float) * np.exp(-0.033 * warming_f)


def equilibrium_moisture_pct(temp_c, rh_pct) -> np.ndarray:
    """Simard (1968) equilibrium moisture content of wood (%), T in deg F, H in %:
        H < 10:        0.03229 + 0.281073 H - 0.000578 H T
        10 <= H < 50:  2.22749 + 0.160107 H - 0.01478 T
        H >= 50:       21.0606 + 0.005565 H^2 - 0.00035 H T - 0.483199 H
    (Not the sign-reversed middle branch in services/spread_rate_moisture.py.)"""
    h = np.clip(np.asarray(rh_pct, dtype=float), 0.0, 100.0)
    t = np.asarray(temp_c, dtype=float) * 1.8 + 32.0
    emc = np.where(h < 10, 0.03229 + 0.281073 * h - 0.000578 * h * t,
                   np.where(h < 50, 2.22749 + 0.160107 * h - 0.01478 * t,
                            21.0606 + 0.005565 * h * h - 0.00035 * h * t - 0.483199 * h))
    return np.clip(emc, FM_MIN_PCT, FM_MAX_PCT)


def dead_fuel_moisture(temp_c, rh_pct, solar_wm2, fuel_wind_ms, precip_mm_1h=None):
    """(1-h, 10-h) dead fuel moisture in percent, with no memory of earlier hours.

    Fosberg & Deeming (1971): 1.03 and 1.28 times the equilibrium moisture content at fuel level.
    An hour with measurable rain sets both to 35 % (NFDRS). `precip_mm_1h=None` means the rain is
    unknown and is not applied; a NaN cell in a given rain field gives NaN."""
    fuel_temp_c, fuel_rh = fuel_level_conditions(temp_c, rh_pct, solar_wm2, fuel_wind_ms)
    emc = equilibrium_moisture_pct(fuel_temp_c, fuel_rh)
    fm1 = np.clip(FM1_PER_EMC * emc, FM_MIN_PCT, FM_MAX_PCT)
    fm10 = np.clip(FM10_PER_EMC * emc, FM_MIN_PCT, FM_MAX_PCT)
    if precip_mm_1h is not None:
        rain = np.asarray(precip_mm_1h, dtype=float)
        fm1 = np.where(np.isnan(rain), np.nan, np.where(rain >= RAIN_WET_MM, RAIN_SATURATED_PCT, fm1))
        fm10 = np.where(np.isnan(rain), np.nan, np.where(rain >= RAIN_WET_MM, RAIN_SATURATED_PCT, fm10))
    return fm1, fm10


# --- 4. live fuel -----------------------------------------------------------------------------------

def live_moisture_pct(when: datetime) -> tuple[float, float]:
    """Assumed Missouri (herbaceous, woody) live moisture for a date: the hybrid's month table
    (behavior.LIVE_MOISTURE_BY_MONTH, documented assumptions) interpolated between mid-month values
    so the cured fraction does not jump on the first of a month."""
    day = float(when.timetuple().tm_yday)
    herb, woody = zip(*(behavior.LIVE_MOISTURE_BY_MONTH[month] for month in range(1, 13)))
    return (float(np.interp(day, _MID_MONTH_DAY, herb, period=365.0)),
            float(np.interp(day, _MID_MONTH_DAY, woody, period=365.0)))


# --- 5. wind ----------------------------------------------------------------------------------------

def design_wind_20ft_ms(wind_10m_ms, gust_10m_ms=None) -> np.ndarray:
    """20 ft wind exceeded 10 % of the time: (U + 1.2816 sigma_u) x the log-profile 10 m -> 20 ft ratio.

    sigma_u = (gust - U) / 3 where a gust is given (never negative), else the neutral surface-layer
    value 0.17 U. Rothermel spread rises monotonically with wind, so fire behaviour at this wind is
    the behaviour exceeded 10 % of the time, assuming the fire front follows the wind quasi-steadily."""
    wind = np.maximum(np.asarray(wind_10m_ms, dtype=float), 0.0)
    sigma = NEUTRAL_TURBULENCE_INTENSITY * wind
    if gust_10m_ms is not None:
        gust = np.asarray(gust_10m_ms, dtype=float)
        sigma = np.where(np.isfinite(gust), np.maximum(gust - wind, 0.0) / GUST_PEAK_FACTOR, sigma)
    return (wind + DESIGN_WIND_Z * sigma) * WIND_10M_TO_20FT


def unsheltered_midflame_factor(fuel_code: int = REFERENCE_FUEL) -> float:
    """Albini & Baughman (1979) unsheltered wind adjustment factor for a fuel bed of depth d (ft):
    1.83 / ln((20 + 0.36 d) / (0.13 d)). Midflame wind = factor x 20 ft wind."""
    depth = float(behavior._TABLE[str(int(fuel_code))]["delta_ft"])
    return float(1.83 / np.log((20.0 + 0.36 * depth) / (0.13 * depth)))


# --- 6-7. fire behaviour and the index --------------------------------------------------------------

def head_fire(fuel_code: int, fm1_pct, fm10_pct, fm100_pct, live_herb_pct, live_woody_pct, wind_20ft_ms):
    """Head-fire spread (m/min), fireline intensity (kW/m) and flame length (m) on flat ground, WITHOUT
    Rothermel's wind-speed limit.

    The limit caps the wind effect at a midflame wind of 0.9 x reaction intensity; Andrews et al. (2013,
    RMRS-GTR-309) found it unsupported by data and recommended not imposing it. With it, flame length in the
    reference grass stops rising near 30 mph, which is exactly where a danger index has to keep discriminating.

    The Rothermel port applies the limit internally, so it is used only where it is exact: for the no-wind
    fire, and at two probe winds far below the limit. Rothermel's wind factor is a pure power law,
    phi_w = C U^B, so those two fix C and B and the factor follows for any wind. Below the limit this
    reproduces the port (tested); above it, it continues the same curve."""
    def port(wind):
        return behavior.surface_fire(code, fm1_pct, fm10_pct, fm100_pct, live_herb_pct, live_woody_pct,
                                     wind, 0.0, 0.0, 0.0, 0.0, 0.0)
    wind = np.asarray(wind_20ft_ms, dtype=float)
    shape = np.broadcast(fm1_pct, fm10_pct, fm100_pct, live_herb_pct, live_woody_pct, wind).shape
    code = np.full(shape, float(fuel_code))
    calm, low, high = port(0.0), port(_PROBE_WINDS_MS[0]), port(_PROBE_WINDS_MS[1])
    burning = calm["ros_m_min"] > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        phi_low = np.where(burning, low["ros_m_min"] / calm["ros_m_min"] - 1.0, 0.0)
        phi_high = np.where(burning, high["ros_m_min"] / calm["ros_m_min"] - 1.0, 0.0)
        exponent = np.where(phi_low > 0, np.log(phi_high / phi_low) / np.log(_PROBE_WINDS_MS[1] / _PROBE_WINDS_MS[0]), 0.0)
        phi = phi_low * np.power(np.maximum(wind, 0.0) / _PROBE_WINDS_MS[0], exponent)
    phi = np.where(np.isnan(wind), np.nan, phi)
    intensity = calm["intensity_kw_m"] * (1.0 + phi)
    return calm["ros_m_min"] * (1.0 + phi), intensity, BYRAM_FLAME_M * np.power(intensity, 0.46)


def index_from_flame_length(flame_length_m) -> np.ndarray:
    """0-100, piecewise linear in flame length: 20 / 40 / 60 / 80 at the handling-chart limits
    (1, 4, 8, 11 ft) and 100 at 15 ft and beyond. floor(index / 20) is the level, 0 Low .. 4 Extreme."""
    flame_ft = np.asarray(flame_length_m, dtype=float) * M_TO_FT
    return np.where(np.isfinite(flame_ft), np.interp(flame_ft, FLAME_LENGTH_ANCHORS_FT, INDEX_ANCHORS), np.nan)


def physical_fire_danger(temp_c, rh_pct, wind_10m_ms, solar_wm2, *, gust_10m_ms=None, precip_mm_1h=None,
                         live_herb_pct=30.0, live_woody_pct=60.0,
                         fm1_pct=None, fm10_pct=None, fm100_pct=None,
                         fuel_code: int = REFERENCE_FUEL) -> dict[str, np.ndarray]:
    """The index and every intermediate it is built from, for one time on any grid shape.

    temp_c, rh_pct      2 m air temperature and relative humidity
    wind_10m_ms         10 m sustained wind speed; gust_10m_ms the gust (optional)
    solar_wm2           global horizontal irradiance at the ground (see `solar_irradiance_wm2`); 0 at night
    precip_mm_1h        rain in the hour (optional, see `dead_fuel_moisture`)
    live_herb_pct       herbaceous live moisture: 30 = fully cured, 120 = fully green (see `live_moisture_pct`)
    fm1/fm10/fm100_pct  dead fuel moisture from a model with memory, where the caller has one; each
                        replaces the no-memory estimate. fm100 defaults to fm10 (the reference grass
                        carries no 100-h fuel).

    Returns float arrays: `index` (0-100), `category` (0 Low .. 4 Extreme), `flame_length_ft`,
    `intensity_kw_m`, `ros_ch_h`, `fm1_pct`, `fm10_pct` and `design_wind_mph` (20 ft). NaN wherever an
    input is missing. A cell too wet to carry fire is a real zero, not NaN."""
    wind_10m_ms = np.asarray(wind_10m_ms, dtype=float)
    mean_midflame = np.maximum(wind_10m_ms, 0.0) * WIND_10M_TO_20FT * unsheltered_midflame_factor(fuel_code)
    est_fm1, est_fm10 = dead_fuel_moisture(temp_c, rh_pct, solar_wm2, mean_midflame, precip_mm_1h)
    fm1 = est_fm1 if fm1_pct is None else np.asarray(fm1_pct, dtype=float)
    fm10 = est_fm10 if fm10_pct is None else np.asarray(fm10_pct, dtype=float)
    fm100 = fm10 if fm100_pct is None else np.asarray(fm100_pct, dtype=float)
    design_wind = design_wind_20ft_ms(wind_10m_ms, gust_10m_ms)

    ros, intensity, flame_m = head_fire(fuel_code, fm1, fm10, fm100, live_herb_pct, live_woody_pct, design_wind)
    shape = flame_m.shape
    return {
        "index": index_from_flame_length(flame_m),
        "category": risk.flame_length_categories(flame_m),
        "flame_length_ft": flame_m * M_TO_FT,
        "intensity_kw_m": intensity,
        "ros_ch_h": ros * M_MIN_TO_CH_H,
        "fm1_pct": np.broadcast_to(fm1, shape),
        "fm10_pct": np.broadcast_to(fm10, shape),
        "design_wind_mph": np.broadcast_to(design_wind * MS_TO_MPH, shape),
    }


def physical_fire_danger_at(temp_c, rh_pct, wind_10m_ms, cloud_cover_pct, lat_deg, lon_deg, when_utc: datetime, *,
                            gust_10m_ms=None, precip_mm_1h=None,
                            fuel_code: int = REFERENCE_FUEL) -> Optional[dict[str, np.ndarray]]:
    """`physical_fire_danger` for a place and time: the sun from the clock and the cloud cover, the
    cured fraction from the date. Also returns `solar_wm2`."""
    solar = solar_irradiance_wm2(lat_deg, lon_deg, when_utc, cloud_cover_pct)
    herb, woody = live_moisture_pct(when_utc)
    out = physical_fire_danger(temp_c, rh_pct, wind_10m_ms, solar, gust_10m_ms=gust_10m_ms, precip_mm_1h=precip_mm_1h,
                               live_herb_pct=herb, live_woody_pct=woody, fuel_code=fuel_code)
    out["solar_wm2"] = np.broadcast_to(solar, out["index"].shape)
    return out
