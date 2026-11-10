"""The feature vector of the fire_escape model, as pure functions of one day's weather at one place.

Everything the model sees is something a forecaster can know on the morning of the day: the day's
forecast weather (rain included), rain through the day before, the slow fuel state from the day before, the
season and the fuels on the ground. A gridMET day is roughly the calendar day (it ends at midnight Mountain
Standard Time). Deliberately NOT features: latitude/longitude, year, cause, discovery time and
reporting agency. They predict the label in the archive (some agencies report only larger fires) but
are not danger, and a model that leans on them would map who reported fires, not where fire escapes.

Units: deg C, percent, m/s, kPa, mm, W/m2. NaN in, NaN out; XGBoost handles a missing feature natively.
"""
from __future__ import annotations

import numpy as np

try:
    from fire_escape import physical_danger
except ImportError:  # the API carries this file as a byte-identical mirror (services/fire_escape/features.py)
    from services.fire_weather import physical_danger

# name, monotone constraint on P(escape) (+1 never lowers it, -1 never raises it, 0 free), meaning.
# The constraints are physics the model may not contradict: drier, windier and more flammable is never
# safer. They cost a little in-sample fit and keep the model from learning noise as a reversal.
FEATURE_SPEC = (
    ("tmax_c", 0, "day's maximum temperature"),
    ("rh_min", -1, "day's minimum relative humidity"),
    ("rh_max", -1, "day's maximum relative humidity (overnight recovery)"),
    ("wind_ms", 1, "day's mean 10 m wind speed"),
    ("vpd_kpa", 1, "day's mean vapor pressure deficit"),
    ("srad_wm2", 0, "day's mean solar irradiance"),
    ("precip_day_mm", -1, "rain on the day itself"),
    ("precip_1d_mm", -1, "rain the day before"),
    ("precip_3d_mm", -1, "rain in the 3 days before"),
    ("precip_7d_mm", -1, "rain in the 7 days before"),
    ("precip_30d_mm", -1, "rain in the 30 days before"),
    ("days_since_rain", 1, "days since 2.5 mm or more fell in a day (1 = yesterday), capped at 30"),
    ("fm100_prev", -1, "100-hour dead fuel moisture, previous day"),
    ("fm1000_prev", -1, "1000-hour dead fuel moisture, previous day"),
    ("erc_prev", 1, "NFDRS energy release component (fuel model G), previous day"),
    ("pfdi_flame_ft", 1, "Physical Fire Danger Index flame length in cured reference grass"),
    ("pfdi_fm1", -1, "PFDI 1-hour fuel moisture of sunlit fuel"),
    ("stn_wind_peak_ms", 1, "afternoon peak sustained wind at the nearest airports"),
    ("stn_gust_peak_ms", 1, "afternoon peak gust at the nearest airports"),
    ("stn_rh_min", -1, "afternoon minimum relative humidity at the nearest airports"),
    ("pfdi_peak_flame_ft", 1, "PFDI flame length at the afternoon peak wind and gust"),
    ("cured_fraction", 1, "share of the grass load that is cured, from the date"),
    ("doy_sin", 0, "season, sine of day of year"),
    ("doy_cos", 0, "season, cosine of day of year"),
    ("frac_grass", 0, "share of burnable grass / grass-shrub fuel within 1 km"),
    ("frac_timber", 0, "share of timber litter / timber understory fuel within 1 km"),
    ("frac_nonburnable", 0, "share of urban, crop, water or barren within 1 km"),
    ("canopy_cover_pct", 0, "mean tree canopy cover within 1 km"),
)
FEATURES = tuple(name for name, _sign, _meaning in FEATURE_SPEC)
MONOTONE = tuple(sign for _name, sign, _meaning in FEATURE_SPEC)
WEATHER_ONLY_FEATURES = FEATURES[:FEATURES.index("frac_grass")]

RAIN_DAY_MM = 2.5
RAIN_LOOKBACK_DAYS = 30
FUEL_STATE_LAG_DAYS = 1       # gridMET's fuel moisture for a day is published after it; the model uses yesterday's


def solar_declination_rad(day_of_year):
    """Cooper (1969): 23.45 deg x sin(2 pi (284 + n) / 365)."""
    return np.radians(23.45) * np.sin(2.0 * np.pi * (284.0 + np.asarray(day_of_year, dtype=float)) / 365.0)


def clear_sky_wm2(lat_deg, day_of_year):
    """(daily mean, solar noon) Haurwitz clear-sky irradiance at a latitude and date."""
    lat = np.radians(np.asarray(lat_deg, dtype=float))[..., None]
    declination = solar_declination_rad(day_of_year)[..., None]
    hour_angle = np.radians(np.arange(-180.0, 180.0, 7.5) + 3.75)        # 48 half-hour steps through the day
    sin_elevation = np.sin(lat) * np.sin(declination) + np.cos(lat) * np.cos(declination) * np.cos(hour_angle)
    clear = physical_danger.clear_sky_irradiance_wm2(np.degrees(np.arcsin(np.clip(sin_elevation, -1.0, 1.0))))
    noon = physical_danger.clear_sky_irradiance_wm2(np.degrees(np.pi / 2 - np.abs(lat - declination)))[..., 0]
    return clear.mean(axis=-1), noon


def midday_solar_wm2(srad_daily_mean_wm2, lat_deg, day_of_year):
    """Solar-noon irradiance implied by a daily-mean irradiance: the daily mean times the clear-sky ratio
    of noon to daily mean at that latitude and date (Haurwitz clear sky, so the cloud reduction of the
    day is carried to noon unchanged)."""
    mean, noon = clear_sky_wm2(lat_deg, day_of_year)
    return np.asarray(srad_daily_mean_wm2, dtype=float) * np.divide(noon, mean, out=np.full(mean.shape, np.nan), where=mean > 0)


def antecedent_precip(precip_mm):
    """From daily rain shaped (..., 31), the 30 days before and then the day itself (last column): the
    day's own rain, the 1, 3, 7 and 30 day totals before it, and the days since a rain day."""
    rain = np.asarray(precip_mm, dtype=float)
    before = rain[..., :-1]
    wet = before[..., ::-1] >= RAIN_DAY_MM                                # column 0 = yesterday
    since = np.where(wet.any(axis=-1), wet.argmax(axis=-1) + 1, RAIN_LOOKBACK_DAYS).astype(float)
    since = np.where(np.isnan(before).any(axis=-1), np.nan, since)
    return {"precip_day_mm": rain[..., -1], "precip_1d_mm": before[..., -1], "precip_3d_mm": before[..., -3:].sum(axis=-1),
            "precip_7d_mm": before[..., -7:].sum(axis=-1), "precip_30d_mm": before.sum(axis=-1),
            "days_since_rain": since}


def cured_fraction(day_of_year):
    """Share of the reference grass load carried as dead fuel (Scott & Burgan 2005 transfer), from the
    assumed Missouri herbaceous moisture for the date."""
    return 1.0 - np.clip(_live_herb(day_of_year) / 90.0 - 1.0 / 3.0, 0.0, 1.0)


def _live_herb(day_of_year):
    herb = [physical_danger.behavior.LIVE_MOISTURE_BY_MONTH[month][0] for month in range(1, 13)]
    return np.interp(np.asarray(day_of_year, dtype=float), physical_danger._MID_MONTH_DAY, herb, period=365.0)


def pfdi(tmax_c, rh_min, wind_ms, srad_daily_mean_wm2, lat_deg, day_of_year, gust_ms=None):
    """The physical index at the day's worst hour: maximum temperature, minimum humidity, noon sun.
    With the day's mean wind and no gust it understates the afternoon; with the peak wind and gust it does not."""
    out = physical_danger.physical_fire_danger(
        tmax_c, rh_min, wind_ms, midday_solar_wm2(srad_daily_mean_wm2, lat_deg, day_of_year),
        gust_10m_ms=gust_ms, live_herb_pct=_live_herb(day_of_year))
    return {"pfdi_flame_ft": out["flame_length_ft"], "pfdi_fm1": out["fm1_pct"]}


def fuel_fractions(fbfm40_codes):
    """Shares of grass, timber and nonburnable fuel in a window of Scott & Burgan FBFM40 codes shaped
    (..., pixels). 101-124 grass and grass-shrub, 161-189 timber understory and litter, 91-99 nonburnable
    (urban, snow, agriculture, water, barren). Negative codes are NoData and are left out of the total."""
    codes = np.asarray(fbfm40_codes)
    valid = codes > 0
    total = valid.sum(axis=-1).astype(float)

    def share(low, high):
        count = ((codes >= low) & (codes <= high)).sum(axis=-1)
        return np.divide(count, total, out=np.full(total.shape, np.nan), where=total > 0)
    return {"frac_grass": share(101, 124), "frac_timber": share(161, 189), "frac_nonburnable": share(91, 99)}


def season(day_of_year):
    angle = 2.0 * np.pi * np.asarray(day_of_year, dtype=float) / 365.25
    return {"doy_sin": np.sin(angle), "doy_cos": np.cos(angle)}


def assemble(*, tmax_c, rh_min, rh_max, wind_ms, vpd_kpa, srad_wm2, precip_window_mm, fm100_prev, fm1000_prev,
             erc_prev, lat_deg, day_of_year, fbfm40_window=None, canopy_cover_pct=None,
             stn_wind_peak_ms=None, stn_gust_peak_ms=None, stn_rh_min=None) -> dict[str, np.ndarray]:
    """Every feature in FEATURES order. The static fuel features are NaN when no window is given, and the
    airport features are NaN when no station reported."""
    shape = np.shape(tmax_c)
    nan = np.full(shape, np.nan)
    peak_wind = nan if stn_wind_peak_ms is None else np.asarray(stn_wind_peak_ms, dtype=float)
    peak_gust = nan if stn_gust_peak_ms is None else np.asarray(stn_gust_peak_ms, dtype=float)
    peak = pfdi(tmax_c, rh_min, peak_wind, srad_wm2, lat_deg, day_of_year, gust_ms=peak_gust)["pfdi_flame_ft"]
    out = {"tmax_c": tmax_c, "rh_min": rh_min, "rh_max": rh_max, "wind_ms": wind_ms, "vpd_kpa": vpd_kpa,
           "srad_wm2": srad_wm2, **antecedent_precip(precip_window_mm),
           "fm100_prev": fm100_prev, "fm1000_prev": fm1000_prev, "erc_prev": erc_prev,
           **pfdi(tmax_c, rh_min, wind_ms, srad_wm2, lat_deg, day_of_year),
           "stn_wind_peak_ms": peak_wind, "stn_gust_peak_ms": peak_gust,
           "stn_rh_min": nan if stn_rh_min is None else stn_rh_min, "pfdi_peak_flame_ft": peak,
           "cured_fraction": cured_fraction(day_of_year), **season(day_of_year),
           **(fuel_fractions(fbfm40_window) if fbfm40_window is not None
              else {"frac_grass": nan, "frac_timber": nan, "frac_nonburnable": nan}),
           "canopy_cover_pct": nan if canopy_cover_pct is None else canopy_cover_pct}
    return {name: np.asarray(out[name], dtype=float) for name in FEATURES}
