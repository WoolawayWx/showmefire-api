"""From one HRRR forecast of the burning period to the model's daily weather features.

The fire_escape models were fitted on gridMET days and airport afternoons. On the day of use neither exists
yet: what exists is the HRRR forecast the daily fire danger forecast already caches (12z run, forecast hours
4-15, 10 AM to 9 PM Central). This module turns that file into the same features in two steps:

1. `aggregate`: the forecast hours reduced to day values per grid cell (maximum temperature, minimum
   humidity, mean and peak wind, peak gust, ...).
2. `adapt`: a linear map from those day values to each gridMET / airport feature, fitted where both exist
   (fire_escape/serving.py: every cached HRRR day against gridMET and the airports). HRRR and gridMET do
   not define a "day" the same way and do not share biases; the map carries one to the other and its fit
   is reported, never assumed.

Pure numpy apart from `read_cache_file`. Mirrored byte-for-byte between model-training and the API.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

PREDICTORS = ("tmax_c", "tmin_c", "rh_min", "rh_max", "rh_first", "wind_mean_ms", "wind_peak_ms", "gust_peak_ms",
              "vpd_mean_kpa", "precip_mm", "clear_sky_wm2")
# The model features the adapter produces. Everything else in the feature vector comes from gridMET history
# (rain before the day, slow fuel moisture), the date or the fuel map.
ADAPTED = ("tmax_c", "rh_min", "rh_max", "wind_ms", "vpd_kpa", "srad_wm2", "precip_day_mm",
           "stn_wind_peak_ms", "stn_gust_peak_ms", "stn_rh_min")
NEUTRAL_GUST_FACTOR = 1.52          # peak = mean + 3 sigma_u with sigma_u = 0.17 U (physical_danger): used where HRRR has no gust


def saturation_vapor_pressure_kpa(temp_c):
    """Bolton (1980): 0.6112 exp(17.67 T / (T + 243.5)) kPa."""
    temp_c = np.asarray(temp_c, dtype=float)
    return 0.6112 * np.exp(17.67 * temp_c / (temp_c + 243.5))


def aggregate(t2m_k, rh_pct, u10, v10, tp_mm=None, gust_ms=None) -> dict[str, np.ndarray]:
    """Day values per cell from hourly fields shaped (hours, ...). `tp_mm` is accumulated since the start of
    the run (its last hour is the day's rain); None leaves `precip_mm` NaN. A cell missing in any hour is NaN."""
    temp_c = np.asarray(t2m_k, dtype=float) - 273.15
    rh = np.clip(np.asarray(rh_pct, dtype=float), 0.0, 100.0)
    wind = np.hypot(np.asarray(u10, dtype=float), np.asarray(v10, dtype=float))
    peak = wind.max(axis=0)
    gust = peak * NEUTRAL_GUST_FACTOR if gust_ms is None else np.maximum(np.asarray(gust_ms, dtype=float).max(axis=0), peak)
    vpd = saturation_vapor_pressure_kpa(temp_c) * (1.0 - rh / 100.0)
    rain = np.full(peak.shape, np.nan) if tp_mm is None else np.maximum(np.asarray(tp_mm, dtype=float)[-1], 0.0)
    return {"tmax_c": temp_c.max(axis=0), "tmin_c": temp_c.min(axis=0), "rh_min": rh.min(axis=0), "rh_max": rh.max(axis=0),
            "rh_first": rh[0], "wind_mean_ms": wind.mean(axis=0), "wind_peak_ms": peak, "gust_peak_ms": gust,
            "vpd_mean_kpa": vpd.mean(axis=0), "precip_mm": rain}


def adapt(adapter: dict, predictors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Each ADAPTED feature as intercept + sum(coefficient x predictor), held inside the range the fit saw.
    `predictors` needs every name in PREDICTORS (`clear_sky_wm2` is the daily-mean clear-sky irradiance)."""
    out = {}
    for name, spec in adapter["features"].items():
        value = spec["intercept"] + sum(spec["coefficients"][p] * np.asarray(predictors[p], dtype=float) for p in adapter["predictors"])
        out[name] = np.clip(value, spec["low"], spec["high"])
    return out


def county_mean(values, cell_county, n_counties: int) -> np.ndarray:
    """Mean of a grid over the cells of each county. `cell_county` holds each cell's county index, -1 outside
    every county. Cells with NaN are left out; a county with no valid cell is NaN."""
    values, index = np.asarray(values, dtype=float).ravel(), np.asarray(cell_county).ravel()
    ok = (index >= 0) & np.isfinite(values)
    total = np.bincount(index[ok], weights=values[ok], minlength=n_counties)
    count = np.bincount(index[ok], minlength=n_counties)
    return np.divide(total, count, out=np.full(n_counties, np.nan), where=count > 0)


def read_cache_file(path: Path) -> dict:
    """The daily forecast's cached HRRR file (hrrr_YYYYMMDD_HHz_fXX-YY.nc): hourly arrays (hours, y, x), 2-D
    latitude/longitude (degrees east in -180..180) and the valid times. `tp` and `gust` are None when the
    file does not carry them."""
    import xarray as xr

    with xr.open_dataset(path, decode_timedelta=False) as dataset:          # the forecast-hour axis is not needed
        def field(name):
            return dataset[name].to_numpy().astype("float32") if name in dataset.data_vars else None
        lon = dataset["longitude"].to_numpy().astype(float)
        return {"t2m": field("t2m"), "r2": field("r2"), "u10": field("u10"), "v10": field("v10"), "tp": field("tp"),
                "gust": field("gust"), "lat": dataset["latitude"].to_numpy().astype(float),
                "lon": np.where(lon > 180.0, lon - 360.0, lon),
                "valid_time": dataset["valid_time"].to_numpy().astype("datetime64[m]")}
