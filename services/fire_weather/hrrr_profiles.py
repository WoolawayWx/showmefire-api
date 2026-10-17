"""HRRR profiles over Missouri for the current hour and the fire-weather fields derived from them.

Each hour uses the previous hour's HRRR run at forecast hour 1 (f01): the analysis
(f00) is only published ~50 minutes after the hour it describes, so it is always
1-2 hours behind the clock, while the f01 valid at the current hour is out a few
minutes before that hour starts. The run has just assimilated the latest balloon,
aircraft and radar observations, so one hour in it stays very close to the analysis.

Herbie downloads only the GRIB messages that match each search (byte ranges
from the .idx file on AWS noaa-hrrr-bdp-pds), and every field is cropped to the
buffered Missouri box (core.domain.crop) right after decoding.
"""
from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import requests

from core.domain import crop
from services.fire_weather import indices as ix

AWS = "https://noaa-hrrr-bdp-pds.s3.amazonaws.com"
FORECAST_HOUR = 1
# 1000-650 mb every 25 mb: Missouri's ground sits at roughly 1000-950 mb and
# afternoon mixing rarely reaches above ~650 mb (~3.5 km).
PRESSURE_LEVELS = tuple(range(1000, 649, -25))
# Dewpoint is only used in the lowest 50 hPa above ground (the HDW layer; Missouri's surface pressure is never
# below ~930 hPa) and at 850 mb (Haines), so the levels above 850 mb are not downloaded. Each level is ~0.75 MB of
# full-CONUS GRIB, so this drops 8 of 75 messages. The dewpoint array is NaN above DEWPOINT_MIN_LEVEL_HPA.
DEWPOINT_MIN_LEVEL_HPA = 850
DEWPOINT_LEVELS = tuple(level for level in PRESSURE_LEVELS if level >= DEWPOINT_MIN_LEVEL_HPA)


def _levels_pattern(levels) -> str:
    return "(?:" + "|".join(str(level) for level in levels) + ") mb:"


PRS_SEARCH = (f":(?:TMP|UGRD|VGRD|HGT):{_levels_pattern(PRESSURE_LEVELS)}"
              f"|:DPT:{_levels_pattern(DEWPOINT_LEVELS)}")
SFC_SEARCHES = {
    "surface": ":(?:HPBL|PRES|HGT):surface:",
    "2m": ":(?:TMP|DPT):2 m above ground:",
    "10m": ":(?:UGRD|VGRD):10 m above ground:",
    "80m": ":(?:UGRD|VGRD):80 m above ground:",
}
HDW_LAYER_HPA = 50.0
# HRRR's Lambert conformal grid (used when the GRIB attributes are missing).
HRRR_CENTRAL_LON = -97.5
HRRR_STANDARD_PARALLEL = 38.5
HEADERS = {"User-Agent": "ShowMeFire/1.0 graphics@showmefire.org"}


def hour_key(valid: datetime) -> str:
    return valid.strftime("%Y%m%d%H")


def run_for(valid: datetime) -> datetime:
    """The HRRR run whose FORECAST_HOUR forecast is valid at `valid`."""
    return valid - timedelta(hours=FORECAST_HOUR)


def _idx_url(run: datetime, product: str) -> str:
    return f"{AWS}/hrrr.{run:%Y%m%d}/conus/hrrr.t{run:%H}z.wrf{product}f{FORECAST_HOUR:02d}.grib2.idx"


def available_hours(hours_back: int = 6, now: Optional[datetime] = None,
                    head: Callable[[str], int] | None = None) -> list[datetime]:
    """Valid hours from the last `hours_back` hours through the current or next
    hour whose prs AND sfc files are published (their .idx exists), oldest first."""
    now = (now or datetime.now(timezone.utc)).replace(minute=0, second=0, microsecond=0)
    head = head or (lambda url: requests.head(url, headers=HEADERS, timeout=20).status_code)
    hours = []
    for back in range(hours_back, -2, -1):  # -1: the next hour's file can appear just before it starts
        valid = now - timedelta(hours=back)
        if all(head(_idx_url(run_for(valid), product)) == 200 for product in ("prs", "sfc")):
            hours.append(valid)
    return hours


@dataclass
class Profiles:
    """Cropped HRRR fields valid at `valid_time`. Profile arrays are (level, y, x), levels
    bottom-up in PRESSURE_LEVELS order; NaN where a level is below ground."""
    valid_time: datetime
    lat: np.ndarray
    lon: np.ndarray
    pressure_hpa: np.ndarray        # (level,)
    height_agl_m: np.ndarray        # (level, y, x)
    temp_c: np.ndarray
    dewpoint_c: np.ndarray
    u: np.ndarray                   # grid-relative m/s
    v: np.ndarray
    raw_temp_c: np.ndarray          # unmasked (for the fixed-level Haines index)
    raw_dewpoint_c: np.ndarray
    surface_pressure_hpa: np.ndarray
    hpbl_m: np.ndarray
    t2_c: np.ndarray
    td2_c: np.ndarray
    u10: np.ndarray
    v10: np.ndarray
    u80: np.ndarray
    v80: np.ndarray
    central_longitude: float = HRRR_CENTRAL_LON
    standard_parallel: float = HRRR_STANDARD_PARALLEL


def _as_list(result):
    return result if isinstance(result, list) else [result]


def _variable(datasets, *names):
    for dataset in datasets:
        for name in names:
            if name in dataset.data_vars:
                return dataset[name]
    raise KeyError(f"none of {names} in HRRR subset")


def _projection_params(dataset) -> tuple[float, float]:
    try:
        attrs = dataset[next(iter(dataset.data_vars))].attrs
        return float(attrs["GRIB_LoVInDegrees"]) - 360.0 * (float(attrs["GRIB_LoVInDegrees"]) > 180), \
            float(attrs["GRIB_Latin1InDegrees"])
    except (KeyError, StopIteration, ValueError):
        return HRRR_CENTRAL_LON, HRRR_STANDARD_PARALLEL


def fetch_profiles(valid: datetime, workdir: Optional[Path] = None, herbie_factory=None,
                   run: Optional[datetime] = None, fxx: int = FORECAST_HOUR) -> Profiles:
    """Download and crop the pressure-level and surface fields valid at `valid`.

    By default that is the previous hour's run at f01. Pass `run` and `fxx` to read a
    specific forecast hour of a specific run instead (hour-by-hour forecast product)."""
    if herbie_factory is None:
        from herbie import Herbie as herbie_factory
    owned = workdir is None
    workdir = Path(workdir or tempfile.mkdtemp(prefix="smf-hrrr-fw-"))
    naive = (run or run_for(valid)).replace(tzinfo=None)
    try:
        def subset(product: str, search: str):
            herbie = herbie_factory(naive, model="hrrr", product=product, fxx=fxx, save_dir=str(workdir),
                                    verbose=False, priority=["aws"])
            return [crop(item) for item in _as_list(herbie.xarray(search, remove_grib=True))]

        prs = subset("prs", PRS_SEARCH)
        surface = {key: subset("sfc", search) for key, search in SFC_SEARCHES.items()}
    finally:
        if owned:
            shutil.rmtree(workdir, ignore_errors=True)

    # cfgrib returns one dataset per set of levels, so dewpoint (fewer levels) is usually its own dataset.
    levels_ds = next(dataset for dataset in prs if "t" in dataset.data_vars)
    pressures = np.sort(np.asarray(levels_ds["isobaricInhPa"].values, dtype=float))[::-1]  # 1000 mb (lowest) first

    def level_field(*names):
        # Reindex onto the full level axis: levels a field was not downloaded for become NaN.
        variable = _variable(prs, *names)
        return np.asarray(variable.reindex(isobaricInhPa=pressures).values, dtype=float)

    temp = level_field("t") - 273.15
    dewpoint = level_field("dpt") - 273.15
    u, v = level_field("u"), level_field("v")
    geopotential_height = level_field("gh")

    surface_pressure = np.asarray(_variable(surface["surface"], "sp").values, dtype=float) / 100.0
    orography = np.asarray(_variable(surface["surface"], "orog").values, dtype=float)
    hpbl = np.asarray(_variable(surface["surface"], "blh", "hpbl").values, dtype=float)

    # HRRR extrapolates pressure levels underground: drop any level whose
    # pressure exceeds the surface pressure, or whose height is not above ground.
    height_agl = geopotential_height - orography[None]
    below = (pressures[:, None, None] > surface_pressure[None]) | (height_agl <= 2.0)
    masked = lambda field: np.where(below, np.nan, field)  # noqa: E731

    central_lon, parallel = _projection_params(levels_ds)
    lon = np.asarray(levels_ds["longitude"].values, dtype=float)
    lon = np.where(lon > 180, lon - 360, lon)
    winds_80 = surface["80m"]
    return Profiles(
        valid_time=valid, lat=np.asarray(levels_ds["latitude"].values, dtype=float), lon=lon,
        pressure_hpa=pressures, height_agl_m=masked(height_agl), temp_c=masked(temp), dewpoint_c=masked(dewpoint),
        u=masked(u), v=masked(v), raw_temp_c=temp, raw_dewpoint_c=dewpoint,
        surface_pressure_hpa=surface_pressure, hpbl_m=hpbl,
        t2_c=np.asarray(_variable(surface["2m"], "t2m").values, dtype=float) - 273.15,
        td2_c=np.asarray(_variable(surface["2m"], "d2m").values, dtype=float) - 273.15,
        u10=np.asarray(_variable(surface["10m"], "u10").values, dtype=float),
        v10=np.asarray(_variable(surface["10m"], "v10").values, dtype=float),
        u80=np.asarray(_variable(winds_80, "u", "u80", "u100").values, dtype=float),
        v80=np.asarray(_variable(winds_80, "v", "v80", "v100").values, dtype=float),
        central_longitude=central_lon, standard_parallel=parallel,
    )


def _sorted_column(heights: np.ndarray, *fields: np.ndarray):
    """Sort stacked levels bottom-up by height in every column (NaN heights last)."""
    order = np.argsort(np.where(np.isfinite(heights), heights, np.inf), axis=0)
    return [np.take_along_axis(array, order, axis=0) for array in (heights, *fields)]


def derive_fields(profiles: Profiles) -> dict[str, np.ndarray]:
    """Every mapped/archived fire-weather field for one hour, on the cropped grid."""
    p = profiles
    shape = p.t2_c.shape
    two_m = np.full(shape, 2.0)

    # Temperature column: the 2 m value plus every above-ground pressure level.
    t_heights, t_col, td_col, p_col = _sorted_column(
        np.concatenate([two_m[None], p.height_agl_m]),
        np.concatenate([p.t2_c[None], p.temp_c]),
        np.concatenate([p.td2_c[None], p.dewpoint_c]),
        np.concatenate([p.surface_pressure_hpa[None],
                        np.where(np.isfinite(p.temp_c), p.pressure_hpa[:, None, None], np.nan)]),
    )
    # Wind column: 10 m and 80 m plus every above-ground pressure level.
    w_heights, u_col, v_col = _sorted_column(
        np.concatenate([np.full(shape, 10.0)[None], np.full(shape, 80.0)[None], p.height_agl_m]),
        np.concatenate([p.u10[None], p.u80[None], p.u]),
        np.concatenate([p.v10[None], p.v80[None], p.v]),
    )

    mixing_height = np.where(np.isfinite(p.hpbl_m), np.maximum(p.hpbl_m, 0.0), np.nan)
    theta = ix.potential_temperature_k(t_col, p_col)
    holzworth = ix.mixing_height_holzworth(t_heights[1:], theta[1:], theta[0])

    mean_u, mean_v = ix.transport_wind(w_heights, u_col, v_col, mixing_height)
    east_u, north_v = ix.earth_relative_winds(mean_u, mean_v, p.lon, p.central_longitude, p.standard_parallel)
    speed = ix.wind_speed(east_u, north_v)
    mixing_height_ft = mixing_height * ix.M_TO_FT
    ventilation = ix.ventilation_rate_kt_ft(speed * ix.MS_TO_KT, mixing_height_ft)

    inversion, inversion_top = ix.surface_inversion(t_heights, t_col)

    level = {int(value): index for index, value in enumerate(p.pressure_hpa)}
    haines = ix.haines_low(p.raw_temp_c[level[950]], p.raw_temp_c[level[850]], p.raw_dewpoint_c[level[850]])

    # HDW layer: the surface plus everything within 50 hPa above it.
    in_layer = p_col >= (p.surface_pressure_hpa - HDW_LAYER_HPA)[None]
    vpd = np.where(in_layer, ix.vapor_pressure_deficit_hpa(t_col, td_col), np.nan)
    wind_pressure = np.concatenate([p.surface_pressure_hpa[None], p.surface_pressure_hpa[None] - 1.0,
                                    np.where(np.isfinite(p.u), p.pressure_hpa[:, None, None], np.nan)])
    wind_levels = np.concatenate([ix.wind_speed(p.u10, p.v10)[None], ix.wind_speed(p.u80, p.v80)[None],
                                  ix.wind_speed(p.u, p.v)])
    wind_in_layer = np.where(wind_pressure >= (p.surface_pressure_hpa - HDW_LAYER_HPA)[None], wind_levels, np.nan)
    hdw = ix.hot_dry_windy(vpd, wind_in_layer)

    return {
        "lat": p.lat, "lon": p.lon,
        "mixing_height_ft": mixing_height_ft,
        "holzworth_ft": holzworth * ix.M_TO_FT,
        "transport_u_ms": east_u, "transport_v_ms": north_v,
        "transport_speed_mph": speed * ix.MS_TO_MPH,
        "transport_dir_deg": ix.wind_direction_from(east_u, north_v),
        "ventilation_kt_ft": ventilation,
        "ventilation_category": ix.ventilation_category(ventilation).astype(float),
        "inversion_strength_f": inversion * 1.8,
        "inversion_top_ft": inversion_top * ix.M_TO_FT,
        "lapse_rate_c_km": ix.lapse_rate_c_per_km(t_heights, t_col),
        "haines": haines,
        "hdw": hdw,
    }
