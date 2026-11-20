"""Day 2 and Day 3 fire danger forecast (beta).

Day 1 (DailyForecast.py) is a fast, good-looking forecast: HRRR -> fuel-moisture model -> the canonical fire-danger
rule -> six Missouri maps. forecast_v1 covers 72 hours too, but it pulls 31 GEFS members plus every field a blend
needs, so a run takes a very long time, and its Day 2/3 graphics do not look like Day 1's.

This job takes Day 1's logic and runs it on the next two days with only the fields it uses:

    Day 2  HRRR blended with RRFS  (fixed 70/30; 06z/12z/18z runs, e.g. f28-f39 of the 12z run: the 16z-03z afternoon,
                                    one day later)
    Day 3  RRFS  f52-f63   (12z run; 18z gives f46-f57. 13 km, so it is regridded to 3 km for drawing)

Each run also saves what every model said at the observation stations, and scores earlier runs against the
observations (forecast/extended_verify.py). The scores say what recent-performance blend weights would be; they are
reported, not applied.

It reuses, unchanged: the stable fuel-moisture model and its features (DailyForecast.predict_fm_grid), the wind
reduction factor, the fire-danger rule with its seasonal softening, the SNODAS snow mask, and the maps' house style
(forecast/map_style.py). Downloads reuse forecast_v1's indexed-GRIB acquisition, so they are the same byte-range reads
clipped to Missouri; only 5 fields are requested.

What it deliberately does NOT do (this is a beta that sits beside Day 1 and cannot disturb it):
  * no station_forecasts rows, v5/risk-fusion/shadow hooks, ensemble hand-off, GIS publication, or forecast_peak
    archive - none of those are written for days 2-3;
  * fuel moisture is not initialized from RAWS (Day 1's spatial model is); it runs the XGBoost model from the
    forecast's own temperature / RH / wind / rain history, started 24 hours before the window;
  * nothing is published to R2. Maps and a JSON summary go to the Testbed images folder.

Usage:  python forecast/ExtendedForecast.py [--day 2] [--day 3] [--cycle 2026-11-19T12] [--out-dir DIR]
                                            [--wait-minutes 90] [--no-render]
Exit status is 0 if at least one requested day was produced.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import xarray as xr

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
for _path in (str(PROJECT_DIR),):
    if _path not in sys.path:
        sys.path.append(_path)

logger = logging.getLogger("extended_forecast")

LOCAL_TZ = ZoneInfo("America/Chicago")
# Hours of model history fed to the fuel-moisture model before the first mapped hour: the model's own rolling
# features reach back 24 hours (precip_24h, hours_since_rain). One more lead is fetched so rain can be differenced.
WARMUP_HOURS = 24
DISPLAY_SPACING_M = 3000          # grid the maps are drawn on; coarser sources are interpolated up to it
SMOOTH_SIGMA = 1.5                # Day 1's summary-map smoothing, in 3 km cells
EXTENT = (-95.8, -89.1, 35.8, 40.8)
PIXEL_W, PIXEL_H, MAP_DPI = 2048, 1152, 144
# A 26 pt bold title is about 26 px per character at this size and ends at the right edge, so past ~44 characters its left end
# reaches the state's northeast corner. Longer titles break onto two lines (same subtitle drop the ensemble maps use).
TITLE_WRAP_CHARS = 44
TWO_LINE_SUBTITLE_Y = 0.84


class SourceNotReady(RuntimeError):
    """The model run does not yet publish every hour this day needs."""


@dataclass(frozen=True)
class DayConfig:
    day: int
    source: str           # "hrrr" | "rrfs"
    start: int            # first mapped forecast hour (inclusive)
    end: int              # last mapped forecast hour (inclusive)
    min_cycle_age_hours: float
    source_label: str
    max_lead: int = 48    # last forecast hour this model run publishes (HRRR's extended runs stop at f48)


# `start`/`end` here are the 12z run's hours. Other cycles shift them: see config_for().
# RRFS's longest lead is set conservatively (only f63 has been confirmed for the 12z run); raise
# SMF_EXTENDED_RRFS_MAX_LEAD once longer leads are confirmed and the 00z/06z runs can feed Day 3 as well.
DAYS = {
    2: DayConfig(2, "hrrr", 28, 39, float(os.getenv("SMF_EXTENDED_HRRR_MIN_AGE_HOURS", "4")), "HRRR Model Forecast (3 km)", 48),
    3: DayConfig(3, "rrfs", 52, 63, float(os.getenv("SMF_EXTENDED_RRFS_MIN_AGE_HOURS", "7")), "RRFS Model Forecast (13 km)",
                 int(os.getenv("SMF_EXTENDED_RRFS_MAX_LEAD", "63"))),
}
SYNOPTIC_HOURS = (0, 6, 12, 18)   # the runs HRRR extends past f18 and RRFS publishes in its 13 km product
WINDOW_START_UTC = 16             # the mapped window opens at 16z (10:00 CST) on the target date
# The maps show conditions in the PEAK_HOURS-long stretch of that window where the model's own danger is highest, chosen
# fresh for each run, not the extremes of the whole window. (The wider window is still modelled: it is the history the
# fuel-moisture model needs, and the rain map's 24 h.)
PEAK_HOURS = int(os.getenv("SMF_EXTENDED_PEAK_HOURS", "2"))
# Fixed HRRR share of the Day 2 blend. The recent-performance (adaptive) weights are only reported, not applied yet.
DAY2_HRRR_WEIGHT = float(os.getenv("SMF_EXTENDED_DAY2_HRRR_WEIGHT", "0.7"))

# One query per level group so a cfgrib problem with one group cannot drop the others. DPT is only the fallback for
# a product that has no RH message (forecast_v1's canonicalizer derives RH from it).
SEARCHES = (
    r":(?:TMP|RH|DPT):2 m above ground:",
    r":(?:UGRD|VGRD):10 m above ground:",
    r":GUST:(?:surface|10 m above ground):",
)


# ----------------------------------------------------------------------------------------------------------------
# Time helpers
# ----------------------------------------------------------------------------------------------------------------
def config_for(day: int, cycle_hour: int) -> DayConfig | None:
    """Day `day`'s window for a run at `cycle_hour` z, or None if that run cannot reach it.

    The window is the afternoon of (the run's UTC date + day - 1), 16z to 03z, so the hours shift with the cycle:
    Day 2 is f34-45 from 06z, f28-39 from 12z and f22-33 from 18z (00z would need f40-51, past HRRR's f48)."""
    base = DAYS[day]
    start = WINDOW_START_UTC + 24 * (day - 1) - cycle_hour
    end = start + (base.end - base.start)
    return None if start < 0 or end > base.max_lead else replace(base, start=start, end=end)


def cycle_hours(day: int) -> tuple[int, ...]:
    """The cycles (UTC hours) that can produce this day."""
    return tuple(hour for hour in SYNOPTIC_HOURS if config_for(day, hour) is not None)


def latest_cycle(now: datetime | None = None, *, min_age_hours: float = 4.0, hours: tuple[int, ...] = (12,)) -> datetime:
    """Newest cycle among `hours` (UTC) that is at least `min_age_hours` old (its last lead takes hours to publish)."""
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    for days_back in range(0, 3):
        day = (current - timedelta(days=days_back)).replace(minute=0, second=0, microsecond=0)
        for hour in sorted(hours, reverse=True):
            candidate = day.replace(hour=hour)
            if candidate + timedelta(hours=min_age_hours) <= current:
                return candidate
    raise ValueError("no cycle is old enough")


def choose_peak_block(risk: list, rh: list, inside: np.ndarray, hours: int = 2) -> int:
    """Index of the first hour of the `hours`-long stretch where Missouri's danger is highest.

    Each hourly step is scored by the state-wide mean danger category (0-4) over cells inside Missouri, with the
    state-wide mean RH as the tie-break (drier is higher), so an all-Low day still gets its driest stretch."""
    inside = np.asarray(inside, dtype=bool)
    scores = []
    for hourly_risk, hourly_rh in zip(risk, rh):
        usable = inside & np.isfinite(hourly_risk)
        mean_risk = float(np.mean(hourly_risk[usable])) if usable.any() else 0.0
        dry = inside & np.isfinite(hourly_rh)
        mean_rh = float(np.mean(hourly_rh[dry])) if dry.any() else 50.0
        scores.append(mean_risk * 1000.0 + (100.0 - mean_rh))
    hours = max(1, min(hours, len(scores)))
    totals = [sum(scores[i:i + hours]) for i in range(len(scores) - hours + 1)]
    return int(np.argmax(totals))


def peak_label(cycle: datetime, first_lead: int, hours: int = 2) -> str:
    """'3:00 PM-4:00 PM CT': the local valid times of the peak block's hourly steps."""
    first = (cycle + timedelta(hours=first_lead)).astimezone(LOCAL_TZ)
    last = (cycle + timedelta(hours=first_lead + hours - 1)).astimezone(LOCAL_TZ)
    clock = lambda t: f"{t:%I:%M %p}".lstrip("0")
    return f"{clock(first)}–{clock(last)} CT"


def missouri_mask(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Cells inside Missouri (all True if the boundary file cannot be read, so scoring still works)."""
    try:
        import geopandas as gpd
        import shapely
        border = gpd.read_file(PROJECT_DIR / "maps/shapefiles/MO_State_Boundary/MO_State_Boundary.shp").to_crs("EPSG:4326")
        return np.asarray(shapely.contains_xy(border.geometry.iloc[0].buffer(0.01), lon, lat), dtype=bool)
    except Exception as error:
        logger.warning("Missouri boundary unavailable for peak-hour selection (%s); using the whole grid", error)
        return np.ones(lon.shape, dtype=bool)


def window_label(cycle: datetime, cfg: DayConfig) -> tuple[str, datetime]:
    """('10:00-21:00 CT', local start time) for the mapped hours, honouring daylight saving."""
    start = (cycle + timedelta(hours=cfg.start)).astimezone(LOCAL_TZ)
    end = (cycle + timedelta(hours=cfg.end)).astimezone(LOCAL_TZ)
    return f"{start:%H}:00–{end:%H}:00 CT", start


# ----------------------------------------------------------------------------------------------------------------
# Day 1 logic, imported lazily so the pure helpers above/below stay importable without loading the model
# ----------------------------------------------------------------------------------------------------------------
def _day1() -> SimpleNamespace:
    """Day 1's own functions and constants. Importing DailyForecast loads the stable fuel-moisture model."""
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))     # DailyForecast does `from export_fire_danger_gis import ...`
    os.environ.setdefault("MODEL_SHADOW_ENABLED", "false")
    from forecast import DailyForecast as df
    from services import model_shadow
    # Beta-vs-stable shadow scoring is calibrated on Day 1 leads; extended-range rows would only dilute it.
    model_shadow._state["enabled"] = False
    from core.fire_danger import RULE_SPEC
    return SimpleNamespace(
        predict_fm_grid=df.predict_fm_grid,
        calculate_fire_danger=df.calculate_fire_danger,
        validate_temperature=df.validate_and_convert_temperature,
        validate_rh=df.validate_relative_humidity,
        get_snodas_swe_grid=df.get_snodas_swe_grid,
        interpolate_swe_to_grid=df.interpolate_swe_to_grid,
        fm_model=df.FM_MODEL,
        precip_contract=df.PRECIPITATION_CONTRACT_VERSION,
        wind_factor=df.HRRR_FIRE_WIND_REDUCTION_FACTOR,
        smoothing=df.APPLY_HOURLY_RISK_SMOOTHING,
        smoothing_sigma=df.HOURLY_RISK_SMOOTHING_SIGMA,
        snow_threshold_in=df.SNOW_THRESHOLD_IN,
        low_fm=float(RULE_SPEC["thresholds"]["low_fm"]),
    )


# ----------------------------------------------------------------------------------------------------------------
# Acquisition
# ----------------------------------------------------------------------------------------------------------------
def _spec(cfg: DayConfig, leads: tuple[int, ...]):
    from forecast_v1.acquisition import APCP_RUN_TOTAL_SEARCH, AcquisitionSpec
    searches = (*SEARCHES, APCP_RUN_TOTAL_SEARCH)
    if cfg.source == "hrrr":
        return AcquisitionSpec("hrrr", "hrrr", "sfc", leads, (None,), required=True, searches=searches,
                               upper_air_on_main=False), None
    product = os.getenv("SMF_RRFS_PRODUCT", "2dfld.13km")
    domain = os.getenv("SMF_RRFS_DOMAIN", "conus") or None
    return AcquisitionSpec("rrfs", "rrfs", product, leads, ("control",), domain=domain, searches=searches,
                           upper_air_on_main=False), "control"


def source_leads(cfg: DayConfig) -> tuple[int, ...]:
    """Every lead from one before the warm-up (rain is a run total, so the first lead only anchors the difference).

    A run that opens the window early (18z Day 2 starts at f22) has fewer than 24 hours before it; the warm-up is
    then cut at f00 and the run is flagged `short_warmup`."""
    return tuple(range(max(0, cfg.start - WARMUP_HOURS - 1), cfg.end + 1))


def precip_name(ds: xr.Dataset) -> str | None:
    """The run-total rain variable: cfgrib names it `tp` for HRRR; forecast_v1 also accepts `apcp`."""
    return next((name for name in ("tp", "apcp") if name in ds), None)


def _validate_source(ds: xr.Dataset, cfg: DayConfig, cycle: datetime) -> None:
    # Rain drives the fuel-moisture model, and HRRR/RRFS always publish it, so a missing field means a failed download.
    rain = precip_name(ds)
    missing = [name for name in ("t2m", "r2", "u10", "v10") if name not in ds] + ([] if rain else ["rain (tp)"])
    if missing:
        raise SourceNotReady(f"{cfg.source} is missing fields: {', '.join(missing)}")
    # A query that fails part-way (network error, unpublished hour) leaves that hour empty rather than raising.
    for name in ("t2m", "r2", "u10", "v10", rain):
        empty = [int(t) for t, hour in zip(range(ds.sizes["time"]), ds[name].values) if not np.isfinite(hour).any()]
        if empty:
            raise SourceNotReady(f"{cfg.source} {name} is empty for {len(empty)} of {ds.sizes['time']} hours")
    expected = [np.datetime64((cycle + timedelta(hours=h)).replace(tzinfo=None), "ns") for h in source_leads(cfg)]
    have = {np.datetime64(t, "ns") for t in ds.time.values}
    absent = [h for h, t in zip(source_leads(cfg), expected) if t not in have]
    if absent:
        raise SourceNotReady(f"{cfg.source} {cycle:%Y-%m-%d %HZ} has not published f{absent[0]:03d}"
                             f"{'' if len(absent) == 1 else f' and {len(absent) - 1} more hours'}")


def load_source(cfg: DayConfig, cycle: datetime, cache_dir: Path) -> xr.Dataset:
    """Clipped, native-grid fields for the warm-up plus mapped hours; cached so a re-run is instant."""
    leads = source_leads(cfg)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{cfg.source}_{cycle:%Y%m%d_%H}z_f{leads[0]:03d}-{leads[-1]:03d}.nc"
    if path.exists():
        try:
            with xr.open_dataset(path) as cached:
                ds = cached.load()
            _validate_source(ds, cfg, cycle)
            logger.info("Day %d: using cached %s", cfg.day, path.name)
            return ds
        except Exception as error:
            logger.warning("Day %d: discarding unusable cache %s (%s)", cfg.day, path.name, error)
            path.unlink(missing_ok=True)

    from forecast_v1.acquisition import acquire_fields
    spec, member = _spec(cfg, leads)
    logger.info("Day %d: downloading %s %s f%03d-f%03d (%d hours)", cfg.day, cfg.source.upper(),
                f"{cycle:%Y-%m-%d %HZ}", leads[0], leads[-1], len(leads))
    started = time.time()
    ds = acquire_fields(spec, cycle, member, cache_dir / "grib")
    _validate_source(ds, cfg, cycle)
    logger.info("Day %d: download took %.0f s", cfg.day, time.time() - started)
    try:
        for name in [*ds.data_vars, *ds.coords]:
            ds[name].attrs.pop("GRIB_missingValue", None)
        ds = ds.drop_vars([n for n in ("gribfile_projection",) if n in ds.variables])
        ds.to_netcdf(path, engine="netcdf4")
    except Exception as error:               # a cache is an optimisation only
        logger.warning("Day %d: could not cache %s (%s)", cfg.day, path.name, error)
        path.unlink(missing_ok=True)
    return ds


def load_source_waiting(cfg: DayConfig, cycle: datetime, cache_dir: Path, wait_minutes: float) -> xr.Dataset:
    """load_source, retrying every 5 minutes while the model run is still publishing, up to `wait_minutes`."""
    deadline = time.time() + wait_minutes * 60
    while True:
        try:
            return load_source(cfg, cycle, cache_dir)
        except Exception as error:
            if time.time() >= deadline:
                raise
            logger.info("Day %d: source not ready (%s); retrying in 5 minutes", cfg.day, str(error)[:200])
            time.sleep(300)


# ----------------------------------------------------------------------------------------------------------------
# Day 2 blend: HRRR (3 km) mixed with RRFS (13 km) before the fuel-moisture model and rule run once
# ----------------------------------------------------------------------------------------------------------------
def _frac_index(coord: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Fractional array index of `values` along a monotonic coordinate; NaN outside its range."""
    index = np.arange(len(coord), dtype="float64")
    if coord[0] > coord[-1]:
        coord, index = coord[::-1], index[::-1]
    out = np.interp(values, coord, index)
    out[(values < coord.min()) | (values > coord.max())] = np.nan
    return out


def regrid_like(partner: xr.Dataset, names: list[str], lon: np.ndarray, lat: np.ndarray) -> dict[str, np.ndarray]:
    """Bilinear (time, y, x) fields of `partner` at the given lon/lat points. NaN outside its domain."""
    import pyproj
    from scipy.ndimage import map_coordinates
    px, py = pyproj.Transformer.from_crs("EPSG:4326", partner.attrs["crs"], always_xy=True).transform(lon, lat)
    fx = _frac_index(np.asarray(partner.x.values, dtype="float64"), np.asarray(px, dtype="float64"))
    fy = _frac_index(np.asarray(partner.y.values, dtype="float64"), np.asarray(py, dtype="float64"))
    outside = ~(np.isfinite(fx) & np.isfinite(fy))
    coords = np.array([np.where(outside, 0.0, fy), np.where(outside, 0.0, fx)])
    out = {}
    for name in names:
        source = np.asarray(partner[name].values, dtype="float64")
        grid = np.empty((source.shape[0], *lon.shape))
        for t in range(source.shape[0]):
            grid[t] = map_coordinates(source[t], coords, order=1, mode="nearest")
            grid[t][outside] = np.nan
        out[name] = grid
    return out


def blend_sources(primary: xr.Dataset, partner: xr.Dataset, weight: float) -> xr.Dataset:
    """`primary`'s grid with temperature, RH, wind speed and run-total rain mixed with `partner`'s.

    weight is the primary's share. Wind is mixed as SPEED (averaging u and v components would shrink it wherever the
    two disagree on direction) and stored as u10 = speed, v10 = 0, which is all the model and rule use. Where the
    partner has no value (outside its domain, or a bad cell) the primary is kept unchanged."""
    lon, lat = native_lonlat(primary)
    partner = partner.sel(time=primary.time)
    rain_p, rain_q = precip_name(primary), precip_name(partner)
    names = ["t2m", "r2", "u10", "v10", *([rain_q] if rain_q else [])]
    other = regrid_like(partner, names, lon, lat)

    def mix(a, b):
        a = np.asarray(a, dtype="float64")
        return np.where(np.isfinite(b), weight * a + (1.0 - weight) * b, a)

    out = primary.copy(deep=True)
    for name in ("t2m", "r2"):
        out[name] = (out[name].dims, mix(primary[name].values, other[name]), out[name].attrs)
    speed = mix(np.hypot(primary.u10.values, primary.v10.values), np.hypot(other["u10"], other["v10"]))
    out["u10"] = (out.u10.dims, speed, out.u10.attrs)
    out["v10"] = (out.v10.dims, np.zeros_like(speed), out.v10.attrs)
    if rain_p and rain_q:
        out[rain_p] = (out[rain_p].dims, mix(primary[rain_p].values, other[rain_q]), out[rain_p].attrs)
    return out


def load_rrfs_partner(cfg: DayConfig, cycle: datetime, cache_dir: Path, wait_minutes: float) -> xr.Dataset | None:
    """The same run's RRFS fields for this day's hours, or None (the day is then HRRR alone and says so)."""
    partner_cfg = replace(cfg, source="rrfs", source_label="RRFS Model Forecast (13 km)")
    try:
        return load_source_waiting(partner_cfg, cycle, cache_dir, wait_minutes)
    except Exception as error:
        logger.warning("Day %d: RRFS is not available for the blend (%s); using HRRR alone", cfg.day, str(error)[:200])
        return None


# ----------------------------------------------------------------------------------------------------------------
# Hour-by-hour calculation (Day 1's logic)
# ----------------------------------------------------------------------------------------------------------------
def precip_increments(total: np.ndarray) -> np.ndarray:
    """Hourly rain (mm) from a run-total series. The first lead has no predecessor so it is 0.

    A drop larger than GRIB packing noise (0.1 mm) means the bucket reset, so that hour's value is the new total."""
    total = np.nan_to_num(np.asarray(total, dtype="float64"), nan=0.0)
    increments = np.zeros_like(total)
    diffs = total[1:] - total[:-1]
    increments[1:] = np.where(diffs < -0.1, total[1:], np.clip(diffs, 0.0, None))
    return increments


def danger_grid(fm, rh, wind_kts, snow_mask, *, calculate, low_fm: float) -> np.ndarray:
    """Per-cell fire danger exactly as Day 1 computes it, skipping the cells the rule must call Low.

    Day 1 loops over every cell in Python; a cell whose fuel moisture is at or above the Low threshold is Low under the
    canonical rule (and the seasonal softening never raises a category), so only the drier cells are evaluated."""
    risk = np.zeros(fm.shape, dtype="float64")
    finite = np.isfinite(fm) & np.isfinite(rh) & np.isfinite(wind_kts)
    risk[~finite] = np.nan
    candidates = finite & ~snow_mask & (fm < low_fm)
    for i, j in zip(*np.nonzero(candidates)):
        value = calculate(fm[i, j], rh[i, j], wind_kts[i, j])
        risk[i, j] = np.nan if value is None else value
    return risk


def _smooth_risk(risk: np.ndarray, snow_mask: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    valid = np.isfinite(risk)
    weight = gaussian_filter(valid.astype(float), sigma=sigma)
    smoothed = gaussian_filter(np.nan_to_num(risk), sigma=sigma) / np.maximum(weight, 1e-9)
    out = np.where(valid, np.clip(np.rint(smoothed), 0, 4), np.nan)
    out[snow_mask] = 0
    return out


def load_swe(day1, cycle: datetime, lon: np.ndarray, lat: np.ndarray):
    """SNODAS snow water equivalent on the model grid, as Day 1 does; today's file, else yesterday's. None if unavailable."""
    bbox = (-95.8, 35.8, -89.1, 40.8)
    for days_back in (0, 1):
        date = pd.Timestamp(cycle.replace(tzinfo=None) - timedelta(days=days_back))
        data, swe_lon, swe_lat = day1.get_snodas_swe_grid(bbox, date)
        if data is not None:
            grid = day1.interpolate_swe_to_grid(data, swe_lon, swe_lat, lon, lat)
            if grid is not None:
                return grid
    logger.warning("SNODAS snow cover unavailable; no snow mask for this run")
    return None


def native_lonlat(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    lon = np.asarray(ds.longitude.values, dtype="float64")
    lat = np.asarray(ds.latitude.values, dtype="float64")
    return np.where(lon > 180, lon - 360, lon), lat


def run_hours(ds: xr.Dataset, cfg: DayConfig, cycle: datetime, day1) -> dict:
    """Walk every hour (warm-up included, in order) building the model's history, and score the mapped hours."""
    lon, lat = native_lonlat(ds)
    swe = load_swe(day1, cycle, lon, lat)
    contract_ok = day1.fm_model.attr("precipitation_contract_version") == day1.precip_contract

    base = pd.Timestamp(cycle.replace(tzinfo=None))
    times = pd.DatetimeIndex(ds.time.values)
    leads = [int(round((t - base) / pd.Timedelta(hours=1))) for t in times]
    order = np.argsort(leads)

    rain = precip_name(ds)
    has_precip = rain is not None
    total_all = ds[rain].values.astype("float64") if has_precip else np.zeros(ds.t2m.shape)
    increments_all = precip_increments(total_all)

    dx = abs(float(ds.x.values[1] - ds.x.values[0]))
    hourly_sigma = day1.smoothing_sigma if dx <= 1.25 * DISPLAY_SPACING_M else 0.0

    t_hist, rh_hist, p_hist = [], [], []
    mapped = {"risk": [], "fm": [], "rh": [], "wind": [], "temp": [], "precip": []}
    snow_mask = np.zeros(lon.shape, dtype=bool)
    for position in order:
        lead = leads[position]
        temp = day1.validate_temperature(ds.t2m.values[position].astype("float64"), expected_unit="kelvin", source=cfg.source)
        rh = day1.validate_rh(ds.r2.values[position].astype("float64"), source=cfg.source)
        u = ds.u10.values[position].astype("float64")
        v = ds.v10.values[position].astype("float64")
        ws_ms = np.hypot(u, v)
        ws_kts = ws_ms * day1.wind_factor * 1.94384
        precip = increments_all[position]
        model_precip = precip if contract_ok else total_all[position]

        if cfg.start <= lead <= cfg.end:
            valid = base + pd.Timedelta(hours=lead)
            fm, snow_mask, _ = day1.predict_fm_grid(
                temp, rh, ws_ms, valid.hour, valid.month, t_hist, rh_hist, p_hist,
                swe_grid=swe, day_of_year=valid.dayofyear,
            )
            risk = danger_grid(fm, rh, ws_kts, snow_mask, calculate=day1.calculate_fire_danger, low_fm=day1.low_fm)
            if day1.smoothing and hourly_sigma > 0:
                risk = _smooth_risk(risk, snow_mask, hourly_sigma)
            mapped["risk"].append(risk)
            mapped["fm"].append(fm)
            mapped["rh"].append(rh)
            mapped["wind"].append(ws_kts)
            mapped["temp"].append(temp)
            mapped["precip"].append(precip)
            logger.info("Day %d f%03d  FM %.1f-%.1f%%  peak category %s", cfg.day, lead, np.nanmin(fm), np.nanmax(fm),
                        int(np.nanmax(risk)) if np.isfinite(risk).any() else "n/a")
        t_hist.append(temp)
        rh_hist.append(rh)
        p_hist.append(model_precip)
        if len(t_hist) > WARMUP_HOURS:
            t_hist.pop(0), rh_hist.pop(0), p_hist.pop(0)

    if len(mapped["risk"]) != cfg.end - cfg.start + 1:
        raise SourceNotReady(f"Day {cfg.day}: {len(mapped['risk'])} of {cfg.end - cfg.start + 1} mapped hours were produced")

    last24 = np.sum(p_hist[-WARMUP_HOURS:], axis=0) if contract_ok else np.nanmax(np.stack(mapped["precip"]), axis=0)
    # Conditions are summarised over the PEAK_HOURS where Missouri's danger is highest, not the whole window.
    first = choose_peak_block(mapped["risk"], mapped["rh"], missouri_mask(lon, lat), PEAK_HOURS)
    last = first + max(1, min(PEAK_HOURS, len(mapped["risk"]))) - 1
    stack = {name: np.stack(values[first:last + 1]) for name, values in mapped.items()}
    first_lead, last_lead = cfg.start + first, cfg.start + last
    logger.info("Day %d peak block: f%03d-f%03d (%s)", cfg.day, first_lead, last_lead, peak_label(cycle, first_lead, last - first + 1))
    return {
        "peak_risk": np.nanmax(stack["risk"], axis=0),
        "min_fm": np.nanmin(stack["fm"], axis=0),
        "min_rh": np.nanmin(stack["rh"], axis=0),
        "max_wind": np.nanmax(stack["wind"], axis=0),
        "max_temp": np.nanmax(stack["temp"], axis=0),
        "precip_mm": last24 if has_precip else None,
        "snow_mask": snow_mask,
        "lon": lon, "lat": lat,
        "peak_block": {"first_lead": first_lead, "last_lead": last_lead, "hours": last - first + 1,
                       "local": peak_label(cycle, first_lead, last - first + 1)},
        "flags": [flag for flag, bad in (("precipitation_missing", not has_precip), ("snow_cover_missing", swe is None),
                                         ("short_warmup", cfg.start - min(leads) - 1 < WARMUP_HOURS)) if bad],
    }


# ----------------------------------------------------------------------------------------------------------------
# Drawing grid: interpolate coarse sources up to 3 km, smooth like Day 1, crop and mask to Missouri
# ----------------------------------------------------------------------------------------------------------------
EDGE_PAD_CELLS = 2                # coarse cells of edge values added around a coarse source before it is interpolated up


def _extend_axis(coord: np.ndarray, cells: int) -> np.ndarray:
    """An ascending, evenly spaced coordinate extended by `cells` steps at each end."""
    step = float(coord[1] - coord[0])
    return np.concatenate([coord[0] - step * np.arange(cells, 0, -1), coord, coord[-1] + step * np.arange(1, cells + 1)])


def to_display_grid(fields: dict[str, np.ndarray], ds: xr.Dataset, lon: np.ndarray, lat: np.ndarray):
    """Return (fields, lon, lat) on a grid no coarser than 3 km. A 13 km source drawn as-is gives blocky contours."""
    x = np.asarray(ds.x.values, dtype="float64")
    y = np.asarray(ds.y.values, dtype="float64")
    if abs(x[1] - x[0]) <= 1.25 * DISPLAY_SPACING_M:
        return fields, lon, lat
    import pyproj
    # Linear interpolation only covers the area between the outermost cell CENTRES, and the source was clipped close to
    # the state, so on its own the data stopped short of Missouri's border (a ragged edge the width of a coarse cell). The
    # grid is extended by EDGE_PAD_CELLS cells, repeating the edge values, so the data reaches past the border.
    order_y, order_x = np.argsort(y), np.argsort(x)
    padded_x, padded_y = _extend_axis(x[order_x], EDGE_PAD_CELLS), _extend_axis(y[order_y], EDGE_PAD_CELLS)
    new_x = np.arange(padded_x.min(), padded_x.max() + 1, DISPLAY_SPACING_M)
    new_y = np.arange(padded_y.min(), padded_y.max() + 1, DISPLAY_SPACING_M)
    out = {}
    for name, grid in fields.items():
        if grid is None:
            out[name] = None
            continue
        values = np.pad(np.asarray(grid, dtype="float64")[np.ix_(order_y, order_x)], EDGE_PAD_CELLS, mode="edge")
        array = xr.DataArray(values, dims=("y", "x"), coords={"y": padded_y, "x": padded_x})
        out[name] = array.interp(x=new_x, y=new_y, method="linear").values
    new_lon, new_lat = pyproj.Transformer.from_crs(ds.attrs["crs"], "EPSG:4326", always_xy=True).transform(
        *np.meshgrid(new_x, new_y))
    return out, new_lon, new_lat


def prepare_maps(result: dict, ds: xr.Dataset) -> dict:
    """Smooth, crop and Missouri-mask the window summaries the way Day 1 does (peak risk gets the weighted smoother)."""
    from scipy.ndimage import gaussian_filter
    import geopandas as gpd
    import shapely

    fields = {k: result[k] for k in ("peak_risk", "min_fm", "min_rh", "max_wind", "max_temp", "precip_mm")}
    fields["snow_mask"] = result["snow_mask"].astype(float)
    fields, lon, lat = to_display_grid(fields, ds, result["lon"], result["lat"])

    peak = fields["peak_risk"]
    valid = np.isfinite(peak)
    weight = gaussian_filter(valid.astype(float), sigma=SMOOTH_SIGMA)
    smooth = {"peak_risk": np.where(valid, gaussian_filter(np.nan_to_num(peak), sigma=SMOOTH_SIGMA) / np.maximum(weight, 1e-9), np.nan)}
    for name in ("min_fm", "min_rh", "max_wind", "max_temp"):
        smooth[name] = gaussian_filter(np.nan_to_num(fields[name], nan=np.nanmean(fields[name])), sigma=SMOOTH_SIGMA)
    if fields["precip_mm"] is not None:
        smooth["precip_in"] = gaussian_filter(np.nan_to_num(fields["precip_mm"]), sigma=0.2) / 25.4

    keep = ((lon >= EXTENT[0] - 0.5) & (lon <= EXTENT[1] + 0.5) & (lat >= EXTENT[2] - 0.5) & (lat <= EXTENT[3] + 0.5))
    rows, cols = np.where(np.any(keep, axis=1))[0], np.where(np.any(keep, axis=0))[0]
    window = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
    lon, lat = lon[window], lat[window]
    smooth = {name: grid[window] for name, grid in smooth.items()}

    border = gpd.read_file(PROJECT_DIR / "maps/shapefiles/MO_State_Boundary/MO_State_Boundary.shp").to_crs("EPSG:4326")
    inside = shapely.contains_xy(border.geometry.iloc[0].buffer(0.01), lon, lat)
    smooth = {name: np.where(inside, grid, np.nan) for name, grid in smooth.items()}
    smooth["lon"], smooth["lat"], smooth["inside"] = lon, lat, inside
    smooth["peak_label"] = (result.get("peak_block") or {}).get("local")
    return smooth


# ----------------------------------------------------------------------------------------------------------------
# Drawing (Day 1's six maps, same palettes / legends / branding)
# ----------------------------------------------------------------------------------------------------------------
def _fm_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    stops = [(0.0, "#4D0000"), (0.1, "#8B0000"), (0.2, "#DC143C"), (0.233, "#FF4500"), (0.267, "#FF6347"),
             (0.30, "#FF8C00"), (0.333, "#FFA500"), (0.367, "#FFB347"), (0.40, "#FFD700"), (0.50, "#FFED4E"),
             (0.60, "#F0E68C"), (0.70, "#C8E6C9"), (0.80, "#81C784"), (0.90, "#4CAF50"), (1.0, "#2E7D32")]
    return LinearSegmentedColormap.from_list("fm_enhanced", stops, N=512)


def map_title(day: int, title: str) -> tuple[str, bool]:
    """('Missouri Day N <title>', False), or the same broken after the day number (and True) when it is too long to sit
    right-aligned without running left over the state."""
    full = f"Missouri Day {day} {title}"
    if len(full) <= TITLE_WRAP_CHARS:
        return full, False
    return f"Missouri Day {day}\n{title}", True


def render_maps(grids: dict, cfg: DayConfig, cycle: datetime, out_dir: Path) -> list[Path]:
    """Draw the Day N maps; returns the PNG paths. A map that fails to draw is logged and skipped."""
    import cartopy.crs as ccrs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from core.temperature_palette import load_wsi_temperature_palette
    from forecast.map_style import add_boundaries, add_title_and_branding, create_base_map

    data_crs = ccrs.PlateCarree()
    map_crs = ccrs.LambertConformal(central_longitude=-92.45, central_latitude=38.3)
    lon, lat = grids["lon"], grids["lat"]
    run_stamp = pd.Timestamp(cycle.replace(tzinfo=None))
    window_hours, start_local = window_label(cycle, cfg)
    hours = grids.get("peak_label") or window_hours                                          # what the map descriptions quote
    subtitle = f"Day {cfg.day} (BETA) | Model Run: {cycle:%Y-%m-%d %HZ} | Valid: {start_local:%Y-%m-%d}"
    # On its own line: added to the line above it, the subtitle ran left over the state's northeast corner.
    peak_line = f"\nPeak {PEAK_HOURS} hours: {hours}" if grids.get("peak_label") else ""
    source = (f"Data Source: {cfg.source_label}, f{cfg.start}–f{cfg.end} | ShowMeFire ML Model\n"
              "BETA: not started from current observations; skill falls with lead time.\n"
              "For More Info, Visit ShowMeFire.org")
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def draw(key: str, title: str, field: str, colorbar_label: str, description: str, *, levels, cmap, norm=None,
             ticks=None, ticklabels=None, alpha=0.7, extend="neither", lines=None, transform=None, dotted=None,
             show_peak=True) -> None:
        data = grids[field]
        if transform:
            data = transform(data)
        fig, ax = create_base_map(EXTENT, map_crs, data_crs, PIXEL_W, PIXEL_H, MAP_DPI)
        cs = ax.contourf(lon, lat, data, transform=data_crs, levels=levels, cmap=cmap, norm=norm, alpha=alpha,
                         zorder=7, antialiased=True, extend=extend)
        if lines is not None:
            ax.contour(lon, lat, data, transform=data_crs, levels=lines, colors="black", linewidths=0.3, alpha=0.2, zorder=8)
        if dotted is not None:
            labelled = ax.contour(lon, lat, data, levels=[dotted], colors="black", linestyles="dotted", linewidths=2,
                                  transform=data_crs, zorder=8, alpha=0.7)
            ax.clabel(labelled, inline=True, fontsize=9, fmt="%g%%", inline_spacing=10)
        add_boundaries(ax, data_crs, PROJECT_DIR)
        cax = fig.add_axes([0.02, 0.08, 0.02, 0.6])
        cbar = plt.colorbar(cs, cax=cax, label=colorbar_label)
        if ticks is not None:
            cbar.set_ticks(ticks)
        if ticklabels is not None:
            cbar.set_ticklabels(ticklabels)
        ax.set_anchor("W")
        plt.subplots_adjust(left=0.05)
        # The title is right-aligned, so a long one runs left over the state's northeast corner. Break it after the day
        # number instead of shrinking it or moving the map; the subtitle drops to clear the second line.
        full_title, wrapped = map_title(cfg.day, title)
        add_title_and_branding(fig, full_title, subtitle + (peak_line if show_peak else ""), f"{description}\n\n{source}",
                               run_stamp, SCRIPT_DIR,
                               subtitle_y=TWO_LINE_SUBTITLE_Y if wrapped else 0.90)
        path = out_dir / f"mo-forecast-day{cfg.day}-{key}.png"
        fig.savefig(path, dpi=MAP_DPI, bbox_inches=None, pad_inches=0)
        plt.close(fig)
        written.append(path)
        gc.collect()

    jobs = []
    jobs.append(lambda: draw(
        "firedanger", "Peak Fire Danger Forecast", "peak_risk", "Fire Danger Level",
        f"Peak Fire Danger Forecast ({hours})\n\nFire Danger Criteria:\nLow:  FM ≥ 15% (fuels too wet to spread significantly)\n\n"
        "Moderate:  FM < 15% AND (RH < 45% OR Wind ≥ 10 kts)\n\n"
        "Elevated:  FM < 9% WITH (RH < 35% and Wind >= 12) or (RH < 25% and Wind >= 5)\n"
        "Critical:  FM < 9% WITH (RH < 25% AND Wind >= 15 kts)\n\n"
        "Extreme:  FM < 7% WITH (RH < 20% AND Wind >= 25 kts)",
        levels=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
        cmap=ListedColormap(["#90EE90", "#FFED4E", "#FFA500", "#FF0000", "#8B0000"]),
        norm=BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], 5), lines=[0.5, 1.5, 2.5, 3.5],
        ticks=[0, 1, 2, 3, 4], ticklabels=["Low", "Moderate", "Elevated \nHigh", "Critical \n Very High", "Extreme"]))
    fm_bounds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 21, 24, 27, 30]
    fm_map = _fm_cmap()
    jobs.append(lambda: draw(
        "fuelmoisture", "Minimum Fuel Moisture Forecast", "min_fm", "Fuel Moisture (%)",
        f"Minimum 10-Hour Fuel Moisture ({hours})\n\nCritical Thresholds:\n< 7%: Extremely dry — potential for extreme fire behavior\n"
        "7–9%: Very dry — critical fire behavior likely\n9–15%: Dry — elevated fire behavior expected\n"
        "15–20%: Moderate — some fire activity possible\n> 20%: Moist — fuels much less receptive to fire\n\n"
        "Areas with snow cover will have substantially higher fuel moisture\nand greatly reduced ignition potential.",
        levels=fm_bounds, cmap=fm_map, norm=mcolors.BoundaryNorm(fm_bounds, fm_map.N), alpha=0.75, extend="both",
        ticks=[0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30], dotted=9))
    rh_levels = [0, 15, 25, 35, 50, 65, 100]
    jobs.append(lambda: draw(
        "minrh", "Minimum Relative Humidity Forecast", "min_rh", "Relative Humidity (%)",
        f"Minimum Relative Humidity ({hours})\n\nCritical Thresholds:\n< 15%: Extremely Dry - Critical fire conditions\n"
        "15-25%: Very Dry - Red Flag criteria with wind\n25-35%: Dry - Elevated fire danger possible\n"
        "35-50%: Moderate - Normal fire activity\n> 50%: Moist - Fire spread limited",
        levels=rh_levels, cmap=ListedColormap(["#8B0000", "#FF0000", "#FFA500", "#FFED4E", "#90EE90", "#228B22"]),
        norm=BoundaryNorm(rh_levels, 6), lines=rh_levels[1:-1], ticks=[7.5, 20, 30, 42.5, 57.5, 82.5],
        ticklabels=["<15%", "15-25%", "25-35%", "35-50%", "50-65%", ">65%"]))
    wind_levels = [0, 10, 15, 20, 25, 30, 50]
    jobs.append(lambda: draw(
        "maxwind", "Maximum Wind Speed Forecast", "max_wind", "Wind Speed (knots)",
        f"Maximum Sustained Wind Speed ({hours})\n\nCritical Thresholds:\n< 10 kts: Light winds - Normal fire behavior\n"
        "10-15 kts: Moderate - Increased fire activity\n15-20 kts: Strong - Red Flag criteria with low RH\n"
        "20-25 kts: Very Strong - Critical fire weather\n> 25 kts: Extreme - Dangerous fire conditions",
        levels=wind_levels, cmap=ListedColormap(["#90EE90", "#FFED4E", "#FFA500", "#FF6347", "#FF0000", "#8B0000"]),
        norm=BoundaryNorm(wind_levels, 6), lines=wind_levels[1:-1], ticks=[5, 12.5, 17.5, 22.5, 27.5, 40],
        ticklabels=["<10 kt\n(<12 mph)", "10-15 kt\n(12-17 mph)", "15-20 kt\n(17-23 mph)", "20-25 kt\n(23-29 mph)",
                    "25-30 kt\n(29-35 mph)", ">30 kt\n(>35 mph)"]))
    temp_cmap, temp_levels_f = load_wsi_temperature_palette()
    jobs.append(lambda: draw(
        "maxtemp", "Maximum Temperature Forecast", "max_temp", "Temperature (°F)",
        f"Maximum Temperature ({hours})\n\nTemperature influences fire behavior:\nHigher temperatures increase fuel dryness\n"
        "and fire spread rates.\n\nCombined with low humidity and wind,\nhigh temperatures create dangerous\nfire weather conditions.",
        levels=temp_levels_f, cmap=temp_cmap, extend="both", lines=np.arange(10, 90, 10), ticks=temp_levels_f,
        transform=lambda c: c * 9 / 5 + 32))
    if "precip_in" in grids:
        from core.precip_palette import load_qpf_palette
        qpf = load_qpf_palette()             # AccuWeather QPF table, shared by every precipitation map
        jobs.append(lambda: draw(
            "rainfall", "Forecast Precipitation", "precip_in", "24-Hour Precipitation (inches)",
            f"24-Hour Precipitation Ending {(cycle + timedelta(hours=cfg.end)).astimezone(LOCAL_TZ):%H}:00 CT\n\n"
            "Precipitation reduces fire danger by:\n• Increasing fuel moisture\n• Raising relative humidity\n• Potentially preventing ignition\n\n"
            "> 0.5\": Significant fire danger reduction\n> 1.0\": Major impact on fire activity",
            levels=list(qpf.levels), cmap=qpf.cmap(), norm=qpf.norm(), alpha=0.8, extend="max",
            lines=list(qpf.line_levels), ticks=list(qpf.ticks), ticklabels=list(qpf.tick_labels), show_peak=False))
    for job in jobs:
        try:
            job()
        except Exception as error:
            logger.error("Day %d: a map failed to draw: %s", cfg.day, error, exc_info=True)
    return written


# ----------------------------------------------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------------------------------------------
def summarize(grids: dict, result: dict, cfg: DayConfig, cycle: datetime, images: list[Path], seconds: float,
              blend: dict | None = None) -> dict:
    inside = grids["inside"]
    peak = grids["peak_risk"][inside]
    hours, start_local = window_label(cycle, cfg)
    share = lambda level: round(float(np.mean(np.rint(peak) >= level)) * 100, 1) if peak.size else None
    return {
        "day": cfg.day,
        "beta": True,
        "source": cfg.source if not blend else "hrrr+rrfs",
        "blend": blend,
        "cycle_utc": cycle.isoformat().replace("+00:00", "Z"),
        "forecast_hours": [cfg.start, cfg.end],
        "valid_local_date": f"{start_local:%Y-%m-%d}",
        "window_local": hours,
        "peak_block": result.get("peak_block"),
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "runtime_seconds": round(seconds, 1),
        "percent_of_state_elevated_or_higher": share(2),
        "percent_of_state_critical_or_higher": share(3),
        "min_fuel_moisture_pct": round(float(np.nanmin(grids["min_fm"])), 1),
        "flags": result["flags"],
        "images": [path.name for path in images],
    }


def update_manifest(out_dir: Path, summaries: list[dict]) -> None:
    try:
        from services.beta_products import BETA_IMAGES_DIR, BETA_ROOT, load_manifest, save_manifest
        manifest = load_manifest()
        entry = manifest.setdefault("products", {}).setdefault("extended_forecast", {})
        entry.update(kind="raster-preview", beta=True, not_for_operations=True,
                     generated_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))
        days = entry.setdefault("days", {})
        for summary in summaries:
            days[str(summary["day"])] = summary
        if out_dir.resolve().is_relative_to(BETA_ROOT.resolve()):
            entry["previews"] = sorted(
                str(path.relative_to(BETA_ROOT)).replace("\\", "/")
                for path in out_dir.glob("mo-forecast-day*.png")
            )
        save_manifest(manifest)
    except Exception as error:
        logger.warning("testbed manifest update failed (non-fatal): %s", error)


def record_snapshots(models: dict[str, xr.Dataset], cfg: DayConfig, cycle: datetime, root: Path | None,
                     peak: dict | None = None) -> None:
    """Save what each model said at the observation stations (for scoring later). Never allowed to break a forecast."""
    try:
        from forecast import extended_verify as verify
        stations = verify.station_list()
        root = root or verify.default_root()
        shots = {}
        for model, data in models.items():
            lon, lat = native_lonlat(data)
            shots[model] = verify.snapshot(data, (cfg.start, cfg.end), cycle, stations, day=cfg.day, model=model, lon=lon,
                                           lat=lat, precip_increments=precip_increments, precip_name=precip_name)
        attrs = {"peak_start_lead": peak["first_lead"], "peak_end_lead": peak["last_lead"]} if peak else None
        written = verify.record(shots, root, cfg.day, cycle, attrs)
        verify.prune_snapshots(root, datetime.now(timezone.utc))
        logger.info("Day %d: saved %d station snapshot(s) for scoring (%s)", cfg.day, len(written),
                    ", ".join(f"{p.stat().st_size / 1024:.1f} KB" for p in written))
    except Exception as error:
        logger.warning("Day %d: could not save station snapshots (non-fatal): %s", cfg.day, error)


def run_scoring(root: Path | None = None) -> None:
    """Score matured snapshots and refresh summary.json (including the shadow blend weights). Non-fatal."""
    try:
        from forecast import extended_verify as verify
        root = root or verify.default_root()
        scored = verify.score_matured(root, verify.hourly_provider())
        summary = verify.write_summary(root)
        weights = summary["weights_day2"]
        logger.info("Verification: scored %d new snapshot(s); shadow Day 2 HRRR share %.2f (%s, not applied)", scored,
                    weights["overall"], weights.get("reason") or "from recent performance")
    except Exception as error:
        logger.warning("Verification scoring skipped (non-fatal): %s", error)


def run_day(cfg: DayConfig, cycle: datetime, out_dir: Path, cache_dir: Path, *, wait_minutes: float, render: bool, day1,
            blend: bool = True, verify_root: Path | None = None, verify: bool = True) -> dict:
    started = time.time()
    ds = load_source_waiting(cfg, cycle, cache_dir, wait_minutes)
    model_ds, blend_info, extra_flags = ds, None, []
    models = {cfg.source: ds}
    if blend and cfg.day == 2 and cfg.source == "hrrr":
        partner = load_rrfs_partner(cfg, cycle, cache_dir, wait_minutes)
        if partner is None:
            extra_flags.append("rrfs_blend_missing")
        else:
            try:
                share = min(1.0, max(0.0, DAY2_HRRR_WEIGHT))
                model_ds = blend_sources(ds, partner, share)
                models.update(rrfs=partner, blend=model_ds)
                blend_info = {"hrrr": round(share, 3), "rrfs": round(1 - share, 3), "weights": "fixed"}
                cfg = replace(cfg, source_label=f"HRRR {share:.0%} / RRFS {1 - share:.0%} Blend (3 km)")
                logger.info("Day %d: blended HRRR %.0f%% / RRFS %.0f%%", cfg.day, share * 100, (1 - share) * 100)
            except Exception as error:
                logger.warning("Day %d: blend failed (%s); using HRRR alone", cfg.day, error, exc_info=True)
                model_ds = ds
                extra_flags.append("blend_failed")
    result = run_hours(model_ds, cfg, cycle, day1)
    result["flags"] = [*result["flags"], *extra_flags]
    if verify:
        record_snapshots(models, cfg, cycle, verify_root, result.get("peak_block"))
    grids = prepare_maps(result, model_ds)
    images = render_maps(grids, cfg, cycle, out_dir) if render else []
    summary = summarize(grids, result, cfg, cycle, images, time.time() - started, blend_info)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"extended-forecast-day{cfg.day}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Day %d done in %.0f s: %d maps, %s%% of the state Elevated+", cfg.day, summary["runtime_seconds"],
                len(images), summary["percent_of_state_elevated_or_higher"])
    return summary


def default_dirs() -> tuple[Path, Path]:
    try:
        from services.beta_products import BETA_IMAGES_DIR
        out_dir = BETA_IMAGES_DIR
    except Exception:
        out_dir = PROJECT_DIR / "images" / "extended"
    return out_dir, Path(os.getenv("SMF_EXTENDED_CACHE_DIR", PROJECT_DIR / "cache" / "extended"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--day", type=int, action="append", choices=sorted(DAYS), help="Day to build (repeatable; default both)")
    parser.add_argument("--cycle", help="model run, e.g. 2026-11-19T12 (default: the newest run that can produce the day and is old enough)")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--wait-minutes", type=float, default=0.0, help="keep retrying while the model run is still publishing")
    parser.add_argument("--no-render", action="store_true", help="calculate and summarize only")
    parser.add_argument("--force", action="store_true", help="rebuild a day even if its newest run is already published")
    parser.add_argument("--no-blend", action="store_true", help="Day 2 from HRRR alone (no RRFS blend)")
    parser.add_argument("--no-verify", action="store_true", help="do not save station snapshots or score earlier ones")
    args = parser.parse_args(argv)

    from core.logging_setup import UtcFormatter
    handler = logging.StreamHandler()
    handler.setFormatter(UtcFormatter())
    logging.getLogger().handlers[:] = [handler]
    logging.getLogger().setLevel(logging.INFO)
    os.environ.setdefault("SMF_HERBIE_THREADS", "4")      # the 5 clipped fields are small; the default of 1 is for the 31-member job

    default_out, cache_dir = default_dirs()
    out_dir = args.out_dir or default_out
    day1 = None                      # loading the model is slow; skip it entirely when every day is already up to date
    summaries, failures, skipped = [], [], 0
    for number in args.day or sorted(DAYS):
        base = DAYS[number]
        cycle = (datetime.fromisoformat(args.cycle).replace(tzinfo=timezone.utc) if args.cycle
                 else latest_cycle(min_age_hours=base.min_cycle_age_hours, hours=cycle_hours(number)))
        cfg = config_for(number, cycle.hour)
        if cfg is None:
            failures.append(number)
            logger.error("Day %d cannot be built from the %s run (supported run hours: %s UTC)", number, f"{cycle:%Y-%m-%d %HZ}",
                         ", ".join(f"{h:02d}" for h in cycle_hours(number)))
            continue
        if not args.force and already_done(out_dir, number, cycle):
            skipped += 1
            logger.info("Day %d is already built from the %s run; nothing to do (--force to rebuild)", number, f"{cycle:%Y-%m-%d %HZ}")
            continue
        try:
            day1 = day1 or _day1()
            summaries.append(run_day(cfg, cycle, out_dir, cache_dir, wait_minutes=args.wait_minutes, render=not args.no_render,
                                     day1=day1, blend=not args.no_blend, verify=not args.no_verify))
        except Exception as error:
            failures.append(number)
            logger.error("Day %d failed: %s", number, error, exc_info=True)
    if summaries and not args.no_render:
        update_manifest(out_dir, summaries)
    if not args.no_verify:
        run_scoring()
    return 0 if (summaries or skipped) and not (failures and not summaries) else 1


def already_done(out_dir: Path, day: int, cycle: datetime) -> bool:
    """True when this day's published summary is from this run and was not a degraded (HRRR-only) result."""
    try:
        existing = json.loads((out_dir / f"extended-forecast-day{day}.json").read_text(encoding="utf-8"))
    except Exception:
        return False
    flags = existing.get("flags") or []
    return (existing.get("cycle_utc") == cycle.isoformat().replace("+00:00", "Z")
            and "rrfs_blend_missing" not in flags and "blend_failed" not in flags)


if __name__ == "__main__":
    sys.exit(main())
