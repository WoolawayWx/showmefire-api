"""Build the daily forecast's HRRR cache file from the hourly product's per-hour data.

Both products read the same 12z HRRR hours. The hourly prefetch (services/hrrr_hourly/job.py `prefetch_cycle`)
already downloads and decodes each hour as it is published and keeps the unprocessed surface fields in its work
files (`raw_*`). This assembles them, with precipitation fetched the way DailyForecast fetches it, into the file
DailyForecast would otherwise download for itself (`hrrr_<date>_12z_f04-15.nc`, see forecast/DailyForecast.py), so it
takes its existing cache-hit path and never loads the full CONUS grids.

Only the four instantaneous fields (2 m temperature and RH, 10 m wind, gust) are shared. Precipitation is a
cumulative record that DailyForecast differences (core/precipitation.py), so it is downloaded with the identical
`:APCP:` call rather than rebuilt from the hourly product's 1-hour windows.

By default the file goes to a shadow directory and DailyForecast is untouched; `scripts/compare_hrrr_cache.py`
checks it against the file DailyForecast downloads for itself. Set FORECAST_USE_SHARED_HRRR=true to write the live
cache path instead, once the comparison has agreed for a few days.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# DailyForecast variable name -> (raw work-file key, units). `i10fg` (gust) is optional, like in DailyForecast.
RAW_FIELDS = {
    "t2m": ("raw_t2m_k", "K"),
    "r2": ("raw_rh2", "%"),
    "u10": ("raw_u10", "m s**-1"),
    "v10": ("raw_v10", "m s**-1"),
    "i10fg": ("raw_gust", "m s**-1"),
}
REQUIRED_VARIABLES = ("t2m", "r2", "u10", "v10")
SHADOW_KEEP_DAYS = 3


class SharedCacheUnavailable(Exception):
    """The per-hour data cannot make a cache file (missing hours or raw fields, grid mismatch, no precipitation)."""


def required_hours() -> tuple[int, ...]:
    """The forecast hours DailyForecast reads (FORECAST_START_HOUR..FORECAST_END_HOUR, default f04-f15)."""
    return tuple(range(int(os.getenv("FORECAST_START_HOUR", "4")), int(os.getenv("FORECAST_END_HOUR", "15")) + 1))


def cache_name(run: datetime, hours) -> str:
    """The file name DailyForecast looks for (forecast/DailyForecast.py, 'hrrr_<date>_<hh>z_f04-15.nc')."""
    return f"hrrr_{run:%Y%m%d_%H}z_f{hours[0]:02d}-{hours[-1]:02d}.nc"


def live_cache_dir() -> Path:
    return Path(os.getenv("SMF_HRRR_CACHE_DIR", "/app/cache/hrrr"))


def shadow_cache_dir() -> Path:
    return Path(os.getenv("SMF_HRRR_SHARED_CACHE_DIR", "cache/hrrr_shared"))


def use_shared() -> bool:
    return os.getenv("FORECAST_USE_SHARED_HRRR", "").lower() in {"1", "true", "yes", "on"}


def stack_raw_fields(work_dir: Path, hours) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """(lat, lon, {variable: (hours, y, x) float32}) from the per-hour work files.

    Raises SharedCacheUnavailable if an hour or a required raw field is missing (work files written before the
    raw fields existed have none). Gust is included only when every hour has it."""
    cubes: dict[str, list[np.ndarray]] = {name: [] for name in RAW_FIELDS}
    lat = lon = None
    for lead in hours:
        path = Path(work_dir) / f"f{lead:02d}.npz"
        if not path.is_file():
            raise SharedCacheUnavailable(f"no work file for f{lead:02d}")
        with np.load(path) as data:
            lat, lon = data["lat"], data["lon"]
            for name, (key, _units) in RAW_FIELDS.items():
                if key in data.files:
                    cubes[name].append(np.asarray(data[key], dtype="float32"))
    stacks = {}
    for name, items in cubes.items():
        if len(items) == len(tuple(hours)):
            stacks[name] = np.stack(items)
        elif name in REQUIRED_VARIABLES:
            raise SharedCacheUnavailable(f"{RAW_FIELDS[name][0]} is missing from {len(tuple(hours)) - len(items)} hour(s)")
    return lat, lon, stacks


def download_precip(run: datetime, hours):
    """Precipitation exactly as forecast/DailyForecast.py gets it: FastHerbie `:APCP:` for the hours, cropped."""
    from herbie import FastHerbie

    from core.domain import crop

    collection = FastHerbie(DATES=[run.replace(tzinfo=None)], fxx=list(hours), model="hrrr", product="sfc")
    dataset = collection.xarray(":APCP:")
    if isinstance(dataset, list):
        dataset = dataset[0]
    return crop(dataset.load())


def assemble(precip, lat: np.ndarray, lon: np.ndarray, stacks: dict[str, np.ndarray], hours):
    """The cache Dataset: the precipitation dataset's own coordinates, plus t2m, r2, u10, v10 (and i10fg)."""
    precip_name = next((n for n in ("tp", "apcp", "APCP") if n in precip.data_vars), None)
    if precip_name is None:
        raise SharedCacheUnavailable("the precipitation download has no tp/apcp variable")
    if precip[precip_name].dims != ("step", "y", "x"):
        raise SharedCacheUnavailable(f"unexpected precipitation dims {precip[precip_name].dims}")
    steps = (np.asarray(precip["step"].values).astype("timedelta64[h]").astype(int)).tolist()
    if steps != list(hours):
        raise SharedCacheUnavailable(f"precipitation covers steps {steps}, expected {list(hours)}")
    if precip["latitude"].shape != lat.shape or not (
            np.allclose(precip["latitude"].values, lat, atol=1e-4) and
            np.allclose(precip["longitude"].values, lon, atol=1e-4)):
        raise SharedCacheUnavailable("the hourly grid does not match the precipitation grid")
    dataset = precip.copy()
    for name, cube in stacks.items():
        dataset[name] = (("step", "y", "x"), cube, {"units": RAW_FIELDS[name][1], "GRIB_stepType": "instant"})
    for variable in dataset.variables:                     # same clean-up DailyForecast does before it writes
        for attr in ("dtype", "source"):
            dataset[variable].attrs.pop(attr, None)
    return dataset


def write_atomic(dataset, path: Path) -> None:
    """Write to a temporary file and rename, so DailyForecast never opens a half-written cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        import netCDF4  # noqa: F401
        dataset.to_netcdf(temporary, engine="netcdf4")
    except ImportError:
        dataset.to_netcdf(temporary)
    os.replace(temporary, path)


def prune_shadow(directory: Path, days: int = SHADOW_KEEP_DAYS, now: Optional[float] = None) -> list[str]:
    cutoff = (now or time.time()) - days * 86400
    removed = []
    for path in Path(directory).glob("hrrr_*.nc"):
        if path.stat().st_mtime < cutoff:
            path.unlink(missing_ok=True)
            removed.append(path.name)
    return removed


def build_shared_cache(cycle_time: datetime, root: Path, live: Optional[bool] = None,
                       precip_loader: Callable = download_precip) -> dict:
    """Assemble and write the cache file for `cycle_time` from `<root>/.work/<cycle>/f##.npz`.

    Writes the shadow directory unless `live` (default: FORECAST_USE_SHARED_HRRR). Raises SharedCacheUnavailable
    when it cannot make a complete file, in which case nothing is written and DailyForecast downloads for itself."""
    from services.hrrr_hourly import cycle_id

    hours = required_hours()
    live = use_shared() if live is None else live
    lat, lon, stacks = stack_raw_fields(Path(root) / ".work" / cycle_id(cycle_time), hours)
    started = time.perf_counter()
    try:
        precip = precip_loader(cycle_time, hours)
    except Exception as exc:  # noqa: BLE001 - a download failure just means "no shared file today"
        raise SharedCacheUnavailable(f"precipitation download failed: {exc}") from exc
    dataset = assemble(precip, lat, lon, stacks, hours)
    directory = live_cache_dir() if live else shadow_cache_dir()
    target = directory / cache_name(cycle_time, hours)
    write_atomic(dataset, target)
    pruned = [] if live else prune_shadow(directory)
    logger.info("hrrr shared cache written to %s (%s, %.1f MB, %.1fs)", target, "LIVE" if live else "shadow",
                target.stat().st_size / 1048576, time.perf_counter() - started)
    return {"path": str(target), "live": live, "variables": sorted(stacks), "bytes": target.stat().st_size,
            "pruned": pruned}


# --- comparison with the file DailyForecast downloads for itself ---------------------------------------------------

def compare_caches(shared: Path, real: Path, atol: float = 1e-3) -> dict:
    """Compare an assembled cache with DailyForecast's own, opened the way DailyForecast opens it (decode_cf=False).

    The four shared fields must match within `atol` (float32 rounding, and K -> degC -> K for temperature);
    precipitation comes from the same download so it must be identical. `ok` is the overall verdict."""
    import xarray as xr

    report: dict = {"shared": str(shared), "real": str(real), "variables": {}, "problems": []}
    with xr.open_dataset(shared, decode_cf=False) as a, xr.open_dataset(real, decode_cf=False) as b:
        if not np.array_equal(np.asarray(a["step"].values), np.asarray(b["step"].values)):
            report["problems"].append("step differs")
        for coord in ("latitude", "longitude"):
            if a[coord].shape != b[coord].shape or not np.allclose(a[coord].values, b[coord].values, atol=1e-4):
                report["problems"].append(f"{coord} differs")
        for name in sorted(set(a.data_vars) | set(b.data_vars)):
            if name not in a.data_vars or name not in b.data_vars:
                if name in REQUIRED_VARIABLES or name in ("tp", "apcp"):
                    report["problems"].append(f"{name} is only in the {'real' if name in b.data_vars else 'shared'} file")
                continue
            x, y = a[name], b[name]
            if x.shape != y.shape:
                report["problems"].append(f"{name} shape {x.shape} vs {y.shape}")
                continue
            xv, yv = np.asarray(x.values, dtype="float64"), np.asarray(y.values, dtype="float64")
            both_nan = np.isnan(xv) & np.isnan(yv)
            diff = np.where(both_nan, 0.0, np.abs(xv - yv))
            worst = float(np.nanmax(diff)) if diff.size else 0.0
            tolerance = 0.0 if name in ("tp", "apcp") else atol
            unit_a, unit_b = x.attrs.get("units"), y.attrs.get("units")
            report["variables"][name] = {"max_abs_diff": worst, "units": [unit_a, unit_b]}
            if not worst <= tolerance:
                report["problems"].append(f"{name} differs by up to {worst:g} (allowed {tolerance:g})")
            if unit_a != unit_b:
                report["problems"].append(f"{name} units {unit_a!r} vs {unit_b!r}")
    report["ok"] = not report["problems"]
    return report
