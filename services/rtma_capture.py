"""Fetch and cache Missouri-area RTMA analyses for training and verification."""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pyproj
import xarray as xr
from herbie import Herbie

logger = logging.getLogger(__name__)
MO_BUFFERED_BBOX = (-96.8, -88.1, 34.8, 41.8)  # west, east, south, north
RTMA_FILE_RE = re.compile(r"^rtma_(\d{8})_(\d{2})z\.nc$")
REQUIRED_RTMA_VARS = frozenset({"t2m", "r2", "u10", "v10", "apcp"})
DEFAULT_SPREAD_RATE_POLL_MINUTES = 15


def _new_herbie(*args, **kwargs):
    """Construct Herbie safely on Windows consoles using a legacy code page."""
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure:
        reconfigure(encoding="utf-8", errors="replace")
    return Herbie(*args, **kwargs)


def _root() -> Path:
    return Path("/app") if Path("/app").exists() else Path(__file__).resolve().parent.parent


def _as_dataset(value):
    if isinstance(value, list):
        if not value:
            raise RuntimeError("Herbie returned no RTMA datasets")
        value = xr.merge([_sanitize_dataset(item) for item in value], compat="override")
    return _sanitize_dataset(value)


def _sanitize_dataset(ds: xr.Dataset) -> xr.Dataset:
    """Remove cfgrib attributes that collide with xarray serialization.

    RTMA analysis has no meaningful forecast lead, so its scalar/length-one
    ``step`` coordinate is dropped after Herbie opens the GRIB.
    """
    ds = ds.copy(deep=False)
    if "step" in ds.dims and ds.sizes["step"] == 1:
        ds = ds.isel(step=0, drop=True)
    elif "step" in ds.coords and "step" not in ds.dims:
        ds = ds.drop_vars("step")
    for name in ds.variables:
        ds[name].attrs.pop("dtype", None)
        ds[name].attrs.pop("source", None)
        # A dtype in encoding is valid, but removing it makes merged datasets
        # deterministic across xarray/cfgrib versions and netCDF engines.
        ds[name].encoding.pop("dtype", None)
    return ds


def _crop(ds: xr.Dataset) -> xr.Dataset:
    lon = ds.longitude
    lat = ds.latitude
    lon180 = xr.where(lon > 180, lon - 360, lon)
    west, east, south, north = MO_BUFFERED_BBOX
    mask = (lon180 >= west) & (lon180 <= east) & (lat >= south) & (lat <= north)
    if mask.ndim != 2:
        return ds.where(mask, drop=True)
    rows, cols = np.where(mask.values)
    if not len(rows):
        raise ValueError("RTMA grid does not intersect the configured domain")
    ydim, xdim = mask.dims
    return ds.isel({ydim: slice(rows.min(), rows.max() + 1), xdim: slice(cols.min(), cols.max() + 1)})


def _relative_humidity(temp_k, dewpoint_k):
    temp_c = temp_k - 273.15
    dewpoint_c = dewpoint_k - 273.15
    rh = 100.0 * np.exp((17.625 * dewpoint_c) / (243.04 + dewpoint_c) - (17.625 * temp_c) / (243.04 + temp_c))
    return rh.clip(0.0, 100.0)


def _projected_axes(ds):
    """Derive regular projected x/y axes from Herbie's 2D lat/lon fields."""
    projection = ds.get("gribfile_projection")
    if projection is None or "grid_mapping_name" not in projection.attrs:
        raise ValueError("RTMA dataset has no CF projection metadata")
    crs = pyproj.CRS.from_cf(projection.attrs)
    longitude = xr.where(ds.longitude > 180, ds.longitude - 360, ds.longitude)
    x_values, y_values = pyproj.Transformer.from_crs(
        "EPSG:4326", crs, always_xy=True
    ).transform(longitude.values, ds.latitude.values)
    return np.asarray(x_values)[0, :], np.asarray(y_values)[:, 0]


def _align_precipitation(precip, target):
    """Put precipitation on the analysis grid when RTMA products differ."""
    if precip.sizes == target.sizes and np.allclose(
        precip.latitude.values, target.latitude.values, equal_nan=True
    ) and np.allclose(
        precip.longitude.values, target.longitude.values, equal_nan=True
    ):
        return precip
    source_x, source_y = _projected_axes(precip)
    target_x, target_y = _projected_axes(target)
    values = precip.reset_coords(drop=True).assign_coords(
        x=("x", source_x), y=("y", source_y)
    )
    aligned = values.interp(
        x=("x", target_x), y=("y", target_y), method="nearest"
    )
    return aligned.assign_coords(
        latitude=target.latitude, longitude=target.longitude
    )


def _fetch_precipitation_mm(run_dt: datetime) -> xr.DataArray:
  """Hourly accumulated precipitation from the separate RTMA pcp product."""
  h = _new_herbie(run_dt, fxx=0, model="rtma", product="pcp")
  try:
    pcp = _as_dataset(h.xarray(":(?:APCP|tp):"))
  except AttributeError as error:
    # Some RTMA precipitation indexes contain only ``range=0-`` and no
    # end_byte. Herbie's byte-range subsetter then raises AttributeError
    # even though the GRIB2 object is present. Fall back to downloading and
    # opening the complete (single-field) precipitation file.
    if "end_byte" not in str(error):
        raise
    full_grib = h.download(search=None)
    pcp = _as_dataset(xr.open_dataset(full_grib, engine="cfgrib"))
  except Exception:
    try:
      pcp = _as_dataset(h.xarray("APCP"))
    except AttributeError as error:
      if "end_byte" not in str(error):
          raise
      full_grib = h.download(search=None)
      pcp = _as_dataset(xr.open_dataset(full_grib, engine="cfgrib"))
  if "tp" in pcp:
    precip = pcp["tp"]
  elif "apcp" in pcp:
    precip = pcp["apcp"]
  else:
    raise KeyError(f"RTMA precipitation variable missing: {list(pcp.data_vars)}")
  precip = precip.where(precip > 0, 0.0)
  precip = precip.rename("apcp")
  precip.attrs.update({"long_name": "1-hour accumulated precipitation", "units": "mm"})
  return precip


def cache_path_for_hour(run_dt: datetime, cache_dir: Path | None = None) -> Path:
    if run_dt.tzinfo is not None:
        run_dt = run_dt.astimezone(timezone.utc).replace(tzinfo=None)
    run_dt = run_dt.replace(minute=0, second=0, microsecond=0)
    cache_dir = Path(cache_dir or (_root() / "cache" / "rtma"))
    return cache_dir / f"rtma_{run_dt:%Y%m%d_%H}z.nc"


def is_analysis_hour_cached(run_dt: datetime, cache_dir: Path | None = None) -> bool:
    target = cache_path_for_hour(run_dt, cache_dir=cache_dir)
    if not target.is_file():
        return False
    try:
        with xr.open_dataset(target) as cached:
            precipitation_missing = bool(
                cached["apcp"].attrs.get("missing", False)
            ) if "apcp" in cached else True
            return REQUIRED_RTMA_VARS.issubset(cached.data_vars) and not precipitation_missing
    except Exception:
        return False


def ensure_analysis_hour_cached(run_dt: datetime | None = None, cache_dir: Path | None = None) -> dict:
    """Best-effort fetch of one RTMA hour into the on-server cache."""
    run_dt = run_dt or latest_complete_hour()
    cache_dir = Path(cache_dir or (_root() / "cache" / "rtma"))
    already_cached = is_analysis_hour_cached(run_dt, cache_dir=cache_dir)
    if already_cached:
        path = cache_path_for_hour(run_dt, cache_dir=cache_dir)
        return {
            "analysis_hour": run_dt.isoformat() + "Z",
            "cached": True,
            "fetched": False,
            "path": str(path),
        }
    try:
        path = fetch_rtma(run_dt, cache_dir=cache_dir)
        return {
            "analysis_hour": run_dt.isoformat() + "Z",
            "cached": True,
            "fetched": True,
            "path": str(path),
        }
    except Exception as exc:
        logger.warning("RTMA fetch failed for %s: %s", run_dt, exc)
        return {
            "analysis_hour": run_dt.isoformat() + "Z",
            "cached": is_analysis_hour_cached(run_dt, cache_dir=cache_dir),
            "fetched": False,
            "error": str(exc),
        }


def ensure_latest_analysis_cached(cache_dir: Path | None = None) -> dict:
    """Poll NOAA for the latest complete RTMA hour and retain it on disk."""
    analysis_hour = latest_complete_hour()
    result = ensure_analysis_hour_cached(analysis_hour, cache_dir=cache_dir)
    result["cache_dir"] = str(cache_dir or (_root() / "cache" / "rtma"))
    return result


def spread_rate_poll_minutes() -> int:
    try:
        minutes = int(os.getenv("SPREAD_RATE_POLL_MINUTES", str(DEFAULT_SPREAD_RATE_POLL_MINUTES)))
    except ValueError:
        logger.warning("Invalid SPREAD_RATE_POLL_MINUTES; using %s", DEFAULT_SPREAD_RATE_POLL_MINUTES)
        minutes = DEFAULT_SPREAD_RATE_POLL_MINUTES
    return max(5, min(minutes, 60))


def fetch_rtma(run_dt: datetime, cache_dir: Path | None = None) -> Path:
    if run_dt.tzinfo is not None:
        run_dt = run_dt.astimezone(timezone.utc).replace(tzinfo=None)
    run_dt = run_dt.replace(minute=0, second=0, microsecond=0)
    cache_dir = Path(cache_dir or (_root() / "cache" / "rtma"))
    target = cache_path_for_hour(run_dt, cache_dir=cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            with xr.open_dataset(target) as cached:
                precipitation_missing = bool(
                    cached["apcp"].attrs.get("missing", False)
                ) if "apcp" in cached else True
                if REQUIRED_RTMA_VARS.issubset(cached.data_vars) and not precipitation_missing:
                    return target
        except Exception:
            pass
        target.unlink(missing_ok=True)

    # A single analysis does not need FastHerbie. FastHerbie always runs
    # combine_nested(time, step), which triggers an xarray/cfgrib collision
    # when RTMA's scalar step coordinate carries a GRIB ``dtype`` attribute.
    h = _new_herbie(run_dt, fxx=0, model="rtma", product="anl")
    try:
        therm = _as_dataset(h.xarray(":(?:TMP|DPT):2 m above ground:"))
    except Exception:
        therm = _as_dataset(h.xarray(":(?:TMP|DPT):2 m"))
    try:
        wind = _as_dataset(h.xarray(":(?:UGRD|VGRD):10 m above ground:"))
    except Exception:
        wind = _as_dataset(h.xarray(":(?:UGRD|VGRD):10 m"))
    ds = _sanitize_dataset(xr.merge([therm, wind], compat="override"))
    if "r2" not in ds:
        if "t2m" not in ds or "d2m" not in ds:
            raise KeyError(f"RTMA variables cannot derive RH: {list(ds.data_vars)}")
        ds["r2"] = _relative_humidity(ds["t2m"], ds["d2m"])
        ds["r2"].attrs.update({"long_name": "relative humidity", "units": "%", "derived_from": "t2m,d2m Magnus formula"})
    try:
        precip = _fetch_precipitation_mm(run_dt)
        precip = _align_precipitation(precip, ds["t2m"])
        ds = _sanitize_dataset(xr.merge([ds, precip], compat="override"))
    except Exception as error:
        logger.warning("RTMA precipitation unavailable for %s: %s", run_dt, error)
        ds["apcp"] = xr.zeros_like(ds["t2m"], dtype="float32")
        ds["apcp"].attrs.update({"long_name": "1-hour accumulated precipitation", "units": "mm", "missing": "true"})
    ds = _crop(ds).load()
    ds.attrs.update({"requested_analysis_time_utc": run_dt.isoformat() + "Z", "domain_bbox": str(MO_BUFFERED_BBOX)})
    ds = _sanitize_dataset(ds)
    temp = target.with_suffix(".nc.tmp")
    ds.to_netcdf(temp, engine="netcdf4")
    with xr.open_dataset(temp) as check:
        if not REQUIRED_RTMA_VARS.issubset(check.data_vars):
            raise RuntimeError("RTMA cache verification failed")
    temp.replace(target)
    logger.info("cached RTMA %s", target)
    return target


def warmup_rtma_cache(
    days: int = 7,
    cache_dir: Path | None = None,
    end_hour: datetime | None = None,
) -> dict:
    """Best-effort backfill of hourly RTMA analyses for spread-rate conditioning."""
    cache_dir = Path(cache_dir or (_root() / "cache" / "rtma"))
    end_hour = end_hour or latest_complete_hour()
    if end_hour.tzinfo is not None:
        end_hour = end_hour.astimezone(timezone.utc).replace(tzinfo=None)
    fetched = failed = 0
    for offset in range(days * 24):
        stamp = end_hour - timedelta(hours=offset)
        try:
            fetch_rtma(stamp, cache_dir=cache_dir)
            fetched += 1
        except Exception:
            failed += 1
            logger.exception("RTMA warm-up failed for %s", stamp)
    return {"fetched": fetched, "failed": failed, "target_hours": days * 24, "end_hour": end_hour.isoformat() + "Z"}


def count_cached_hours(
    end_hour: datetime,
    hours: int,
    cache_dir: Path | None = None,
) -> int:
    cache_dir = Path(cache_dir or (_root() / "cache" / "rtma"))
    if end_hour.tzinfo is not None:
        end_hour = end_hour.astimezone(timezone.utc).replace(tzinfo=None)
    available = 0
    for offset in range(hours):
        stamp = end_hour - timedelta(hours=offset)
        path = cache_dir / f"rtma_{stamp:%Y%m%d_%H}z.nc"
        if path.is_file():
            available += 1
    return available


def latest_complete_hour(now: datetime | None = None) -> datetime:
    now = now or datetime.now(timezone.utc)
    return now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)


def cleanup_rtma_cache(cache_dir: Path | None = None, now: datetime | None = None, retention_days: int | None = None):
    """Remove expired operational RTMA analyses and nothing else."""
    cache_dir = Path(cache_dir or (_root() / "cache" / "rtma"))
    if not cache_dir.is_dir():
        return {"removed_files": 0, "removed_bytes": 0}
    if retention_days is None:
        try:
            retention_days = int(os.getenv("RTMA_RETENTION_DAYS", "7"))
        except ValueError:
            logger.warning("Invalid RTMA_RETENTION_DAYS; using 7")
            retention_days = 7
    if retention_days < 1:
        raise ValueError("RTMA_RETENTION_DAYS must be at least 1")
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    cutoff = now.astimezone(timezone.utc) - timedelta(days=retention_days)
    removed_files = removed_bytes = 0
    for path in cache_dir.iterdir():
        if not path.is_file():
            continue
        match = RTMA_FILE_RE.fullmatch(path.name)
        if not match:
            continue
        analysis_time = datetime.strptime("".join(match.groups()), "%Y%m%d%H").replace(tzinfo=timezone.utc)
        if analysis_time >= cutoff:
            continue
        size = path.stat().st_size
        path.unlink()
        removed_files += 1
        removed_bytes += size
    logger.info("RTMA retention cleanup removed %d file(s), %.1f MB", removed_files, removed_bytes / 1e6)
    return {"removed_files": removed_files, "removed_bytes": removed_bytes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", help="UTC analysis time, e.g. 2026-07-12T14:00Z")
    args = parser.parse_args()
    run_dt = datetime.fromisoformat(args.time.replace("Z", "+00:00")) if args.time else latest_complete_hour()
    print(fetch_rtma(run_dt))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
