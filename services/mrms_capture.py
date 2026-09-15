"""Download and normalize hourly MRMS precipitation for verification.

MRMS is optional.  The downloader uses NOAA's public HTTP directory and
stores a small Missouri-area NetCDF cache consumed by the verification
pipeline.  It never turns a failed download into zero precipitation.
"""
from __future__ import annotations

import gzip
import logging
import os
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import requests
import xarray as xr

logger = logging.getLogger(__name__)

MO_BUFFERED_BBOX = (-96.8, -88.1, 34.8, 41.8)
MRMS_PRODUCT = os.getenv("VERIFICATION_MRMS_PRODUCT", "MultiSensor_QPE_01H_Pass2")
MRMS_ROOT_URL = os.getenv(
    "VERIFICATION_MRMS_ROOT_URL",
    f"https://mrms.ncep.noaa.gov/2D/{MRMS_PRODUCT}/",
)
DEFAULT_RETENTION_DAYS = 7


def _root() -> Path:
    return Path(os.getenv("VERIFICATION_MRMS_ROOT", "cache/mrms"))


def _product_name(valid_time: datetime) -> str:
    stamp = valid_time.astimezone(timezone.utc).strftime("%Y%m%d-%H0000")
    return f"MRMS_{MRMS_PRODUCT}_00.00_{stamp}.grib2.gz"


def cache_path_for_hour(valid_time: datetime, cache_dir: Path | None = None) -> Path:
    valid_time = valid_time.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return Path(cache_dir or _root()) / f"mrms_{valid_time:%Y%m%d_%H}z.nc"


def _download_source(valid_time: datetime, target: Path) -> Path:
    valid_time = valid_time.astimezone(timezone.utc)
    filename = _product_name(valid_time)
    url = f"{MRMS_ROOT_URL.rstrip('/')}/{filename}"
    partial = target.with_suffix(target.suffix + ".partial")
    target.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=180) as response:
        response.raise_for_status()
        with partial.open("wb") as output:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    output.write(chunk)
    partial.replace(target)
    return target


def _find_variable(ds: xr.Dataset) -> str:
    for name in ("unknown", "precipitation", "precip_mm", "apcp"):
        if name in ds.data_vars:
            return name
    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars))
    raise ValueError(f"MRMS file has no recognizable precipitation variable: {list(ds.data_vars)}")


def _normalize_coordinates(ds: xr.Dataset) -> xr.Dataset:
    lat_name = next((name for name in ("latitude", "lat") if name in ds), None)
    lon_name = next((name for name in ("longitude", "lon") if name in ds), None)
    if not lat_name or not lon_name:
        raise ValueError("MRMS dataset has no latitude/longitude coordinates")
    variable = _find_variable(ds)
    values = ds[variable].squeeze(drop=True)
    lat = ds[lat_name]
    lon = ds[lon_name]
    if lat.ndim == 1 and lon.ndim == 1:
        values = values.assign_coords(
            latitude=(lat.dims[0], np.asarray(lat.values)),
            longitude=(lon.dims[0], np.asarray(lon.values)),
        )
    else:
        values = values.assign_coords(latitude=lat, longitude=lon)
    values = values.rename("precipitation")
    values.attrs.update({"units": "mm", "long_name": "MRMS 1-hour accumulated precipitation"})
    return values.to_dataset()


def _crop(ds: xr.Dataset) -> xr.Dataset:
    values = ds["precipitation"]
    lat = ds["latitude"] if "latitude" in ds else ds.coords.get("latitude")
    lon = ds["longitude"] if "longitude" in ds else ds.coords.get("longitude")
    west, east, south, north = MO_BUFFERED_BBOX
    if lat is not None and lon is not None and lat.ndim == 2:
        mask = (lon >= west) & (lon <= east) & (lat >= south) & (lat <= north)
        rows, cols = np.where(mask.values)
        if not len(rows):
            raise ValueError("MRMS grid does not intersect Missouri bbox")
        ydim, xdim = mask.dims
        return ds.isel({ydim: slice(rows.min(), rows.max() + 1), xdim: slice(cols.min(), cols.max() + 1)})
    if lat is None or lon is None:
        raise ValueError("MRMS dataset has no usable coordinates")
    lat_slice = slice(south, north) if float(lat[0]) < float(lat[-1]) else slice(north, south)
    lon_slice = slice(west, east) if float(lon[0]) < float(lon[-1]) else slice(east, west)
    return ds.sel({lat.dims[0]: lat_slice, lon.dims[0]: lon_slice})


def fetch_mrms(valid_time: datetime | None = None, *, cache_dir: Path | None = None) -> Path:
    """Fetch, decode, crop, and cache one MRMS hour."""
    valid_time = valid_time or (datetime.now(timezone.utc) - timedelta(hours=1))
    target = cache_path_for_hour(valid_time, cache_dir)
    if target.is_file() and target.stat().st_size > 0:
        return target
    with tempfile.TemporaryDirectory(prefix="mrms-") as temporary:
        source = Path(temporary) / _product_name(valid_time)
        _download_source(valid_time, source)
        with gzip.open(source, "rb") as compressed, tempfile.NamedTemporaryFile(suffix=".grib2") as grib:
            shutil.copyfileobj(compressed, grib)
            grib.flush()
            with xr.open_dataset(
                grib.name,
                engine="cfgrib",
                backend_kwargs={"indexpath": ""},
            ) as raw:
                normalized = _crop(_normalize_coordinates(raw.load()))
                normalized.attrs.update({
                    "provider": "MRMS",
                    "product": MRMS_PRODUCT,
                    "valid_time": valid_time.astimezone(timezone.utc).isoformat(),
                })
                target.parent.mkdir(parents=True, exist_ok=True)
                normalized.to_netcdf(target, engine="netcdf4")
    return target


def cleanup_mrms_cache(
    cache_dir: Path | None = None,
    retention_days: int | None = None,
) -> int:
    days = retention_days or int(os.getenv("MRMS_RETENTION_DAYS", DEFAULT_RETENTION_DAYS))
    cutoff = datetime.now(timezone.utc).timestamp() - days * 86400
    removed = 0
    for path in Path(cache_dir or _root()).glob("mrms_*.nc"):
        if path.stat().st_mtime < cutoff:
            path.unlink(missing_ok=True)
            removed += 1
    return removed


def mrms_enabled() -> bool:
    return os.getenv("VERIFICATION_MRMS_ENABLED", "false").lower() == "true"
