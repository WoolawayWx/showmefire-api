"""Fetch and cache Missouri-area RTMA analyses for training and verification."""
from __future__ import annotations

import argparse
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xarray as xr
from herbie import FastHerbie

logger = logging.getLogger(__name__)
MO_BUFFERED_BBOX = (-96.8, -88.1, 34.8, 41.8)  # west, east, south, north


def _root() -> Path:
    return Path("/app") if Path("/app").exists() else Path(__file__).resolve().parent.parent


def _as_dataset(value):
    if isinstance(value, list):
        if not value:
            raise RuntimeError("Herbie returned no RTMA datasets")
        return xr.merge(value, compat="override")
    return value


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


def fetch_rtma(run_dt: datetime, cache_dir: Path | None = None) -> Path:
    if run_dt.tzinfo is not None:
        run_dt = run_dt.astimezone(timezone.utc).replace(tzinfo=None)
    run_dt = run_dt.replace(minute=0, second=0, microsecond=0)
    cache_dir = Path(cache_dir or (_root() / "cache" / "rtma"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"rtma_{run_dt:%Y%m%d_%H}z.nc"
    if target.exists():
        try:
            with xr.open_dataset(target) as cached:
                required = {"t2m", "r2", "u10", "v10"}
                if required.issubset(cached.data_vars):
                    return target
        except Exception:
            pass
        target.unlink(missing_ok=True)

    fh = FastHerbie(DATES=[run_dt], fxx=[0], model="rtma", product="anl")
    try:
        therm = _as_dataset(fh.xarray(":(TMP|DPT):2 m above ground:"))
    except Exception:
        therm = _as_dataset(fh.xarray(":(TMP|DPT):2 m"))
    try:
        wind = _as_dataset(fh.xarray(":(UGRD|VGRD):10 m above ground:"))
    except Exception:
        wind = _as_dataset(fh.xarray(":(UGRD|VGRD):10 m"))
    ds = xr.merge([therm, wind], compat="override")
    if "r2" not in ds:
        if "t2m" not in ds or "d2m" not in ds:
            raise KeyError(f"RTMA variables cannot derive RH: {list(ds.data_vars)}")
        ds["r2"] = _relative_humidity(ds["t2m"], ds["d2m"])
        ds["r2"].attrs.update({"long_name": "relative humidity", "units": "%", "derived_from": "t2m,d2m Magnus formula"})
    ds = _crop(ds).load()
    ds.attrs.update({"requested_analysis_time_utc": run_dt.isoformat() + "Z", "domain_bbox": str(MO_BUFFERED_BBOX)})
    for var in ds.variables:
        for attr in ("dtype", "source"):
            ds[var].attrs.pop(attr, None)
    temp = target.with_suffix(".nc.tmp")
    ds.to_netcdf(temp, engine="netcdf4")
    with xr.open_dataset(temp) as check:
        if not {"t2m", "r2", "u10", "v10"}.issubset(check.data_vars):
            raise RuntimeError("RTMA cache verification failed")
    temp.replace(target)
    logger.info("cached RTMA %s", target)
    return target


def latest_complete_hour(now: datetime | None = None) -> datetime:
    now = now or datetime.now(timezone.utc)
    return now.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", help="UTC analysis time, e.g. 2026-07-12T14:00Z")
    args = parser.parse_args()
    run_dt = datetime.fromisoformat(args.time.replace("Z", "+00:00")) if args.time else latest_complete_hour()
    print(fetch_rtma(run_dt))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
