"""
MRMS multi-sensor hourly precipitation, used ONLY to fill hours whose RTMA precipitation was flagged missing in the
cache (services/rtma_capture.py stores zeros with apcp.attrs['missing']='true' when the precip product is unavailable,
and a plain replay would treat those hours as dry).

Product: noaa-mrms-pds CONUS MultiSensor_QPE_01H_Pass2 (gauge-corrected), falling back to Pass1 (available sooner,
less corrected). RadarOnly is never used (biased low in places). Key time is the END of the 1-hour accumulation, same
convention as the RTMA hourly precipitation. Grid 7000 x 3500 at 0.01 deg (lon 230.005..299.995 E); mm; negative codes
(-3 = no coverage) are NoData. ~0.6 MB per hour, ~0.3 s to decode; each fetched hour is cropped to the domain and cached.

Area-mean sampling: a 3 km cell spans ~3x3 MRMS pixels, so each target cell is the mean of sub-samples taken inside it
(bilinear at each), not one point sample.
"""
from __future__ import annotations

import gzip
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import requests

logger = logging.getLogger(__name__)

BASE_URL = os.getenv("SMF_FIRE_OUTLOOK_MRMS_BASE", "https://noaa-mrms-pds.s3.amazonaws.com/CONUS")
PRODUCTS = ("MultiSensor_QPE_01H_Pass2_00.00", "MultiSensor_QPE_01H_Pass1_00.00")
CROP_MARGIN_DEG = 0.3


def url(product: str, hour: datetime) -> str:
    h = hour.astimezone(timezone.utc) if hour.tzinfo else hour
    return f"{BASE_URL}/{product}/{h:%Y%m%d}/MRMS_{product}_{h:%Y%m%d-%H}0000.grib2.gz"


def decode(gz_bytes: bytes) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """gzip'd GRIB2 -> (mm (ny, nx) with NaN for negative codes, lat1d, lon1d in -180..180)."""
    import eccodes
    from services.ensemble_fire_danger.grib_idx import _ECCODES_LOCK
    raw = gzip.decompress(gz_bytes)
    with _ECCODES_LOCK:
        gid = eccodes.codes_new_from_message(raw)
        try:
            ni, nj = int(eccodes.codes_get(gid, "Ni")), int(eccodes.codes_get(gid, "Nj"))
            values = np.asarray(eccodes.codes_get_values(gid), dtype="float64").reshape(nj, ni)
            lat1 = float(eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees"))
            lat2 = float(eccodes.codes_get(gid, "latitudeOfLastGridPointInDegrees"))
            lon1 = float(eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees"))
            lon2 = float(eccodes.codes_get(gid, "longitudeOfLastGridPointInDegrees"))
        finally:
            eccodes.codes_release(gid)
    lat = np.linspace(lat1, lat2, nj)
    lon = np.linspace(lon1, lon2, ni)
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    return np.where(values < 0, np.nan, values), lat, lon


def _crop(mm: np.ndarray, lat: np.ndarray, lon: np.ndarray, bbox: Tuple[float, float, float, float]):
    south, north, west, east = bbox
    rows = np.where((lat >= south) & (lat <= north))[0]
    cols = np.where((lon >= west) & (lon <= east))[0]
    if rows.size < 2 or cols.size < 2:
        raise ValueError("domain does not overlap the MRMS grid")
    return mm[np.ix_(rows, cols)].astype("float32"), lat[rows], lon[cols]


def fetch_hour(hour: datetime, cache_dir: Path, bbox: Tuple[float, float, float, float], *,
               session: Optional[requests.Session] = None, products: Sequence[str] = PRODUCTS) -> Optional[dict]:
    """One hour cropped to bbox=(south, north, west, east): {'mm', 'lat', 'lon', 'product'} or None if no product
    is published for that hour. Cached per hour (the product that was used is cached with it)."""
    hour = hour.astimezone(timezone.utc) if hour.tzinfo else hour.replace(tzinfo=timezone.utc)
    cache = Path(cache_dir) / f"mrms_{hour:%Y%m%d%H}.npz"
    if cache.exists():
        try:
            with np.load(cache, allow_pickle=False) as z:
                return {"mm": z["mm"], "lat": z["lat"], "lon": z["lon"], "product": str(z["product"])}
        except Exception:  # noqa: BLE001 - corrupt cache entry: refetch
            cache.unlink(missing_ok=True)
    session = session or requests.Session()
    for product in products:
        try:
            response = session.get(url(product, hour), timeout=60)
        except requests.exceptions.RequestException as error:
            logger.warning("MRMS %s %s: %s", product, hour, error)
            continue
        if response.status_code == 404:
            continue
        if response.status_code != 200:
            logger.warning("MRMS %s %s: HTTP %s", product, hour, response.status_code)
            continue
        mm, lat, lon = _crop(*decode(response.content), bbox)
        cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache.with_name(cache.name + ".tmp.npz")
        np.savez_compressed(tmp, mm=mm, lat=lat, lon=lon, product=np.array(product))
        tmp.replace(cache)
        return {"mm": mm, "lat": lat, "lon": lon, "product": product}
    return None


def _frac(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    idx = np.arange(axis.size, dtype="float64")
    return np.interp(values, axis, idx) if axis[0] < axis[-1] else np.interp(values, axis[::-1], idx[::-1])


def sample_to_grid(mm: np.ndarray, lat1d: np.ndarray, lon1d: np.ndarray,
                   sub_lat: Sequence[np.ndarray], sub_lon: Sequence[np.ndarray]) -> np.ndarray:
    """Area mean over sub-sample points inside each target cell. NaN samples are ignored; a cell with no valid
    sample is NaN (never 0)."""
    from scipy.ndimage import map_coordinates
    total = np.zeros(sub_lat[0].shape)
    count = np.zeros(sub_lat[0].shape)
    filled = np.nan_to_num(mm.astype("float64"), nan=0.0)
    valid = (~np.isnan(mm)).astype("float64")
    for la, lo in zip(sub_lat, sub_lon):
        coords = np.stack([_frac(lat1d, la), _frac(lon1d, lo)])
        v = map_coordinates(valid, coords, order=1, mode="nearest")
        total += map_coordinates(filled, coords, order=1, mode="nearest")
        count += v
    with np.errstate(all="ignore"):
        out = np.where(count > 1e-6, total / count, np.nan)   # bilinear of (value*valid) / bilinear of valid
    return out.astype("float32")


def subcell_samples(lat: np.ndarray, lon: np.ndarray, n: int = 3) -> Tuple[list, list]:
    """n x n sample points inside each target cell, spaced from the grid's own local spacing (works for any grid)."""
    dlat = np.gradient(lat, axis=0)
    dlon = np.gradient(lon, axis=1)
    offsets = (np.arange(n) - (n - 1) / 2.0) / n
    subs_lat, subs_lon = [], []
    for i in offsets:
        for j in offsets:
            subs_lat.append(lat + i * dlat)
            subs_lon.append(lon + j * dlon)
    return subs_lat, subs_lon


def make_filler(cache_dir: Path, bbox: Tuple[float, float, float, float], sub_lat: Sequence[np.ndarray],
                sub_lon: Sequence[np.ndarray], *, fetch: Callable = fetch_hour, session=None) -> Callable:
    """hour -> (mm on the target grid, product label) or None. Hours are fetched lazily (only flagged ones)."""
    def fill(hour: datetime):
        got = fetch(hour, cache_dir, bbox, session=session)
        if got is None:
            return None
        grid = sample_to_grid(got["mm"], got["lat"], got["lon"], sub_lat, sub_lon)
        return grid, got["product"]
    return fill
