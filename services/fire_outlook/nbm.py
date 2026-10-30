"""
National Blend of Models (NBM) reference forecast at station locations: the plan's "separate detailed reference forecast
and comparison baseline" for weather verification. It is NEVER mixed into the GEFS scenario agreement, and nothing here
touches the public grids; it only adds `<variable>_nbm` columns to the verification rows so the training tool can compare
the GEFS weather against a deterministic blend.

Source: noaa-nbm-grib2-pds `blend.YYYYMMDD/HH/core/blend.tHHz.core.fFFF.co.grib2` (CONUS 2.5 km). Cadence (probed lead by
lead, 2026-10-28 00Z): HOURLY to f048, then every 3 h to f192 (f049, f050, f052 ... do not exist). Fields per lead, by byte
range (the file is ~200 MB, never fetch it whole): TMP 2 m, RH 2 m, WIND 10 m (sustained speed, m/s), and at multiples of 6 h
the 6-hour APCP window ending at the lead (aligned with the outlook's windows, so no 1-hour sums are needed). Only the
nearest grid point to each station is kept.
Window statistics use the forecast side's definition and hour convention (step k = valid time cycle+k h belongs to window
(k-1)//6): max temperature, min RH, max wind, 6 h precipitation total. After f048 the instantaneous fields are linearly
interpolated to hourly, exactly as the GEFS 3-hourly fields are, so the two sources are treated alike (a window max from
3-hourly data can still sit below the true max; the same limitation applies to both).
Anything missing returns None for the whole cycle (a partial baseline would bias the comparison); the outlook is unaffected.
"""
from __future__ import annotations

import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from services.fire_outlook import acquisition as _acq  # session helper, workers default
from services.fire_outlook import windows

logger = logging.getLogger(__name__)

BASE_URL = os.getenv("SMF_FIRE_OUTLOOK_NBM_BASE", "https://noaa-nbm-grib2-pds.s3.amazonaws.com")
HORIZON = windows.HORIZON_HOURS
LEADS = tuple(range(1, 49)) + tuple(range(51, HORIZON + 1, 3))          # published leads (hourly, then 3-hourly)
SEARCHES = {
    "t2m": r"^:TMP:2 m above ground:\d+ hour fcst:?$",
    "rh": r"^:RH:2 m above ground:\d+ hour fcst:?$",
    "wind": r"^:WIND:10 m above ground:\d+ hour fcst:?$",
    "apcp": r"^:APCP:surface:\d+-\d+ hour acc fcst:?$",
}
NEAREST_MAX_KM = 6.0


class NBMUnavailable(RuntimeError):
    """A needed NBM lead/field is missing; no baseline is produced for this cycle."""


def grib_url(cycle: datetime, lead: int) -> str:
    c = cycle.astimezone(timezone.utc)
    return f"{BASE_URL}/blend.{c:%Y%m%d}/{c:%H}/core/blend.t{c:%H}z.core.f{lead:03d}.co.grib2"


def nearest_cells(lat2d: np.ndarray, lon2d: np.ndarray, stations: List[dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest NBM (row, col) for each station and the distance in km. Searches only a crop around the stations."""
    from scipy.spatial import cKDTree
    s_lat = np.array([s["lat"] for s in stations])
    s_lon = np.array([s["lon"] for s in stations])
    margin = 0.5
    inside = ((lat2d >= s_lat.min() - margin) & (lat2d <= s_lat.max() + margin) &
              (lon2d >= s_lon.min() - margin) & (lon2d <= s_lon.max() + margin))
    rr, cc = np.where(inside)
    if rr.size == 0:
        raise NBMUnavailable("stations are outside the NBM grid")
    coslat = np.cos(np.radians(s_lat.mean()))
    pts = np.column_stack([lat2d[rr, cc], lon2d[rr, cc] * coslat])
    dist, idx = cKDTree(pts).query(np.column_stack([s_lat, s_lon * coslat]))
    return rr[idx], cc[idx], dist * 111.2


def fetch_lead(session, cycle: datetime, lead: int, stations: List[dict], cells: Optional[Tuple], tmp_dir: Path):
    """One lead -> dict(values at stations; 'precip6' only at multiples of 6 h) and the cells used (from the first lead)."""
    from services.ensemble_fire_danger import grib_idx
    url = grib_url(cycle, lead)
    try:
        found, lat, lon = grib_idx.fetch_fields(session, url, SEARCHES, tmp_dir)
    except grib_idx.SourceUnavailable as error:
        raise NBMUnavailable(f"NBM f{lead:03d} not published: {error}") from error
    if lat.size == 0:
        raise NBMUnavailable(f"NBM f{lead:03d}: no matching fields")
    if cells is None:
        rows, cols, km = nearest_cells(lat, lon, stations)
        if (km > NEAREST_MAX_KM).any():
            raise NBMUnavailable("a station is more than %.0f km from the nearest NBM point" % NEAREST_MAX_KM)
        cells = (rows, cols)
    rows, cols = cells
    out = {}
    for key in ("t2m", "rh", "wind"):
        if len(found[key]) != 1:
            raise NBMUnavailable(f"NBM f{lead:03d}: expected one {key} field, found {len(found[key])}")
        out[key] = found[key][0][1][rows, cols].astype("float64")
    if lead % windows.WINDOW_HOURS == 0:
        six = grib_idx.apcp_windows(found["apcp"]).get((lead - windows.WINDOW_HOURS, lead))
        if six is None:
            raise NBMUnavailable(f"NBM f{lead:03d}: no {windows.WINDOW_HOURS}-hour precipitation window ending at the lead")
        out["precip6"] = six[rows, cols].astype("float64")
    return out, cells


def station_series(cycle: datetime, stations: List[dict], cache_dir: Path, *, workers: int = _acq.DEFAULT_WORKERS,
                   fetch=fetch_lead) -> Dict[str, np.ndarray]:
    """Station series: hourly (168, S) t_c, rh, wind_ms (interpolated between published leads) and precip6 (28, S)
    6-hour totals. Cached per cycle + station list."""
    cache_dir = Path(cache_dir)
    ids = [s["station_id"] for s in stations]
    c = cycle.astimezone(timezone.utc)
    path = cache_dir / f"nbm_{c:%Y%m%dT%HZ}.npz"
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as z:
                if list(z["station_ids"]) == ids:
                    return {k: z[k] for k in ("t_c", "rh", "wind_ms", "precip6")}
        except Exception:  # noqa: BLE001 - corrupt/stale cache: refetch
            path.unlink(missing_ok=True)
    session = _acq._session(workers)
    tmp_dir = Path(tempfile.gettempdir())
    first, cells = fetch(session, cycle, LEADS[0], stations, None, tmp_dir)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rest = list(pool.map(lambda lead: fetch(session, cycle, lead, stations, cells, tmp_dir)[0], LEADS[1:]))
    rows = [first] + rest

    def hourly(key, transform=lambda a: a):
        data = transform(np.stack([r[key] for r in rows]))                       # (len(LEADS), S)
        target = np.arange(1, HORIZON + 1)
        return np.stack([np.interp(target, LEADS, data[:, s]) for s in range(data.shape[1])], axis=1)
    six = [r["precip6"] for lead, r in zip(LEADS, rows) if lead % windows.WINDOW_HOURS == 0]
    if len(six) != windows.N_WINDOWS:
        raise NBMUnavailable(f"expected {windows.N_WINDOWS} six-hour precipitation windows, got {len(six)}")
    series = {
        "t_c": hourly("t2m", lambda a: a - 273.15),
        "rh": np.clip(hourly("rh"), 0.0, 100.0),
        "wind_ms": hourly("wind"),
        "precip6": np.clip(np.stack(six), 0.0, None),
    }
    if any(np.isnan(v).any() for v in series.values()):
        raise NBMUnavailable("NBM series has missing values at a station")
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, station_ids=np.array(ids), **series)
    tmp.replace(path)
    return series


def window_stats(series: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """-> (28, S) per window: t2m_c (max), rh_pct (min), wind_ms (max), precip_mm (the 6-hour accumulation)."""
    def by_window(a):
        return a.reshape((windows.N_WINDOWS, windows.WINDOW_HOURS) + a.shape[1:])
    return {"t2m_c": by_window(series["t_c"]).max(axis=1), "rh_pct": by_window(series["rh"]).min(axis=1),
            "wind_ms": by_window(series["wind_ms"]).max(axis=1), "precip_mm": series["precip6"]}


def reference_for(cycle: datetime, stations: List[dict], cache_dir: Path, **kwargs) -> Optional[Dict[str, np.ndarray]]:
    """Window statistics at the stations, or None when NBM cannot supply a complete baseline (never partial)."""
    try:
        return window_stats(station_series(cycle, stations, cache_dir, **kwargs))
    except NBMUnavailable as error:
        logger.warning("NBM baseline unavailable for %s: %s", cycle, error)
        return None
