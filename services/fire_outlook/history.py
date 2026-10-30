"""
Causal recent-weather replay that conditions fuel moisture before an outlook cycle.

Replaces services/spread_rate_moisture.condition_moisture for the outlook (same return keys, plus an audit) because
that function (a) replays RTMA hours whose precipitation was flagged missing in the cache (rtma_capture stores zeros with
apcp.attrs['missing']='true') as DRY hours, (b) picks the nearest RTMA pixel instead of interpolating, and (c) is not
parameterized for the 3 km public grid. The moisture step itself is moisture.advance (parity-tested against it).

Per hour (oldest first, ending at the cycle hour, nothing later is read):
  * temperature and RH come from the RTMA cache, linearly interpolated (cached Delaunay weights) to the grid; target
    cells outside the cached RTMA hull take their nearest inside cell (the public grid is ~0.3 deg wider than the cached
    RTMA crop on the west edge) and are counted in the audit.
  * precipitation is the RTMA hourly value, unless the hour is flagged missing: then MRMS (Pass2, else Pass1) is used and the
    hour is counted per product. If neither exists the hour is SKIPPED (state carried, like a missing file) and counted;
    it is never replayed as dry.
Insufficient history, or more than `max_gap_hours` unfilled precipitation hours, raises RuntimeError, which
conditioning.condition_initial_state turns into ConditioningUnavailable (or an opted-in, labelled EMC seed).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

from services.fire_outlook import moisture

logger = logging.getLogger(__name__)

TARGET_HOURS = 7 * 24
MIN_HOURS = 5 * 24
MAX_GAP_HOURS = 12


def _squeeze2d(values) -> np.ndarray:
    values = np.asarray(values)
    while values.ndim > 2:
        values = np.take(values, 0, axis=0)
    return values


def load_rtma_hour(path: Path) -> Optional[dict]:
    """One cached RTMA hour -> dict(t_c, rh, precip, precip_missing, lat, lon) or None if absent/unreadable."""
    import xarray as xr
    if not path.is_file():
        return None
    try:
        with xr.open_dataset(path) as ds:
            t = _squeeze2d(ds["t2m"].values).astype("float64")
            t = np.where(t > 150, t - 273.15, t)
            lon = _squeeze2d(ds["longitude"].values).astype("float64")
            return {
                "t_c": t, "rh": _squeeze2d(ds["r2"].values).astype("float64"),
                "precip": _squeeze2d(ds["apcp"].values).astype("float64") if "apcp" in ds else None,
                "precip_missing": ("apcp" not in ds) or str(ds["apcp"].attrs.get("missing", "")).lower() in ("true", "1"),
                "lat": _squeeze2d(ds["latitude"].values).astype("float64"),
                "lon": np.where(lon > 180.0, lon - 360.0, lon),
            }
    except Exception as error:  # noqa: BLE001 - corrupt file == missing hour
        logger.warning("RTMA cache file unreadable %s: %s", path, error)
        return None


def edge_fill_index(inside: np.ndarray) -> np.ndarray:
    """Flat index of the nearest inside cell for every cell (identity for inside cells)."""
    from scipy.ndimage import distance_transform_edt
    _, (iy, ix) = distance_transform_edt(~inside, return_indices=True)
    return (iy * inside.shape[1] + ix).ravel()


class _Grid:
    """Regrid RTMA fields to the target grid with cached weights and nearest-inside edge fill."""

    def __init__(self, src_lat, src_lon, lat, lon, cache_dir: Optional[Path]):
        from services.ensemble_fire_danger.regrid import Regridder
        self.regrid = Regridder(src_lat, src_lon, lat, lon, cache_dir=cache_dir)
        inside = self.regrid.inside.reshape(lat.shape)
        self.edge_cells = int((~inside).sum())
        self.nearest = edge_fill_index(inside) if self.edge_cells else None
        self.key = src_lat.shape

    def __call__(self, field: np.ndarray) -> np.ndarray:
        out = self.regrid(field)
        if self.nearest is not None:
            out = out.ravel()[self.nearest].reshape(out.shape)
        return out


def condition(lat: np.ndarray, lon: np.ndarray, analysis_hour: Optional[datetime] = None, raws_payload=None, *,
              hours: int = TARGET_HOURS, min_hours: int = MIN_HOURS, max_gap_hours: int = MAX_GAP_HOURS,
              rtma_dir: Optional[Path] = None, cache_dir: Optional[Path] = None,
              precip_fill: Optional[Callable] = None) -> Dict:
    """Replay up to `hours` ending at analysis_hour. raws_payload is accepted for signature parity and ignored
    (anchoring is done causally by conditioning.condition_initial_state). precip_fill(hour) -> (mm_grid, product)|None."""
    if analysis_hour is None:
        raise ValueError("analysis_hour is required (the replay is anchored on the cycle time, never on the clock)")
    end = analysis_hour.astimezone(timezone.utc).replace(tzinfo=None) if analysis_hour.tzinfo else analysis_hour
    end = end.replace(minute=0, second=0, microsecond=0)
    if rtma_dir is None:
        from services.rtma_capture import _root
        rtma_dir = _root() / "cache" / "rtma"
    rtma_dir = Path(rtma_dir)
    if cache_dir is None:
        cache_dir = Path(os.getenv("SMF_FIRE_OUTLOOK_DIR", "data/fire-outlook")) / "cache"
    cache_dir = Path(cache_dir)

    stamps = [end - timedelta(hours=offset) for offset in range(hours - 1, -1, -1)]
    present = [s for s in stamps if (rtma_dir / f"rtma_{s:%Y%m%d_%H}z.nc").is_file()]
    if len(present) < min_hours:
        raise RuntimeError(f"insufficient RTMA history for outlook conditioning: {len(present)}/{min_hours} hours")

    grid: Optional[_Grid] = None
    state = None
    audit = {"hours_replayed": 0, "hours_file_missing": 0, "precip_rtma_hours": 0, "precip_mrms_hours": 0,
             "precip_mrms_products": {}, "precip_unfilled_hours": 0, "edge_filled_cells": 0}
    unfilled: list = []
    for stamp in stamps:
        data = load_rtma_hour(rtma_dir / f"rtma_{stamp:%Y%m%d_%H}z.nc")
        if data is None:
            audit["hours_file_missing"] += 1
            continue
        if grid is None or grid.key != data["lat"].shape:
            grid = _Grid(data["lat"], data["lon"], lat, lon, cache_dir)
            audit["edge_filled_cells"] = grid.edge_cells
        t, rh = grid(data["t_c"]), grid(data["rh"])
        if data["precip_missing"]:
            filled = precip_fill(stamp) if precip_fill is not None else None
            if filled is None:
                audit["precip_unfilled_hours"] += 1
                unfilled.append(stamp.isoformat() + "Z")
                continue                                  # skip the hour: never replay missing rain as dry
            precip, product = filled
            audit["precip_mrms_hours"] += 1
            audit["precip_mrms_products"][product] = audit["precip_mrms_products"].get(product, 0) + 1
        else:
            precip = grid(np.where(np.isfinite(data["precip"]), data["precip"], 0.0))
            audit["precip_rtma_hours"] += 1
        if state is None:
            emc = moisture.nelson_emc(t, rh)
            state = {"fm1": emc.copy(), "fm10": emc.copy(), "fm100": emc.copy()}
        else:
            state = moisture.advance(state, t, rh, precip)
        audit["hours_replayed"] += 1
    if state is None:
        raise RuntimeError("no usable RTMA hours for moisture conditioning")
    if audit["precip_unfilled_hours"] > max_gap_hours:
        raise RuntimeError(f"{audit['precip_unfilled_hours']} hours had no usable precipitation (RTMA flagged missing, "
                           f"no MRMS); limit {max_gap_hours}")
    if np.isnan(state["fm10"]).any():
        raise RuntimeError("conditioned fuel moisture has NoData cells")
    audit["unfilled_hours"] = unfilled[:48]
    return {
        "analysis_hour": end.isoformat(), "fm1_pct": state["fm1"], "fm10_pct": state["fm10"], "fm100_pct": state["fm100"],
        "conditioning_hours_available": len(present), "conditioning_hours_missing": len(stamps) - len(present),
        "precip_audit": audit,
    }


def make_condition(bbox_for_mrms: Optional[tuple] = None, sub_lat=None, sub_lon=None, *, fetch_mrms=None,
                   **fixed) -> Callable:
    """A `condition(lat, lon, analysis_hour=..., raws_payload=None)` callable with MRMS gap-filling wired in, for
    conditioning.condition_initial_state(condition=...). Without a bbox no MRMS fill is attempted."""
    from services.fire_outlook import mrms

    def run(lat, lon, analysis_hour=None, raws_payload=None):
        fill = None
        cache_dir = fixed.get("cache_dir") or Path(os.getenv("SMF_FIRE_OUTLOOK_DIR", "data/fire-outlook")) / "cache"
        if bbox_for_mrms is not None:
            fill = mrms.make_filler(Path(cache_dir) / "mrms", bbox_for_mrms, sub_lat or [lat], sub_lon or [lon],
                                    **({"fetch": fetch_mrms} if fetch_mrms else {}))
        return condition(lat, lon, analysis_hour, raws_payload, precip_fill=fill, **fixed)
    return run
