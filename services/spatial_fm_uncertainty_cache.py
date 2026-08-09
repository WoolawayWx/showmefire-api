"""
Persist the fuel-moisture spatial model's p10/p90/confidence grids (and
the two station-geometry diagnostics) to a small per-run cache file.

spatial_fm.py::try_predict already computes these every run - the public
forecast path only ever reads result["p50"] and throws the rest away
(DailyForecast.py:932). This module makes the discarded quantiles
available to the future risk-fusion feature builder without touching
anything on the public path: it is called once, read-only, right after
try_predict_spatial_fm() returns, and every failure is swallowed so a
caching bug can never affect forecast generation.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data") / "spatial_fm_uncertainty"
RETENTION_DAYS = 14


def _run_id(run_date) -> str:
    stamp = pd.Timestamp(run_date) if run_date is not None else pd.Timestamp(datetime.now(timezone.utc))
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.strftime("%Y%m%d_%H")


def persist(spatial_prediction: Optional[dict], run_date=None) -> Optional[Path]:
    """
    Write spatial_prediction's p10/p90/confidence/station-geometry arrays
    to CACHE_DIR/{run_id}.npz. Returns the written path, or None if there
    was nothing to persist or persistence failed - callers should not
    branch on the return value beyond logging.
    """
    if not spatial_prediction:
        return None
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        run_id = _run_id(run_date)
        target = CACHE_DIR / f"{run_id}.npz"
        np.savez_compressed(
            target,
            p10=spatial_prediction["p10"],
            p90=spatial_prediction["p90"],
            confidence=spatial_prediction["confidence"],
            nearest_station_distance_deg=spatial_prediction["nearest_station_distance_deg"],
            effective_station_count=spatial_prediction["effective_station_count"],
        )
        (CACHE_DIR / f"{run_id}.json").write_text(json.dumps({
            "run_id": run_id,
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "steps": int(spatial_prediction["p10"].shape[0]),
            "grid_shape": list(spatial_prediction["p10"].shape[1:]),
        }), encoding="utf-8")
        return target
    except Exception as exc:
        logger.warning("Could not persist spatial FM uncertainty (non-fatal): %s", exc)
        return None


def purge_stale(retention_days: int = RETENTION_DAYS) -> int:
    """Delete cache files older than retention_days. Never raises."""
    if not CACHE_DIR.exists():
        return 0
    cutoff = datetime.now(timezone.utc).timestamp() - retention_days * 86400
    removed = 0
    for path in CACHE_DIR.glob("*"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            continue
    return removed
