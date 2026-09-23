"""
Hand-off of the operational forecast's own hourly fuel-moisture field (and
grid) from forecast/DailyForecast.py to the ensemble product.

The ensemble anchors every member's fuel moisture on the operational FM
(RAWS-initialized XGBoost, or the spatial ONNX override when it is
active) - see core.anchored_fm. DailyForecast already holds that field in
memory; it writes it here once per run (best-effort, never raising into
the forecast), and the ensemble runner reads it back. When the file is
absent (DailyForecast failed, or the ensemble is run for a day that
DailyForecast did not run), the ensemble falls back to each member's raw
XGBoost FM and records "fm_anchor": "none" in its evidence.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

STATE_DIR = Path(os.getenv("SMF_FORECAST_STATE_DIR") or (Path(os.getenv("DATA_DIR", "data")) / "forecast-state"))
KEEP_FILES = 14


def _path(run_date: datetime, state_dir: Optional[Path] = None) -> Path:
    return Path(state_dir or STATE_DIR) / f"forecast_state_{run_date:%Y%m%d_%H}z.npz"


def write(run_date, lat: np.ndarray, lon: np.ndarray, hourly_fm, swe_grid: Optional[np.ndarray],
          snow_threshold_in: float, state_dir: Optional[Path] = None) -> Optional[Path]:
    """Never raises. hourly_fm: list/array of (y, x) grids, one per forecast lead."""
    try:
        run_date = datetime(run_date.year, run_date.month, run_date.day, run_date.hour)
        path = _path(run_date, state_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            tmp,
            lat=np.asarray(lat, dtype="float64"),
            lon=np.asarray(lon, dtype="float64"),
            hourly_fm=np.stack([np.asarray(g, dtype="float32") for g in hourly_fm]),
            swe_grid=(np.asarray(swe_grid, dtype="float32") if swe_grid is not None else np.empty((0, 0), "float32")),
            snow_threshold_in=np.float64(snow_threshold_in),
        )
        os.replace(tmp, path)
        for old in sorted(path.parent.glob("forecast_state_*.npz"))[:-KEEP_FILES]:
            old.unlink(missing_ok=True)
        return path
    except Exception as error:
        logger.warning("forecast-state hand-off write failed (non-fatal): %s", error)
        return None


def read(run_date: datetime, state_dir: Optional[Path] = None) -> Optional[dict]:
    path = _path(run_date, state_dir)
    if not path.exists():
        return None
    try:
        with np.load(path) as data:
            swe = data["swe_grid"]
            return {
                "path": str(path),
                "lat": data["lat"],
                "lon": data["lon"],
                "hourly_fm": data["hourly_fm"].astype("float64"),
                "swe_grid": swe.astype("float64") if swe.size else None,
                "snow_threshold_in": float(data["snow_threshold_in"]),
            }
    except Exception as error:
        logger.warning("forecast-state %s unreadable: %s", path, error)
        return None
