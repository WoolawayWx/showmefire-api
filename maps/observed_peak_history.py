"""Tracks the running maximum observed fire-danger category grid for "today".

Companion to station_danger_history.py's station-count rolling history, but
per-pixel: realtimefiredanger.py runs on an external ~15-minute cadence and
each run only reflects a single instant. This module folds each run's grid
into a running elementwise max for the current local day (America/Chicago),
so at any point you can see the worst danger category any pixel has reached
so far today - the observed counterpart to the forecast pipeline's
peak_risk_smooth (gis/peak_fire_danger.tif).
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Dict

import numpy as np
from zoneinfo import ZoneInfo

try:
    from core.config import GIS_DIR
except ModuleNotFoundError:
    # Allow running map scripts directly (python ./maps/<script>.py)
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_DIR = SCRIPT_DIR.parent
    GIS_DIR = PROJECT_DIR / "gis"
    sys.path.append(str(PROJECT_DIR))

from realtime_geotiff import export_discrete_rgba_geotiff

try:
    from scripts.upload_cdn import upload_to_cdn
except Exception:
    # CDN upload is a best-effort side artifact - never let a missing
    # dependency (boto3, R2 credentials) break the running-max tracking.
    upload_to_cdn = None

CHICAGO_TZ = ZoneInfo("America/Chicago")

OBSERVED_PEAK_DIR = Path(GIS_DIR) / "observed_peak"
ARCHIVE_DIR = OBSERVED_PEAK_DIR / "archive"
STATE_FILE = OBSERVED_PEAK_DIR / "state.json"
CURRENT_DAY_FILE = OBSERVED_PEAK_DIR / "current_day.npy"
TODAY_TIF = Path(GIS_DIR) / "observed_peak_today.tif"

DANGER_CLASS_COLORS = {
    0: (144, 238, 144, 255),  # Low
    1: (255, 237, 78, 255),   # Moderate
    2: (255, 165, 0, 255),    # Elevated
    3: (255, 0, 0, 255),      # Critical
    4: (139, 0, 0, 255),      # Extreme
}
BOUNDS_TOLERANCE = 1e-6


def _atomic_write_bytes(file_path: Path, write_fn) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("wb", delete=False, dir=file_path.parent) as tmp_file:
        write_fn(tmp_file)
        tmp_path = Path(tmp_file.name)
    tmp_path.replace(file_path)


def _atomic_write_text(file_path: Path, content: str) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", delete=False, dir=file_path.parent, encoding="utf-8") as tmp_file:
        tmp_file.write(content)
        tmp_path = Path(tmp_file.name)
    tmp_path.replace(file_path)


def _save_running_grid(grid: np.ndarray) -> None:
    def _write(tmp_file):
        np.save(tmp_file, grid)
    # np.save appends a .npy suffix unless the file object is already open,
    # which is the case here (tmp_file is a real file handle).
    _atomic_write_bytes(CURRENT_DAY_FILE, _write)


def _load_state() -> Dict[str, Any] | None:
    if not STATE_FILE.exists():
        return None
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _save_state(state: Dict[str, Any]) -> None:
    _atomic_write_text(STATE_FILE, json.dumps(state, indent=2))


def _load_running_grid() -> np.ndarray | None:
    if not CURRENT_DAY_FILE.exists():
        return None
    try:
        return np.load(CURRENT_DAY_FILE)
    except (OSError, ValueError):
        return None


def _grid_metadata(grid_values: np.ndarray, lon_mesh: np.ndarray, lat_mesh: np.ndarray) -> Dict[str, Any]:
    return {
        "shape": list(np.asarray(grid_values).shape),
        "lon_min": float(np.nanmin(lon_mesh)),
        "lon_max": float(np.nanmax(lon_mesh)),
        "lat_min": float(np.nanmin(lat_mesh)),
        "lat_max": float(np.nanmax(lat_mesh)),
    }


def _matches_state(state: Dict[str, Any], metadata: Dict[str, Any]) -> bool:
    if state.get("shape") != metadata["shape"]:
        return False
    for key in ("lon_min", "lon_max", "lat_min", "lat_max"):
        if abs(float(state.get(key, float("nan"))) - metadata[key]) > BOUNDS_TOLERANCE:
            return False
    return True


def _export_tif(grid: np.ndarray, lon_mesh: np.ndarray, lat_mesh: np.ndarray, out_path: Path, description: str) -> bool:
    return export_discrete_rgba_geotiff(
        grid_values=grid,
        lon_mesh=lon_mesh,
        lat_mesh=lat_mesh,
        out_path=out_path,
        class_colors=DANGER_CLASS_COLORS,
        description=description,
        source="Synoptic observations + ShowMeFire fire danger model",
        legend="Low=green, Moderate=yellow, Elevated=orange, Critical=red, Extreme=dark red",
    )


def _upload_daily_archive_to_cdn(archive_path: Path, date_local: str) -> None:
    """Push the finished day's archive tif to the CDN's ml-support/ folder.

    Filename is the compact date + a fixed suffix (e.g. 20260813-obs_peak_fd.tif)
    so downstream ML tooling can find historical observed-peak grids without
    depending on this API server's local disk retention.
    """
    if upload_to_cdn is None:
        return
    date_compact = date_local.replace("-", "")
    dest_key = f"ml-support/{date_compact}-obs_peak_fd.tif"
    try:
        upload_to_cdn(
            [archive_path],
            [dest_key],
            content_types=["image/tiff"],
            cache_controls=["public, max-age=31536000, immutable"],
        )
    except Exception as e:
        print(f"Warning: Failed to upload observed peak archive to CDN: {e}")


def _peak_class_from_grid(grid: np.ndarray) -> int | None:
    finite = np.asarray(grid, dtype=float)
    if not np.isfinite(finite).any():
        return None
    return int(np.nanmax(finite))


def load_today_running_grid() -> np.ndarray | None:
    """Return today's in-progress running-max grid, if any.

    Used by realtimefiredanger.py right after update_observed_peak_grid() to
    render a styled PNG of the same array already folded/saved to disk here.
    """
    return _load_running_grid()


def update_observed_peak_grid(
    grid_values: np.ndarray,
    lon_mesh: np.ndarray,
    lat_mesh: np.ndarray,
    run_time_utc: datetime | None = None,
) -> Dict[str, Any]:
    """Fold this run's grid into the running max for today; archive on day rollover.

    Also tracks the single highest class reached anywhere in the grid today
    and when it was first reached, so a lightweight "today's peak" summary
    can be read from state.json without touching the raster.

    Returns a status dict: {"date_local", "rolled_over", "archived_path",
    "today_path", "shape", "peak_class", "peak_reached_at_utc"}.
    """
    run_time_utc = (run_time_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    today_local = run_time_utc.astimezone(CHICAGO_TZ).strftime("%Y-%m-%d")
    run_time_iso = run_time_utc.isoformat().replace("+00:00", "Z")

    metadata = _grid_metadata(grid_values, lon_mesh, lat_mesh)
    state = _load_state()
    running_grid = _load_running_grid() if state is not None else None

    rolled_over = False
    archived_path = None

    state_valid = state is not None and running_grid is not None and _matches_state(state, metadata)

    if not state_valid or state.get("date_local") != today_local:
        # Day rollover (or missing/incompatible/first-ever state): archive the
        # previous day's finished grid, if there was one, then start fresh.
        if state_valid and state.get("date_local"):
            archive_path = ARCHIVE_DIR / f"{state['date_local']}.tif"
            ok = _export_tif(
                running_grid, lon_mesh, lat_mesh, archive_path,
                description=f"Missouri observed peak fire danger classes (RGBA) - {state['date_local']}",
            )
            if ok:
                archived_path = str(archive_path)
                _upload_daily_archive_to_cdn(archive_path, state["date_local"])
        running_grid = np.asarray(grid_values, dtype=float).copy()
        rolled_over = True
        peak_class = _peak_class_from_grid(running_grid)
        peak_reached_at_utc = run_time_iso if peak_class is not None else None
    else:
        running_grid = np.fmax(running_grid, np.asarray(grid_values, dtype=float))
        peak_class = _peak_class_from_grid(running_grid)
        previous_peak_class = state.get("peak_class")
        if peak_class is not None and (previous_peak_class is None or peak_class > previous_peak_class):
            peak_reached_at_utc = run_time_iso
        else:
            peak_reached_at_utc = state.get("peak_reached_at_utc")

    _save_running_grid(running_grid)
    _save_state({
        "date_local": today_local,
        "peak_class": peak_class,
        "peak_reached_at_utc": peak_reached_at_utc,
        **metadata,
    })
    _export_tif(
        running_grid, lon_mesh, lat_mesh, TODAY_TIF,
        description=f"Missouri observed peak fire danger classes (RGBA) - {today_local} (in progress)",
    )

    return {
        "date_local": today_local,
        "rolled_over": rolled_over,
        "archived_path": archived_path,
        "today_path": str(TODAY_TIF),
        "shape": metadata["shape"],
        "peak_class": peak_class,
        "peak_reached_at_utc": peak_reached_at_utc,
    }
