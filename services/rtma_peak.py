"""Build an additive, RTMA-driven daily peak fire-danger surface.

This is intentionally separate from observed_peak_history.py.  The existing
peak is the running maximum of the station-interpolated operational surface;
this product uses RTMA's hourly gridded weather to provide a second spatial
diagnostic without replacing that established artifact.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from zoneinfo import ZoneInfo

from core.config import GIS_DIR, IMAGES_DIR
from core.fire_danger import calculate_fire_danger
from maps.observed_peak_history import DANGER_CLASS_COLORS
from maps.realtime_geotiff import export_discrete_rgba_geotiff
from services.rtma_capture import fetch_rtma

logger = logging.getLogger(__name__)

CHICAGO_TZ = ZoneInfo("America/Chicago")
RTMA_PEAK_DIR = Path(GIS_DIR) / "rtma_peak"
RTMA_PEAK_ARCHIVE_DIR = RTMA_PEAK_DIR / "archive"
RTMA_PEAK_TODAY_TIF = Path(GIS_DIR) / "rtma_peak_today.tif"
RTMA_PEAK_IMAGE_DIR = Path(IMAGES_DIR) / "rtma_peak"
RTMA_PEAK_IMAGE_ARCHIVE_DIR = RTMA_PEAK_IMAGE_DIR / "archive"
RTMA_PEAK_TODAY_PNG = Path(IMAGES_DIR) / "mo-rtma-observedpeakfiredanger.png"


def _hours_for_local_date(target_date: date):
    start = datetime.combine(target_date, datetime.min.time(), tzinfo=CHICAGO_TZ)
    return [start.astimezone(timezone.utc) + timedelta(hours=i) for i in range(24)]


def _classify_grid(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return class grid, longitude mesh, and latitude mesh for one RTMA hour."""
    lat = np.asarray(ds["latitude"].values)
    lon = np.asarray(ds["longitude"].values)
    rh = np.asarray(ds["r2"].values, dtype=float)
    wind = np.hypot(
        np.asarray(ds["u10"].values, dtype=float),
        np.asarray(ds["v10"].values, dtype=float),
    ) * 1.9438444924406

    # RTMA has no fuel-moisture field.  This transparent RH-based estimate is
    # only the weather-driven spatial diagnostic; it is not a training label.
    fuel_moisture = 3.0 + 0.25 * rh
    valid = np.isfinite(fuel_moisture) & np.isfinite(rh) & np.isfinite(wind)
    result = np.full(rh.shape, np.nan, dtype=float)
    classify = np.vectorize(calculate_fire_danger, otypes=[float])
    result[valid] = classify(fuel_moisture[valid], rh[valid], wind[valid])
    return result, lon, lat


def _render_png(grid: np.ndarray, lon: np.ndarray, lat: np.ndarray, out_path: Path, target_date: date) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    colors = [tuple(channel / 255 for channel in DANGER_CLASS_COLORS[i][:3]) for i in range(5)]
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    cmap.set_bad((0, 0, 0, 0))
    fig, ax = plt.subplots(figsize=(12, 7), dpi=144)
    ax.pcolormesh(lon, lat, grid, cmap=cmap, vmin=0, vmax=4, shading="auto")
    ax.set_title(f"RTMA Weather-Derived Peak Fire Danger — {target_date.isoformat()}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, transparent=False)
    plt.close(fig)
    return out_path


def generate_rtma_peak(target_date: str | date | None = None) -> dict:
    """Generate and archive the RTMA peak for a local date.

    Missing individual RTMA hours are skipped.  The run fails only when no
    usable hours exist, which makes historical retries safe and resumable.
    """
    if target_date is None:
        local_date = datetime.now(CHICAGO_TZ).date()
    elif isinstance(target_date, date):
        local_date = target_date
    else:
        try:
            local_date = datetime.strptime(target_date, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError(f"date must be YYYY-MM-DD, got: {target_date}") from exc

    peak = None
    lon = lat = None
    used_hours = []
    for hour in _hours_for_local_date(local_date):
        try:
            path = fetch_rtma(hour)
            with xr.open_dataset(path) as ds:
                current, current_lon, current_lat = _classify_grid(ds)
                if peak is None:
                    peak = current
                    lon, lat = current_lon, current_lat
                elif current.shape == peak.shape and np.allclose(current_lon, lon) and np.allclose(current_lat, lat):
                    peak = np.fmax(peak, current)
                else:
                    logger.warning("Skipping RTMA hour %s because its grid does not match the first hour", hour)
                    continue
                used_hours.append(hour.isoformat())
        except Exception:
            logger.exception("Unable to process RTMA hour %s for %s", hour, local_date)

    if peak is None or not np.isfinite(peak).any():
        raise RuntimeError(f"No usable RTMA analyses found for {local_date}")

    tif_path = RTMA_PEAK_ARCHIVE_DIR / f"{local_date.isoformat()}.tif"
    png_path = RTMA_PEAK_IMAGE_ARCHIVE_DIR / f"{local_date.isoformat()}.png"
    export_discrete_rgba_geotiff(
        peak, lon, lat, tif_path, DANGER_CLASS_COLORS,
        f"RTMA weather-derived peak fire danger - {local_date}",
        "RTMA t2m/r2/u10/v10 + RH-derived fuel-moisture diagnostic",
        "Low=green, Moderate=yellow, Elevated=orange, Critical=red, Extreme=dark red",
    )
    export_discrete_rgba_geotiff(
        peak, lon, lat, RTMA_PEAK_TODAY_TIF, DANGER_CLASS_COLORS,
        f"RTMA weather-derived peak fire danger - {local_date}",
        "RTMA t2m/r2/u10/v10 + RH-derived fuel-moisture diagnostic",
    )
    _render_png(peak, lon, lat, png_path, local_date)
    _render_png(peak, lon, lat, RTMA_PEAK_TODAY_PNG, local_date)
    return {
        "date": local_date.isoformat(),
        "hours_used": len(used_hours),
        "peak_class": int(np.nanmax(peak)),
        "tif": f"rtma_peak/archive/{local_date.isoformat()}.tif",
        "png": f"rtma_peak/archive/{local_date.isoformat()}.png",
    }


async def run_rtma_peak_job():
    try:
        import asyncio
        result = await asyncio.to_thread(generate_rtma_peak)
        logger.info("RTMA peak generated: %s", result)
    except Exception:
        logger.exception("Scheduled RTMA peak generation failed")
