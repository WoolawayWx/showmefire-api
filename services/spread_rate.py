"""Hourly Rothermel surface head-fire spread-rate product for the Testbed."""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from services.beta_products import BETA_ROOT, load_manifest, save_manifest
from services.fire_behavior_static import diagnostics as static_diagnostics, load_static_fields
from services.rtma_capture import (
    ensure_latest_analysis_cached,
    is_analysis_hour_cached,
    latest_complete_hour,
    spread_rate_poll_minutes,
    warmup_rtma_cache,
)
from services.spread_rate_moisture import (
    MIN_CONDITIONING_HOURS,
    TARGET_CONDITIONING_HOURS,
    condition_moisture,
)

logger = logging.getLogger(__name__)

CHICAGO_TZ = ZoneInfo("America/Chicago")
PRODUCT_VERSION = "1.0.0"
PYROTECHNICS_VERSION = "2025.5.15"
REFRESH_INTERVAL_MINUTES = spread_rate_poll_minutes()

SPREAD_RATE_DIR = BETA_ROOT / "spread_rate"
SPREAD_RATE_GIS_DIR = BETA_ROOT / "gis" / "spread_rate"
SPREAD_RATE_IMAGE_DIR = BETA_ROOT / "images"
STATUS_PATH = SPREAD_RATE_DIR / "status.json"
TIF_PATH = SPREAD_RATE_GIS_DIR / "spread_rate_latest.tif"
PNG_PATH = SPREAD_RATE_IMAGE_DIR / "spread_rate_latest.png"

M_PER_MIN_TO_CH_PER_H = 60.0 / 20.1168
FT_PER_M = 3.28084
MS_TO_FT_PER_MIN = FT_PER_M * 60.0
WIND_10M_TO_20FT = 0.9

ROS_CLASS_BOUNDS = (0.0, 2.0, 5.0, 20.0, 50.0, 150.0)
ROS_CLASS_LABELS = ("Very Low", "Low", "Moderate", "High", "Very High", "Extreme")
ROS_CLASS_COLORS = ("#90EE90", "#ADFF2F", "#FFFF00", "#FFA500", "#FF4500", "#8B0000")

NODATA_CLASS = 255
NODATA_FLOAT = -9999.0


def _now_local() -> datetime:
    return datetime.now(CHICAGO_TZ)


def _atomic_replace(temp_path: Path, final_path: Path) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    os.replace(temp_path, final_path)


def _atomic_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        _atomic_replace(Path(temporary), path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_status() -> dict:
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def m_per_min_to_ch_per_h(rate_m_per_min: float) -> float:
    return float(rate_m_per_min) * M_PER_MIN_TO_CH_PER_H


def classify_ros_ch_per_h(rate: np.ndarray) -> np.ndarray:
    """Map continuous chains/hour to Scott–Burgan six-class indices."""
    classes = np.full(rate.shape, NODATA_CLASS, dtype=np.uint8)
    valid = np.isfinite(rate) & (rate >= 0.0)
    if not np.any(valid):
        return classes
    values = rate[valid]
    bucket = np.digitize(values, ROS_CLASS_BOUNDS, right=False) - 1
    bucket = np.clip(bucket, 0, len(ROS_CLASS_LABELS) - 1)
    classes[valid] = bucket.astype(np.uint8)
    return classes


def wind_from_degrees(u_ms: np.ndarray, v_ms: np.ndarray) -> np.ndarray:
    """Meteorological wind-from direction in degrees clockwise from north."""
    return (np.degrees(np.arctan2(-u_ms, -v_ms)) + 360.0) % 360.0


def aspect_degrees(aspect_sin: np.ndarray, aspect_cos: np.ndarray) -> np.ndarray:
    return (np.degrees(np.arctan2(aspect_sin, aspect_cos)) + 360.0) % 360.0


def spread_direction_degrees(direction_vector: tuple[float, float, float]) -> float:
    x, y, _z = direction_vector
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def _percent_to_fraction(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float) / 100.0, 0.0, 1.5)


def _compute_cell_ros(
    fuel_code: int,
    fm1: float,
    fm10: float,
    fm100: float,
    live_herb: float,
    live_woody: float,
    wind_ms: float,
    wind_from_deg: float,
    slope_deg: float,
    aspect_deg: float,
    canopy_cover_pct: float,
    canopy_height_m: float,
) -> tuple[float, float] | None:
    from pyretechnics.fuel_models import fuel_model_exists, get_fuel_model, moisturize
    from pyretechnics.surface_fire import (
        calc_midflame_wind_speed,
        calc_surface_fire_behavior_max,
        calc_surface_fire_behavior_no_wind_no_slope,
    )

    if fuel_code < 1 or fuel_code > 999 or not fuel_model_exists(int(fuel_code)):
        return None
    fuel_model = get_fuel_model(int(fuel_code))
    if not fuel_model.get("burnable", True):
        return None
    moisture = (
        fm1,
        fm10,
        fm100,
        live_herb,
        live_woody,
        live_woody,
    )
    moisturized = moisturize(fuel_model, moisture)
    surface_min = calc_surface_fire_behavior_no_wind_no_slope(moisturized)
    if surface_min["base_spread_rate"] <= 0.0:
        return None
    wind_20ft_ft_min = max(0.0, float(wind_ms)) * MS_TO_FT_PER_MIN * WIND_10M_TO_20FT
    bed_depth_ft = float(moisturized["delta"])
    canopy_cover = float(np.clip(canopy_cover_pct / 100.0, 0.0, 1.0))
    canopy_height_ft = max(0.0, float(canopy_height_m) * FT_PER_M)
    midflame = calc_midflame_wind_speed(wind_20ft_ft_min, bed_depth_ft, canopy_height_ft, canopy_cover)
    slope_fraction = math.tan(math.radians(max(0.0, float(slope_deg))))
    behavior = calc_surface_fire_behavior_max(
        surface_min,
        midflame,
        float(wind_from_deg),
        slope_fraction,
        float(aspect_deg),
    )
    ros_ch_h = m_per_min_to_ch_per_h(behavior["max_spread_rate"])
    direction = spread_direction_degrees(behavior["max_spread_direction"])
    return ros_ch_h, direction


def compute_spread_rate_grid(static: dict, moisture: dict) -> dict[str, np.ndarray]:
    shape = static["lat"].shape
    ros = np.full(shape, np.nan, dtype=np.float32)
    direction = np.full(shape, np.nan, dtype=np.float32)

    fm1 = _percent_to_fraction(moisture["fm1_pct"])
    fm10 = _percent_to_fraction(moisture["fm10_pct"])
    fm100 = _percent_to_fraction(moisture["fm100_pct"])
    live_herb = _percent_to_fraction(moisture["live_herbaceous_pct"])
    live_woody = _percent_to_fraction(moisture["live_woody_pct"])
    wind_ms = np.asarray(moisture["wind_ms"], dtype=float)
    wind_from = wind_from_degrees(np.asarray(moisture["u_ms"]), np.asarray(moisture["v_ms"]))
    aspect = aspect_degrees(static["aspect_sin"], static["aspect_cos"])

    valid = static["valid_mask"] & np.isfinite(wind_ms)
    rows, cols = np.where(valid)
    for row, col in zip(rows, cols):
        fuel_code = int(static["fuel_model_code"][row, col])
        if 91 <= fuel_code <= 99:
            continue
        try:
            result = _compute_cell_ros(
                fuel_code,
                float(fm1[row, col]),
                float(fm10[row, col]),
                float(fm100[row, col]),
                float(live_herb[row, col]),
                float(live_woody[row, col]),
                float(wind_ms[row, col]),
                float(wind_from[row, col]),
                float(static["slope_deg"][row, col]),
                float(aspect[row, col]),
                float(static["canopy_cover_pct"][row, col]),
                float(static["canopy_height_m"][row, col]),
            )
        except Exception:
            continue
        if result is None:
            continue
        ros[row, col], direction[row, col] = result
    classes = classify_ros_ch_per_h(ros)
    return {
        "ros_ch_per_h": ros,
        "class": classes,
        "spread_direction_deg": direction,
        "fm10_pct": np.asarray(moisture["fm10_pct"], dtype=np.float32),
    }


def _write_geotiff(static: dict, grids: dict[str, np.ndarray], out_path: Path) -> None:
    import rasterio
    from rasterio.transform import from_bounds

    lat = static["lat"]
    lon = static["lon"]
    west, east = float(np.nanmin(lon)), float(np.nanmax(lon))
    south, north = float(np.nanmin(lat)), float(np.nanmax(lat))
    height, width = lat.shape
    transform = from_bounds(west, south, east, north, width, height)

    ros = grids["ros_ch_per_h"].astype(np.float32)
    ros[~np.isfinite(ros)] = NODATA_FLOAT
    direction = grids["spread_direction_deg"].astype(np.float32)
    direction[~np.isfinite(direction)] = NODATA_FLOAT
    fm10 = grids["fm10_pct"].astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".spread_rate.", suffix=".tif", dir=out_path.parent)
    os.close(fd)
    temp_path = Path(temporary)
    try:
        with rasterio.open(
            temp_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=4,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
            nodata=NODATA_FLOAT,
        ) as dataset:
            dataset.write(ros, 1)
            dataset.set_band_description(1, "head_fire_ros_ch_per_h")
            class_band = grids["class"].astype(np.float32)
            class_band[class_band == NODATA_CLASS] = NODATA_FLOAT
            dataset.write(class_band, 2)
            dataset.set_band_description(2, "ros_class_index")
            dataset.write(direction, 3)
            dataset.set_band_description(3, "max_spread_direction_deg")
            dataset.write(fm10, 4)
            dataset.set_band_description(4, "fm10_pct")
        _atomic_replace(temp_path, out_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


def _render_png(grids: dict[str, np.ndarray], static: dict, out_path: Path, metadata: dict) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    classes = grids["class"].astype(np.float32)
    classes[classes == NODATA_CLASS] = np.nan
    cmap = ListedColormap(ROS_CLASS_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, len(ROS_CLASS_LABELS) + 0.5, 1.0), len(ROS_CLASS_LABELS))

    fig, ax = plt.subplots(figsize=(10, 6), dpi=144)
    lon = static["lon"]
    lat = static["lat"]
    mesh = ax.pcolormesh(lon, lat, classes, cmap=cmap, norm=norm, shading="auto")
    cbar = fig.colorbar(mesh, ax=ax, ticks=np.arange(len(ROS_CLASS_LABELS)))
    cbar.ax.set_yticklabels(ROS_CLASS_LABELS)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Observed Rothermel Spread Rate (Experimental)")
    subtitle = (
        f"Analysis {metadata.get('analysis_hour', 'n/a')} · "
        f"Generated {metadata.get('generated_at', 'n/a')} · "
        f"Max {metadata.get('max_ros_ch_per_h', 'n/a')} ch/h"
    )
    ax.text(0.5, 1.02, subtitle, transform=ax.transAxes, ha="center", fontsize=9)
    fig.tight_layout()
    fd, temporary = tempfile.mkstemp(prefix=".spread_rate.", suffix=".png", dir=out_path.parent)
    os.close(fd)
    temp_path = Path(temporary)
    try:
        fig.savefig(temp_path, bbox_inches="tight")
        plt.close(fig)
        _atomic_replace(temp_path, out_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        plt.close("all")


def _update_manifest(status: dict) -> None:
    manifest = load_manifest()
    manifest["spread_rate_updated_at"] = status.get("generated_at")
    manifest.setdefault("products", {})["spread_rate"] = {
        "kind": "raster",
        "path": "gis/spread_rate/spread_rate_latest.tif",
        "preview": "images/spread_rate_latest.png",
        "metadata": "spread_rate/status.json",
        "status": status.get("status"),
        "generated_at": status.get("generated_at"),
        "analysis_hour": status.get("analysis_hour"),
        "max_ros_ch_per_h": status.get("max_ros_ch_per_h"),
    }
    save_manifest(manifest)


def _build_status_payload(
    *,
    status: str,
    moisture: dict | None,
    static: dict | None,
    grids: dict[str, np.ndarray] | None,
    error: str | None = None,
    previous: dict | None = None,
) -> dict:
    generated_at = _now_local().isoformat()
    analysis_hour = moisture.get("analysis_hour") if moisture else (previous or {}).get("analysis_hour")
    max_ros = None
    if grids is not None:
        finite = grids["ros_ch_per_h"][np.isfinite(grids["ros_ch_per_h"])]
        if finite.size:
            max_ros = float(np.nanmax(finite))
    payload = {
        "product": "observed_rothermel_spread_rate",
        "product_version": PRODUCT_VERSION,
        "library": {"name": "pyretechnics", "version": PYROTECHNICS_VERSION},
        "methodology": (
            "Modeled surface head-fire spread-rate potential from Rothermel (Pyretechnics), "
            "hourly RTMA weather, causal Nelson dead-fuel moisture with RAWS 10-hour anchoring, "
            "and versioned LANDFIRE/topography static inputs. Not observed fire motion, GFDI, "
            "spotting, crown fire, or a public warning."
        ),
        "status": status,
        "generated_at": generated_at,
        "analysis_hour": analysis_hour,
        "refresh_cadence_minutes": REFRESH_INTERVAL_MINUTES,
        "rtma_expected_delay_minutes": 60,
        "rtma_cache": (moisture or {}).get("rtma_cache"),
        "conditioning_hours_available": (moisture or {}).get("conditioning_hours_available"),
        "conditioning_hours_required": MIN_CONDITIONING_HOURS,
        "conditioning_hours_target": TARGET_CONDITIONING_HOURS,
        "raws_correction": (moisture or {}).get("raws_correction"),
        "static_bundle_version": (static or {}).get("bundle_version"),
        "static_bundle_diagnostics": static_diagnostics(),
        "live_fuel_assumption": "Statewide GDD proxy mapped to Scott–Burgan live-moisture scenarios",
        "max_ros_ch_per_h": max_ros,
        "class_breaks_ch_per_h": list(ROS_CLASS_BOUNDS),
        "class_labels": list(ROS_CLASS_LABELS),
        "artifacts": {
            "geotiff": "gis/spread_rate/spread_rate_latest.tif",
            "preview_png": "images/spread_rate_latest.png",
            "status_json": "spread_rate/status.json",
        },
        "error": error,
    }
    if status in {"stale", "error"} and previous:
        payload["previous_success"] = {
            "generated_at": previous.get("generated_at"),
            "analysis_hour": previous.get("analysis_hour"),
            "max_ros_ch_per_h": previous.get("max_ros_ch_per_h"),
        }
    return payload


def generate_spread_rate(
    raws_payload: dict | None = None,
    *,
    analysis_hour: datetime | None = None,
    allow_warming: bool = True,
    rtma_cache: dict | None = None,
) -> dict:
    """Compute and publish spread-rate artifacts atomically."""
    previous = _load_status()
    analysis_hour = analysis_hour or latest_complete_hour()
    if analysis_hour.tzinfo is not None:
        analysis_hour = analysis_hour.astimezone(timezone.utc).replace(tzinfo=None)
    if not is_analysis_hour_cached(analysis_hour):
        message = f"latest RTMA analysis hour not cached yet: {analysis_hour.isoformat()}Z"
        status = _build_status_payload(
            status="waiting_for_rtma",
            moisture={"analysis_hour": analysis_hour.isoformat(), "rtma_cache": rtma_cache},
            static=None,
            grids=None,
            error=message,
            previous=previous if previous.get("status") == "ready" else None,
        )
        _atomic_json_write(STATUS_PATH, status)
        _update_manifest(status)
        return status
    try:
        static = load_static_fields()
        moisture = condition_moisture(
            static["lat"],
            static["lon"],
            analysis_hour=analysis_hour,
            raws_payload=raws_payload,
        )
    except RuntimeError as exc:
        message = str(exc)
        if allow_warming and "insufficient RTMA history" in message:
            status = _build_status_payload(status="warming", moisture=None, static=None, grids=None, error=message, previous=previous)
            _atomic_json_write(STATUS_PATH, status)
            _update_manifest(status)
            return status
        status = _build_status_payload(status="error", moisture=None, static=None, grids=None, error=message, previous=previous)
        _atomic_json_write(STATUS_PATH, status)
        _update_manifest(status)
        return status
    except Exception as exc:
        logger.exception("Spread-rate generation failed during input preparation")
        status = _build_status_payload(
            status="stale" if previous.get("status") == "ready" else "error",
            moisture=None,
            static=None,
            grids=None,
            error=str(exc),
            previous=previous,
        )
        _atomic_json_write(STATUS_PATH, status)
        _update_manifest(status)
        return status

    try:
        grids = compute_spread_rate_grid(static, moisture)
        if not np.isfinite(grids["ros_ch_per_h"]).any():
            raise RuntimeError("Rothermel produced no finite spread-rate cells")
        _write_geotiff(static, grids, TIF_PATH)
        status = _build_status_payload(status="ready", moisture={**moisture, "rtma_cache": rtma_cache}, static=static, grids=grids)
        _render_png(grids, static, PNG_PATH, status)
        _atomic_json_write(STATUS_PATH, status)
        _update_manifest(status)
        return status
    except Exception as exc:
        logger.exception("Spread-rate generation failed during Rothermel export")
        status = _build_status_payload(
            status="stale" if previous.get("status") == "ready" else "error",
            moisture=moisture,
            static=static,
            grids=None,
            error=str(exc),
            previous=previous,
        )
        _atomic_json_write(STATUS_PATH, status)
        _update_manifest(status)
        return status


def spread_rate_status() -> dict:
    status = _load_status()
    if status:
        return status
    return {
        "product": "observed_rothermel_spread_rate",
        "status": "waiting",
        "generated_at": None,
        "artifacts": {
            "geotiff": "gis/spread_rate/spread_rate_latest.tif",
            "preview_png": "images/spread_rate_latest.png",
            "status_json": "spread_rate/status.json",
        },
    }


async def run_spread_rate_job(raws_payload: dict | None = None):
    try:
        import asyncio

        result = await asyncio.to_thread(run_spread_rate_pipeline, raws_payload)
        logger.info(
            "Spread-rate job finished with status=%s analysis_hour=%s",
            result.get("status"),
            result.get("analysis_hour"),
        )
    except Exception:
        logger.exception("Scheduled spread-rate generation failed")


def run_spread_rate_pipeline(raws_payload: dict | None = None) -> dict:
    """Poll RTMA into the on-server cache, then refresh spread-rate when inputs allow."""
    rtma_cache = ensure_latest_analysis_cached()
    return generate_spread_rate(raws_payload, rtma_cache=rtma_cache)


def warmup_spread_rate_inputs(days: int = 7) -> dict:
    return warmup_rtma_cache(days=days)
