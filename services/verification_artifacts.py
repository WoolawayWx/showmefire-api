"""Build server-side verification raster products.

The browser receives these as ordinary single-band categorical GeoTIFFs.  All
pixel arithmetic happens here so station and RTMA products remain aligned.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from core.config import GIS_DIR
from services.verification_rainfall import CONTRACT_VERSION, combine_category_grids

logger = logging.getLogger(__name__)


def _read_category(path: Path, target_profile: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    with rasterio.open(path) as src:
        profile = src.profile.copy()
        values = src.read(1).astype(float)
        values[values >= 255] = np.nan
        if target_profile is None:
            return values, profile
        aligned = np.full(
            (target_profile["height"], target_profile["width"]),
            np.nan,
            dtype=np.float32,
        )
        reproject(
            values,
            aligned,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target_profile["transform"],
            dst_crs=target_profile["crs"],
            src_nodata=np.nan,
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )
        return aligned, target_profile


def _write_category(path: Path, values: np.ndarray, profile: dict[str, Any], description: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = profile.copy()
    output.update(dtype=rasterio.uint8, count=1, nodata=255, compress="lzw")
    encoded = np.full(values.shape, 255, dtype=np.uint8)
    valid = np.isfinite(values)
    encoded[valid] = np.clip(np.rint(values[valid]), 0, 4).astype(np.uint8)
    with rasterio.open(path, "w", **output) as dst:
        dst.write(encoded, 1)
        dst.set_band_description(1, description)
        dst.update_tags(
            SOURCE="station observed peak + RTMA rainfall-adjusted peak",
            VERIFICATION_RAINFALL_CONTRACT=CONTRACT_VERSION,
        )


def _station_adjustment_from_rtma(
    station: np.ndarray,
    raw_rtma: np.ndarray,
    adjusted_rtma: np.ndarray,
) -> np.ndarray:
    """Transfer only the observed rainfall category reduction to station cells.

    The station map remains the source of the observed category; RTMA supplies
    the spatially resolved rainfall reduction when station rain is sparse.
    """
    reduction = np.clip(np.rint(raw_rtma) - np.rint(adjusted_rtma), 0, 2)
    result = station.copy()
    valid = np.isfinite(result) & np.isfinite(reduction)
    result[valid] = np.maximum(0, result[valid] - reduction[valid])
    return result


def build_combined_verification_artifacts(
    date: str,
    *,
    gis_dir: Path | None = None,
) -> dict[str, Any]:
    """Create combined and aligned verification products when inputs exist."""
    root = Path(gis_dir or GIS_DIR)
    observed = root / "observed_peak" / "archive" / f"{date}.tif"
    rtma_raw = root / "rtma_peak" / "archive" / f"{date}.tif"
    rtma_adjusted = root / "rtma_peak_rainfall_adjusted" / "archive" / f"{date}.tif"
    result: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "combined_tif": None,
        "adjusted_station_tif": None,
        "adjusted_rtma_tif": (
            f"rtma_peak_rainfall_adjusted/archive/{date}.tif"
            if rtma_adjusted.exists() else None
        ),
        "inputs": {
            "observed": observed.exists(),
            "rtma_raw": rtma_raw.exists(),
            "rtma_adjusted": rtma_adjusted.exists(),
        },
        "fallback_reason": None,
    }
    if not observed.exists() or not rtma_adjusted.exists():
        result["fallback_reason"] = "station_or_adjusted_rtma_raster_unavailable"
        return result
    try:
        with rasterio.open(rtma_adjusted) as target:
            profile = target.profile.copy()
        rtma, _ = _read_category(rtma_adjusted, profile)
        station, _ = _read_category(observed, profile)
        if rtma_raw.exists():
            raw_rtma, _ = _read_category(rtma_raw, profile)
            adjusted_station = _station_adjustment_from_rtma(station, raw_rtma, rtma)
        else:
            adjusted_station = station
        adjusted_station_path = root / "station_peak_rainfall_adjusted" / "archive" / f"{date}.tif"
        _write_category(
            adjusted_station_path,
            adjusted_station,
            profile,
            "Rainfall-adjusted station observed peak category",
        )
        result["adjusted_station_tif"] = f"station_peak_rainfall_adjusted/archive/{date}.tif"
        combined = combine_category_grids(adjusted_station, rtma)
        output = root / "verification_combined" / "archive" / f"{date}.tif"
        _write_category(output, combined, profile, "Combined rainfall-adjusted verification category")
        result["combined_tif"] = f"verification_combined/archive/{date}.tif"
        result["coverage"] = {
            "station_pixels": int(np.isfinite(station).sum()),
            "adjusted_station_pixels": int(np.isfinite(adjusted_station).sum()),
            "rtma_pixels": int(np.isfinite(rtma).sum()),
            "combined_pixels": int(np.isfinite(combined).sum()),
        }
        manifest = output.with_suffix(".json")
        manifest.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        result["manifest"] = str(manifest.relative_to(root))
    except Exception as exc:
        logger.exception("Unable to build combined verification artifacts for %s", date)
        result["fallback_reason"] = f"artifact_generation_failed:{exc}"
    return result
