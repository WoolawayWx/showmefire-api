"""Compare a dated 09z peak fire-danger raster with the matching RTMA peak."""
from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

API_ROOT = Path(__file__).resolve().parent.parent
FORECAST_DIR = API_ROOT / "gis" / "forecast_peak_09z" / "archive"
LIVE_FORECAST_PATH = API_ROOT / "gis" / "peak_fire_danger_09z.tif"
LIVE_FORECAST_PNG = API_ROOT / "images" / "mo-forecastfiredanger_09z.png"
RTMA_DIR = API_ROOT / "gis" / "rtma_peak" / "archive"
METRICS_DIR = API_ROOT / "archive" / "forecast_09z_rtma_metrics"
LATEST_PATH = API_ROOT / "archive" / "forecast_09z_vs_rtma_summary.json"
NODATA = 255
CATEGORY_COUNT = 5


def _read_on_forecast_grid(forecast_path: Path, rtma_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(forecast_path) as forecast_source:
        forecast = forecast_source.read(1)
        forecast_profile = forecast_source.profile

    with rasterio.open(rtma_path) as rtma_source:
        if (
            rtma_source.shape == forecast.shape
            and rtma_source.crs == forecast_profile["crs"]
            and rtma_source.transform == forecast_profile["transform"]
        ):
            rtma = rtma_source.read(1)
        else:
            rtma = np.full(forecast.shape, NODATA, dtype=np.uint8)
            reproject(
                source=rasterio.band(rtma_source, 1),
                destination=rtma,
                src_transform=rtma_source.transform,
                src_crs=rtma_source.crs,
                src_nodata=rtma_source.nodata,
                dst_transform=forecast_profile["transform"],
                dst_crs=forecast_profile["crs"],
                dst_nodata=NODATA,
                resampling=Resampling.nearest,
            )
    return forecast, rtma


def build_comparison(forecast_path: Path, rtma_path: Path, date: str) -> dict:
    forecast, rtma = _read_on_forecast_grid(forecast_path, rtma_path)
    valid = (
        (forecast != NODATA)
        & (rtma != NODATA)
        & (forecast >= 0)
        & (forecast < CATEGORY_COUNT)
        & (rtma >= 0)
        & (rtma < CATEGORY_COUNT)
    )
    forecast_values = forecast[valid].astype(float)
    rtma_values = rtma[valid].astype(float)
    if not forecast_values.size:
        raise RuntimeError(f"No overlapping valid pixels for {date}")

    differences = forecast_values - rtma_values
    confusion = np.zeros((CATEGORY_COUNT, CATEGORY_COUNT), dtype=int)
    for observed, predicted in zip(rtma_values.astype(int), forecast_values.astype(int)):
        confusion[observed, predicted] += 1

    count = int(differences.size)
    matches = int(np.count_nonzero(differences == 0))
    within_one = int(np.count_nonzero(np.abs(differences) <= 1))
    rtma_metadata = {}
    try:
        payload = json.loads(rtma_path.with_suffix(".json").read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("fuel_moisture"), dict):
            rtma_metadata = payload["fuel_moisture"]
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        pass

    return {
        "schema_version": 2,
        "date": date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "comparison": "09z_minus_rtma",
        "forecast_source": forecast_path.name,
        "rtma_source": rtma_path.name,
        "rtma_fuel_moisture": rtma_metadata or {"mode": "unknown"},
        "valid_pixel_count": count,
        "fire_danger_category_agreement": {
            "matches": matches,
            "within_one": within_one,
            "total": count,
            "agreement_rate": matches / count,
            "within_one_rate": within_one / count,
            "mean_bias": float(np.mean(differences)),
            "mean_abs_diff": float(np.mean(np.abs(differences))),
            "rmse": math.sqrt(float(np.mean(np.square(differences)))),
            "forecast_higher_rate": float(np.count_nonzero(differences > 0) / count),
            "forecast_lower_rate": float(np.count_nonzero(differences < 0) / count),
        },
        "confusion_matrix": {
            "labels": ["Low", "Moderate", "Elevated", "Critical", "Extreme"],
            "matrix": confusion.tolist(),
            "rows": "rtma",
            "columns": "09z_forecast",
        },
        "gis": {
            "forecast_peak_tif": f"forecast_peak_09z/archive/{date}.tif",
            "forecast_peak_png": f"forecast_peak_09z/archive/{date}.png",
            "rtma_peak_tif": f"rtma_peak/archive/{date}.tif",
            "rtma_peak_png": f"rtma_peak/archive/{date}.png",
        },
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    logger.info("Wrote %s", path)


def _forecast_path_for_date(date: str) -> Path:
    archived = FORECAST_DIR / f"{date}.tif"
    if archived.exists():
        return archived
    if not LIVE_FORECAST_PATH.exists():
        return archived

    try:
        with rasterio.open(LIVE_FORECAST_PATH) as source:
            model_run = str(source.tags().get("MODEL_RUN", ""))
        if not model_run.startswith(date):
            logger.warning("Live 09z raster belongs to %s, not %s", model_run or "an unknown run", date)
            return archived
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(LIVE_FORECAST_PATH, archived)
        if LIVE_FORECAST_PNG.exists():
            image_archive = API_ROOT / "images" / "forecast_peak_09z" / "archive" / f"{date}.png"
            image_archive.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(LIVE_FORECAST_PNG, image_archive)
        logger.info("Backfilled dated 09z peak artifacts from the matching live product")
    except (OSError, rasterio.errors.RasterioError):
        logger.exception("Unable to validate the live 09z raster for %s", date)
    return archived


def main(date: str) -> int:
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except ValueError as error:
        raise ValueError(f"date must be YYYY-MM-DD, got: {date}") from error

    forecast_path = _forecast_path_for_date(date)
    rtma_path = RTMA_DIR / f"{date}.tif"
    if not forecast_path.exists():
        logger.warning("No archived 09z forecast peak for %s", date)
        return 0
    if not rtma_path.exists():
        logger.warning("No archived RTMA peak for %s", date)
        return 0

    summary = build_comparison(forecast_path, rtma_path, date)
    _write_json(METRICS_DIR / f"{date}.json", summary)
    _write_json(LATEST_PATH, summary)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True, help="Local verification date in YYYY-MM-DD format")
    arguments = parser.parse_args()
    raise SystemExit(main(arguments.date))
