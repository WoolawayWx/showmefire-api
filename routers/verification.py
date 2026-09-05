"""Public forecast-verification API.

Read-only summary of how past days' forecasts performed, backed by the
existing nightly validation pipeline (forecast/endOfDayReport.py). Exposes a
trimmed view of reports/validation_history.json and reports/{date}/
validation_summary.json - no raw file paths, no station-level data.
"""
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException

from core.config import ARCHIVE_RAW_DATA_DIR, ARCHIVE_DIR, GIS_DIR, IMAGES_DIR, REPORTS_DIR

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/verification", tags=["verification"])

HISTORY_FILE = Path(REPORTS_DIR) / "validation_history.json"

_METRIC_FIELD_MAP = {
    "temp_mae": "Temperature (C)",
    "rh_mae": "Relative Humidity (%)",
    "wind_mae": "Wind Speed (m/s)",
    "fm_mae": "Fuel Moisture (%)",
    "fire_danger_mae": "Fire Danger Index",
}


def _report_covers_closed_window(
    entry: Dict[str, Any],
    fallback_date: str | None = None,
    now: datetime | None = None,
) -> bool:
    """Reject current-day or prematurely generated verification reports."""
    raw_date = str(entry.get("date") or fallback_date or "").strip()
    try:
        date_value = datetime.strptime(raw_date.replace("-", ""), "%Y%m%d").date()
    except ValueError:
        return False

    central = ZoneInfo("America/Chicago")
    window_close = datetime.combine(date_value, datetime.min.time(), tzinfo=central).replace(hour=21)
    current = (now or datetime.now(central)).astimezone(central)
    if current < window_close:
        return False

    generated_at = entry.get("generated_at")
    if not generated_at:
        return True
    try:
        generated = datetime.fromisoformat(str(generated_at).replace("Z", "+00:00"))
        if generated.tzinfo is None:
            generated = generated.replace(tzinfo=ZoneInfo("UTC"))
        return generated.astimezone(central) >= window_close
    except (TypeError, ValueError):
        return False


def _load_history() -> List[Dict[str, Any]]:
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


def _observed_peak_tif_path(date: str) -> Path:
    return Path(GIS_DIR) / "observed_peak" / "archive" / f"{date}.tif"


def _forecast_peak_tif_path(date: str) -> Path:
    return Path(GIS_DIR) / "forecast_peak" / "archive" / f"{date}.tif"


def _observed_peak_png_path(date: str) -> Path:
    return Path(IMAGES_DIR) / "observed_peak" / "archive" / f"{date}.png"


def _forecast_peak_png_path(date: str) -> Path:
    return Path(IMAGES_DIR) / "forecast_peak" / "archive" / f"{date}.png"


def _rtma_peak_tif_path(date: str) -> Path:
    return Path(GIS_DIR) / "rtma_peak" / "archive" / f"{date}.tif"


def _rtma_peak_png_path(date: str) -> Path:
    return Path(IMAGES_DIR) / "rtma_peak" / "archive" / f"{date}.png"


def _mae_for(entry: Dict[str, Any], metric_label: str) -> Optional[float]:
    value = entry.get("metrics", {}).get(metric_label, {}).get("mae")
    return value if isinstance(value, (int, float)) else None


def _fire_danger_accuracy(entry: Dict[str, Any]) -> Optional[float]:
    value = entry.get("metrics", {}).get("Fire Danger Index", {}).get("exact_match_rate")
    return value if isinstance(value, (int, float)) else None


@router.get("/history")
async def get_verification_history(limit: int = 90):
    history = _load_history()
    history_sorted = sorted(history, key=lambda e: e.get("date", ""), reverse=True)[:limit]

    dates = []
    for entry in history_sorted:
        date = entry.get("date")
        if not date or not _report_covers_closed_window(entry):
            continue
        row = {"date": date, "record_count": entry.get("record_count", 0)}
        for field, label in _METRIC_FIELD_MAP.items():
            row[field] = _mae_for(entry, label)
        row["fire_danger_accuracy"] = _fire_danger_accuracy(entry)
        row["has_confusion_matrix"] = bool(entry.get("confusion_matrix"))
        row["has_observed_peak"] = _observed_peak_tif_path(date).exists()
        row["has_rtma_peak"] = _rtma_peak_tif_path(date).exists()
        dates.append(row)

    return {"dates": dates}


@router.get("/report/{date}")
async def get_verification_report(date: str):
    summary_path = Path(REPORTS_DIR) / date / "validation_summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail=f"No validation report available for {date}")

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to read validation summary for %s: %s", date, exc)
        raise HTTPException(status_code=500, detail="Failed to read validation report") from exc

    if not _report_covers_closed_window(summary, date):
        raise HTTPException(status_code=404, detail=f"No completed validation report available for {date}")

    try:
        from maps.observed_peak_history import snapshot_observed_peak_for_date
        snapshot_observed_peak_for_date(date)
    except Exception:
        logger.exception("Failed to snapshot observed peak archive for %s", date)

    observed_peak_tif_path = _observed_peak_tif_path(date)
    forecast_peak_tif_path = _forecast_peak_tif_path(date)
    observed_peak_png_path = _observed_peak_png_path(date)
    forecast_peak_png_path = _forecast_peak_png_path(date)
    rtma_peak_tif_path = _rtma_peak_tif_path(date)
    rtma_peak_png_path = _rtma_peak_png_path(date)

    return {
        "date": summary.get("date", date),
        "generated_at": summary.get("generated_at"),
        "record_count": summary.get("record_count", 0),
        "stations_count": summary.get("stations_count"),
        "metrics": summary.get("metrics", {}),
        "confusion_matrix": summary.get("confusion_matrix"),
        "wind_confusion_matrix": summary.get("wind_confusion_matrix"),
        "neighborhood_verification": summary.get("neighborhood_verification"),
        "gis": {
            "forecast_peak_tif": (
                f"forecast_peak/archive/{date}.tif" if forecast_peak_tif_path.exists() else None
            ),
            "observed_peak_tif": (
                f"observed_peak/archive/{date}.tif" if observed_peak_tif_path.exists() else None
            ),
            "forecast_peak_png": (
                f"forecast_peak/archive/{date}.png" if forecast_peak_png_path.exists() else None
            ),
            "observed_peak_png": (
                f"observed_peak/archive/{date}.png" if observed_peak_png_path.exists() else None
            ),
            "rtma_peak_tif": (
                f"rtma_peak/archive/{date}.tif" if rtma_peak_tif_path.exists() else None
            ),
            "rtma_peak_png": (
                f"rtma_peak/archive/{date}.png" if rtma_peak_png_path.exists() else None
            ),
        },
    }


@router.get("/report/{date}/comparisons")
async def get_verification_comparisons(date: str, station: Optional[str] = None, limit: int = 500):
    """Return canonical hourly forecast/observation pairs for a report date."""
    summary_path = Path(REPORTS_DIR) / date / "validation_summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail=f"No validation report available for {date}")

    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        if not _report_covers_closed_window(summary, date):
            raise HTTPException(status_code=404, detail=f"No completed validation report available for {date}")
        # Reports generated by the current end-of-day pipeline persist the
        # exact comparison rows used for scoring. Prefer those rows because
        # the source archives may have already been moved off local disk.
        # Older reports without this field continue through the reconstruction
        # path below.
        stored_rows = summary.get("comparison_rows")
        if isinstance(stored_rows, list) and stored_rows:
            rows = []
            for row in stored_rows:
                if not isinstance(row, dict):
                    continue
                if station and str(row.get("station", "")).upper() != station.strip().upper():
                    continue
                forecast = row.get("forecast") if isinstance(row.get("forecast"), dict) else {}
                observed = row.get("observed") if isinstance(row.get("observed"), dict) else {}
                rows.append({
                    "station": row.get("station"),
                    "timestamp": row.get("timestamp"),
                    "forecast": {
                        "temperature_c": _json_number(forecast.get("temperature_c")),
                        "relative_humidity_pct": _json_number(forecast.get("relative_humidity_pct")),
                        "wind_speed_ms": _json_number(forecast.get("wind_speed_ms")),
                        "fuel_moisture_pct": _json_number(forecast.get("fuel_moisture_pct")),
                        "fire_danger": _json_number(forecast.get("fire_danger")),
                    },
                    "observed": {
                        "temperature_c": _json_number(observed.get("temperature_c")),
                        "relative_humidity_pct": _json_number(observed.get("relative_humidity_pct")),
                        "wind_speed_ms": _json_number(observed.get("wind_speed_ms")),
                        "fuel_moisture_pct": _json_number(observed.get("fuel_moisture_pct")),
                        "fire_danger": _json_number(observed.get("fire_danger")),
                    },
                })
            return {"date": date, "count": len(rows[:min(max(limit, 1), 2000)]), "rows": rows[:min(max(limit, 1), 2000)]}

        from forecast.endOfDayReport import (
            find_matching_files,
            get_forecast_dataframe,
            get_observation_dataframe,
        )
        forecast_name = summary.get("forecast_source")
        observation_name = summary.get("observation_source")
        forecast_path = Path(ARCHIVE_DIR) / "forecasts" / forecast_name if forecast_name else None
        observation_path = Path(ARCHIVE_RAW_DATA_DIR) / observation_name if observation_name else None
        if not forecast_path or not observation_path or not forecast_path.exists() or not observation_path.exists():
            forecast_data, forecast_path, raw_data, observation_path = find_matching_files(
                Path(ARCHIVE_DIR) / "forecasts", Path(ARCHIVE_RAW_DATA_DIR)
            )
        else:
            with open(forecast_path, "r", encoding="utf-8") as f:
                forecast_data = json.load(f)
            with open(observation_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        if not forecast_data or not raw_data:
            raise HTTPException(status_code=404, detail="Archived comparison data is unavailable")

        import pandas as pd
        run_date = pd.Timestamp(forecast_data.get("run_date"))
        start = pd.Timestamp(run_date.date(), tz="UTC") + pd.Timedelta(hours=16)
        end = start + pd.Timedelta(hours=11)
        forecast_df = get_forecast_dataframe(forecast_data, start, end)
        observation_df, _, _ = get_observation_dataframe(raw_data, start, end)
        merged = pd.merge(forecast_df, observation_df, on=["stid", "timestamp"], how="inner")
        if station:
            merged = merged[merged["stid"].str.upper() == station.strip().upper()]
        rows = []
        for _, row in merged.sort_values(["timestamp", "stid"]).head(min(max(limit, 1), 2000)).iterrows():
            rows.append({
                "station": row["stid"],
                "timestamp": row["timestamp"].isoformat(),
                "forecast": {
                    "temperature_c": _json_number(row.get("pred_temp")),
                    "relative_humidity_pct": _json_number(row.get("pred_rh")),
                    "wind_speed_ms": _json_number(row.get("pred_wind")),
                    "fuel_moisture_pct": _json_number(row.get("pred_fm")),
                    "fire_danger": _json_number(row.get("pred_fire_danger")),
                },
                "observed": {
                    "temperature_c": _json_number(row.get("obs_temp")),
                    "relative_humidity_pct": _json_number(row.get("obs_rh")),
                    "wind_speed_ms": _json_number(row.get("obs_wind")),
                    "fuel_moisture_pct": _json_number(row.get("obs_fm")),
                    "fire_danger": _json_number(row.get("obs_fire_danger")),
                },
            })
        return {"date": date, "count": len(rows), "rows": rows}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to build comparisons for %s: %s", date, exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to build forecast comparisons") from exc


def _json_number(value):
    try:
        if value is None:
            return None
        value = float(value)
        return value if value == value else None
    except (TypeError, ValueError):
        return None
