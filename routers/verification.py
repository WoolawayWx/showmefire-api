"""Public forecast-verification API.

Read-only summary of how past days' forecasts performed, backed by the
existing nightly validation pipeline (forecast/endOfDayReport.py). Exposes a
trimmed view of reports/validation_history.json and reports/{date}/
validation_summary.json - no raw file paths, no station-level data.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

from core.config import GIS_DIR, REPORTS_DIR

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


def _mae_for(entry: Dict[str, Any], metric_label: str) -> Optional[float]:
    value = entry.get("metrics", {}).get(metric_label, {}).get("mae")
    return value if isinstance(value, (int, float)) else None


@router.get("/history")
async def get_verification_history(limit: int = 90):
    history = _load_history()
    history_sorted = sorted(history, key=lambda e: e.get("date", ""), reverse=True)[:limit]

    dates = []
    for entry in history_sorted:
        date = entry.get("date")
        if not date:
            continue
        row = {"date": date, "record_count": entry.get("record_count", 0)}
        for field, label in _METRIC_FIELD_MAP.items():
            row[field] = _mae_for(entry, label)
        row["has_confusion_matrix"] = bool(entry.get("confusion_matrix"))
        row["has_observed_peak"] = _observed_peak_tif_path(date).exists()
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

    observed_peak_path = _observed_peak_tif_path(date)
    forecast_peak_path = Path(GIS_DIR) / "peak_fire_danger.tif"

    return {
        "date": summary.get("date", date),
        "generated_at": summary.get("generated_at"),
        "record_count": summary.get("record_count", 0),
        "stations_count": summary.get("stations_count"),
        "metrics": summary.get("metrics", {}),
        "confusion_matrix": summary.get("confusion_matrix"),
        "gis": {
            "forecast_peak_tif": "peak_fire_danger.tif" if forecast_peak_path.exists() else None,
            "observed_peak_tif": (
                f"observed_peak/archive/{date}.tif" if observed_peak_path.exists() else None
            ),
        },
    }
