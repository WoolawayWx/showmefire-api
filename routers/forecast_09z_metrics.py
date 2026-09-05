"""Public API for 09z comparisons against 12z, stations, and RTMA."""
from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from core.config import ARCHIVE_DIR, GIS_DIR, REPORTS_DIR
from routers.verification import _report_covers_closed_window


router = APIRouter(prefix="/api/forecast-09z/metrics", tags=["forecast-09z-metrics"])
METRICS_DIR = Path(ARCHIVE_DIR) / "forecast_09z_metrics"
STATIONS_DIR = Path(GIS_DIR) / "forecast_09z_vs_12z" / "archive"
OBSERVED_HISTORY_FILE = Path(REPORTS_DIR) / "validation_history_09z.json"
RTMA_METRICS_DIR = Path(ARCHIVE_DIR) / "forecast_09z_rtma_metrics"
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail="09z comparison metrics are not available") from error
    except (OSError, json.JSONDecodeError) as error:
        raise HTTPException(status_code=500, detail="Unable to read 09z comparison metrics") from error
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="Invalid 09z comparison metrics")
    return payload


def _validate_date(date: str) -> str:
    if not DATE_PATTERN.fullmatch(date):
        raise HTTPException(status_code=400, detail="date must use YYYY-MM-DD format")
    return date


def _summary_files() -> list[Path]:
    return sorted(METRICS_DIR.glob("????-??-??.json"), reverse=True)


def _difference_metric(metric: dict | None) -> dict:
    metric = metric if isinstance(metric, dict) else {}
    return {
        "count": metric.get("count", 0),
        "mean_bias": metric.get("bias"),
        "mean_abs_diff": metric.get("mae"),
        "rmse": metric.get("rmse"),
        "correlation": metric.get("correlation"),
    }


def _station_report_public(summary: dict, include_rows: bool = False) -> dict:
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    danger = metrics.get("Fire Danger Index") if isinstance(metrics.get("Fire Danger Index"), dict) else {}
    count = int(danger.get("count", 0) or 0)
    agreement = danger.get("exact_match_rate")
    within_one = danger.get("within_one_category_rate")
    result = {
        "date": summary.get("date"),
        "generated_at": summary.get("generated_at"),
        "comparison": "09z_minus_station_observed",
        "station_count": summary.get("stations_count", 0),
        "record_count": summary.get("record_count", 0),
        "metrics": {
            "temp_c": _difference_metric(metrics.get("Temperature (C)")),
            "rh": _difference_metric(metrics.get("Relative Humidity (%)")),
            "wind_speed_ms": _difference_metric(metrics.get("Wind Speed (m/s)")),
            "fuel_moisture": _difference_metric(metrics.get("Fuel Moisture (%)")),
        },
        "fire_danger_category_agreement": {
            "matches": round(agreement * count) if isinstance(agreement, (int, float)) else 0,
            "within_one": round(within_one * count) if isinstance(within_one, (int, float)) else 0,
            "total": count,
            "agreement_rate": agreement,
            "within_one_rate": within_one,
            "mean_bias": danger.get("bias"),
            "mean_abs_diff": danger.get("mean_absolute_category_error", danger.get("mae")),
        },
        "confusion_matrix": summary.get("confusion_matrix"),
        "neighborhood_verification": summary.get("neighborhood_verification"),
        "qc_exclusions": summary.get("qc_exclusions", []),
    }
    if include_rows:
        result["comparison_rows"] = summary.get("comparison_rows", [])
    return result


def _observed_history() -> list[dict]:
    if not OBSERVED_HISTORY_FILE.exists():
        return []
    try:
        payload = json.loads(OBSERVED_HISTORY_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


@router.get("/history")
async def get_09z_metrics_history(limit: int = Query(default=90, ge=1, le=365)):
    rows = []
    for path in _summary_files()[:limit]:
        summary = _load_json(path)
        rows.append({
            "date": summary.get("date", path.stem),
            "generated_at": summary.get("generated_at"),
            "station_count": summary.get("station_count", 0),
            "record_count": summary.get("record_count", 0),
            "metrics": summary.get("metrics", {}),
            "fire_danger_category_agreement": summary.get("fire_danger_category_agreement", {}),
        })
    return {"dates": rows}


@router.get("/latest")
async def get_latest_09z_metrics():
    files = _summary_files()
    if not files:
        raise HTTPException(status_code=404, detail="No 09z comparison metrics are available")
    return _load_json(files[0])


@router.get("/observed/history")
async def get_09z_observed_history(limit: int = Query(default=90, ge=1, le=365)):
    completed = [entry for entry in _observed_history() if _report_covers_closed_window(entry)]
    completed.sort(key=lambda entry: str(entry.get("date", "")), reverse=True)
    return {"dates": [_station_report_public(entry) for entry in completed[:limit]]}


@router.get("/observed/{date}")
async def get_09z_observed_metrics(date: str):
    safe_date = _validate_date(date)
    summary = _load_json(Path(REPORTS_DIR) / safe_date / "validation_summary_09z.json")
    if not _report_covers_closed_window(summary, safe_date):
        raise HTTPException(status_code=404, detail=f"No completed 09z observation report available for {safe_date}")
    return _station_report_public(summary, include_rows=True)


@router.get("/rtma/history")
async def get_09z_rtma_history(limit: int = Query(default=90, ge=1, le=365)):
    rows = []
    for path in sorted(RTMA_METRICS_DIR.glob("????-??-??.json"), reverse=True):
        summary = _load_json(path)
        if _report_covers_closed_window(summary, path.stem):
            rows.append(summary)
        if len(rows) >= limit:
            break
    return {"dates": rows}


@router.get("/rtma/{date}")
async def get_09z_rtma_metrics(date: str):
    safe_date = _validate_date(date)
    summary = _load_json(RTMA_METRICS_DIR / f"{safe_date}.json")
    if not _report_covers_closed_window(summary, safe_date):
        raise HTTPException(status_code=404, detail=f"No completed 09z RTMA report available for {safe_date}")
    return summary


@router.get("/{date}/stations")
async def get_09z_station_metrics(date: str):
    safe_date = _validate_date(date)
    return _load_json(STATIONS_DIR / f"{safe_date}.geojson")


@router.get("/{date}")
async def get_09z_metrics(date: str):
    safe_date = _validate_date(date)
    return _load_json(METRICS_DIR / f"{safe_date}.json")
