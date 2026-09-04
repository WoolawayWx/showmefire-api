"""Public API for archived 09z-versus-12z forecast comparison metrics."""
from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from core.config import ARCHIVE_DIR, GIS_DIR


router = APIRouter(prefix="/api/forecast-09z/metrics", tags=["forecast-09z-metrics"])
METRICS_DIR = Path(ARCHIVE_DIR) / "forecast_09z_metrics"
STATIONS_DIR = Path(GIS_DIR) / "forecast_09z_vs_12z" / "archive"
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


@router.get("/{date}/stations")
async def get_09z_station_metrics(date: str):
    safe_date = _validate_date(date)
    return _load_json(STATIONS_DIR / f"{safe_date}.geojson")


@router.get("/{date}")
async def get_09z_metrics(date: str):
    safe_date = _validate_date(date)
    return _load_json(METRICS_DIR / f"{safe_date}.json")
