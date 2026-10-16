"""Hourly HRRR fire-weather data (the numbers behind maps/fire_weather_maps.py).

GET /api/fire-weather/latest                 statewide + regional summary for the newest hour
GET /api/fire-weather/history?hours=48       hourly summaries, oldest first
GET /api/fire-weather/at?lat=&lon=           values at a point from an hourly grid
GET /api/fire-weather/stations/{stid}        one station's hourly values over the last N hours
GET /api/fire-weather/verification           balloon vs HRRR comparison (bias, MAE)

Metric definitions and their sources: services/fire_weather/indices.py.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Query

from services.fire_weather import history
from services.fire_weather import summary as fire_summary

router = APIRouter(prefix="/api/fire-weather", tags=["fire-weather"])


@router.get("/latest")
def latest():
    data = fire_summary.load_latest()
    if data is None:
        raise HTTPException(status_code=404, detail="No fire-weather analysis yet")
    return data


@router.get("/history")
def summary_history(hours: int = Query(48, ge=1, le=24 * 10)):
    records = history.read_records(hours)
    return {
        "hours": hours,
        "count": len(records),
        "records": [{key: record[key] for key in ("valid_time", "source", "units", "statewide", "regions")
                     if key in record} for record in records],
    }


@router.get("/stations/{stid}")
def station_history(stid: str, hours: int = Query(48, ge=1, le=24 * 10)):
    stid = stid.upper()
    rows = []
    for record in history.read_records(hours):
        values = (record.get("stations") or {}).get(stid)
        if values:
            rows.append({"valid_time": record["valid_time"], **values})
    if not rows:
        raise HTTPException(status_code=404, detail=f"No fire-weather values for station {stid}")
    return {"station": stid, "hours": hours, "count": len(rows), "records": rows}


def _grid_path(valid: Optional[datetime]) -> Path:
    if valid is not None:
        return history.GRID_DIR / f"{valid:%Y%m%d}" / f"{valid:%H}z.nc"
    candidates = sorted(history.GRID_DIR.glob("*/*z.nc"))
    if not candidates:
        raise HTTPException(status_code=404, detail="No hourly fire-weather grids yet")
    return candidates[-1]


@lru_cache(maxsize=8)
def _load_grid(path: str, mtime: float) -> dict:
    """Decoded grids for one hour, cached per file version."""
    import xarray as xr

    with xr.open_dataset(path) as dataset:
        # Older files may lack fields added later (e.g. holzworth_ft): use what each file has.
        fields = {name: np.asarray(dataset[name].values, dtype=float) for name in history.GRID_FIELDS
                  if name in dataset.data_vars}
        return {"lat": np.asarray(dataset["latitude"].values, dtype=float),
                "lon": np.asarray(dataset["longitude"].values, dtype=float),
                "valid_time": dataset.attrs.get("valid_time"), "source": dataset.attrs.get("source"),
                "fields": fields}


@router.get("/at")
def at_point(lat: float = Query(..., ge=-90, le=90), lon: float = Query(..., ge=-180, le=180),
             time: Optional[str] = Query(None, description="Valid hour, YYYYMMDDHH (UTC); default newest")):
    valid = None
    if time:
        try:
            valid = datetime.strptime(time, "%Y%m%d%H").replace(tzinfo=timezone.utc)
        except ValueError:
            raise HTTPException(status_code=422, detail="time must be YYYYMMDDHH (UTC)")
    path = _grid_path(valid)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="No fire-weather grid for that hour")
    grid = _load_grid(str(path), path.stat().st_mtime)
    distance = (grid["lat"] - lat) ** 2 + ((grid["lon"] - lon) * np.cos(np.radians(lat))) ** 2
    i, j = np.unravel_index(int(np.argmin(distance)), distance.shape)
    km = float(np.sqrt(distance[i, j])) * 111.0
    if km > 10:  # outside the Missouri grid
        raise HTTPException(status_code=404, detail="Point is outside the fire-weather grid")
    values = {}
    for name, field in grid["fields"].items():
        value = float(field[i, j])
        values[name] = None if not np.isfinite(value) else round(value, 1)
    category = values.get("ventilation_kt_ft")
    if category is not None:
        from services.fire_weather import indices as ix

        values["ventilation_category"] = ix.VENTILATION_CATEGORIES[int(ix.ventilation_category(category))][0]
    latest = fire_summary.load_latest() or {}
    return {"valid_time": grid["valid_time"], "source": grid["source"],
            "grid_point": {"lat": round(float(grid["lat"][i, j]), 4), "lon": round(float(grid["lon"][i, j]), 4),
                           "distance_km": round(km, 1)},
            "units": latest.get("units"), "values": values}


@router.get("/verification")
def verification(days: int = Query(30, ge=1, le=365)):
    from services.fire_weather import verification as fw_verification

    return fw_verification.summarize(days=days)
