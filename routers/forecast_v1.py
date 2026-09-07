from __future__ import annotations

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException

from forecast_v1.contracts import QUALITY_BITS
from forecast_v1.repository import ensure_schema, resolve_run, run_assets, transaction
from core.fire_danger import CATEGORY_LABELS


router = APIRouter(prefix="/api/forecast/v1", tags=["forecast-v1"])


def _decode(row) -> dict:
    value = dict(row)
    for key in tuple(value):
        if key.endswith("_json"):
            value[key[:-5]] = json.loads(value.pop(key) or "null")
    value["is_public"] = bool(value.get("is_public"))
    return value


def _manifest_for(row) -> dict:
    with transaction() as connection:
        assets = run_assets(connection, row["run_id"])
    manifest_asset = next((asset for asset in assets if asset["kind"] == "manifest"), None)
    if manifest_asset and manifest_asset["local_path"] and Path(manifest_asset["local_path"]).is_file():
        return json.loads(Path(manifest_asset["local_path"]).read_text(encoding="utf-8"))
    result = _decode(row)
    result["assets"] = [_decode(asset) for asset in assets]
    return result


@router.get("/runs/latest")
def latest_run():
    ensure_schema()
    with transaction() as connection:
        row = resolve_run(connection)
    if not row:
        raise HTTPException(status_code=404, detail="No public 72-hour forecast run is available")
    return _manifest_for(row)


@router.get("/runs/{run_id}")
def get_run(run_id: str):
    ensure_schema()
    with transaction() as connection:
        row = resolve_run(connection, run_id)
    if not row:
        raise HTTPException(status_code=404, detail="Forecast run not found")
    return _manifest_for(row)


def _run_id_or_404(connection, run_id: str | None) -> str:
    row = resolve_run(connection, run_id)
    if not row:
        raise HTTPException(status_code=404, detail="Forecast run not found")
    return str(row["run_id"])


@router.get("/stations")
def get_stations(bbox: str | None = None, run_id: str | None = None):
    ensure_schema()
    bounds = None
    if bbox:
        try:
            bounds = tuple(float(value) for value in bbox.split(","))
            if len(bounds) != 4:
                raise ValueError
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="bbox must be west,south,east,north") from exc
    with transaction() as connection:
        resolved = _run_id_or_404(connection, run_id)
        sql = "SELECT * FROM forecast_station_metadata WHERE is_active=1"
        params: list = []
        if bounds:
            sql += " AND longitude BETWEEN ? AND ? AND latitude BETWEEN ? AND ?"
            params.extend((bounds[0], bounds[2], bounds[1], bounds[3]))
        stations = connection.execute(sql + " ORDER BY station_id", params).fetchall()
        features = []
        for station in stations:
            days = connection.execute("SELECT * FROM station_forecast_days WHERE run_id=? AND station_id=? ORDER BY day_index", (resolved, station["station_id"])).fetchall()
            features.append({
                "type": "Feature", "geometry": {"type": "Point", "coordinates": [station["longitude"], station["latitude"]]},
                "properties": {"stationId": station["station_id"], "name": station["name"], "network": station["network_type"], "days": [_day_payload(day) for day in days]},
            })
    return {"type": "FeatureCollection", "runId": resolved, "features": features}


def _hour_payload(row) -> dict:
    flags = [name for name, bit in QUALITY_BITS.items() if row["quality_mask"] & bit]
    danger = row["danger_class"]
    return {
        "validTime": row["valid_time_utc"], "leadHour": row["lead_hour"],
        "temperatureC": row["temperature_c"], "dewpointC": row["dewpoint_c"], "relativeHumidity": row["relative_humidity"],
        "wind": {"uMs": row["wind_u_ms"], "vMs": row["wind_v_ms"], "speedMs": row["wind_speed_ms"], "gustMs": row["wind_gust_ms"]},
        "precipitationMm": row["precipitation_mm"], "solarWm2": row["solar_wm2"], "cloudCover": row["cloud_cover_pct"], "mixingHeightM": row["mixing_height_m"],
        "fuelMoisture": {"p10": row["fuel_moisture_p10"], "p50": row["fuel_moisture_p50"], "p90": row["fuel_moisture_p90"]},
        "risk": {"category": danger, "label": CATEGORY_LABELS[danger] if danger is not None else None, "score": row["danger_score"], "probabilityRhLe25": row["probability_rh_le_25"], "probabilityGustGe30Mph": row["probability_gust_ge_30mph"], "probabilityConcurrent": row["probability_concurrent"]},
        "confidence": {"meteorological": row["meteorological_confidence"], "category": row["category_confidence"]},
        "qualityFlags": flags,
    }


def _day_payload(row) -> dict:
    danger = row["peak_category"]
    return {
        "localDate": row["local_date"], "dayIndex": row["day_index"], "temperatureLowC": row["temperature_low_c"], "temperatureHighC": row["temperature_high_c"],
        "minimumRh": row["minimum_rh"], "maximumWindMs": row["maximum_wind_ms"], "maximumGustMs": row["maximum_gust_ms"], "precipitationTotalMm": row["precipitation_total_mm"],
        "fuelMoisture": {"minimumP50": row["minimum_fuel_moisture_p50"], "p10": row["fuel_moisture_p10"], "p90": row["fuel_moisture_p90"]},
        "risk": {"peakCategory": danger, "label": CATEGORY_LABELS[danger] if danger is not None else None, "criticalWindowStart": row["critical_window_start_utc"], "criticalWindowEnd": row["critical_window_end_utc"], "concurrentThresholdHours": row["concurrent_threshold_hours"]},
        "confidence": {"meteorological": row["meteorological_confidence"], "category": row["category_confidence"]},
    }


@router.get("/stations/{station_id}")
def get_station(station_id: str, run_id: str | None = None):
    ensure_schema()
    with transaction() as connection:
        resolved = _run_id_or_404(connection, run_id)
        station = connection.execute("SELECT * FROM forecast_station_metadata WHERE station_id=?", (station_id,)).fetchone()
        if not station:
            raise HTTPException(status_code=404, detail="Station not found")
        hours = connection.execute("SELECT * FROM station_forecast_hours WHERE run_id=? AND station_id=? ORDER BY lead_hour", (resolved, station_id)).fetchall()
        days = connection.execute("SELECT * FROM station_forecast_days WHERE run_id=? AND station_id=? ORDER BY day_index", (resolved, station_id)).fetchall()
        observations = connection.execute("SELECT * FROM station_observations_v2 WHERE station_id=? AND qc_state='pass' ORDER BY observed_at_utc DESC LIMIT 168", (station_id,)).fetchall()
        verification = connection.execute("SELECT * FROM forecast_verification WHERE station_id=? AND qc_eligible=1 ORDER BY valid_time_utc DESC LIMIT 250", (station_id,)).fetchall()
    return {
        "schemaVersion": "forecast-v1", "runId": resolved,
        "station": {"stationId": station["station_id"], "name": station["name"], "latitude": station["latitude"], "longitude": station["longitude"], "elevationM": station["elevation_m"], "network": station["network_type"], "timezone": station["timezone"], "sensorCapabilities": json.loads(station["sensor_capabilities_json"])},
        "hours": [_hour_payload(row) for row in hours], "days": [_day_payload(row) for row in days],
        "recentObservations": [_decode(row) for row in observations], "verification": [_decode(row) for row in verification],
        "confidenceComponents": {"modelAgreement": 0.30, "ensembleSpread": 0.25, "cycleConsistency": 0.20, "rolling30DayVerification": 0.15, "leadTime": 0.10},
        "units": {"temperature": "degC", "wind": "m s-1", "precipitation": "mm", "fuelMoisture": "%", "confidence": "%"},
    }
