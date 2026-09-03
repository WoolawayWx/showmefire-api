"""Publish current point feeds for MapServer without changing source APIs."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.config import GIS_DIR, MISSOURI_FIRES_GEOJSON
from services.gis_publisher import publish_vectors


def _observation(station: dict, name: str) -> Any:
    value = (station.get("observations") or {}).get(name)
    return value.get("value") if isinstance(value, dict) else value


def publish_weather_stations(stations_payload: dict, raws_payload: dict) -> dict:
    raws = {row.get("stid"): row for row in (raws_payload.get("stations") or []) if row.get("stid")}
    features = []
    for station in stations_payload.get("stations") or []:
        if station.get("state") not in (None, "MO"):
            continue
        try:
            longitude = float(station.get("longitude"))
            latitude = float(station.get("latitude"))
        except (TypeError, ValueError):
            continue
        stid = station.get("stid")
        raw = raws.get(stid, {})
        properties = {
            "stid": stid,
            "name": station.get("name") or stid,
            "county": station.get("county"),
            "observed_at": station.get("date_time") or station.get("timestamp"),
            "temperature_f": _observation(station, "air_temp"),
            "relative_humidity": _observation(station, "relative_humidity"),
            "wind_speed_mph": _observation(station, "wind_speed"),
            "wind_gust_mph": _observation(station, "wind_gust"),
            "fuel_moisture": _observation(raw, "fuel_moisture"),
        }
        features.append({
            "type": "Feature", "geometry": {"type": "Point", "coordinates": [longitude, latitude]},
            "properties": properties,
        })
    paths = publish_vectors("weather_stations", features)
    return {"feature_count": len(features), **{key: str(value) for key, value in paths.items()}}


def _load_features(path: Path, source: str) -> list[dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []
    result = []
    for feature in payload.get("features") or []:
        if not feature.get("geometry"):
            continue
        properties = dict(feature.get("properties") or {})
        # GeoPackage scalar fields cannot contain nested JSON structures.
        properties = {
            key: json.dumps(value, separators=(",", ":")) if isinstance(value, (dict, list)) else value
            for key, value in properties.items()
        }
        properties["source_feed"] = source
        result.append({**feature, "properties": properties})
    return result


def publish_fire_detections() -> dict:
    features = _load_features(Path(GIS_DIR) / "satfiredetection.geojson", "FIRMS")
    features.extend(_load_features(Path(MISSOURI_FIRES_GEOJSON), "NIFC/NGFS"))
    paths = publish_vectors(
        "fire_detections", features,
        generated_at=datetime.now(timezone.utc),
    )
    return {"feature_count": len(features), **{key: str(value) for key, value in paths.items()}}
