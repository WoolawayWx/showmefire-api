"""Isolated data products for the public Testbed.

This module deliberately writes below the isolated Testbed product root.
It never updates production map files, status.json, or production database
tables.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

from core.beta_fire_danger import BETA_SCORER_VERSION, score_fire_danger
from core.config import DATA_DIR
from core.fire_danger import CATEGORY_LABELS, miles_per_hour_to_knots


CHICAGO_TZ = ZoneInfo("America/Chicago")
BETA_ROOT = Path(os.getenv("TESTBED_PRODUCTS_DIR", str(DATA_DIR / "testbed")))
BETA_GIS_DIR = Path(os.getenv("TESTBED_GIS_DIR", str(BETA_ROOT / "gis")))
BETA_IMAGES_DIR = Path(os.getenv("TESTBED_IMAGES_DIR", str(BETA_ROOT / "images")))
BETA_MANIFEST_PATH = BETA_ROOT / "manifest.json"
BETA_OBSERVATION_STATE_PATH = BETA_ROOT / "observation_state.json"


def _now() -> datetime:
    return datetime.now(CHICAGO_TZ)


def _atomic_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _observation_value(station: dict, key: str) -> Any:
    value = (station.get("observations") or {}).get(key)
    return value.get("value") if isinstance(value, dict) else value


def _station_features(stations: Iterable[dict], raws_by_stid: dict[str, dict]) -> list[dict]:
    features = []
    for station in stations:
        if station.get("state") not in (None, "MO"):
            continue
        stid = station.get("stid")
        if not stid:
            continue
        raw_station = raws_by_stid.get(stid, {})
        fm = _observation_value(raw_station, "fuel_moisture")
        rh = _observation_value(station, "relative_humidity")
        wind_mph = _observation_value(station, "wind_speed")
        try:
            if fm is None or rh is None or wind_mph is None:
                continue
            fm, rh, wind_mph = float(fm), float(rh), float(wind_mph)
            if not (0 <= rh <= 100 and fm >= 0 and wind_mph >= 0):
                continue
            wind_kts = miles_per_hour_to_knots(wind_mph)
            result = score_fire_danger(fm, rh, wind_kts)
        except (TypeError, ValueError):
            continue

        properties = {
            "stid": stid,
            "name": station.get("name") or stid,
            "county": station.get("county"),
            "fuel_moisture": round(fm, 2),
            "relative_humidity": round(rh, 2),
            "wind_speed_mph": round(wind_mph, 2),
            "wind_speed_knots": round(wind_kts, 2),
            "official_category": result["official_category"],
            "official_label": result["official_label"],
            "beta_category": result["beta_category"],
            "beta_label": result["beta_label"],
            "beta_score": result["score"],
            "criteria": result["criteria"],
        }
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [station.get("longitude"), station.get("latitude")],
            },
            "properties": properties,
        })
    return features


def _feature_collection(name: str, features: list[dict], timestamp: str) -> dict:
    return {
        "type": "FeatureCollection",
        "name": name,
        "metadata": {
            "product": name,
            "scorer_version": BETA_SCORER_VERSION,
            "generated_at": timestamp,
            "units": {"fuel_moisture": "percent", "relative_humidity": "percent", "wind": "knots"},
            "official_categories": list(CATEGORY_LABELS),
        },
        "features": features,
    }


def _load_state() -> dict:
    try:
        return json.loads(BETA_OBSERVATION_STATE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {"date_local": _now().date().isoformat(), "peak_by_stid": {}}


def refresh_observation_products(stations_payload: dict, raws_payload: dict) -> dict:
    """Build current, observed-peak, and difference station products."""
    stations = stations_payload.get("stations") or []
    raw_stations = raws_payload.get("stations") or []
    raws_by_stid = {s.get("stid"): s for s in raw_stations if s.get("stid")}
    generated_at = _now().isoformat()
    current_features = _station_features(stations, raws_by_stid)

    state = _load_state()
    today = _now().date().isoformat()
    if state.get("date_local") != today:
        state = {"date_local": today, "peak_by_stid": {}}
    peak_by_stid = state.setdefault("peak_by_stid", {})
    for feature in current_features:
        stid = feature["properties"]["stid"]
        current = feature["properties"]
        previous = peak_by_stid.get(stid)
        if previous is None or current["beta_score"] > previous["beta_score"]:
            peak_by_stid[stid] = feature

    peak_features = list(peak_by_stid.values())
    difference_features = []
    for feature in current_features:
        properties = dict(feature["properties"])
        properties["category_difference"] = (
            properties["beta_category"] - properties["official_category"]
        )
        difference_features.append({**feature, "properties": properties})

    products = {
        "realtime_current": _feature_collection("Beta realtime current", current_features, generated_at),
        "observed_peak": _feature_collection("Beta observed peak", peak_features, generated_at),
        "production_vs_beta": _feature_collection(
            "Production category versus beta score", difference_features, generated_at
        ),
    }
    BETA_GIS_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    filenames = {
        "realtime_current": "realtime_current.geojson",
        "observed_peak": "observed_peak.geojson",
        "production_vs_beta": "production_vs_beta.geojson",
    }
    for product, payload in products.items():
        output = BETA_GIS_DIR / filenames[product]
        _atomic_json_write(output, payload)
        paths[product] = str(output)
    _atomic_json_write(BETA_OBSERVATION_STATE_PATH, state)

    manifest = load_manifest()
    manifest.update({
        "manifest_version": "1.0.0",
        "scorer_version": BETA_SCORER_VERSION,
        "observation_updated_at": generated_at,
        "products": {
            **(manifest.get("products") or {}),
            **{
                product: {
                    "kind": "geojson",
                    "path": f"gis/{filenames[product]}",
                    "feature_count": len(payload["features"]),
                    "generated_at": generated_at,
                }
                for product, payload in products.items()
            },
        },
    })
    save_manifest(manifest)
    return {"manifest": manifest, "products": products}


def load_manifest() -> dict:
    try:
        return json.loads(BETA_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {
            "manifest_version": "1.0.0",
            "scorer_version": BETA_SCORER_VERSION,
            "status": "waiting",
            "products": {},
        }


def save_manifest(manifest: dict) -> None:
    _atomic_json_write(BETA_MANIFEST_PATH, manifest)

