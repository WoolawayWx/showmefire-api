"""Compare a 9z secondary forecast run against the same-day 12z operational run.

Reads the two per-station archive files DailyForecast.py already writes
(archive/forecasts/station_forecasts_{YYYYMMDD}_09.json and ..._12.json),
joins them by (station_id, valid_time), and writes:

  - gis/forecast_09z_vs_12z.geojson: one feature per station (its peak-risk
    hour from each run) with both runs' fuel moisture/fire-danger side by
    side, for mapping.
  - archive/forecast_09z_vs_12z_summary.json: per-metric mean-absolute-
    difference/bias across all matched (station, hour) pairs, plus a
    fire-danger category agreement count, for a quick numeric read.

If the same-day 12z archive doesn't exist yet (e.g. the 9z job ran before the
12z job that day), this logs and exits 0 rather than failing the 9z pipeline.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.fire_danger import category_label  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_ROOT = Path(__file__).resolve().parent.parent
ARCHIVE_DIR = API_ROOT / "archive" / "forecasts"
GIS_DIR = API_ROOT / "gis"

METRICS = ("temp_c", "rh", "wind_speed_ms", "fuel_moisture")


def _latest_run_file(cycle_suffix: str, date_str: str | None = None) -> Path | None:
    pattern = f"station_forecasts_{date_str}_{cycle_suffix}.json" if date_str else f"station_forecasts_*_{cycle_suffix}.json"
    matches = sorted(ARCHIVE_DIR.glob(pattern))
    return matches[-1] if matches else None


def _load_stations(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("stations") or {}


def _peak_forecast(forecasts: list[dict]) -> dict | None:
    """Pick the hour with the highest fire-danger category (ties broken by lowest fuel moisture)."""
    best = None
    for forecast in forecasts:
        if forecast.get("fire_danger") is None:
            continue
        if best is None or (
            forecast["fire_danger"], -(forecast.get("fuel_moisture") or 0)
        ) > (
            best["fire_danger"], -(best.get("fuel_moisture") or 0)
        ):
            best = forecast
    return best


def build_comparison(file_09z: Path, file_12z: Path) -> tuple[dict, dict]:
    stations_09 = _load_stations(file_09z)
    stations_12 = _load_stations(file_12z)
    shared_station_ids = sorted(set(stations_09) & set(stations_12))

    features = []
    deltas = {metric: [] for metric in METRICS}
    category_matches = 0
    category_total = 0

    for station_id in shared_station_ids:
        station_09 = stations_09[station_id]
        station_12 = stations_12[station_id]
        peak_09 = _peak_forecast(station_09.get("forecasts") or [])
        peak_12 = _peak_forecast(station_12.get("forecasts") or [])
        if not peak_09 or not peak_12:
            continue
        if station_09.get("lat") is None or station_09.get("lon") is None:
            continue

        for metric in METRICS:
            v09, v12 = peak_09.get(metric), peak_12.get(metric)
            if isinstance(v09, (int, float)) and isinstance(v12, (int, float)):
                deltas[metric].append(v09 - v12)

        if peak_09.get("fire_danger") is not None and peak_12.get("fire_danger") is not None:
            category_total += 1
            if peak_09["fire_danger"] == peak_12["fire_danger"]:
                category_matches += 1

        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [station_09["lon"], station_09["lat"]]},
            "properties": {
                "station_id": station_id,
                "run_09z": {
                    "valid_time": peak_09.get("time"),
                    "temp_c": peak_09.get("temp_c"),
                    "rh": peak_09.get("rh"),
                    "wind_speed_ms": peak_09.get("wind_speed_ms"),
                    "fuel_moisture": peak_09.get("fuel_moisture"),
                    "fire_danger_category": peak_09.get("fire_danger"),
                    "fire_danger_label": category_label(peak_09.get("fire_danger")),
                },
                "run_12z": {
                    "valid_time": peak_12.get("time"),
                    "temp_c": peak_12.get("temp_c"),
                    "rh": peak_12.get("rh"),
                    "wind_speed_ms": peak_12.get("wind_speed_ms"),
                    "fuel_moisture": peak_12.get("fuel_moisture"),
                    "fire_danger_category": peak_12.get("fire_danger"),
                    "fire_danger_label": category_label(peak_12.get("fire_danger")),
                },
                "fuel_moisture_delta": (
                    (peak_09.get("fuel_moisture") - peak_12.get("fuel_moisture"))
                    if isinstance(peak_09.get("fuel_moisture"), (int, float))
                    and isinstance(peak_12.get("fuel_moisture"), (int, float))
                    else None
                ),
                "fire_danger_category_delta": (
                    (peak_09.get("fire_danger") - peak_12.get("fire_danger"))
                    if peak_09.get("fire_danger") is not None and peak_12.get("fire_danger") is not None
                    else None
                ),
            },
        })

    geojson = {
        "type": "FeatureCollection",
        "name": "9z vs 12z forecast peak comparison",
        "metadata": {
            "run_09z_file": file_09z.name,
            "run_12z_file": file_12z.name,
            "station_count": len(features),
        },
        "features": features,
    }

    def _summary(values: list[float]) -> dict:
        if not values:
            return {"count": 0, "mean_bias": None, "mean_abs_diff": None}
        count = len(values)
        return {
            "count": count,
            "mean_bias": sum(values) / count,
            "mean_abs_diff": sum(abs(v) for v in values) / count,
        }

    summary = {
        "run_09z_file": file_09z.name,
        "run_12z_file": file_12z.name,
        "station_count": len(shared_station_ids),
        "metrics": {metric: _summary(values) for metric, values in deltas.items()},
        "fire_danger_category_agreement": {
            "matches": category_matches,
            "total": category_total,
            "agreement_rate": (category_matches / category_total) if category_total else None,
        },
    }
    return geojson, summary


def main() -> int:
    file_09z = _latest_run_file("09")
    if not file_09z:
        logger.info("No 9z station forecast archive found yet - skipping comparison.")
        return 0

    date_str = file_09z.stem.split("_")[2]  # station_forecasts_{YYYYMMDD}_09
    file_12z = _latest_run_file("12", date_str)
    if not file_12z:
        logger.info(f"No same-day 12z station forecast archive ({date_str}) found yet - skipping comparison.")
        return 0

    geojson, summary = build_comparison(file_09z, file_12z)

    GIS_DIR.mkdir(parents=True, exist_ok=True)
    geojson_path = GIS_DIR / "forecast_09z_vs_12z.geojson"
    with geojson_path.open("w", encoding="utf-8") as handle:
        json.dump(geojson, handle, indent=2)
    logger.info(f"Wrote {geojson_path}")

    ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = ARCHIVE_DIR / "forecast_09z_vs_12z_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logger.info(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
