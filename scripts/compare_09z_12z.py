"""Compare same-day 09z and operational 12z station forecasts.

Numeric metrics use only exact (station_id, valid_time) matches. Dated summary
and GeoJSON artifacts are retained for the metrics page, while latest copies
remain available for existing consumers.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.fire_danger import category_label  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

API_ROOT = Path(__file__).resolve().parent.parent
FORECAST_ARCHIVE_DIR = API_ROOT / "archive" / "forecasts"
METRICS_ARCHIVE_DIR = API_ROOT / "archive" / "forecast_09z_metrics"
GIS_DIR = API_ROOT / "gis"
GIS_ARCHIVE_DIR = GIS_DIR / "forecast_09z_vs_12z" / "archive"
LATEST_SUMMARY_PATH = API_ROOT / "archive" / "forecast_09z_vs_12z_summary.json"
LATEST_GEOJSON_PATH = GIS_DIR / "forecast_09z_vs_12z.geojson"

METRICS = ("temp_c", "rh", "wind_speed_ms", "fuel_moisture")
RUN_FILE_PATTERN = re.compile(r"^station_forecasts_(\d{8})_(09|12)\.json$")


def _run_file(cycle_suffix: str, date_key: str | None = None) -> Path | None:
    matches = []
    for path in FORECAST_ARCHIVE_DIR.glob(f"station_forecasts_*_{cycle_suffix}.json"):
        match = RUN_FILE_PATTERN.fullmatch(path.name)
        if not match or match.group(2) != cycle_suffix:
            continue
        if date_key and match.group(1) != date_key:
            continue
        matches.append(path)
    matches.sort(key=_date_key)
    return matches[-1] if matches else None


def _date_key(path: Path) -> str:
    return path.stem.split("_")[2]


def _display_date(date_key: str) -> str:
    return datetime.strptime(date_key, "%Y%m%d").strftime("%Y-%m-%d")


def _load_stations(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    stations = payload.get("stations")
    return stations if isinstance(stations, dict) else {}


def _number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _by_valid_time(forecasts: list[dict]) -> dict[str, dict]:
    return {
        str(forecast["time"]): forecast
        for forecast in forecasts
        if isinstance(forecast, dict) and forecast.get("time")
    }


def _peak_forecast(forecasts: list[dict]) -> dict | None:
    eligible = [forecast for forecast in forecasts if _number(forecast.get("fire_danger"))]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda forecast: (
            forecast["fire_danger"],
            -(forecast.get("fuel_moisture") if _number(forecast.get("fuel_moisture")) else math.inf),
        ),
    )


def _correlation(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 2:
        return None
    left = [pair[0] for pair in pairs]
    right = [pair[1] for pair in pairs]
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in pairs)
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left)
        * sum((b - right_mean) ** 2 for b in right)
    )
    return numerator / denominator if denominator else None


def _summary(pairs: list[tuple[float, float]]) -> dict:
    if not pairs:
        return {"count": 0, "mean_bias": None, "mean_abs_diff": None, "rmse": None, "correlation": None}
    differences = [run_09 - run_12 for run_09, run_12 in pairs]
    count = len(differences)
    return {
        "count": count,
        "mean_bias": sum(differences) / count,
        "mean_abs_diff": sum(abs(value) for value in differences) / count,
        "rmse": math.sqrt(sum(value * value for value in differences) / count),
        "correlation": _correlation(pairs),
    }


def _forecast_view(forecast: dict | None) -> dict | None:
    if not forecast:
        return None
    return {
        "valid_time": forecast.get("time"),
        "temp_c": forecast.get("temp_c"),
        "rh": forecast.get("rh"),
        "wind_speed_ms": forecast.get("wind_speed_ms"),
        "fuel_moisture": forecast.get("fuel_moisture"),
        "fire_danger_category": forecast.get("fire_danger"),
        "fire_danger_label": category_label(forecast.get("fire_danger")),
    }


def build_comparison(file_09z: Path, file_12z: Path) -> tuple[dict, dict]:
    stations_09 = _load_stations(file_09z)
    stations_12 = _load_stations(file_12z)
    metric_pairs = {metric: [] for metric in METRICS}
    features = []
    matched_times: list[str] = []
    category_differences: list[float] = []
    category_matches = 0
    category_within_one = 0
    compared_stations = 0
    record_count = 0

    for station_id in sorted(set(stations_09) & set(stations_12)):
        station_09 = stations_09[station_id]
        station_12 = stations_12[station_id]
        forecasts_09 = _by_valid_time(station_09.get("forecasts") or [])
        forecasts_12 = _by_valid_time(station_12.get("forecasts") or [])
        common_times = sorted(set(forecasts_09) & set(forecasts_12))
        if not common_times:
            continue

        compared_stations += 1
        record_count += len(common_times)
        matched_times.extend(common_times)
        station_pairs = {metric: [] for metric in METRICS}
        station_category_differences = []

        for valid_time in common_times:
            forecast_09 = forecasts_09[valid_time]
            forecast_12 = forecasts_12[valid_time]
            for metric in METRICS:
                value_09 = forecast_09.get(metric)
                value_12 = forecast_12.get(metric)
                if _number(value_09) and _number(value_12):
                    pair = (float(value_09), float(value_12))
                    metric_pairs[metric].append(pair)
                    station_pairs[metric].append(pair)

            danger_09 = forecast_09.get("fire_danger")
            danger_12 = forecast_12.get("fire_danger")
            if _number(danger_09) and _number(danger_12):
                difference = float(danger_09 - danger_12)
                category_differences.append(difference)
                station_category_differences.append(difference)
                category_matches += int(difference == 0)
                category_within_one += int(abs(difference) <= 1)

        latitude = station_09.get("lat")
        longitude = station_09.get("lon")
        if not _number(latitude) or not _number(longitude):
            continue

        common_09 = [forecasts_09[valid_time] for valid_time in common_times]
        common_12 = [forecasts_12[valid_time] for valid_time in common_times]
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [longitude, latitude]},
            "properties": {
                "station_id": station_id,
                "matched_hours": len(common_times),
                "overlap_start": common_times[0],
                "overlap_end": common_times[-1],
                "metrics": {metric: _summary(pairs) for metric, pairs in station_pairs.items()},
                "fire_danger": {
                    "count": len(station_category_differences),
                    "agreement_rate": (
                        sum(value == 0 for value in station_category_differences) / len(station_category_differences)
                        if station_category_differences else None
                    ),
                    "mean_bias": (
                        sum(station_category_differences) / len(station_category_differences)
                        if station_category_differences else None
                    ),
                },
                "peak_09z": _forecast_view(_peak_forecast(common_09)),
                "peak_12z": _forecast_view(_peak_forecast(common_12)),
            },
        })

    date_key = _date_key(file_12z)
    date = _display_date(date_key)
    category_count = len(category_differences)
    generated_at = datetime.now(timezone.utc).isoformat()
    summary = {
        "schema_version": 2,
        "date": date,
        "generated_at": generated_at,
        "comparison": "09z_minus_12z",
        "run_09z": {"cycle": 9, "forecast_hours": [5, 18], "file": file_09z.name},
        "run_12z": {"cycle": 12, "file": file_12z.name},
        "station_count": compared_stations,
        "record_count": record_count,
        "overlap": {
            "start": min(matched_times) if matched_times else None,
            "end": max(matched_times) if matched_times else None,
        },
        "metrics": {metric: _summary(pairs) for metric, pairs in metric_pairs.items()},
        "fire_danger_category_agreement": {
            "matches": category_matches,
            "within_one": category_within_one,
            "total": category_count,
            "agreement_rate": category_matches / category_count if category_count else None,
            "within_one_rate": category_within_one / category_count if category_count else None,
            "mean_bias": sum(category_differences) / category_count if category_count else None,
            "mean_abs_diff": sum(abs(value) for value in category_differences) / category_count if category_count else None,
        },
    }
    geojson = {
        "type": "FeatureCollection",
        "name": "09z vs 12z matched-time forecast comparison",
        "metadata": summary,
        "features": features,
    }
    return geojson, summary


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    logger.info("Wrote %s", path)


def main(date: str | None = None) -> int:
    requested_key = date.replace("-", "") if date else None
    file_12z = _run_file("12", requested_key)
    if not file_12z:
        logger.info("No 12z station forecast archive found%s - skipping comparison.", f" for {date}" if date else "")
        return 0

    date_key = _date_key(file_12z)
    file_09z = _run_file("09", date_key)
    if not file_09z:
        logger.info("No same-day 09z station forecast archive (%s) - skipping comparison.", date_key)
        return 0

    geojson, summary = build_comparison(file_09z, file_12z)
    display_date = summary["date"]
    _write_json(METRICS_ARCHIVE_DIR / f"{display_date}.json", summary)
    _write_json(GIS_ARCHIVE_DIR / f"{display_date}.geojson", geojson)
    _write_json(LATEST_SUMMARY_PATH, summary)
    _write_json(LATEST_GEOJSON_PATH, geojson)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", help="Forecast date in YYYY-MM-DD format (defaults to latest 12z archive)")
    arguments = parser.parse_args()
    raise SystemExit(main(arguments.date))
