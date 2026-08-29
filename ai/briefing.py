"""Build a numeric, model-readable briefing from forecast artifacts.

The forecast maps remain useful for people, but they are deliberately not used
as the source of numeric values for AI-generated text. This module reduces the
station forecast artifact to one daily summary per station and then reports
regional and statewide ranges/medians.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from core.fire_events import MO_LAT_MAX, MO_LAT_MIN, MO_LON_MAX, MO_LON_MIN

DANGER_LABELS = ("Low", "Moderate", "Elevated", "Critical", "Extreme")
MS_TO_MPH = 2.2369362920544


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _round(value: float | None, digits: int = 1) -> float | None:
    return round(value, digits) if value is not None else None


def _region(latitude: float, longitude: float) -> str:
    # A central Missouri box prevents Columbia/Jefferson City/central Ozarks
    # stations from being forced into a broad east/west quadrant.
    if 37.2 <= latitude <= 39.6 and -93.8 <= longitude <= -91.4:
        return "Central"
    if latitude >= 38.5:
        return "NW" if longitude < -92.5 else "NE"
    return "SW" if longitude < -92.5 else "SE"


def _latest_forecast_file(archive_dir: Path) -> Path:
    candidates = sorted(
        (
            path
            for path in archive_dir.glob("station_forecasts_*.json")
            if re.fullmatch(r"station_forecasts_\d{8}_\d{2}\.json", path.name)
        ),
        key=lambda path: path.name,
    )
    if not candidates:
        raise FileNotFoundError(f"No station forecast JSON found in {archive_dir}")
    return candidates[-1]


def _daily_station_rows(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for station_id, station in (payload.get("stations") or {}).items():
        latitude = _finite(station.get("lat"))
        longitude = _finite(station.get("lon"))
        if latitude is None or longitude is None:
            continue
        if not (MO_LAT_MIN <= latitude <= MO_LAT_MAX and MO_LON_MIN <= longitude <= MO_LON_MAX):
            continue

        forecasts = station.get("forecasts") or []
        def values(key: str) -> list[float]:
            return [number for item in forecasts if (number := _finite(item.get(key))) is not None]

        precip = values("precip_in")
        fuel = values("fuel_moisture")
        rh = values("rh")
        wind = values("wind_speed_ms")
        temp_c = values("temp_c")
        danger = [int(number) for number in values("fire_danger") if 0 <= int(number) <= 4]
        if not any((precip, fuel, rh, wind, temp_c, danger)):
            continue

        rows.append({
            "station_id": str(station_id),
            "region": _region(latitude, longitude),
            "latitude": latitude,
            "longitude": longitude,
            # Cumulative precipitation is reduced to the maximum forecast
            # accumulation for the station, while weather variables use the
            # daily extrema relevant to fire danger.
            "precip_in": max(precip) if precip else None,
            "fuel_moisture": min(fuel) if fuel else None,
            "rh": min(rh) if rh else None,
            "wind_mph": max(wind) * MS_TO_MPH if wind else None,
            "temp_f": max(temp_c) * 9 / 5 + 32 if temp_c else None,
            "fire_danger": max(danger) if danger else None,
        })
    return rows


def _summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    row_list = list(rows)
    metric_keys = ("precip_in", "fuel_moisture", "rh", "wind_mph", "temp_f")
    metric_digits = {"precip_in": 3, "fuel_moisture": 1, "rh": 1, "wind_mph": 1, "temp_f": 1}
    result: dict[str, Any] = {"station_count": len(row_list)}
    for key in metric_keys:
        values = sorted(
            number for row in row_list if (number := _finite(row.get(key))) is not None
        )
        result[key] = {
            "min": _round(values[0], metric_digits[key]) if values else None,
            "p50": _round(_percentile(values, 0.5), metric_digits[key]) if values else None,
            "max": _round(values[-1], metric_digits[key]) if values else None,
        }
    danger_counts = {label: 0 for label in DANGER_LABELS}
    for row in row_list:
        danger = row.get("fire_danger")
        if isinstance(danger, int) and 0 <= danger < len(DANGER_LABELS):
            danger_counts[DANGER_LABELS[danger]] += 1
    result["fire_danger_station_counts"] = {
        label: count for label, count in danger_counts.items() if count
    }
    result["fire_danger_present"] = [
        label for label, count in danger_counts.items() if count
    ]
    danger_values = [
        row["fire_danger"] for row in row_list
        if isinstance(row.get("fire_danger"), int)
    ]
    result["highest_fire_danger"] = DANGER_LABELS[max(danger_values)] if danger_values else None
    result["precipitation"] = (
        "trace" if result["precip_in"]["max"] is not None and result["precip_in"]["max"] < 0.05
        else "measurable"
        if result["precip_in"]["max"] is not None
        else "unavailable"
    )
    return result


def _county_summary(county_path: Path | None) -> dict[str, Any] | None:
    if county_path is None or not county_path.exists():
        return None
    try:
        with county_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None

    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, Mapping):
        entries = payload.get("counties", [])
    else:
        entries = []
    values = []
    for entry in entries:
        value = _finite(entry.get("max_fire_danger")) if isinstance(entry, Mapping) else None
        if value is not None and 0 <= int(value) < len(DANGER_LABELS):
            values.append(int(value))
    if not values:
        return None
    counts = {label: values.count(index) for index, label in enumerate(DANGER_LABELS)}
    return {
        "county_count": len(values),
        "fire_danger_class_counts": {label: count for label, count in counts.items() if count},
        "fire_danger_present": [label for label, count in counts.items() if count],
        "highest_fire_danger": DANGER_LABELS[max(values)],
    }


def build_briefing(
    archive_dir: str | Path,
    forecast_path: str | Path | None = None,
    county_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a JSON-serializable briefing from the latest station forecast."""
    archive_path = Path(archive_dir)
    path = Path(forecast_path) if forecast_path else _latest_forecast_file(archive_path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    rows = _daily_station_rows(payload)
    regions = {name: _summary(row for row in rows if row["region"] == name)
               for name in ("NW", "NE", "Central", "SW", "SE")}
    resolved_county_path = (
        Path(county_path)
        if county_path
        else archive_path.parent.parent / "gis" / "dangerbycounty.json"
    )
    briefing = {
        "source_file": path.name,
        "run_date": payload.get("run_date"),
        "units": {
            "precip_in": "inches",
            "fuel_moisture": "percent",
            "rh": "percent",
            "wind_mph": "mph",
            "temp_f": "degrees Fahrenheit",
        },
        "regions": regions,
        "statewide": _summary(rows),
    }
    county_summary = _county_summary(resolved_county_path)
    if county_summary:
        briefing["county_danger"] = county_summary
        # County danger provides broader spatial coverage than stations. Merge
        # its valid classes into the allowed statewide vocabulary while keeping
        # station and county counts separate for transparency.
        present = set(briefing["statewide"]["fire_danger_present"])
        present.update(county_summary["fire_danger_present"])
        briefing["statewide"]["fire_danger_present"] = [
            label for label in DANGER_LABELS if label in present
        ]
        danger_values = [
            index for index, label in enumerate(DANGER_LABELS) if label in present
        ]
        briefing["statewide"]["highest_fire_danger"] = DANGER_LABELS[max(danger_values)]
    return briefing


def briefing_json(briefing: Mapping[str, Any]) -> str:
    """Serialize a briefing consistently for a model prompt and logs."""
    return json.dumps(briefing, sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_briefing_text(text: str, briefing: Mapping[str, Any]) -> bool:
    """Check model prose for unsupported danger classes and rainfall totals."""
    allowed = set(briefing["statewide"].get("fire_danger_present") or ["Low"])
    for label in DANGER_LABELS:
        danger_context = (
            rf"\b{label}\b(?:\s*(?:-|to)\s*\w+)?\s+"
            rf"(?:fire\s+)?(?:danger|risk|conditions?|class)\b"
        )
        class_context = (
            rf"\b(?:fire[- ]danger|fire risk|danger class|class)\b"
            rf"\s*(?:is|of|at|include|includes|:)?\s*\b{label}\b"
        )
        if (
            re.search(danger_context, text, re.IGNORECASE)
            or re.search(class_context, text, re.IGNORECASE)
        ) and label not in allowed:
            return False

    maximum = briefing["statewide"]["precip_in"].get("max")
    if maximum is None:
        return True
    return not any(
        float(match.group(1)) > maximum * 1.2
        for match in re.finditer(
            r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:inches?|in\b|in\.)",
            text,
            re.IGNORECASE,
        )
    )
