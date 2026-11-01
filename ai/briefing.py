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
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

from core.fire_events import MO_LAT_MAX, MO_LAT_MIN, MO_LON_MAX, MO_LON_MIN

DANGER_LABELS = ("Low", "Moderate", "Elevated", "Critical", "Extreme")
REGION_NAMES = {
    "NW": "Northwest",
    "NE": "Northeast",
    "Central": "Central",
    "SW": "Southwest",
    "SE": "Southeast",
}
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


def region_for(latitude: float, longitude: float) -> str:
    # A central Missouri box prevents Columbia/Jefferson City/central Ozarks
    # stations from being forced into a broad east/west quadrant.
    if 37.2 <= latitude <= 39.6 and -93.8 <= longitude <= -91.4:
        return "Central"
    if latitude >= 38.5:
        return "NW" if longitude < -92.5 else "NE"
    return "SW" if longitude < -92.5 else "SE"


def _direction_sector(degrees: float) -> str:
    directions = (
        "north", "northeast", "east", "southeast",
        "south", "southwest", "west", "northwest",
    )
    return directions[int((degrees + 22.5) % 360 // 45)]


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
        wind_directions = values("wind_direction_deg")
        danger_peak_times = []
        if danger:
            peak_danger = max(danger)
            for item in forecasts:
                item_danger = _finite(item.get("fire_danger"))
                if item_danger is not None and int(item_danger) == peak_danger:
                    timestamp = item.get("time")
                    if timestamp:
                        danger_peak_times.append(timestamp)
        if not any((precip, fuel, rh, wind, temp_c, danger)):
            continue

        def at_extreme(key: str, pick) -> str | None:
            """Timestamp of the hour where `key` is at its min/max for this station."""
            best = None
            for item in forecasts:
                number = _finite(item.get(key))
                if number is not None and item.get("time") and (best is None or pick(number, best[0])):
                    best = (number, item["time"])
            return best[1] if best else None

        # Hours with measurable rain: the forecast field is cumulative, so a rise
        # between consecutive hours marks the rain window.
        rain_hours = []
        previous_precip = 0.0
        for item in forecasts:
            number = _finite(item.get("precip_in"))
            if number is None:
                continue
            if number - previous_precip >= 0.01 and item.get("time"):
                rain_hours.append(item["time"])
            previous_precip = max(previous_precip, number)
        danger_by_time = [
            (item.get("time"), int(number)) for item in forecasts
            if item.get("time") and (number := _finite(item.get("fire_danger"))) is not None
            and 0 <= int(number) <= 4
        ]

        rows.append({
            "station_id": str(station_id),
            "region": region_for(latitude, longitude),
            "latitude": latitude,
            "longitude": longitude,
            # Cumulative precipitation is reduced to the maximum forecast
            # accumulation for the station, while weather variables use the
            # daily extrema relevant to fire danger.
            "precip_in": max(precip) if precip else None,
            "fuel_moisture": min(fuel) if fuel else None,
            "rh": min(rh) if rh else None,
            "wind_mph": max(wind) * MS_TO_MPH if wind else None,
            "wind_direction_sectors": [
                _direction_sector(direction) for direction in wind_directions
            ],
            "temp_f": max(temp_c) * 9 / 5 + 32 if temp_c else None,
            "fire_danger": max(danger) if danger else None,
            "danger_peak_times": danger_peak_times,
            "min_rh_time": at_extreme("rh", lambda new, old: new < old),
            "max_wind_time": at_extreme("wind_speed_ms", lambda new, old: new > old),
            "max_temp_time": at_extreme("temp_c", lambda new, old: new > old),
            "rain_hours": rain_hours,
            "danger_by_time": danger_by_time,
        })
    return rows


def _central_label(timestamp: str) -> str | None:
    try:
        parsed = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ZoneInfo("UTC"))
    return parsed.astimezone(ZoneInfo("America/Chicago")).strftime("%-I %p %Z")


def _central_span(timestamps: Iterable[str | None]) -> dict[str, str] | None:
    """Earliest-to-latest Central-time span of the given ISO timestamps (middle half of
    stations' values would hide outliers; the full span is what the data supports)."""
    parsed = []
    for timestamp in timestamps:
        if not timestamp:
            continue
        try:
            value = datetime.fromisoformat(str(timestamp).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue
        parsed.append(value if value.tzinfo else value.replace(tzinfo=ZoneInfo("UTC")))
    if not parsed:
        return None
    return {"from": _central_label(min(parsed).isoformat()), "to": _central_label(max(parsed).isoformat())}


def _danger_trend(rows: list[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Highest danger class among stations in the first and last third of the period,
    so prose can say whether danger builds or eases through the day."""
    times = sorted({time for row in rows for time, _ in row.get("danger_by_time", [])})
    if len(times) < 3:
        return None
    third = max(1, len(times) // 3)
    early, late = set(times[:third]), set(times[-third:])

    def highest(window: set[str]) -> int | None:
        values = [level for row in rows for time, level in row.get("danger_by_time", []) if time in window]
        return max(values) if values else None

    start, end = highest(early), highest(late)
    if start is None or end is None:
        return None
    return {
        "early": DANGER_LABELS[start],
        "late": DANGER_LABELS[end],
        "direction": "building" if end > start else "easing" if end < start else "steady",
    }


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
    direction_counts: dict[str, int] = {}
    direction_start_counts: dict[str, int] = {}
    direction_end_counts: dict[str, int] = {}
    for row in row_list:
        directions = row.get("wind_direction_sectors", [])
        for direction in directions:
            direction_counts[direction] = direction_counts.get(direction, 0) + 1
        if directions:
            direction_start_counts[directions[0]] = direction_start_counts.get(directions[0], 0) + 1
            direction_end_counts[directions[-1]] = direction_end_counts.get(directions[-1], 0) + 1
    result["wind_direction_counts"] = direction_counts
    result["predominant_wind_direction"] = (
        max(direction_counts, key=direction_counts.get)
        if direction_counts
        else None
    )
    result["wind_direction_start"] = (
        max(direction_start_counts, key=direction_start_counts.get)
        if direction_start_counts
        else None
    )
    result["wind_direction_end"] = (
        max(direction_end_counts, key=direction_end_counts.get)
        if direction_end_counts
        else None
    )
    danger_values = [
        row["fire_danger"] for row in row_list
        if isinstance(row.get("fire_danger"), int)
    ]
    result["highest_fire_danger"] = DANGER_LABELS[max(danger_values)] if danger_values else None
    result["highest_station_fire_danger"] = result["highest_fire_danger"]
    peak_times = [
        timestamp for row in row_list for timestamp in row.get("danger_peak_times", [])
    ]
    if peak_times:
        try:
            parsed_times = sorted(
                datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                for timestamp in peak_times
            )
            central = ZoneInfo("America/Chicago")
            result["fire_danger_peak_start"] = parsed_times[0].astimezone(central).strftime("%I:%M %p %Z").lstrip("0")
            result["fire_danger_peak_end"] = parsed_times[-1].astimezone(central).strftime("%I:%M %p %Z").lstrip("0")
        except (TypeError, ValueError):
            result["fire_danger_peak_start"] = None
            result["fire_danger_peak_end"] = None
    else:
        result["fire_danger_peak_start"] = None
        result["fire_danger_peak_end"] = None
    result["min_rh_time"] = _central_span(row.get("min_rh_time") for row in row_list)
    result["max_wind_time"] = _central_span(row.get("max_wind_time") for row in row_list)
    result["max_temp_time"] = _central_span(row.get("max_temp_time") for row in row_list)
    rain_stations = [row for row in row_list if row.get("rain_hours")]
    result["precip_station_share"] = (
        round(len(rain_stations) / len(row_list), 2) if row_list else None
    )
    result["precip_window"] = _central_span(
        hour for row in rain_stations for hour in row["rain_hours"]
    )
    result["danger_trend"] = _danger_trend(row_list)
    result["precipitation"] = (
        "trace" if result["precip_in"]["max"] is not None and result["precip_in"]["max"] < 0.05
        else "measurable"
        if result["precip_in"]["max"] is not None
        else "unavailable"
    )
    return result


COUNTY_SHAPEFILE = Path(__file__).resolve().parent.parent / "maps" / "shapefiles" / "MO_County_Boundaries" / "MO_County_Boundaries.shp"
# Shapefile COUNTYNAME -> the spelling PerCounty.py writes to dangerbycounty.json.
_COUNTY_NAME_OVERRIDES = {
    "Dekalb": "DeKalb", "St Charles": "St. Charles", "St Clair": "St. Clair", "St Francois": "St. Francois",
    "St Louis": "St. Louis", "St Louis City": "St. Louis City", "Ste Genevieve": "Ste. Genevieve",
}
MAX_COUNTIES_PER_CLASS = 15
MAX_COUNTIES_PER_REGION_CLASS = 5  # the prose may name only five counties per sentence; class_counts keeps the totals
MAX_DISCUSSION_SENTENCES = 16


@lru_cache(maxsize=1)
def county_regions() -> dict[str, dict[str, str]]:
    """County name -> {"region": "NW"|..., "fips": "29xxx"} from each county's bounding-box
    centre (the shapefile is Web Mercator), using the same region_for() as the stations."""
    import shapefile

    try:
        reader = shapefile.Reader(str(COUNTY_SHAPEFILE))
    except Exception:
        return {}
    fields = [field[0] for field in reader.fields[1:]]
    result: dict[str, dict[str, str]] = {}
    for shape_record in reader.iterShapeRecords():
        row = dict(zip(fields, shape_record.record))
        x_min, y_min, x_max, y_max = shape_record.shape.bbox
        radius = 6378137.0
        longitude = math.degrees(((x_min + x_max) / 2) / radius)
        latitude = math.degrees(math.atan(math.sinh(((y_min + y_max) / 2) / radius)))
        name = str(row.get("COUNTYNAME") or "").strip()
        name = _COUNTY_NAME_OVERRIDES.get(name, name)
        raw = str(row.get("COUNTYFIPS") or "").strip()
        result[name] = {
            "region": region_for(latitude, longitude),
            "fips": raw if len(raw) == 5 else f"29{raw.zfill(3)}",
        }
    return result


def _read_county_entries(county_path: Path | None) -> list[Mapping[str, Any]]:
    if county_path is None or not county_path.exists():
        return []
    try:
        with county_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, Mapping):
        return payload.get("counties", [])
    return []


def _county_entries(
    county_path: Path | None,
    source_mtime: float | None,
    tif_path: Path | None,
) -> list[Mapping[str, Any]]:
    """County danger classes that belong to THIS forecast, or [] when none can be shown.

    dangerbycounty.json is used only if it is not older than the station forecast it
    accompanies (the 12z script runs PerCounty first; the 9z run never writes it).
    Otherwise the classes are computed in memory from the peak-danger raster, again
    only if that raster is fresh. Stale county data is never used."""
    def fresh(path: Path | None) -> bool:
        if path is None or not path.exists():
            return False
        return source_mtime is None or path.stat().st_mtime >= source_mtime

    if fresh(county_path):
        entries = _read_county_entries(county_path)
        if entries:
            return entries
    if tif_path is not None and fresh(tif_path):
        try:
            from forecast.PerCounty import compute_county_danger

            return [
                {"county": entry["county"], "max_fire_danger": entry["max_fire_danger"]}
                for entry in compute_county_danger(tif_path)
            ]
        except Exception as exc:  # optional detail: never block the discussion
            print(f"County danger could not be computed for the briefing: {exc}")
    return []


def _county_summary(entries: list[Mapping[str, Any]]) -> dict[str, Any] | None:
    values = []
    counties_by_class: dict[str, list[str]] = {}
    by_region: dict[str, dict[str, list[str]]] = {}
    regions = county_regions()
    for entry in entries:
        value = _finite(entry.get("max_fire_danger")) if isinstance(entry, Mapping) else None
        if value is None or not 0 <= int(value) < len(DANGER_LABELS):
            continue
        level = int(value)
        values.append(level)
        county_name = entry.get("county")
        if not county_name:
            continue
        label = DANGER_LABELS[level]
        counties_by_class.setdefault(label, []).append(county_name)
        region = regions.get(county_name, {}).get("region")
        if region:
            by_region.setdefault(REGION_NAMES[region], {}).setdefault(label, []).append(county_name)
    if not values:
        return None
    counts = {label: values.count(index) for index, label in enumerate(DANGER_LABELS)}
    # Low counties are the default; only name the ones above it, capped per class.
    region_detail = {
        region: {
            "class_counts": {label: len(names) for label, names in classes.items()},
            # Only the region's highest class above Low is listed (five names at most): one short list per
            # region keeps the prose to one county sentence each; lower classes stay in class_counts.
            "counties_above_low": {
                label: sorted(names)[:MAX_COUNTIES_PER_REGION_CLASS]
                for label, names in list(
                    sorted(
                        ((l, n) for l, n in classes.items() if l != DANGER_LABELS[0]),
                        key=lambda item: DANGER_LABELS.index(item[0]),
                        reverse=True,
                    )
                )[:1]
            },
        }
        for region, classes in by_region.items()
    }
    return {
        "county_count": len(values),
        "fire_danger_class_counts": {label: count for label, count in counts.items() if count},
        "fire_danger_present": [label for label, count in counts.items() if count],
        "highest_fire_danger": DANGER_LABELS[max(values)],
        "by_region": region_detail,
    }


def _red_flag(alerts: list[Mapping[str, Any]] | None) -> list[dict[str, Any]]:
    """Active Red Flag Warnings / Fire Weather Watches as event -> counties. Empty list
    when none are active (the key is then left out of the briefing entirely)."""
    if alerts is None:
        try:
            from services.mobile_content import active_fire_weather_alerts

            alerts = active_fire_weather_alerts()
        except Exception:
            return []
    names_by_fips = {info["fips"]: name for name, info in county_regions().items()}
    grouped: dict[str, dict[str, Any]] = {}
    for alert in alerts:
        event = alert.get("event")
        if not event:
            continue
        group = grouped.setdefault(event, {"event": event, "counties": set(), "ends": None})
        group["counties"].update(
            names_by_fips[fips] for fips in alert.get("countyFips") or [] if fips in names_by_fips
        )
        ends = alert.get("expires")
        label = _central_label(ends) if ends else None
        if label and group["ends"] is None:
            group["ends"] = label
    return [
        {"event": group["event"], "counties": sorted(group["counties"]), "ends": group["ends"]}
        for group in grouped.values() if group["counties"]
    ]


def build_briefing(
    archive_dir: str | Path,
    forecast_path: str | Path | None = None,
    county_path: str | Path | None = None,
    alerts: list[Mapping[str, Any]] | None = None,
    county_tif_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a JSON-serializable briefing from the latest station forecast.

    Only forecast products go in: station forecast values, county danger classes for the
    same run, and active Red Flag/Fire Weather Watch counties. Current-conditions data
    (smoke dispersal, mixing height) is deliberately excluded."""
    archive_path = Path(archive_dir)
    path = Path(forecast_path) if forecast_path else _latest_forecast_file(archive_path)
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    rows = _daily_station_rows(payload)
    regions = {
        REGION_NAMES[name]: _summary(row for row in rows if row["region"] == name)
        for name in ("NW", "NE", "Central", "SW", "SE")
    }
    gis_dir = archive_path.parent.parent / "gis"
    resolved_county_path = Path(county_path) if county_path else gis_dir / "dangerbycounty.json"
    resolved_tif = Path(county_tif_path) if county_tif_path else gis_dir / "peak_fire_danger.tif"
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
    county_summary = _county_summary(
        _county_entries(
            resolved_county_path,
            path.stat().st_mtime if not county_path else None,
            resolved_tif,
        )
    )
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
    red_flag = _red_flag(alerts)
    if red_flag:
        briefing["red_flag"] = red_flag
    return briefing


def briefing_json(briefing: Mapping[str, Any]) -> str:
    """Serialize a briefing consistently for a model prompt and logs."""
    return json.dumps(briefing, sort_keys=True, separators=(",", ":"), allow_nan=False)


MAX_COUNTIES_PER_SENTENCE = 5


def _county_placement_problem(text: str, briefing: Mapping[str, Any]) -> str | None:
    """A sentence may name at most five counties, and when it names exactly one region every county in it
    must belong to that region."""
    regions = county_regions()
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        counties = [n for n in regions if re.search(rf"(?<![\w.]){re.escape(n)}(?![\w])", sentence)]
        if len(counties) > MAX_COUNTIES_PER_SENTENCE:
            return f"more than {MAX_COUNTIES_PER_SENTENCE} counties in one sentence ({', '.join(counties)})"
        named = [r for r in briefing.get("regions", {}) if re.search(rf"\b{r}\b", sentence)]
        if len(named) == 1 and any(REGION_NAMES.get(regions[c]["region"]) != named[0] for c in counties):
            return "county named under the wrong region"
    return None


def briefing_text_problem(text: str, briefing: Mapping[str, Any]) -> str | None:
    """Why model prose is unacceptable (unsupported class, place, wording or rainfall), or None if it is fine."""
    problem = _county_placement_problem(text, briefing)
    if problem:
        return problem
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
            return "unsupported danger class"

    if re.search(r"dispersal|mixing height|transport wind|smoke", text, re.IGNORECASE):
        return "smoke/dispersal wording"
    # Talk about the briefing itself, or about data that is missing, is never forecast content.
    meta = re.search(r"\bbriefing\b|\bjson\b|not available|unavailable|no data|not provided", text, re.IGNORECASE)
    if meta:
        return f"talk about the briefing or missing data ({meta.group(0)!r})"
    # The forecast carries sustained wind only.
    if re.search(r"\bgust", text, re.IGNORECASE):
        return "gust wording"
    # Warnings/watches may be mentioned only when one is active (and then only as the briefing lists them).
    if not briefing.get("red_flag") and re.search(
        r"red flag|fire weather watch|\bwarnings?\b", text, re.IGNORECASE
    ):
        return "red flag/warning wording with none active"
    if not _named_places_supported(text, briefing):
        return "county or region not in the briefing"

    # Rain is forecast per region (amount and window), never per county: a county named in a sentence
    # about rain would be an invented location.
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if re.search(r"\b(?:rain\w*|precipitation|showers?)\b", sentence, re.IGNORECASE) and any(
            re.search(rf"(?<![\w.]){re.escape(name)}(?![\w])", sentence) for name in county_regions()
        ):
            return "rain sentence names a county"

    maximum = briefing["statewide"]["precip_in"].get("max")
    if maximum is None:
        return None
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        # A sentence about exactly one region is held to that region's own forecast maximum.
        named = [
            name for name in briefing.get("regions", {})
            if re.search(rf"\b{name}\b", sentence)
        ]
        cap = maximum
        if len(named) == 1:
            region_max = briefing["regions"][named[0]]["precip_in"].get("max")
            cap = region_max if region_max is not None else 0.0
        for match in re.finditer(
            r"(?<![\w.])(\d+(?:\.\d+)?)\s*(?:inches?|in\b|in\.)", sentence, re.IGNORECASE
        ):
            if float(match.group(1)) > cap * 1.2 + 0.005:
                return "rain amount above forecast maximum"
    return None


def validate_briefing_text(text: str, briefing: Mapping[str, Any]) -> bool:
    """Check model prose for unsupported danger classes, places and rainfall totals."""
    return briefing_text_problem(text, briefing) is None


def _named_places_supported(text: str, briefing: Mapping[str, Any]) -> bool:
    """Every county the prose names must appear in the supplied briefing, and a region
    with no stations may not be described."""
    supplied = json.dumps(briefing, sort_keys=True)
    for name in county_regions():
        # A county named anywhere in the text (with or without "County") must be one the briefing supplied.
        if re.search(rf"(?<![\w.]){re.escape(name)}(?![\w])", text) and f'"{name}"' not in supplied:
            return False
    for region, summary in briefing.get("regions", {}).items():
        if summary.get("station_count") == 0 and re.search(rf"\b{region} Missouri\b", text):
            return False
    return True


def contains_core_weather_facts(text: str, briefing: Mapping[str, Any]) -> bool:
    """Require generated forecast prose to carry the key numeric observations."""
    state = briefing["statewide"]
    for metric in ("rh", "fuel_moisture", "wind_mph"):
        values = state[metric]
        available = [values.get("min"), values.get("max")]
        if not any(value is not None for value in available):
            continue
        candidates = {
            candidate
            for value in available
            if value is not None
            for candidate in (
                str(value).rstrip("0").rstrip(".") or "0",
                str(round(value)),
            )
        }
        if not any(
            re.search(rf"(?<![\d.]){re.escape(candidate)}(?![\d.])", text)
            for candidate in candidates
        ):
            return False
    return True


def validate_operational_style(text: str, briefing: Mapping[str, Any]) -> bool:
    """Keep AI prose concise and appropriate for operational readers."""
    if len(re.findall(r"[.!?](?:\s|$)", text)) > MAX_DISCUSSION_SENTENCES:
        return False
    if re.search(r"\d+\.\d+\s*(?:°?\s*F|degrees?|mph|%)\b", text, re.IGNORECASE):
        return False
    if "average" in text.lower():
        return False
    if briefing["statewide"]["precipitation"] != "measurable" and re.search(
        r"\b(?:precipitation|rainfall|rain|showers?)\b", text, re.IGNORECASE
    ):
        return False
    return True
