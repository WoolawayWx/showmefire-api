"""The next eight hours of weather at an incident, for the incident card.

Read from this server's own hourly HRRR product (services/hrrr_hourly), the same one the burn check uses, at the grid cell
nearest the incident's centre: temperature, humidity, wind (speed, gusts, direction it blows from), rain and the HRRR-only
fire-danger estimate for each hour from the current hour on. It is a model forecast, not a station reading, and says so.

Only looked up when an incident's detail is opened (never for the list), and cached per location and forecast hour.
Never raises: returns None when there is no fresh forecast or the point is off the grid.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from typing import Any, Optional
from zoneinfo import ZoneInfo

from core.fire_danger import CATEGORY_LABELS

logger = logging.getLogger(__name__)

HOURS = 8
MAX_AGE_HOURS = 8                     # same freshness the burn check demands of the hourly forecast
LOCAL_TZ = ZoneInfo("America/Chicago")
ELEMENTS = ("temp_f", "rh_pct", "wind_speed_mph", "wind_gust_mph", "wind_dir_deg", "precip_mm_1h", "fire_danger_est")
COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
NOTE = "Hourly HRRR model forecast at the incident's location, not a station reading. Fire danger is an estimate."


def cardinal(degrees: Optional[float]) -> Optional[str]:
    if degrees is None or degrees != degrees:
        return None
    return COMPASS[int((degrees % 360) / 22.5 + 0.5) % 16]


def _num(value: Any, digits: int = 0) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if not math.isfinite(number) else round(number, digits)


def _hour_label(moment: datetime) -> str:
    local = moment.astimezone(LOCAL_TZ)
    return f"{local.hour % 12 or 12} {'AM' if local.hour < 12 else 'PM'}"


def _hour(row: dict) -> dict:
    danger = _num(row.get("fire_danger_est"))
    degrees = _num(row.get("wind_dir_deg"))
    precip = _num(row.get("precip_mm_1h"), 1)
    return {
        "time_utc": row["time_utc"].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hour_label": _hour_label(row["time_utc"]),
        "temp_f": _num(row.get("temp_f")),
        "rh_pct": _num(row.get("rh_pct")),
        "wind_mph": _num(row.get("wind_speed_mph")),
        "gust_mph": _num(row.get("wind_gust_mph")),
        "wind_from_deg": degrees,
        "wind_from": cardinal(degrees),
        "precip_mm": precip,
        "fire_danger": None if danger is None else int(danger),
        "fire_danger_label": None if danger is None or int(danger) not in CATEGORY_LABELS else CATEGORY_LABELS[int(danger)],
    }


def _summary(hours: list[dict]) -> dict:
    def values(key):
        return [h[key] for h in hours if h.get(key) is not None]

    danger = values("fire_danger")
    winds = [h["wind_from_deg"] for h in hours if h.get("wind_from_deg") is not None]
    shift = False
    if len(winds) >= 2:
        delta = abs(winds[-1] - winds[0]) % 360
        shift = min(delta, 360 - delta) >= 60          # the wind turns 60 degrees or more across the window
    peak = max(danger) if danger else None
    return {
        "temp_min_f": min(values("temp_f"), default=None), "temp_max_f": max(values("temp_f"), default=None),
        "rh_min_pct": min(values("rh_pct"), default=None), "rh_max_pct": max(values("rh_pct"), default=None),
        "wind_max_mph": max(values("wind_mph"), default=None), "gust_max_mph": max(values("gust_mph"), default=None),
        "peak_fire_danger": peak,
        "peak_fire_danger_label": None if peak is None or peak not in CATEGORY_LABELS else CATEGORY_LABELS[peak],
        "wind_shift": shift,
        "rain": any((h.get("precip_mm") or 0) >= 0.1 for h in hours),
    }


def _forecast(latitude: float, longitude: float, now: datetime) -> Optional[dict]:
    from pathlib import Path

    from services.hrrr_hourly import root_dir, store

    root = Path(root_dir())
    cycle = store._latest_cycle_id(root)
    manifest = store.read_manifest(cycle, root) if cycle else None
    if not manifest:
        return None
    cycle_time = datetime.strptime(manifest["cycle_time_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    if (now - cycle_time).total_seconds() > MAX_AGE_HOURS * 3600:
        return None
    point = store.point_series(cycle, latitude, longitude, root, names=ELEMENTS)
    if not point or point.get("error"):
        return None
    series = point["series"]
    rows = []
    for index, stamp in enumerate(point["valid_times_utc"]):
        row = {"time_utc": datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)}
        for name in ELEMENTS:
            values = series.get(name) or []
            row[name] = values[index] if index < len(values) else None
        rows.append(row)
    return {"cycle": cycle, "cycle_time_utc": manifest["cycle_time_utc"], "rows": rows}


@lru_cache(maxsize=512)
def _cached(latitude: float, longitude: float, hour_key: str) -> Optional[dict]:
    now = datetime.strptime(hour_key, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
    return outlook_at(latitude, longitude, now)


def outlook_at(latitude: float, longitude: float, now: datetime, forecast=_forecast) -> Optional[dict]:
    """The card for one point and moment (uncached). `forecast` is injectable for tests."""
    found = forecast(latitude, longitude, now)
    if not found:
        return None
    start = now.replace(minute=0, second=0, microsecond=0)
    hours = [_hour(row) for row in found["rows"] if start <= row["time_utc"] < start + timedelta(hours=HOURS)]
    if not hours:
        return None
    return {"available": True, "hours": hours, "summary": _summary(hours), "cycle_time_utc": found["cycle_time_utc"],
            "source": "HRRR model forecast", "note": NOTE}


def weather_outlook(latitude: Any, longitude: Any, now: Optional[datetime] = None) -> Optional[dict]:
    """The next eight hours at a point, or None. Never raises."""
    try:
        lat, lon = float(latitude), float(longitude)
        if not (math.isfinite(lat) and math.isfinite(lon)):
            return None
        now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        result = _cached(round(lat, 2), round(lon, 2), now.strftime("%Y-%m-%dT%H"))
        return dict(result) if result else None
    except Exception:  # noqa: BLE001 - the card is optional; the incident renders without it
        logger.info("incident weather outlook unavailable", exc_info=True)
        return None
