"""Compact county fire-danger summary for the mobile home-screen widgets (GET /api/mobile/widget).

Each widget fetches this itself every 30+ minutes (mobile/src/widgets for Android, mobile/targets/widget for iOS), so it has
to stay small and cheap: the answer per county is cached for CACHE_SECONDS, which keeps the work at 115 counties (with St. Louis City) however
many widgets exist. Built from the burn check's assessment at a point inside the county (services.burn_check.assess), so a
widget and /v1/burn-check never disagree about current conditions, alerts or the burn ban.
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Callable, Optional

from pyproj import Transformer

from core.fire_danger import CATEGORY_HEX_COLORS, CATEGORY_LABELS
from services.cdn_thumbnails import thumbnail_name, thumbnails_enabled
from services.mobile_content import PUBLIC_CDN_BASE_URL

CACHE_SECONDS = 600
FORECAST_MAP_FILE = "mo-forecastfiredanger.png"

_cache: dict[str, tuple[float, dict]] = {}
_cache_lock = threading.Lock()


@lru_cache(maxsize=1)
def county_points() -> dict[str, tuple[float, float, str]]:
    """{fips: (lat, lon, name)}: a point guaranteed inside each county (representative_point, not the centroid, which can
    fall outside an irregular county)."""
    from services.county_lookup import _county_polygons

    to_lonlat = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
    points: dict[str, tuple[float, float, str]] = {}
    for prepared, _bounds, fips, name in _county_polygons():
        inside = prepared.context.representative_point()
        lon, lat = to_lonlat.transform(inside.x, inside.y)
        points[fips] = (lat, lon, name)
    return points


def _level(value: Any) -> Optional[dict]:
    if value is None:
        return None
    level = int(value)
    return {"level": level, "label": CATEGORY_LABELS[level], "color": CATEGORY_HEX_COLORS[level]}


def _peak(hours: list[dict]) -> Optional[dict]:
    """The highest fire danger forecast for the rest of today and the hours it spans ("11 AM" to "10 PM")."""
    rated = [hour for hour in hours if hour.get("fire_danger") is not None]
    if not rated:
        return None
    top = max(hour["fire_danger"] for hour in rated)
    at_top = [hour for hour in rated if hour["fire_danger"] == top]
    first, last = at_top[0], at_top[-1]
    end = datetime.fromisoformat(last["time_utc"].replace("Z", "+00:00")) + timedelta(hours=1)
    from services.burn_check import hour_label

    return {**_level(top), "start": first["hour_label"], "end": hour_label(end)}


def _map_url(now: datetime) -> str:
    filename = (thumbnails_enabled() and thumbnail_name(FORECAST_MAP_FILE)) or FORECAST_MAP_FILE
    # Same URL all day as the forecast is republished; the hourly query keeps device and CDN caches from going stale.
    return f"{PUBLIC_CDN_BASE_URL}/{filename}?v={now:%Y%m%d%H}"


def build_summary(fips: str, *, assess: Optional[Callable[..., dict]] = None) -> dict:
    """The widget payload for one county (uncached). Raises KeyError for an unknown FIPS."""
    lat, lon, name = county_points()[fips]
    if assess is None:
        from services.burn_check import assess
    result = assess(lat, lon)
    current = result.get("current")
    generated = datetime.fromisoformat(result["generated_at"].replace("Z", "+00:00"))
    return {
        "county": {"fips": fips, "name": name},
        "generatedAt": result["generated_at"],
        "peak": _peak(result.get("today", {}).get("hours") or []),
        "current": None if current is None else {
            **(_level(current.get("fire_danger")) or {"level": None, "label": None, "color": None}),
            "rhPct": current.get("rh_pct"),
            "windMph": current.get("wind_mph"),
            "tempF": current.get("temp_f"),
            "fuelMoisturePct": current.get("fuel_moisture_pct"),
            "observedAt": current.get("observed_at"),
        },
        "alerts": [{"event": str(alert.get("event") or "Fire weather alert")} for alert in result.get("alerts", {}).get("active") or []],
        "burnBan": bool(result.get("burn_ban", {}).get("active")),
        "mapUrl": _map_url(generated),
    }


def widget_summary(fips: str, *, clock: Callable[[], float] = time.monotonic,
                   build: Callable[[str], dict] = build_summary) -> dict:
    """build_summary, cached per county for CACHE_SECONDS."""
    now = clock()
    with _cache_lock:
        hit = _cache.get(fips)
        if hit and now - hit[0] < CACHE_SECONDS:
            return hit[1]
    summary = build(fips)
    with _cache_lock:
        _cache[fips] = (now, summary)
    return summary


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()
