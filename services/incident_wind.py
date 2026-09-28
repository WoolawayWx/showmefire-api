"""Current wind at a fire's location, for the incident popup.

Uses the NWS hourly forecast for the location's grid cell (the first period is
the current hour) - free, keyless, and already how this project talks to NWS.
This is a forecast-model estimate for the current hour, not a station reading,
and is labeled that way. Looked up only when someone opens an incident's
detail, never per map refresh, and cached per ~5 km cell so repeat clicks and
neighboring fires share one lookup.

Never raises: returns None if NWS is unreachable or the response is unusable.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

NWS_BASE = "https://api.weather.gov"
HEADERS = {"User-Agent": "ShowMeFire (https://showmefire.org)", "Accept": "application/geo+json"}
CACHE_TTL_S = 20 * 60
TIMEOUT_S = 6.0

COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
_cache: dict[tuple[float, float], tuple[float, Optional[dict]]] = {}
_lock = threading.Lock()


def _cardinal_to_degrees(cardinal: str) -> Optional[float]:
    try:
        return COMPASS.index(cardinal.strip().upper()) * 22.5
    except ValueError:
        return None


def _opposite(cardinal: str) -> str:
    return COMPASS[(COMPASS.index(cardinal) + 8) % 16]


def parse_period(period: dict) -> Optional[dict]:
    """NWS hourly period -> wind dict. windSpeed looks like '10 mph' or '5 to 15 mph'."""
    direction = str(period.get("windDirection") or "").strip().upper()
    speeds = [int(n) for n in re.findall(r"\d+", str(period.get("windSpeed") or ""))]
    if direction not in COMPASS or not speeds:
        return None
    return {
        "from_cardinal": direction,
        "from_degrees": _cardinal_to_degrees(direction),
        "toward_cardinal": _opposite(direction),
        "speed_mph": max(speeds),
        "speed_range_mph": [min(speeds), max(speeds)],
        "valid_at": period.get("startTime"),
        "source": "NWS hourly forecast (current hour)",
    }


def get_wind(latitude: float, longitude: float) -> Optional[dict]:
    key = (round(latitude / 0.05) * 0.05, round(longitude / 0.05) * 0.05)
    now = time.monotonic()
    with _lock:
        cached = _cache.get(key)
        if cached and now - cached[0] < CACHE_TTL_S:
            return cached[1]
    wind = None
    try:
        with httpx.Client(timeout=TIMEOUT_S, headers=HEADERS, follow_redirects=True) as client:
            point = client.get(f"{NWS_BASE}/points/{latitude:.4f},{longitude:.4f}")
            point.raise_for_status()
            hourly_url = point.json()["properties"]["forecastHourly"]
            hourly = client.get(hourly_url)
            hourly.raise_for_status()
            periods = hourly.json()["properties"]["periods"]
            wind = parse_period(periods[0]) if periods else None
    except Exception as exc:  # network, HTTP, or shape errors - wind is optional context
        logger.warning("incident_wind: lookup failed for %.3f,%.3f: %s", latitude, longitude, exc)
        return None  # don't cache failures, so a transient NWS blip retries next click
    with _lock:
        _cache[key] = (now, wind)
    return wind
