"""Turn what a person typed (an address, a place, or coordinates) into Missouri points.

Order: coordinates are read directly; otherwise the US Census geocoder (street addresses, free, no key) and, if that finds
nothing, OpenStreetMap Nominatim (towns, zip codes, landmarks), always limited to Missouri. Results are cached for a day so the same
address never goes to a third party twice, and nothing typed is logged or stored. Network calls are injectable for tests.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from typing import Callable, Optional

import requests

from core.fire_events import MO_LAT_MAX, MO_LAT_MIN, MO_LON_MAX, MO_LON_MIN

logger = logging.getLogger(__name__)

USER_AGENT = "ShowMeFire/1.0 burn-check (https://showmefire.org)"
CENSUS_URL = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
TIMEOUT_SECONDS = 4.0
CACHE_SECONDS = 24 * 3600
CACHE_MAX = 2000
MAX_QUERY_LENGTH = 200

_NUMBER = r"[-+]?\d{1,3}(?:\.\d+)?"
_PAIR = re.compile(rf"^\s*({_NUMBER})\s*°?\s*([NSns])?\s*[,;\s]\s*({_NUMBER})\s*°?\s*([EWew])?\s*$")


class GeocodeUnavailable(Exception):
    """Every geocoder that was tried failed (as opposed to finding nothing)."""


def in_missouri(lat: float, lon: float) -> bool:
    return MO_LAT_MIN <= lat <= MO_LAT_MAX and MO_LON_MIN <= lon <= MO_LON_MAX


def parse_coordinates(text: str) -> Optional[tuple[float, float]]:
    """'38.5767, -92.1735', '38.5767 N 92.1735 W' and '38.5767 92.1735' -> (lat, lon), or None if it is not a coordinate pair.

    A positive longitude is taken as west when it is a Missouri longitude: nobody in Missouri means 92 degrees east."""
    match = _PAIR.match(text or "")
    if not match:
        return None
    lat, lat_hemisphere, lon, lon_hemisphere = float(match[1]), match[2], float(match[3]), match[4]
    if lat_hemisphere and lat_hemisphere.upper() == "S":
        lat = -abs(lat)
    if lon_hemisphere and lon_hemisphere.upper() == "W":
        lon = -abs(lon)
    elif lon > 0 and not lon_hemisphere and 88 <= lon <= 96:
        lon = -lon
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon


def _http_json(url: str, params: dict) -> dict | list:
    response = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def _census(query: str, get: Callable) -> list[dict]:
    data = get(CENSUS_URL, {"address": query, "benchmark": "Public_AR_Current", "format": "json"})
    out = []
    for match in (data.get("result") or {}).get("addressMatches") or []:
        try:
            out.append({"label": str(match["matchedAddress"]).title() if str(match["matchedAddress"]).isupper() else str(match["matchedAddress"]),
                        "lat": float(match["coordinates"]["y"]), "lon": float(match["coordinates"]["x"]), "source": "census"})
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _nominatim(query: str, get: Callable) -> list[dict]:
    data = get(NOMINATIM_URL, {"q": query, "format": "json", "limit": 5, "countrycodes": "us",
                               "viewbox": f"{MO_LON_MIN},{MO_LAT_MAX},{MO_LON_MAX},{MO_LAT_MIN}", "bounded": 1})
    out = []
    for item in data or []:
        try:
            out.append({"label": str(item.get("display_name") or query)[:200], "lat": float(item["lat"]), "lon": float(item["lon"]),
                        "source": "openstreetmap"})
        except (KeyError, TypeError, ValueError):
            continue
    return out


_cache: dict[str, tuple[float, list[dict]]] = {}
_lock = threading.Lock()


def clear_cache() -> None:
    with _lock:
        _cache.clear()


def geocode(query: str, *, get: Optional[Callable] = None, now: Callable[[], float] = time.time) -> list[dict]:
    """Up to 5 Missouri candidates [{label, lat, lon, source}], best first. Raises GeocodeUnavailable if every service failed."""
    query = " ".join((query or "").split())[:MAX_QUERY_LENGTH]
    if not query:
        return []
    if _PAIR.match(query):                   # two numbers: coordinates, never an address to send to a geocoder
        coordinates = parse_coordinates(query)
        if not coordinates or not in_missouri(*coordinates):
            return []
        lat, lon = coordinates
        return [{"label": f"{lat:.4f}, {lon:.4f}", "lat": lat, "lon": lon, "source": "coordinates"}]
    key = query.lower()
    with _lock:
        hit = _cache.get(key)
        if hit and now() - hit[0] < CACHE_SECONDS:
            return [dict(item) for item in hit[1]]
    get = get or _http_json
    failures = 0
    found: list[dict] = []
    for service in (_census, _nominatim):
        try:
            found = [c for c in service(query, get) if in_missouri(c["lat"], c["lon"])]
        except Exception:  # noqa: BLE001 - a geocoder being down must not break the tool; the next one is tried
            logger.info("burn-check geocoder %s unavailable", service.__name__)
            failures += 1
            continue
        if found:
            break
    if not found and failures == 2:
        raise GeocodeUnavailable()
    found = [{**c, "lat": round(c["lat"], 4), "lon": round(c["lon"], 4)} for c in found[:5]]
    if found or failures == 0:             # remember "nothing found" too, but not a failure
        with _lock:
            if len(_cache) >= CACHE_MAX:
                _cache.pop(next(iter(_cache)))
            _cache[key] = (now(), found)
    return [dict(item) for item in found]
