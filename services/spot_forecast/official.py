"""The push toward an official NWS spot forecast. The same block goes with every result: this page is a model estimate, and a real
burn or incident should have a forecaster's spot forecast.

NWS spot forecasts are for government agencies, not private citizens, so the block says that plainly and tells everyone else who
to ask instead of implying they can file one. Each office keeps its own request link on its fire weather page, so we link there.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Optional

import requests

from services.spot_forecast.narrative import RULES

logger = logging.getLogger(__name__)

POINTS_URL = "https://api.weather.gov/points/{lat:.4f},{lon:.4f}"
USER_AGENT = "ShowMeFire/1.0 spot-forecast (https://showmefire.org)"
TIMEOUT_SECONDS = 4.0
CACHE_SECONDS = 7 * 24 * 3600
CACHE_MAX = 2000
OFFICE_NAMES = {"EAX": "Kansas City/Pleasant Hill", "SGF": "Springfield", "LSX": "St. Louis", "PAH": "Paducah", "DVN": "Quad Cities"}

_cache: dict[tuple[float, float], tuple[float, Optional[str]]] = {}
_lock = threading.Lock()


def fire_page(code: str) -> str:
    return f"https://www.weather.gov/{code.lower()}/fire"


def _http_json(url: str) -> dict:
    response = requests.get(url, headers={"User-Agent": USER_AGENT, "Accept": "application/geo+json"}, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()


def office_for_point(lat: float, lon: float, get: Callable[[str], dict] = _http_json, now: Optional[float] = None) -> Optional[str]:
    """The NWS forecast office (county warning area) for the point, e.g. "LSX", or None if it cannot be found. Cached a week by
    ~1 km cell; a failed lookup is not cached."""
    key = (round(lat, 2), round(lon, 2))
    now = time.time() if now is None else now
    with _lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < CACHE_SECONDS:
            return hit[1]
    try:
        code = ((get(POINTS_URL.format(lat=lat, lon=lon)).get("properties") or {}).get("cwa") or "").upper() or None
    except Exception:  # noqa: BLE001 - the page then lists every Missouri office instead
        logger.warning("NWS office lookup failed", exc_info=True)
        return None
    with _lock:
        if len(_cache) >= CACHE_MAX:
            _cache.clear()
        _cache[key] = (now, code)
    return code


def block(office: Optional[str]) -> dict:
    """The official-spot section of the response."""
    offices = RULES["nws_offices"]
    known = office in offices
    return {
        "headline": "Planning a burn or working a fire? Request an official NWS spot forecast.",
        "not_a_substitute": "This page is an automated model estimate. An NWS spot forecast is written by a forecaster for your exact site, can be updated, and can be talked through by phone.",
        "who_can_request": "NWS spot forecasts are for government agencies (fire departments and districts, the Missouri Department of Conservation, the U.S. Forest Service, and other federal, state, tribal and local land and fire agencies). The NWS does not provide them to private individuals.",
        "if_you_are_not_an_agency": "Ask your fire department or fire district, or your land manager, to request one for you.",
        "incident_note": "For a wildfire, have the incident commander or dispatch request it. Call 911 first if life or property is at risk.",
        "have_ready": ["Exact site location (latitude/longitude) and elevation", "Burn or incident start date and time, and how long it will run",
                       "Fuel type and acres", "Observed on-site wind, temperature and humidity if you have them", "A phone number the forecaster can reach"],
        "office": None if not office else {"code": office, "name": OFFICE_NAMES.get(office), "request_page": fire_page(office) if known else None},
        "all_offices": [{"code": c, "name": OFFICE_NAMES.get(c), "request_page": fire_page(c)} for c in offices],
        "office_note": None if known else "We could not tell which NWS office covers this point. Missouri is covered by several offices; find yours below.",
    }
