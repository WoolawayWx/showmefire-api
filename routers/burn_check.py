"""Public burn check: give a Missouri location (coordinates or an address) and get "Do not burn", "Use extreme caution" or
"Conditions look okay" with the reasons. A screening aid only; every response carries the disclaimer (see
services/burn_check/disclaimer.py). Public and read-only, rate limited; nothing about the request is stored.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from core.rate_limit import RateLimiter
from services import burn_check
from services.burn_check import disclaimer, geocode

logger = logging.getLogger(__name__)

check_limiter = RateLimiter("burn_check", 20)
geocode_limiter = RateLimiter("burn_check_geocode", 12)

router = APIRouter(prefix="/v1/burn-check", tags=["burn-check"])

OUTSIDE = "The burn check covers Missouri only. That location is outside the state."


def _terms(response: Response) -> None:
    """Point every caller at the terms of use (agencies and developers accept them by using the API or the embed)."""
    response.headers["Link"] = f'<{disclaimer.TERMS_URL}>; rel="terms-of-service"'


@router.get("", dependencies=[Depends(check_limiter)])
def burn_check_for_location(
    response: Response,
    lat: Optional[float] = Query(None, ge=-90, le=90, description="WGS84 latitude (use with lon)"),
    lon: Optional[float] = Query(None, ge=-180, le=180, description="WGS84 longitude (use with lat)"),
    address: Optional[str] = Query(None, max_length=200, description="A street address, town or zip code in Missouri (or 'lat, lon')"),
):
    """Burn screening for one point. Pass `lat` and `lon`, or `address`. Never cached: it describes right now."""
    response.headers["Cache-Control"] = "no-store"
    _terms(response)
    candidates: list[dict] = []
    matched: Optional[str] = None
    if (lat is None) != (lon is None):
        raise HTTPException(status_code=422, detail="Give both lat and lon.")
    if lat is None:
        if not (address or "").strip():
            raise HTTPException(status_code=422, detail="Give lat and lon, or an address.")
        try:
            candidates = geocode.geocode(address)
        except geocode.GeocodeUnavailable:
            raise HTTPException(status_code=503, detail="Address lookup is unavailable right now. Try your location or coordinates.")
        if not candidates:
            raise HTTPException(status_code=404, detail="We could not find that address in Missouri. Try a street address with a town, or coordinates.")
        lat, lon, matched = candidates[0]["lat"], candidates[0]["lon"], candidates[0]["label"]
    try:
        result = burn_check.assess(lat, lon, matched_address=matched)
    except burn_check.OutsideMissouri:
        raise HTTPException(status_code=422, detail=OUTSIDE)
    result["alternatives"] = candidates[1:]
    return result


@router.get("/geocode", dependencies=[Depends(geocode_limiter)])
def geocode_address(response: Response, q: str = Query(..., min_length=3, max_length=200)):
    """Missouri candidates for an address, place or 'lat, lon', so a page can let the person pick one."""
    response.headers["Cache-Control"] = "public, max-age=300"
    _terms(response)
    try:
        candidates = geocode.geocode(q)
    except geocode.GeocodeUnavailable:
        raise HTTPException(status_code=503, detail="Address lookup is unavailable right now.")
    return {"success": True, "candidates": candidates}


@router.get("/disclaimer", dependencies=[Depends(check_limiter)])
def get_disclaimer(response: Response):
    response.headers["Cache-Control"] = "public, max-age=3600"
    _terms(response)
    return {"success": True, **disclaimer.payload()}
