"""Public unofficial spot-style forecast (see services/spot_forecast). Read-only, rate limited, nothing about the request is
stored. The response always leads with the official-spot request guidance and the disclaimer."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response

from core.rate_limit import RateLimiter
from services import burn_check, spot_forecast
from services.burn_check import geocode
from services.spot_forecast import disclaimer

forecast_limiter = RateLimiter("spot_forecast", 15)

router = APIRouter(prefix="/v1/spot-forecast", tags=["spot-forecast"])

OUTSIDE = "The spot forecast covers Missouri only. That location is outside the state."


def _terms(response: Response) -> None:
    response.headers["Link"] = f'<{disclaimer.TERMS_URL}>; rel="terms-of-service"'


@router.get("", dependencies=[Depends(forecast_limiter)])
def spot_forecast_for_location(
    response: Response,
    lat: Optional[float] = Query(None, ge=-90, le=90, description="WGS84 latitude (use with lon)"),
    lon: Optional[float] = Query(None, ge=-180, le=180, description="WGS84 longitude (use with lat)"),
    address: Optional[str] = Query(None, max_length=200, description="A street address, town or zip code in Missouri (or 'lat, lon')"),
    window: str = Query("today", pattern="^(today|tomorrow)$"),
):
    """Unofficial spot-style forecast for one point. Pass `lat` and `lon`, or `address`. Never cached."""
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
            raise HTTPException(status_code=503, detail="Address lookup is unavailable right now. Try coordinates.")
        if not candidates:
            raise HTTPException(status_code=404, detail="We could not find that address in Missouri. Try a street address with a town, or coordinates.")
        lat, lon, matched = candidates[0]["lat"], candidates[0]["lon"], candidates[0]["label"]
    try:
        result = spot_forecast.assess(lat, lon, window=window, matched_address=matched)
    except burn_check.OutsideMissouri:
        raise HTTPException(status_code=422, detail=OUTSIDE)
    result["alternatives"] = candidates[1:]
    return result


@router.get("/disclaimer", dependencies=[Depends(forecast_limiter)])
def get_disclaimer(response: Response):
    response.headers["Cache-Control"] = "public, max-age=3600"
    _terms(response)
    return {"success": True, **disclaimer.payload()}
