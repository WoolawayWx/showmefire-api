"""Admin controls and status for the additive RTMA peak product."""
from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.security import verify_token
from services.rtma_peak import generate_rainfall_impact_map, generate_rtma_peak

router = APIRouter(prefix="/api/admin/rtma-peak", tags=["rtma-peak-admin"])


class GenerateRequest(BaseModel):
    date: str


class RainfallImpactGenerateRequest(BaseModel):
    date: Optional[str] = None
    days: int = 7


def _require_admin(token: Optional[str] = None):
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return email


@router.get("/status")
async def rtma_peak_status(date: str, token: Optional[str] = None):
    _require_admin(token)
    from services.rtma_peak import RTMA_PEAK_IMAGE_ARCHIVE_DIR, RTMA_PEAK_ARCHIVE_DIR
    metadata = {}
    try:
        payload = json.loads((RTMA_PEAK_ARCHIVE_DIR / f"{date}.json").read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            metadata = payload.get("fuel_moisture", {})
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        pass
    return {
        "date": date,
        "available": (
            (RTMA_PEAK_ARCHIVE_DIR / f"{date}.tif").exists()
            and (RTMA_PEAK_IMAGE_ARCHIVE_DIR / f"{date}.png").exists()
        ),
        "png": f"rtma_peak/archive/{date}.png",
        "tif": f"rtma_peak/archive/{date}.tif",
        "fuel_moisture": metadata or {"mode": "unknown"},
    }


@router.post("/generate")
async def generate_rtma_peak_admin(payload: GenerateRequest, token: Optional[str] = None):
    email = _require_admin(token)
    try:
        result = await __import__("asyncio").to_thread(generate_rtma_peak, payload.date)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RTMA peak generation failed: {exc}")
    return {"success": True, "requested_by": email, "result": result}


@router.get("/fuel-moisture/status")
async def fuel_moisture_status(date: str, token: Optional[str] = None):
    """Status of the calibrated (RTMA + RAWS) fuel-moisture map for one date.

    Generated automatically as part of /generate; there is no separate
    fuel-moisture generate endpoint.
    """
    _require_admin(token)
    from services.rtma_peak import RTMA_FUEL_MOISTURE_ARCHIVE_DIR, RTMA_FUEL_MOISTURE_IMAGE_ARCHIVE_DIR
    return {
        "date": date,
        "available": (
            (RTMA_FUEL_MOISTURE_ARCHIVE_DIR / f"{date}.tif").exists()
            and (RTMA_FUEL_MOISTURE_IMAGE_ARCHIVE_DIR / f"{date}.png").exists()
        ),
        "png": f"rtma_fuel_moisture/archive/{date}.png",
        "tif": f"rtma_fuel_moisture/archive/{date}.tif",
    }


@router.get("/rainfall-impact/status")
async def rainfall_impact_status(date: str, token: Optional[str] = None):
    """Status of the trailing-window rainfall-suppression map for one generation date."""
    _require_admin(token)
    from services.rtma_peak import RTMA_IMPACT_ARCHIVE_DIR, RTMA_IMPACT_IMAGE_ARCHIVE_DIR
    return {
        "date": date,
        "available": (
            (RTMA_IMPACT_ARCHIVE_DIR / f"{date}.tif").exists()
            and (RTMA_IMPACT_IMAGE_ARCHIVE_DIR / f"{date}.png").exists()
        ),
        "png": f"rtma_rainfall_impact/archive/{date}.png",
        "tif": f"rtma_rainfall_impact/archive/{date}.tif",
    }


@router.post("/rainfall-impact/generate")
async def generate_rainfall_impact_admin(payload: RainfallImpactGenerateRequest, token: Optional[str] = None):
    """Rebuild the trailing-window rainfall-impact map without rerunning the full day's RTMA peak.

    Useful for backfilling with a different window size, or after a fix to
    this map's own rendering - the underlying daily rtma_rainfall_reduction
    archives it reads are only ever (re)written by /generate.
    """
    email = _require_admin(token)
    try:
        result = await __import__("asyncio").to_thread(
            generate_rainfall_impact_map, payload.date, payload.days
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Rainfall impact map generation failed: {exc}")
    return {"success": True, "requested_by": email, "result": result}
