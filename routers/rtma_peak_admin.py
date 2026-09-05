"""Admin controls and status for the additive RTMA peak product."""
from __future__ import annotations

import json
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.security import verify_token
from services.rtma_peak import generate_rtma_peak

router = APIRouter(prefix="/api/admin/rtma-peak", tags=["rtma-peak-admin"])


class GenerateRequest(BaseModel):
    date: str


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
