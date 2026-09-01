"""Admin controls and warm-up for the Testbed spread-rate product."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.security import verify_token
from services.spread_rate import (
    PNG_PATH,
    STATUS_PATH,
    TIF_PATH,
    generate_spread_rate,
    run_spread_rate_pipeline,
    spread_rate_status,
    warmup_spread_rate_inputs,
)
from core.scheduler import raws_station_data


router = APIRouter(prefix="/api/admin/testbed/spread-rate", tags=["spread-rate-admin"])


class WarmupRequest(BaseModel):
    days: int = Field(default=7, ge=1, le=14)


def _require_admin(token: Optional[str] = None) -> str:
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return email


@router.get("/status")
async def spread_rate_admin_status(token: Optional[str] = None):
    _require_admin(token)
    status = spread_rate_status()
    return {
        **status,
        "artifacts_available": {
            "geotiff": TIF_PATH.is_file(),
            "preview_png": PNG_PATH.is_file(),
            "status_json": STATUS_PATH.is_file(),
        },
    }


@router.post("/generate")
async def spread_rate_admin_generate(token: Optional[str] = None):
    email = _require_admin(token)
    try:
        result = await __import__("asyncio").to_thread(
            run_spread_rate_pipeline,
            raws_station_data if raws_station_data.get("stations") else None,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Spread-rate generation failed: {exc}") from exc
    return {"success": True, "requested_by": email, "result": result}


@router.post("/warmup")
async def spread_rate_admin_warmup(payload: WarmupRequest, token: Optional[str] = None):
    email = _require_admin(token)
    try:
        result = await __import__("asyncio").to_thread(warmup_spread_rate_inputs, payload.days)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"RTMA warm-up failed: {exc}") from exc
    return {"success": True, "requested_by": email, "result": result}
