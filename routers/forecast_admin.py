"""Admin controls for isolated Testbed forecast generation."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, status

from core.security import verify_token
from services.forecast_jobs import get_beta_forecast_status, trigger_beta_forecast


router = APIRouter(prefix="/api/admin/testbed/forecast", tags=["testbed-admin"])


def _require_admin(token: Optional[str] = None) -> str:
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return email


@router.post("/run", status_code=status.HTTP_202_ACCEPTED)
async def run_beta_forecast(token: Optional[str] = None):
    email = _require_admin(token)
    try:
        return trigger_beta_forecast(email)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/status")
async def beta_forecast_status():
    return get_beta_forecast_status()

