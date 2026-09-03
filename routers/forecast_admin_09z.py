"""Admin controls for the on-demand 9z secondary forecast run."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, status

from core.security import verify_token
from services.forecast_09z_jobs import get_09z_forecast_status, trigger_09z_forecast


router = APIRouter(prefix="/api/admin/forecast-09z", tags=["forecast-09z-admin"])


def _require_admin(token: Optional[str] = None) -> str:
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return email


@router.post("/run", status_code=status.HTTP_202_ACCEPTED)
async def run_09z_forecast(token: Optional[str] = None):
    email = _require_admin(token)
    try:
        return trigger_09z_forecast(email)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/status")
async def forecast_09z_status(token: Optional[str] = None):
    _require_admin(token)
    return get_09z_forecast_status()
