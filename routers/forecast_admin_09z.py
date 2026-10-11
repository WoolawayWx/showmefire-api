"""Admin controls for the on-demand 9z secondary forecast run."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from core.security import require_admin, verify_confirm_token
from services.forecast_09z_jobs import get_09z_forecast_status, trigger_09z_forecast


router = APIRouter(prefix="/api/admin/forecast-09z", tags=["forecast-09z-admin"])

RUN_ACTION = "run_forecast_09z"


class RunConfirmation(BaseModel):
    confirm_token: Optional[str] = None


@router.post("/run", status_code=status.HTTP_202_ACCEPTED)
async def run_09z_forecast(payload: RunConfirmation = RunConfirmation(), email: str = Depends(require_admin)):
    if not verify_confirm_token(payload.confirm_token, RUN_ACTION):
        raise HTTPException(status_code=401, detail="Password confirmation required or expired")
    try:
        return trigger_09z_forecast(email)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/status")
async def forecast_09z_status(_admin: str = Depends(require_admin)):
    return get_09z_forecast_status()
