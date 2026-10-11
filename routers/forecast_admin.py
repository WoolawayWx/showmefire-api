"""Admin controls for isolated Testbed forecast generation."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from core.security import require_admin, verify_confirm_token
from services.forecast_jobs import get_beta_forecast_status, trigger_beta_forecast


router = APIRouter(prefix="/api/admin/testbed/forecast", tags=["testbed-admin"])

RUN_ACTION = "run_forecast_testbed"


class RunConfirmation(BaseModel):
    confirm_token: Optional[str] = None


@router.post("/run", status_code=status.HTTP_202_ACCEPTED)
async def run_beta_forecast(payload: RunConfirmation = RunConfirmation(), email: str = Depends(require_admin)):
    if not verify_confirm_token(payload.confirm_token, RUN_ACTION):
        raise HTTPException(status_code=401, detail="Password confirmation required or expired")
    try:
        return trigger_beta_forecast(email)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/status")
async def beta_forecast_status(_admin: str = Depends(require_admin)):
    return get_beta_forecast_status()
