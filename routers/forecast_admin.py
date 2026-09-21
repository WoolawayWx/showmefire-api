"""Admin controls for isolated Testbed forecast generation."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from core.security import verify_confirm_token, verify_token
from services.forecast_jobs import get_beta_forecast_status, trigger_beta_forecast


router = APIRouter(prefix="/api/admin/testbed/forecast", tags=["testbed-admin"])

RUN_ACTION = "run_forecast_testbed"


def _require_admin(token: Optional[str] = None) -> str:
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return email


class RunConfirmation(BaseModel):
    confirm_token: Optional[str] = None


@router.post("/run", status_code=status.HTTP_202_ACCEPTED)
async def run_beta_forecast(payload: RunConfirmation = RunConfirmation(), token: Optional[str] = None):
    email = _require_admin(token)
    if not verify_confirm_token(payload.confirm_token, RUN_ACTION):
        raise HTTPException(status_code=401, detail="Password confirmation required or expired")
    try:
        return trigger_beta_forecast(email)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.get("/status")
async def beta_forecast_status(token: Optional[str] = None):
    _require_admin(token)
    return get_beta_forecast_status()
