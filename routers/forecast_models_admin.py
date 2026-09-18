"""Admin console for the forecast_v1 source-model registry.

Lets an admin add a model (whose adapter/Herbie fetch parameters already
exist in forecast_v1.registry.ADAPTER_ACQUISITION), move it between
disabled/shadow/active, set its acquisition schedule, and see its logged
performance - without touching contracts.BLEND_WEIGHTS or redeploying.
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from core.security import verify_token
from forecast_v1 import registry
from forecast_v1.adapters import ADAPTERS

router = APIRouter(prefix="/api/admin/forecast-models", tags=["forecast-models-admin"])


def _require_admin(token: Optional[str] = None) -> str:
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return email


class AddModelRequest(BaseModel):
    key: str = Field(min_length=1, max_length=40, pattern=r"^[a-z0-9_]+$")
    display_name: str = Field(min_length=1, max_length=120)
    adapter_key: str
    notes: str = ""


class StatusRequest(BaseModel):
    status: str
    weight: Optional[float] = Field(default=None, gt=0, le=1)


class ScheduleRequest(BaseModel):
    schedule_minutes: Optional[int] = Field(default=None, gt=0)


@router.get("")
def list_forecast_models(token: Optional[str] = None):
    _require_admin(token)
    return {"models": registry.list_models(), "available_adapters": sorted(ADAPTERS)}


@router.post("")
def add_forecast_model(body: AddModelRequest, token: Optional[str] = None):
    _require_admin(token)
    if body.adapter_key not in ADAPTERS:
        raise HTTPException(status_code=400, detail=f"unknown adapter_key: {body.adapter_key}")
    try:
        return registry.add_model(body.key, body.display_name, body.adapter_key, body.notes)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/{key}")
def get_forecast_model(key: str, token: Optional[str] = None):
    _require_admin(token)
    model = registry.get_model(key)
    if not model:
        raise HTTPException(status_code=404, detail="not found")
    return model


@router.post("/{key}/status")
def set_forecast_model_status(key: str, body: StatusRequest, token: Optional[str] = None):
    """Move a model between disabled/shadow/active. 'active' requires `weight`."""
    _require_admin(token)
    try:
        return registry.set_status(key, body.status, body.weight)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/{key}/schedule")
def set_forecast_model_schedule(key: str, body: ScheduleRequest, token: Optional[str] = None):
    _require_admin(token)
    if not registry.get_model(key):
        raise HTTPException(status_code=404, detail="not found")
    return registry.set_schedule(key, body.schedule_minutes)


@router.get("/{key}/performance")
def get_forecast_model_performance(key: str, limit_cycles: int = 20, token: Optional[str] = None):
    _require_admin(token)
    if not registry.get_model(key):
        raise HTTPException(status_code=404, detail="not found")
    return registry.performance_summary(key, limit_cycles)
