import asyncio
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.security import verify_token
from models.versioning import get_model_entry, validate_promotion_candidate
from services.beta_operations import build_beta_operations_status
from services.beta_verification import run_beta_verification
from services.model_shadow import diagnostics as fuel_moisture_shadow_diagnostics
from services.model_shadow import evaluate_shadow_evidence
from services.v4_shadow import diagnostics as v4_shadow_diagnostics
from services.v5_shadow import diagnostics as v5_shadow_diagnostics
from services.risk_fusion_shadow import diagnostics as risk_fusion_shadow_diagnostics
from services.risk_fusion_glm_shadow import diagnostics as risk_fusion_glm_shadow_diagnostics

router = APIRouter(prefix="/api/admin/models", tags=["model-admin"])

# Model types tracked in the shared stable/beta/history registry
# (models/versioning.py). The guarded v4/v5/risk-fusion shadow bundles are
# separate, non-registry evidence paths - see the top-level README's
# "V4 guarded shadow boundary" section - and are reported alongside instead.
REGISTRY_MODEL_TYPES = ["fuel_moisture", "fire_danger", "fuel_moisture_spatial", "fire_behavior_static"]

MAX_HISTORY_ENTRIES = 5


def _require_admin(token: Optional[str] = None) -> str:
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return email


def _registry_summary(model_type: str) -> dict:
    entry = get_model_entry(model_type)
    history = entry.get("history") or []
    beta = entry.get("beta")
    blockers = []
    if beta:
        try:
            blockers = validate_promotion_candidate(model_type, beta)
        except Exception as error:
            blockers = [f"promotion validation failed: {error}"]
    return {
        "model_type": model_type,
        "stable": entry.get("stable"),
        "beta": beta,
        "promotion": {
            "ready": bool(beta) and not blockers,
            "blockers": blockers,
        },
        "history": history[-MAX_HISTORY_ENTRIES:],
    }


class BetaVerificationRequest(BaseModel):
    date: Optional[str] = None


@router.get("/status")
async def get_model_status(token: Optional[str] = None):
    _require_admin(token)
    shadows = {
        "fuel_moisture": fuel_moisture_shadow_diagnostics(),
        "v4": v4_shadow_diagnostics(),
        "v5": v5_shadow_diagnostics(),
        "risk_fusion": risk_fusion_shadow_diagnostics(),
        "risk_fusion_glm": risk_fusion_glm_shadow_diagnostics(),
    }
    return {
        "registry": [_registry_summary(model_type) for model_type in REGISTRY_MODEL_TYPES],
        "fuel_moisture_shadow": {
            "diagnostics": shadows["fuel_moisture"],
            "promotion_gate": evaluate_shadow_evidence(),
        },
        "guarded_shadows": {
            "v4": shadows["v4"],
            "v5": shadows["v5"],
            "risk_fusion": shadows["risk_fusion"],
            "risk_fusion_glm": shadows["risk_fusion_glm"],
        },
        "operations": build_beta_operations_status(shadows=shadows),
    }


@router.post("/verify-beta")
async def verify_beta_forecast(payload: BetaVerificationRequest, token: Optional[str] = None):
    _require_admin(token)
    try:
        return await asyncio.to_thread(run_beta_verification, payload.date)
    except RuntimeError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Beta verification failed: {error}") from error
