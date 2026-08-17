from typing import Optional

from fastapi import APIRouter, HTTPException

from core.security import verify_token
from models.versioning import get_model_entry
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
REGISTRY_MODEL_TYPES = ["fuel_moisture", "fire_danger", "fuel_moisture_spatial"]

MAX_HISTORY_ENTRIES = 5


def _require_admin(token: Optional[str] = None) -> str:
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return email


def _registry_summary(model_type: str) -> dict:
    entry = get_model_entry(model_type)
    history = entry.get("history") or []
    return {
        "model_type": model_type,
        "stable": entry.get("stable"),
        "beta": entry.get("beta"),
        "history": history[-MAX_HISTORY_ENTRIES:],
    }


@router.get("/status")
async def get_model_status(token: Optional[str] = None):
    _require_admin(token)
    return {
        "registry": [_registry_summary(model_type) for model_type in REGISTRY_MODEL_TYPES],
        "fuel_moisture_shadow": {
            "diagnostics": fuel_moisture_shadow_diagnostics(),
            "promotion_gate": evaluate_shadow_evidence(),
        },
        "guarded_shadows": {
            "v4": v4_shadow_diagnostics(),
            "v5": v5_shadow_diagnostics(),
            "risk_fusion": risk_fusion_shadow_diagnostics(),
            "risk_fusion_glm": risk_fusion_glm_shadow_diagnostics(),
        },
    }
