from fastapi import APIRouter

from services.spatial_fm import diagnostics
from services.model_shadow import diagnostics as shadow_diagnostics
from services.v4_shadow import diagnostics as v4_shadow_diagnostics
from services.v5_shadow import diagnostics as v5_shadow_diagnostics
from services.risk_fusion_shadow import diagnostics as risk_fusion_shadow_diagnostics
from core.fire_danger import missing_input_diagnostics

router = APIRouter(prefix="/api/model/spatial", tags=["model-diagnostics"])


@router.get("/diagnostics")
def spatial_diagnostics():
    return diagnostics()


@router.get("/shadow-diagnostics")
def model_shadow_diagnostics():
    return shadow_diagnostics()


@router.get("/v4-shadow-diagnostics")
def guarded_v4_shadow_diagnostics():
    return v4_shadow_diagnostics()


@router.get("/v5-shadow-diagnostics")
def summer_guarded_v5_shadow_diagnostics():
    return v5_shadow_diagnostics()


@router.get("/fire-danger-diagnostics")
def fire_danger_diagnostics():
    return {"missing_inputs": missing_input_diagnostics()}


@router.get("/risk-fusion-shadow-diagnostics")
def risk_fusion_shadow_diagnostics_endpoint():
    return risk_fusion_shadow_diagnostics()
