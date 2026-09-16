import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from core.security import verify_token
from models import shadow_bundles
from models.versioning import get_model_entry, promote, rollback, validate_promotion_candidate
from services.beta_operations import build_beta_operations_status
from services.beta_verification import run_beta_verification
from services.model_shadow import diagnostics as fuel_moisture_shadow_diagnostics
from services.model_shadow import evaluate_shadow_evidence
from services.v4_shadow import diagnostics as v4_shadow_diagnostics
from services.v5_shadow import diagnostics as v5_shadow_diagnostics
from services.risk_fusion_shadow import diagnostics as risk_fusion_shadow_diagnostics
from services.risk_fusion_glm_shadow import diagnostics as risk_fusion_glm_shadow_diagnostics
from services.fire_weather_ml_shadow import diagnostics as fire_weather_ml_shadow_diagnostics
from services.fire_weather_index_shadow import diagnostics as fire_weather_index_shadow_diagnostics

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/models", tags=["model-admin"])

# Model types tracked in the shared stable/beta/history registry
# (models/versioning.py) - promote()/rollback() apply to these.
REGISTRY_MODEL_TYPES = ["fuel_moisture", "fire_danger", "fuel_moisture_spatial", "fire_behavior_static"]

# Guarded shadow bundle families - not in the registry above, each scored
# from a single fixed directory pointed at by an SMF_<X>_BUNDLE env var (see
# models/shadow_bundles.py for why a real version history needed building
# for these). risk_fusion (Phase A) has no bundle - it's the live rule's own
# Monte Carlo, reported alongside for monitoring but not upload/activate-able.
GUARDED_SHADOW_TYPES = shadow_bundles.SHADOW_FAMILIES  # v4, v5, risk_fusion_glm, fire_weather_ml, fire_weather_index

MAX_HISTORY_ENTRIES = 5
MAX_BUNDLE_UPLOAD_BYTES = 200 * 1024 * 1024  # generous for a multi-file model bundle archive, still bounded


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
        # Previously missing here despite already being wired into
        # routers/spatial_model.py's own diagnostics endpoints - these two
        # families were invisible on this admin page for no reason other
        # than never having been added to this dict.
        "fire_weather_ml": fire_weather_ml_shadow_diagnostics(),
        "fire_weather_index": fire_weather_index_shadow_diagnostics(),
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
            "fire_weather_ml": shadows["fire_weather_ml"],
            "fire_weather_index": shadows["fire_weather_index"],
        },
        "operations": build_beta_operations_status(shadows=shadows),
    }


@router.get("/families")
async def list_model_families(token: Optional[str] = None):
    """Every model family this page can show/manage, with which kind it is
    (registry vs guarded-shadow) - the single source the frontend's family
    dropdown is built from, so a new family only needs adding here."""
    _require_admin(token)
    return {
        "registry": REGISTRY_MODEL_TYPES,
        "guarded_shadow": GUARDED_SHADOW_TYPES,
    }


@router.get("/{family}/versions")
async def list_family_versions(family: str, token: Optional[str] = None):
    _require_admin(token)
    if family in REGISTRY_MODEL_TYPES:
        return _registry_summary(family)
    if family in GUARDED_SHADOW_TYPES:
        return {
            "model_type": family,
            "versions": shadow_bundles.list_versions(family),
            "active": shadow_bundles.get_active(family),
        }
    raise HTTPException(status_code=404, detail=f"Unknown model family: {family}")


@router.post("/{family}/upload", status_code=201)
async def upload_shadow_bundle(family: str, token: Optional[str] = None, file: UploadFile = File(...)):
    email = _require_admin(token)
    if family not in GUARDED_SHADOW_TYPES:
        raise HTTPException(status_code=400, detail=f"{family} is not an uploadable guarded-shadow family")
    content = await file.read(MAX_BUNDLE_UPLOAD_BYTES + 1)
    if len(content) > MAX_BUNDLE_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Bundle archive too large")
    suffix = Path(file.filename or "").suffix or ".zip"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temporary:
        temporary.write(content)
        archive_path = Path(temporary.name)
    try:
        installed = shadow_bundles.install_bundle(family, archive_path, uploaded_by=email)
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=f"Bundle validation failed: {exc}")
    except Exception as exc:
        logger.error("Failed to install %s bundle upload: %s", family, exc)
        raise HTTPException(status_code=500, detail="Failed to install bundle")
    finally:
        archive_path.unlink(missing_ok=True)
    return {"success": True, "installed": installed}


class ActivateRequest(BaseModel):
    version: str


@router.post("/{family}/activate")
async def activate_family_version(family: str, payload: ActivateRequest, token: Optional[str] = None):
    """Guarded-shadow families: flips the live active bundle pointer with no
    restart required. Registry families: this IS promotion (stable takes on
    the named beta candidate) - wraps the existing, previously CLI-only
    models.versioning.promote()."""
    _require_admin(token)
    if family in GUARDED_SHADOW_TYPES:
        try:
            return {"success": True, "active": shadow_bundles.set_active(family, payload.version)}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Activation failed: {exc}")
    if family in REGISTRY_MODEL_TYPES:
        try:
            promote(family, payload.version)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error("Failed to promote %s: %s", family, exc)
            raise HTTPException(status_code=500, detail="Promotion failed")
        return {"success": True, **_registry_summary(family)}
    raise HTTPException(status_code=404, detail=f"Unknown model family: {family}")


class RollbackRequest(BaseModel):
    version: Optional[str] = None


@router.post("/{family}/rollback")
async def rollback_family(family: str, payload: RollbackRequest, token: Optional[str] = None):
    """Registry families only - wraps the existing, previously CLI-only
    models.versioning.rollback(). A guarded-shadow family "rolls back" by
    activating an older already-installed version through /activate
    instead - there is no separate stable/beta distinction to roll back
    from for those families."""
    _require_admin(token)
    if family not in REGISTRY_MODEL_TYPES:
        raise HTTPException(status_code=400,
                            detail=f"{family} has no stable/beta distinction - use /activate with an older version instead")
    try:
        rollback(family, payload.version)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error("Failed to roll back %s: %s", family, exc)
        raise HTTPException(status_code=500, detail="Rollback failed")
    return {"success": True, **_registry_summary(family)}


@router.post("/verify-beta")
async def verify_beta_forecast(payload: BetaVerificationRequest, token: Optional[str] = None):
    _require_admin(token)
    try:
        return await asyncio.to_thread(run_beta_verification, payload.date)
    except RuntimeError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Beta verification failed: {error}") from error
