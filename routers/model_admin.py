import asyncio
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel

from core.database import get_shadow_model_setting, list_shadow_model_settings, set_shadow_model_setting
from core.security import verify_confirm_token, verify_token
from models import shadow_bundles
from models.versioning import get_model_entry, promote, rollback, validate_promotion_candidate
from pipelines import import_model
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
# fire_risk_fusion is import-only in practice: validate_promotion_candidate()
# hard-blocks promotion unless metadata["advisory_only"] is True (a
# deliberate v1 boundary, not a satisfiable gate) - listing it here makes
# attempting a promotion possible (it used to crash on any bundle lacking a
# model/checkpoint/static_bundle asset role) rather than guaranteed to succeed.
REGISTRY_MODEL_TYPES = ["fuel_moisture", "fire_danger", "fuel_moisture_spatial", "fire_behavior_static",
                        "fire_risk_fusion", "ensemble_fire_danger"]

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


def _require_confirmation(confirm_token: Optional[str], action: str) -> None:
    if not verify_confirm_token(confirm_token, action):
        raise HTTPException(status_code=401, detail="Password confirmation required or expired")


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
    guarded_shadows = {
        "v4": shadows["v4"],
        "v5": shadows["v5"],
        "risk_fusion": shadows["risk_fusion"],
        "risk_fusion_glm": shadows["risk_fusion_glm"],
        "fire_weather_ml": shadows["fire_weather_ml"],
        "fire_weather_index": shadows["fire_weather_index"],
    }
    # Merge the DB-backed live enable/disable setting into each guarded
    # family so the dashboard doesn't need a second round-trip.
    # `requested_enabled` is deliberately a distinct field from `enabled` -
    # `enabled` already means "actually running right now, accounting for
    # configured/auto_disabled/failures" per each module's own diagnostics(),
    # while `requested_enabled` is specifically "what the admin asked for."
    # risk_fusion (Phase A) has no settings row (it's not in
    # shadow_bundles.SHADOW_FAMILIES - no trained artifact at all).
    db_settings = list_shadow_model_settings()
    for family in shadow_bundles.SHADOW_FAMILIES:
        setting = db_settings.get(family)
        if setting:
            guarded_shadows[family]["requested_enabled"] = setting["enabled"]
            guarded_shadows[family]["settings_updated_at"] = setting["updated_at"]
            guarded_shadows[family]["settings_updated_by"] = setting["updated_by"]
    return {
        "registry": [_registry_summary(model_type) for model_type in REGISTRY_MODEL_TYPES],
        "fuel_moisture_shadow": {
            "diagnostics": shadows["fuel_moisture"],
            "promotion_gate": evaluate_shadow_evidence(),
        },
        "guarded_shadows": guarded_shadows,
        "operations": build_beta_operations_status(shadows=shadows),
    }


@router.get("/families")
async def list_model_families(token: Optional[str] = None):
    """Every model family this page can show/manage, with which kind it is
    (registry vs guarded-shadow) - the single source the frontend's family
    dropdown is built from, so a new family only needs adding here.

    `importable` is deliberately its own list, not just `registry`: two of
    the guarded-shadow families (fire_weather_ml/fire_weather_index) DO have
    a training-side GitHub-release path and are accepted by
    pipelines/import_model.py, so the website's Import panel should show for
    them too - see import_model.IMPORTABLE_MODEL_TYPES, the single source
    of truth this mirrors rather than re-deriving from `registry` alone."""
    _require_admin(token)
    return {
        "registry": REGISTRY_MODEL_TYPES,
        "guarded_shadow": GUARDED_SHADOW_TYPES,
        "importable": import_model.IMPORTABLE_MODEL_TYPES,
    }


class ShadowSettingsRequest(BaseModel):
    enabled: bool


@router.get("/{family}/settings")
async def get_family_settings(family: str, token: Optional[str] = None):
    """Live, no-restart enable/disable state for a shadow/advisory family -
    see core.database.shadow_model_settings. Registry families
    (fuel_moisture, fuel_moisture_spatial, etc.) and risk_fusion (Phase A,
    no trained artifact) have no such setting - they're always on."""
    _require_admin(token)
    if family not in shadow_bundles.SHADOW_FAMILIES:
        raise HTTPException(status_code=404, detail=f"{family} has no live enable/disable setting")
    setting = get_shadow_model_setting(family)
    if setting is None:
        raise HTTPException(status_code=404, detail=f"{family} has no settings row (database not initialized?)")
    return setting


@router.post("/{family}/settings")
async def set_family_settings(family: str, payload: ShadowSettingsRequest, token: Optional[str] = None):
    email = _require_admin(token)
    if family not in shadow_bundles.SHADOW_FAMILIES:
        raise HTTPException(status_code=404, detail=f"{family} has no live enable/disable setting")
    return {"success": True, "setting": set_shadow_model_setting(family, payload.enabled, updated_by=email)}


@router.get("/schedule")
async def get_model_schedule(request: Request, token: Optional[str] = None):
    """Read-only introspection of the live APScheduler instance, filtered to
    the curated core.scheduler.MODEL_RELEVANT_JOBS allowlist - no editing
    capability is exposed here by design (see the plan this shipped under)."""
    _require_admin(token)
    from core.scheduler import MODEL_RELEVANT_JOBS

    scheduler = getattr(request.app.state, "scheduler", None)
    if scheduler is None:
        return {"scheduler_running": False, "jobs": []}

    jobs = []
    for job in scheduler.get_jobs():
        meta = MODEL_RELEVANT_JOBS.get(job.id)
        if meta is None:
            continue
        jobs.append({
            "id": job.id,
            "category": meta["category"],
            "description": meta["description"],
            "cadence": str(job.trigger),
            "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
        })
    jobs.sort(key=lambda row: (row["category"], row["id"]))
    return {"scheduler_running": True, "jobs": jobs}


def _guarded_shadow_versions(family: str) -> dict:
    """shadow_bundles.py's own version list/active pointer, extended with
    any registry-only entries for migrated families - i.e. candidates that
    arrived via GitHub-release import (pipelines/import_model.py), which
    never calls shadow_bundles.install_bundle() at all, so they'd otherwise
    be invisible here even though `import_model_release()` above reported
    success. Registry entries dual-written by a zip upload already have a
    matching shadow_bundles entry and are skipped to avoid double-listing."""
    versions = shadow_bundles.list_versions(family)
    active = shadow_bundles.get_active(family)

    if family in shadow_bundles._REGISTRY_MIGRATED_FAMILIES:
        known = {v.get("version") for v in versions}
        entry = get_model_entry(family)
        records = [entry.get("beta"), entry.get("stable"), *entry.get("history", [])]
        seen = set()
        for record in records:
            if not record:
                continue
            display_version = (record.get("metadata") or {}).get("shadow_bundle_version") or record["version"]
            if display_version in known or display_version in seen:
                continue
            seen.add(display_version)
            source_release_tag = (record.get("performance") or {}).get("source_release_tag")
            versions.append({
                "version": display_version,
                "installed_at": record.get("trained_at") or record.get("promoted_at") or record.get("recorded_at"),
                "uploaded_by": f"github-import ({source_release_tag})" if source_release_tag else "github-import",
                "registry_version": record["version"],
                "origin": "registry-import",
            })
        stable = entry.get("stable")
        if stable:
            active = {
                "version": (stable.get("metadata") or {}).get("shadow_bundle_version") or stable["version"],
                "path": active.get("path") if active else None,
                "activated_at": stable.get("promoted_at"),
            }

    return {"model_type": family, "versions": versions, "active": active}


@router.get("/{family}/versions")
async def list_family_versions(family: str, token: Optional[str] = None):
    _require_admin(token)
    if family in REGISTRY_MODEL_TYPES:
        return _registry_summary(family)
    if family in GUARDED_SHADOW_TYPES:
        return _guarded_shadow_versions(family)
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


# Repo(s) import is allowed to pull from - never trust an arbitrary
# client-supplied repo string, since this ultimately shells out to `gh`.
# Defaults to SMF_GITHUB_REPO alone if unset, preserving today's
# single-repo CLI behavior; set SMF_ALLOWED_IMPORT_REPOS (comma-separated)
# to widen it deliberately.
def _allowed_import_repos() -> set:
    raw = os.getenv("SMF_ALLOWED_IMPORT_REPOS") or os.getenv("SMF_GITHUB_REPO", "")
    return {entry.strip() for entry in raw.split(",") if entry.strip()}


IMPORT_TIMEOUT_SECONDS = 300


class ImportRequest(BaseModel):
    tag: str
    repo: Optional[str] = None
    bump: str = "patch"


@router.post("/{family}/import", status_code=201)
async def import_model_release(family: str, payload: ImportRequest, token: Optional[str] = None):
    """Server-side equivalent of pipelines/import_model.py's CLI: pulls a
    GitHub release, verifies its assets, and registers it as a beta
    candidate - the one step in the model lifecycle that was previously
    100% CLI/SSH-only despite promote/rollback/upload already having a UI."""
    _require_admin(token)
    if family not in import_model.IMPORTABLE_MODEL_TYPES:
        raise HTTPException(status_code=400, detail=f"{family} is not importable into the registry")
    if not payload.tag.strip():
        raise HTTPException(status_code=400, detail="tag is required")
    if payload.bump not in ("major", "minor", "patch"):
        raise HTTPException(status_code=400, detail="bump must be major, minor, or patch")

    repo = payload.repo or os.getenv("SMF_GITHUB_REPO")
    if not repo:
        raise HTTPException(status_code=400, detail="No source repo configured (set SMF_GITHUB_REPO)")
    if repo not in _allowed_import_repos():
        raise HTTPException(status_code=403, detail=f"Repo {repo!r} is not in the allowlist")

    try:
        version = await asyncio.to_thread(
            import_model.import_release, family, payload.tag, repo,
            bump=payload.bump, timeout=IMPORT_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Import timed out contacting GitHub")
    except import_model.ImportValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except subprocess.CalledProcessError as exc:
        raise HTTPException(status_code=502, detail=f"gh release download failed: {exc}")
    except Exception as exc:
        logger.error("Failed to import %s %s from %s: %s", family, payload.tag, repo, exc)
        raise HTTPException(status_code=500, detail="Import failed")
    return {"success": True, "version": version, **_registry_summary(family)}


class ActivateRequest(BaseModel):
    version: str
    confirm_token: Optional[str] = None


def _registry_action_for_shadow_bundle(family: str, version: str):
    """Maps a version identifier to what activating it means in the unified
    registry (models/versioning.py), for families in
    shadow_bundles._REGISTRY_MIGRATED_FAMILIES: promoting the current beta,
    a no-op (already stable), rolling back to an older stable, or None if
    this version was never registered at all.

    `version` can be in EITHER of two identifier spaces, since a migrated
    family's candidates can arrive two ways:
      - zip upload (shadow_bundles.install_bundle's dual-write) - `version`
        is the shadow_bundles version directory name, stored back onto the
        registry record as metadata["shadow_bundle_version"].
      - GitHub-release import (pipelines/import_model.py) - never touches
        shadow_bundles at all, so there is no shadow_bundle_version; the
        only identifier that exists is the registry's own assigned version
        string (e.g. "0.0.1-beta.1").
    Matching only the first (as an earlier version of this function did)
    made an imported candidate's Activate button 404 - it had no
    shadow_bundle_version to match against. Checking both, in order, covers
    each origin.

    Beta and stable version strings live in different spaces (beta carries
    a "-beta.N" suffix, promote() strips it for stable) - matching on the
    registry's own version field would compare the wrong string depending
    on which channel happened to match first, so this checks each channel
    explicitly and returns which action applies, rather than a bare record
    for the caller to guess at.

    Returns (action, registry_version) where action is "promote", "current",
    or "rollback"; or None.
    """
    entry = get_model_entry(family)

    def _matches(record):
        if not record:
            return False
        if (record.get("metadata") or {}).get("shadow_bundle_version") == version:
            return True
        return record.get("version") == version

    beta = entry.get("beta")
    if _matches(beta):
        return "promote", beta["version"]
    stable = entry.get("stable")
    if _matches(stable):
        return "current", None
    for record in entry.get("history", []):
        if record.get("channel") == "stable" and _matches(record):
            return "rollback", record["version"]
    return None


@router.post("/{family}/activate")
async def activate_family_version(family: str, payload: ActivateRequest, token: Optional[str] = None):
    """Guarded-shadow families: flips the live active bundle pointer with no
    restart required. Registry families: this IS promotion (stable takes on
    the named beta candidate) - wraps the existing, previously CLI-only
    models.versioning.promote()."""
    _require_admin(token)
    _require_confirmation(payload.confirm_token, f"activate_model:{family}")
    if family in shadow_bundles._REGISTRY_MIGRATED_FAMILIES:
        # Migrated shadow family: activating a version is a real, gated
        # promotion in the unified registry (per the approved design -
        # "Upload = beta, Activate = promote"), not just an ungated pointer
        # flip. shadow_bundles.set_active() still runs afterward, best-effort,
        # purely to keep its own active.json/BUNDLE_ENV in sync for the
        # existing version-history display and any operator-set env var
        # override during this transition - the read side
        # (risk_fusion_glm_shadow.load_bundle()) already resolves correctly
        # from the registry alone once BUNDLE_ENV is unset.
        action = _registry_action_for_shadow_bundle(family, payload.version)
        if action is None:
            raise HTTPException(status_code=404,
                                detail=f"{family} version {payload.version!r} has no matching registry candidate "
                                       f"(it may predate this family's registry migration)")
        action_name, registry_version = action
        try:
            if action_name == "promote":
                promote(family, registry_version)
            elif action_name == "rollback":
                rollback(family, registry_version)
            # "current": already the active stable version - nothing to do.
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except Exception as exc:
            logger.error("Failed to promote/rollback %s to %s: %s", family, payload.version, exc)
            raise HTTPException(status_code=500, detail="Activation failed")
        try:
            shadow_bundles.set_active(family, payload.version)
        except Exception as exc:
            logger.warning("Registry-side activation of %s %s succeeded, but shadow_bundles.set_active "
                           "failed (non-fatal - registry fallback still serves it): %s", family, payload.version, exc)
        return {"success": True, **_registry_summary(family)}
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
    confirm_token: Optional[str] = None


@router.post("/{family}/rollback")
async def rollback_family(family: str, payload: RollbackRequest, token: Optional[str] = None):
    """Registry families only - wraps the existing, previously CLI-only
    models.versioning.rollback(). A guarded-shadow family "rolls back" by
    activating an older already-installed version through /activate
    instead - there is no separate stable/beta distinction to roll back
    from for those families."""
    _require_admin(token)
    _require_confirmation(payload.confirm_token, f"rollback_model:{family}")
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
