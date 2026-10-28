"""Admin endpoints for the fire-danger model performance tracker (services/fire_danger_tracking.py)."""
from typing import Literal

from fastapi import APIRouter, Depends, Query, Response

from core.security import require_admin
from services import fire_danger_tracking as tracking

router = APIRouter(tags=["fire-danger-tracking"])


@router.get("/api/admin/models/fire-danger-tracking")
def fire_danger_tracking(response: Response, days: int = Query(90, ge=7, le=730), lag_days: int = Query(3, ge=0, le=30),
                         min_tier: Literal["unverified", "admin_reviewed", "official_source_confirmed"] = "admin_reviewed",
                         _admin: str = Depends(require_admin)):
    """How each model's county-day predictions line up with confirmed fires. Reads stored scores; nothing here runs a model."""
    response.headers["Cache-Control"] = "no-store"
    return {"success": True, **tracking.evaluate(days, lag_days, min_tier)}


@router.post("/api/admin/models/fire-danger-tracking/refresh")
def refresh_fire_danger_tracking(_admin: str = Depends(require_admin)):
    """Copy any new model evidence into the tracking table now (the nightly job does this too)."""
    return {"success": True, **tracking.ingest_scores()}
