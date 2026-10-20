import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from core.security import require_admin
from services import outlook as outlook_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["outlook"])


class OutlookSaveRequest(BaseModel):
    geojson: Dict[str, Any]


@router.get("/outlook/published")
def get_published_outlook(day: int = 2):
    return outlook_service.get_published(day)


@router.get("/api/admin/outlook/draft")
def get_outlook_draft(day: int = 2, email: str = Depends(require_admin)):
    return outlook_service.get_draft(day)


@router.post("/api/admin/outlook/draft")
def save_outlook_draft(payload: OutlookSaveRequest, day: int = 2, email: str = Depends(require_admin)):
    try:
        return outlook_service.save_draft(payload.geojson, email, day)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to save outlook draft: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to save outlook draft") from exc


@router.post("/api/admin/outlook/publish")
def publish_outlook(day: int = 2, email: str = Depends(require_admin)):
    try:
        return outlook_service.publish_draft(email, day)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to publish outlook draft: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to publish outlook draft") from exc


@router.delete("/api/admin/outlook/draft")
def delete_outlook_draft(day: int = 2, email: str = Depends(require_admin)):
    try:
        return outlook_service.clear_draft(email, day)
    except Exception as exc:
        logger.error("Failed to delete outlook draft: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to delete outlook draft") from exc


@router.get("/api/admin/outlook/status")
def get_outlook_status(day: int = 2, email: str = Depends(require_admin)):
    return outlook_service.get_status(day)


@router.post("/api/admin/outlook/generate-graphic")
def generate_outlook_graphic(day: int = 2, valid_date: Optional[str] = None, issue_time: Optional[str] = None, email: str = Depends(require_admin)):
    try:
        return outlook_service.generate_graphic(email, day, valid_date, issue_time)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to generate outlook graphic: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to generate outlook graphic") from exc


