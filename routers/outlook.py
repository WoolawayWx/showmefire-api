import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core.security import verify_token
from services import outlook as outlook_service

logger = logging.getLogger(__name__)

router = APIRouter(tags=["outlook"])


class OutlookSaveRequest(BaseModel):
    geojson: Dict[str, Any]


@router.get("/outlook/published")
async def get_published_outlook(day: int = 2):
    return outlook_service.get_published(day)


@router.get("/api/admin/outlook/draft")
async def get_outlook_draft(token: str, day: int = 2):
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")

    return outlook_service.get_draft(day)


@router.post("/api/admin/outlook/draft")
async def save_outlook_draft(payload: OutlookSaveRequest, token: str, day: int = 2):
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        return outlook_service.save_draft(payload.geojson, email, day)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to save outlook draft: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to save outlook draft") from exc


@router.post("/api/admin/outlook/publish")
async def publish_outlook(token: str, day: int = 2):
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        return outlook_service.publish_draft(email, day)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to publish outlook draft: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to publish outlook draft") from exc


@router.delete("/api/admin/outlook/draft")
async def delete_outlook_draft(token: str, day: int = 2):
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        return outlook_service.clear_draft(email, day)
    except Exception as exc:
        logger.error("Failed to delete outlook draft: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to delete outlook draft") from exc


@router.get("/api/admin/outlook/status")
async def get_outlook_status(token: str, day: int = 2):
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")

    return outlook_service.get_status(day)


@router.post("/api/admin/outlook/generate-graphic")
async def generate_outlook_graphic(token: str, day: int = 2, valid_date: Optional[str] = None, issue_time: Optional[str] = None):
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        return outlook_service.generate_graphic(email, day, valid_date, issue_time)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to generate outlook graphic: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to generate outlook graphic") from exc


