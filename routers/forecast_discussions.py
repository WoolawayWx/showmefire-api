import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from core.database import (
    archive_forecast_discussion,
    create_forecast_discussion,
    delete_forecast_discussion,
    get_forecast_discussion,
    get_latest_forecast_discussion,
    list_forecast_discussions,
    publish_forecast_discussion,
    update_forecast_discussion,
)
from core.security import verify_token

logger = logging.getLogger(__name__)
router = APIRouter(tags=["forecast-discussions"])


def _require_admin(token: Optional[str] = None) -> str:
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return email


class ForecastDiscussionCreate(BaseModel):
    title: str
    body: str
    author_name: Optional[str] = None
    status: str = "draft"

    @field_validator("title", "body")
    @classmethod
    def require_text(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("status")
    @classmethod
    def require_status(cls, value: str) -> str:
        if value not in {"draft", "published"}:
            raise ValueError("status must be draft or published")
        return value


class ForecastDiscussionUpdate(BaseModel):
    title: Optional[str] = None
    body: Optional[str] = None
    author_name: Optional[str] = None
    status: Optional[str] = None

    @field_validator("title", "body")
    @classmethod
    def strip_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = str(value).strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and value not in {"draft", "published", "archived"}:
            raise ValueError("invalid status")
        return value


def _public_or_404(discussion_id: int):
    discussion = get_forecast_discussion(discussion_id)
    if not discussion or discussion["status"] == "draft":
        raise HTTPException(status_code=404, detail="Forecast discussion not found")
    return discussion


@router.get("/api/forecast-discussions/latest")
def public_latest_forecast_discussion():
    discussion = get_latest_forecast_discussion()
    if not discussion:
        raise HTTPException(status_code=404, detail="No current forecast discussion")
    return {"discussion": discussion}


@router.get("/api/forecast-discussions")
def public_list_forecast_discussions(limit: int = 50, offset: int = 0):
    discussions = list_forecast_discussions(status="published", limit=100, offset=0)
    discussions += list_forecast_discussions(status="archived", limit=100, offset=0)
    discussions.sort(key=lambda item: (item.get("issued_at") or item.get("created_at") or "", item["id"]), reverse=True)
    return {"discussions": discussions[offset:offset + min(max(limit, 1), 100)]}


@router.get("/api/forecast-discussions/{discussion_id}")
def public_get_forecast_discussion(discussion_id: int):
    return {"discussion": _public_or_404(discussion_id)}


@router.get("/api/admin/forecast-discussions")
def admin_list_forecast_discussions(token: Optional[str] = None, limit: int = 100, offset: int = 0):
    _require_admin(token)
    return {"success": True, "discussions": list_forecast_discussions(limit=limit, offset=offset)}


@router.post("/api/admin/forecast-discussions")
def admin_create_forecast_discussion(payload: ForecastDiscussionCreate, token: Optional[str] = None):
    _require_admin(token)
    discussion = create_forecast_discussion(
        payload.title, payload.body, payload.author_name, payload.status
    )
    if payload.status == "published":
        discussion = publish_forecast_discussion(discussion["id"])
    return {"success": True, "discussion": discussion}


@router.get("/api/admin/forecast-discussions/{discussion_id}")
def admin_get_forecast_discussion(discussion_id: int, token: Optional[str] = None):
    _require_admin(token)
    discussion = get_forecast_discussion(discussion_id)
    if not discussion:
        raise HTTPException(status_code=404, detail="Forecast discussion not found")
    return {"success": True, "discussion": discussion}


@router.put("/api/admin/forecast-discussions/{discussion_id}")
def admin_update_forecast_discussion(
    discussion_id: int, payload: ForecastDiscussionUpdate, token: Optional[str] = None
):
    _require_admin(token)
    discussion = update_forecast_discussion(
        discussion_id, payload.title, payload.body, payload.author_name, payload.status
    )
    if not discussion:
        raise HTTPException(status_code=404, detail="Forecast discussion not found")
    if payload.status == "published":
        discussion = publish_forecast_discussion(discussion_id)
    return {"success": True, "discussion": discussion}


@router.post("/api/admin/forecast-discussions/{discussion_id}/publish")
def admin_publish_forecast_discussion(discussion_id: int, token: Optional[str] = None):
    _require_admin(token)
    if not get_forecast_discussion(discussion_id):
        raise HTTPException(status_code=404, detail="Forecast discussion not found")
    return {"success": True, "discussion": publish_forecast_discussion(discussion_id)}


@router.post("/api/admin/forecast-discussions/{discussion_id}/archive")
def admin_archive_forecast_discussion(discussion_id: int, token: Optional[str] = None):
    _require_admin(token)
    discussion = archive_forecast_discussion(discussion_id)
    if not discussion:
        raise HTTPException(status_code=404, detail="Forecast discussion not found")
    return {"success": True, "discussion": discussion}


@router.delete("/api/admin/forecast-discussions/{discussion_id}")
def admin_delete_forecast_discussion(discussion_id: int, token: Optional[str] = None):
    _require_admin(token)
    if not delete_forecast_discussion(discussion_id):
        raise HTTPException(status_code=404, detail="Forecast discussion not found")
    return {"success": True}
