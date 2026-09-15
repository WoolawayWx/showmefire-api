"""Admin controls and review endpoints for Broadcastify transcription."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from core.database import (
    add_bcfy_channel,
    delete_bcfy_channel,
    get_bcfy_call,
    get_bcfy_channel,
    get_bcfy_config,
    list_bcfy_calls,
    list_bcfy_channels,
    list_bcfy_job_events,
    review_bcfy_fire_signal,
    update_bcfy_channel,
    update_bcfy_config,
)
from core.security import verify_token
from services.transcription import poll_bcfy_calls, reprocess_call, test_bcfy_connection

router = APIRouter(prefix="/api/admin/transcription", tags=["transcription"])


def _admin(token: Optional[str]) -> str:
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return email


class ConfigUpdate(BaseModel):
    enabled: Optional[bool] = None
    department_name: Optional[str] = Field(default=None, min_length=1, max_length=160)
    poll_minutes: Optional[int] = Field(default=None, ge=1, le=60)
    model_name: Optional[str] = Field(default=None, min_length=1, max_length=80)
    classification_threshold: Optional[float] = Field(default=None, ge=0, le=1)
    retention_days: Optional[int] = Field(default=None, ge=1, le=90)


class ChannelCreate(BaseModel):
    group_id: str = Field(min_length=1, max_length=80)
    label: str = Field(default="", max_length=160)


class ChannelUpdate(BaseModel):
    label: Optional[str] = Field(default=None, max_length=160)
    enabled: Optional[bool] = None


class ReviewRequest(BaseModel):
    status: str
    linked_fire_event_id: Optional[int] = None


@router.get("/config")
async def config(token: Optional[str] = None):
    _admin(token)
    current = get_bcfy_config()
    from services.transcription import configured
    return {
        "success": True,
        "config": current,
        "credentials_configured": configured(),
    }


@router.post("/config")
async def update_config(payload: ConfigUpdate, token: Optional[str] = None):
    email = _admin(token)
    values = payload.model_dump(exclude_none=True)
    values["updated_by"] = email
    if "department_name" in values:
        values["department_name"] = values["department_name"].strip()
    return {"success": True, "config": update_bcfy_config(**values)}


@router.get("/status")
async def status(token: Optional[str] = None):
    _admin(token)
    config_data = get_bcfy_config()
    calls = list_bcfy_calls(limit=1)
    counts = {}
    for state in ("discovered", "downloading", "transcribing", "classifying", "accepted", "rejected", "failed"):
        counts[state] = len(list_bcfy_calls(limit=200, status=state))
    from services.transcription import configured
    return {
        "success": True,
        "enabled": bool(config_data.get("enabled")),
        "credentials_configured": configured(),
        "department_name": config_data.get("department_name", ""),
        "channels": list_bcfy_channels(),
        "latest_call": calls[0] if calls else None,
        "counts": counts,
    }


@router.get("/channels")
async def channels(token: Optional[str] = None):
    _admin(token)
    return {"success": True, "channels": list_bcfy_channels()}


@router.post("/channels")
async def create_channel(payload: ChannelCreate, token: Optional[str] = None):
    _admin(token)
    group_id = payload.group_id.strip()
    if any(existing["group_id"] == group_id for existing in list_bcfy_channels()):
        raise HTTPException(status_code=400, detail="That channel is already being monitored")
    return {"success": True, "channel": add_bcfy_channel(group_id, payload.label.strip())}


@router.post("/channels/{channel_id}")
async def edit_channel(channel_id: int, payload: ChannelUpdate, token: Optional[str] = None):
    _admin(token)
    if not get_bcfy_channel(channel_id):
        raise HTTPException(status_code=404, detail="Channel not found")
    values = payload.model_dump(exclude_none=True)
    if "label" in values:
        values["label"] = values["label"].strip()
    return {"success": True, "channel": update_bcfy_channel(channel_id, **values)}


@router.delete("/channels/{channel_id}")
async def remove_channel(channel_id: int, token: Optional[str] = None):
    _admin(token)
    if not delete_bcfy_channel(channel_id):
        raise HTTPException(status_code=404, detail="Channel not found")
    return {"success": True}


@router.post("/channels/{channel_id}/test")
async def test_channel(channel_id: int, token: Optional[str] = None):
    _admin(token)
    channel = get_bcfy_channel(channel_id)
    if not channel:
        raise HTTPException(status_code=404, detail="Channel not found")
    return {"success": True, "result": await test_bcfy_connection(channel["group_id"])}


@router.get("/calls")
async def calls(token: Optional[str] = None, limit: int = 50, offset: int = 0, status: Optional[str] = None):
    _admin(token)
    return {"success": True, "calls": list_bcfy_calls(limit=limit, offset=offset, status=status)}


@router.get("/calls/{call_id}")
async def call_detail(call_id: int, token: Optional[str] = None):
    _admin(token)
    call = get_bcfy_call(call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    call["evidence"] = __import__("json").loads(call.get("evidence_json") or "[]")
    call["events"] = list_bcfy_job_events(call_id)
    return {"success": True, "call": call}


@router.get("/calls/{call_id}/audio")
async def call_audio(call_id: int, token: Optional[str] = None):
    _admin(token)
    call = get_bcfy_call(call_id)
    if not call or not call.get("audio_path"):
        raise HTTPException(status_code=404, detail="Audio not available")
    path = Path(call["audio_path"]).resolve()
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path, media_type="audio/mpeg", filename=path.name)


@router.post("/poll")
async def poll_now(token: Optional[str] = None):
    _admin(token)
    return {"success": True, "result": await poll_bcfy_calls()}


@router.post("/calls/{call_id}/retry")
async def retry_call(call_id: int, token: Optional[str] = None):
    _admin(token)
    call = get_bcfy_call(call_id)
    if not call:
        raise HTTPException(status_code=404, detail="Call not found")
    try:
        result = await reprocess_call(call_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"success": True, "call": result}


@router.post("/calls/reprocess-failed")
async def reprocess_failed_calls(token: Optional[str] = None, limit: int = 50):
    _admin(token)
    limit = max(1, min(limit, 200))
    targets = list_bcfy_calls(limit=limit, status="failed") + list_bcfy_calls(limit=limit, status="discovered")
    results = []
    for call in targets[:limit]:
        try:
            results.append(await reprocess_call(call["id"]))
        except Exception as exc:
            results.append({"id": call["id"], "external_id": call.get("external_id"), "error": str(exc)})
    return {"success": True, "count": len(results), "calls": results}


@router.post("/signals/{signal_id}/review")
async def review_signal(signal_id: int, payload: ReviewRequest, token: Optional[str] = None):
    reviewer = _admin(token)
    try:
        signal = review_bcfy_fire_signal(
            signal_id, payload.status, reviewer, payload.linked_fire_event_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not signal:
        raise HTTPException(status_code=404, detail="Fire signal not found")
    return {"success": True, "signal": signal}
