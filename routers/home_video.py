"""Homepage featured video: a YouTube link or a locally uploaded file."""
import logging
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import parse_qs, urlparse

from fastapi import APIRouter, File, HTTPException, Response, UploadFile
from pydantic import BaseModel, Field

from core.database import get_db_path
from core.security import verify_token

logger = logging.getLogger(__name__)
router = APIRouter(tags=["home-video"])

HOME_VIDEO_DIR = Path("data/home_video")
HOME_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
ALLOWED_EXTENSIONS = {".mp4", ".webm"}
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
YOUTUBE_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")


def _require_admin(token: Optional[str]) -> str:
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return email


def parse_youtube_id(value: str) -> Optional[str]:
    """Video ID from a watch / youtu.be / embed / shorts / live URL, or a bare ID."""
    value = (value or "").strip()
    if YOUTUBE_ID.match(value):
        return value
    parsed = urlparse(value if "//" in value else f"https://{value}")
    host = (parsed.hostname or "").lower().removeprefix("www.").removeprefix("m.")
    candidate = None
    if host == "youtu.be":
        candidate = parsed.path.strip("/").split("/")[0]
    elif host in {"youtube.com", "youtube-nocookie.com"}:
        parts = [part for part in parsed.path.split("/") if part]
        if parts and parts[0] in {"embed", "shorts", "live", "v"} and len(parts) > 1:
            candidate = parts[1]
        else:
            candidate = (parse_qs(parsed.query).get("v") or [None])[0]
    return candidate if candidate and YOUTUBE_ID.match(candidate) else None


def _row() -> dict:
    with sqlite3.connect(get_db_path()) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute("SELECT * FROM home_video WHERE id = 1").fetchone()
    return dict(row) if row else {
        "enabled": 0, "source": "youtube", "youtube_id": "", "filename": "", "title": "", "caption": "",
        "expires_at": "",
    }


def _parse_expires_at(value: str) -> Optional[datetime]:
    value = (value or "").strip()
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(status_code=400, detail="That doesn't look like a valid expiration date/time.")
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _is_expired(row: dict) -> bool:
    expires_at = _parse_expires_at(row.get("expires_at") or "")
    return expires_at is not None and expires_at <= datetime.now(timezone.utc)


def _present(row: dict) -> dict:
    source = row.get("source") or "youtube"
    playable = bool(row.get("youtube_id")) if source == "youtube" else bool(row.get("filename"))
    expired = _is_expired(row)
    return {
        "enabled": bool(row.get("enabled")) and playable and not expired,
        "source": source,
        "youtube_id": row.get("youtube_id") or "",
        "video_url": f"/home-video/{row['filename']}" if source == "upload" and row.get("filename") else "",
        "has_upload": bool(row.get("filename")),
        "title": row.get("title") or "",
        "caption": row.get("caption") or "",
        "expires_at": row.get("expires_at") or "",
        "expired": expired,
        "updated_at": row.get("updated_at"),
    }


@router.get("/api/home-video")
def public_home_video(response: Response):
    response.headers["Cache-Control"] = "public, max-age=60"
    return _present(_row())


@router.get("/api/admin/home-video")
def admin_get_home_video(token: Optional[str] = None):
    _require_admin(token)
    return {"success": True, **_present(_row()), "enabled_setting": bool(_row().get("enabled"))}


class HomeVideoUpdate(BaseModel):
    enabled: bool = False
    source: Literal["youtube", "upload"] = "youtube"
    youtube_url: str = Field(default="", max_length=300)
    title: str = Field(default="", max_length=120)
    caption: str = Field(default="", max_length=500)
    expires_at: str = Field(default="", max_length=40)


def _save(email: str, **fields) -> None:
    current = _row()
    merged = {**current, **fields}
    with sqlite3.connect(get_db_path()) as connection:
        connection.execute(
            """INSERT INTO home_video (id, enabled, source, youtube_id, filename, title, caption, expires_at, updated_by, updated_at)
               VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(id) DO UPDATE SET enabled=excluded.enabled, source=excluded.source,
                 youtube_id=excluded.youtube_id, filename=excluded.filename, title=excluded.title,
                 caption=excluded.caption, expires_at=excluded.expires_at, updated_by=excluded.updated_by,
                 updated_at=CURRENT_TIMESTAMP""",
            (int(bool(merged["enabled"])), merged["source"], merged["youtube_id"], merged["filename"],
             merged["title"], merged["caption"], merged["expires_at"], email),
        )


@router.post("/api/admin/home-video")
def admin_update_home_video(payload: HomeVideoUpdate, token: Optional[str] = None):
    email = _require_admin(token)
    current = _row()
    youtube_id = current.get("youtube_id") or ""
    if payload.youtube_url.strip():
        youtube_id = parse_youtube_id(payload.youtube_url) or ""
        if not youtube_id:
            raise HTTPException(status_code=400, detail="That doesn't look like a YouTube link.")
    if payload.enabled and payload.source == "youtube" and not youtube_id:
        raise HTTPException(status_code=400, detail="Add a YouTube link before enabling the video.")
    if payload.enabled and payload.source == "upload" and not current.get("filename"):
        raise HTTPException(status_code=400, detail="Upload a video file before enabling the video.")
    expires_at = payload.expires_at.strip()
    expires_at_parsed = _parse_expires_at(expires_at)
    if payload.enabled and expires_at_parsed is not None and expires_at_parsed <= datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="The expiration date/time is in the past.")
    _save(email, enabled=payload.enabled, source=payload.source, youtube_id=youtube_id,
          title=payload.title.strip(), caption=payload.caption.strip(), expires_at=expires_at)
    return {"success": True, **_present(_row())}


def _remove_file(filename: str) -> None:
    if filename:
        (HOME_VIDEO_DIR / Path(filename).name).unlink(missing_ok=True)


@router.post("/api/admin/home-video/upload")
async def admin_upload_home_video(token: Optional[str] = None, upload: UploadFile = File(...)):
    email = _require_admin(token)
    extension = Path(upload.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only .mp4 or .webm files are allowed.")
    filename = f"{uuid.uuid4().hex}{extension}"
    target = HOME_VIDEO_DIR / filename
    size = 0
    try:
        with open(target, "wb") as handle:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Video is larger than 200 MB.")
                handle.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    previous = _row().get("filename") or ""
    _save(email, filename=filename, source="upload")
    if previous != filename:
        _remove_file(previous)
    return {"success": True, **_present(_row())}


@router.delete("/api/admin/home-video/upload")
def admin_delete_home_video_upload(token: Optional[str] = None):
    email = _require_admin(token)
    current = _row()
    has_youtube = bool(current.get("youtube_id"))
    # Without a YouTube fallback there is nothing left to show, so switch it off.
    _save(email, filename="", source="youtube" if has_youtube else current.get("source"),
          enabled=bool(current.get("enabled")) and has_youtube)
    _remove_file(current.get("filename") or "")
    return {"success": True, **_present(_row())}
