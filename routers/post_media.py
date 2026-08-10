import logging
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from core.security import verify_token
from services.post_media import POST_MEDIA_DIR, list_post_media, save_post_media

logger = logging.getLogger(__name__)

router = APIRouter(tags=["post-media"])


def _require_admin(token: Optional[str] = None) -> str:
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return email


@router.get("/api/admin/post-media")
def admin_list_post_media(token: Optional[str] = None, limit: int = 100, offset: int = 0):
    _require_admin(token)
    try:
        return {"success": True, "media": list_post_media(limit=limit, offset=offset)}
    except Exception as exc:
        logger.error("Failed to list post media: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to list media")


@router.post("/api/admin/post-media/upload", status_code=201)
async def admin_upload_post_media(token: Optional[str] = None, file: UploadFile = File(...)):
    email = _require_admin(token)
    content = await file.read(10 * 1024 * 1024 + 1)
    try:
        record = save_post_media(
            content=content,
            content_type=file.content_type or "",
            original_name=file.filename or "upload",
            uploaded_by=email,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Failed to upload post media: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to upload media")

    return {"success": True, "media": record}


@router.get("/files/post-media/{filename}")
def serve_post_media(filename: str):
    safe_name = filename.replace("/", "").replace("\\", "")
    target = POST_MEDIA_DIR / safe_name
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="Media not found")
    return FileResponse(target)
