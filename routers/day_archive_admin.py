"""Admin endpoints for the R2 day archive. Everything is read from R2, never the server's disk.

No web page lives here by design: these are JSON endpoints for the CLI, scripts or the website.
"""
import logging
import re
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from core.security import require_admin
from services import day_archive

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/day-archive", tags=["day-archive-admin"])
DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _s3():
    from services.archive_bundler import _r2_client, _r2_credentials_present

    if not _r2_credentials_present():
        raise HTTPException(status_code=503, detail="R2 credentials are not configured")
    return _r2_client()


def _day(value: str) -> str:
    if not DAY.fullmatch(value):
        raise HTTPException(status_code=422, detail="date must be YYYY-MM-DD")
    return value


class MarkDownloaded(BaseModel):
    files: List[str] = Field(default_factory=list, description="Names of the files that were verified")
    verified: bool = Field(default=True, description="True only if every checksum matched the manifest")


@router.get("")
def list_day_archives(_admin: str = Depends(require_admin)):
    return {"window_days": day_archive.keep_days(), "days": day_archive.list_days(_s3())}


@router.get("/{date}")
def get_day_manifest(date: str, _admin: str = Depends(require_admin)):
    manifest = day_archive.read_manifest(_s3(), _day(date))
    if manifest is None:
        raise HTTPException(status_code=404, detail=f"No day archive for {date}")
    return manifest


@router.get("/{date}/links")
def get_day_links(date: str, expires_in: int = Query(3600, ge=60, le=86400),
                  _admin: str = Depends(require_admin)):
    """Short-lived presigned R2 download URLs for every file in the bundle."""
    links = day_archive.presigned_links(_s3(), _day(date), expires_in=expires_in)
    if links is None:
        raise HTTPException(status_code=404, detail=f"No day archive for {date}")
    return {"day": date, "expires_in": expires_in, "files": links}


@router.post("/{date}/mark-downloaded")
def mark_day_downloaded(date: str, payload: Optional[MarkDownloaded] = None, admin: str = Depends(require_admin)):
    """Record a verified download (writes a receipt to R2). Only unlocks R2 expiry after the window."""
    s3 = _s3()
    day = _day(date)
    if day_archive.read_manifest(s3, day) is None:
        raise HTTPException(status_code=404, detail=f"No day archive for {date}")
    payload = payload or MarkDownloaded()
    receipt = day_archive.write_receipt(s3, day, admin, files=payload.files, verified=payload.verified)
    return {"success": True, "receipt": receipt}


@router.post("/run", status_code=202)
def run_day_archive(background_tasks: BackgroundTasks, _admin: str = Depends(require_admin)):
    """Trigger a build/prune/reminder pass now (the scheduler also runs it nightly)."""
    background_tasks.add_task(_run_logged)
    return {"success": True, "message": "Day archive run started"}


def _run_logged():
    try:
        logger.info("manual day archive run: %s", day_archive.run())
    except Exception:  # noqa: BLE001
        logger.exception("manual day archive run failed")
