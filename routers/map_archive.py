"""Daily zips of the 15-minute CDN map copies (services/map_archive.py).

GET /api/map-archives    zips currently available to download, newest first
"""
from fastapi import APIRouter

from services import map_archive

router = APIRouter(prefix="/api/map-archives", tags=["archives"])


@router.get("")
def list_archives():
    return {
        "cdn_retention_days": map_archive.keep_days(),
        "zip_retention_days": map_archive.zip_keep_days(),
        "archives": map_archive.available(),
    }
