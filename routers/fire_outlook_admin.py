"""Staff-only view of the seven-day fire outlook (the shadow pilot).

Same data as the public routes in routers/fire_outlook.py, behind an admin session, so staff can see exactly what the worker is
producing while the public routes answer 404 (SMF_FIRE_OUTLOOK_PUBLIC unset). Nothing here changes anything: it only reads the worker's
folder. To go public later: review the pilot, set SMF_FIRE_OUTLOOK_PUBLIC=true and (for CDN delivery) SMF_FIRE_OUTLOOK_PUBLISH=true.

    GET /api/admin/fire-outlook/status                     settings, the worker's full status (with error text), the latest run's
                                                           conditioning / precipitation audit / sources, verification export counts
    GET /api/admin/fire-outlook/runs                       runs on disk
    GET /api/admin/fire-outlook/runs/{run_id}/manifest     the immutable manifest
    GET /api/admin/fire-outlook/runs/{run_id}/point        28-window series at a lat/lon
    GET /api/admin/fire-outlook/runs/{run_id}/files/...    contour files (private cache headers)

Not aliased to /fire-outlook: /api/admin/* is excluded from the unprefixed aliases (core/api_aliases.py).
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, Query

from core.rate_limit import RateLimiter
from core.security import require_admin
from routers import fire_outlook as public
from services.fire_outlook import worker

admin_limiter = RateLimiter("fire_outlook_admin", 1200)
router = APIRouter(prefix="/api/admin/fire-outlook", tags=["fire-outlook-admin"],
                   dependencies=[Depends(admin_limiter)])

GEFS_READY_AFTER_HOURS = 5   # measured: all 31 members' f168 are published about 5 h after the cycle


def _flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


def _dir_stats(path: Path) -> dict:
    files = [p for p in path.rglob("*") if p.is_file()] if path.is_dir() else []
    return {"files": len(files), "bytes": sum(p.stat().st_size for p in files)}


def _run_summary(root: Path, run_id: Optional[str]) -> Optional[dict]:
    if not run_id:
        return None
    manifest = public._read_json(root / "runs" / run_id / "manifest.json")
    if not manifest:
        return None
    stats = _dir_stats(root / "runs" / run_id)
    conditioning = manifest.get("conditioning") or {}
    return {
        "run_id": manifest["run_id"], "cycle_time_utc": manifest["cycle_time_utc"], "published_utc": manifest.get("published_utc"),
        "member_count": manifest["member_count"], "windows": len(manifest["windows"]), "serving": public._serving(manifest),
        "conditioning": conditioning, "rules": manifest.get("rules"), "driver": manifest.get("driver"),
        "source_cycles": manifest.get("source_cycles"), "coverage": manifest.get("coverage"), "runtime": manifest.get("runtime"),
        "nodata_cells_any_window": manifest.get("nodata_cells_any_window"),
        "layers": [layer["id"] for layer in manifest.get("layers", [])], "files": stats["files"], "bytes": stats["bytes"],
    }


def build_admin_status(root: Path, now: Optional[datetime] = None) -> dict:
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    config = worker.Config.from_env(root)
    latest = public._latest_id(root)
    cycle = worker.latest_cycle(now)
    verification_dir = root / "verification"
    records = sorted((verification_dir / "records").glob("*")) if (verification_dir / "records").is_dir() else []
    runs_dir = root / "runs"
    on_disk = [p.name for p in sorted(runs_dir.iterdir(), reverse=True)
               if p.is_dir() and public.RUN_ID_PATTERN.fullmatch(p.name)] if runs_dir.is_dir() else []
    published = public._read_json(root / "published.json") or {}
    return {
        "mode": "public" if public.public_enabled() else "admin_preview",
        "settings": {
            "worker_enabled": _flag("SMF_FIRE_OUTLOOK_ENABLED"), "public": public.public_enabled(), "publish_to_cdn": config.publish,
            "threads": config.threads, "memory_limit_gb": config.memory_limit_gb, "public_keep_days": config.public_keep_days,
            "cache_keep_days": config.cache_keep_days,
        },
        "schedule": {
            "latest_cycle_utc": cycle.isoformat().replace("+00:00", "Z"),
            "inputs_expected_utc": (cycle + timedelta(hours=GEFS_READY_AFTER_HOURS)).isoformat().replace("+00:00", "Z"),
            "checks": "every 30 minutes (:07, :37)",
        },
        "worker": public._read_json(root / "status.json") or {"state": "never_run"},
        "latest_run": _run_summary(root, latest),
        "runs_on_disk": on_disk,
        "uploaded_to_cdn": sorted(published.get("run_ids", []))[-5:],
        "verification": {
            "forecast_side": _dir_stats(verification_dir / "forecast"),
            "records": {**_dir_stats(verification_dir / "records"), "newest": records[-1].name if records else None},
        },
        "storage": {"runs": _dir_stats(runs_dir), "cache": _dir_stats(root / "cache")},
    }


@router.get("/status")
def get_status(_admin: str = Depends(require_admin)):
    return build_admin_status(public._root())


@router.get("/runs")
def list_runs(_admin: str = Depends(require_admin)):
    return public.list_runs()


@router.get("/runs/{run_id}/manifest")
def get_manifest(run_id: str, _admin: str = Depends(require_admin)):
    return public.get_manifest(run_id)


@router.get("/runs/{run_id}/point")
def get_point(run_id: str, lat: float = Query(..., ge=-90, le=90), lon: float = Query(..., ge=-180, le=180),
              _admin: str = Depends(require_admin)):
    return public.get_point(run_id, lat, lon)


@router.get("/runs/{run_id}/files/{path:path}")
def get_file(run_id: str, path: str, _admin: str = Depends(require_admin)):
    return public.file_response(run_id, path, private=True)
