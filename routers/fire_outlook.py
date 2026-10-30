"""Read-only access to the seven-day fire outlook (services/fire_outlook/).

    GET /api/fire-outlook/status                     worker state, cycle age, stage, runtime, memory, source coverage
    GET /api/fire-outlook/runs                       runs on disk (newest first) with the current latest
    GET /api/fire-outlook/runs/{run_id}/manifest     the run's immutable manifest ('latest' is accepted for run_id)
    GET /api/fire-outlook/runs/{run_id}/point        28-window series at a lat/lon, read from the stored arrays
    GET /api/fire-outlook/runs/{run_id}/files/{path} one contour file (contours/<layer>/wNN.geojson), for use without the CDN

These public routes answer 404 until SMF_FIRE_OUTLOOK_PUBLIC=true (the staff-only pilot uses the admin copy in
routers/fire_outlook_admin.py instead). Nothing here runs a forecast calculation. Grid and contour assets are served by the CDN dataset `fire-outlook`
(services/cdn_datasets.py), not by this API. Percentages are "scenario agreement" (share of GEFS weather
scenarios), not a probability of fire; the label text travels with every response.
"""
from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from core.rate_limit import RateLimiter
from services.fire_outlook import assemble, layers, point, worker

RUN_ID_PATTERN = re.compile(r"\d{8}T\d{2}Z")
# The only run files this API serves (the grids themselves are read through /point): the contour GeoJSON the map draws.
_LAYER_IDS = "|".join(["elevated", "critical", *layers.LAYER_IDS])
FILE_PATTERN = re.compile(rf"contours/({_LAYER_IDS})/w(?:0\d|1\d|2[0-7])\.geojson")

status_limiter = RateLimiter("fire_outlook_status", 120)
runs_limiter = RateLimiter("fire_outlook_runs", 60)
manifest_limiter = RateLimiter("fire_outlook_manifest", 120)
point_limiter = RateLimiter("fire_outlook_point", 30)
file_limiter = RateLimiter("fire_outlook_files", 600)

router = APIRouter(prefix="/api/fire-outlook", tags=["fire-outlook"])


def public_enabled() -> bool:
    """The outlook is staff-only (admin routes) until SMF_FIRE_OUTLOOK_PUBLIC is true: the plan's shadow-pilot gate."""
    return os.getenv("SMF_FIRE_OUTLOOK_PUBLIC", "false").strip().lower() in {"1", "true", "yes", "on"}


def require_public() -> None:
    if not public_enabled():
        raise HTTPException(status_code=404, detail="The seven-day outlook is not public yet")

# status.json fields that are safe and useful to show publicly; error MESSAGES stay in the worker's own file.
_PUBLIC_STATUS = ("state", "stage", "cycle", "cycle_age_hours", "started_utc", "updated_utc", "completed_utc",
                  "runtime_s", "peak_rss_mb", "threads", "memory_limit_gb", "within_4h_budget", "run_id",
                  "last_complete", "published")


def _root() -> Path:
    return worker.Config.from_env().root


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _latest_id(root: Path):
    return (_read_json(root / "latest.json") or {}).get("run_id")


def _resolve(run_id: str, root: Path) -> Path:
    if run_id == "latest":
        run_id = _latest_id(root)
        if not run_id:
            raise HTTPException(status_code=404, detail="No fire outlook run has been published yet")
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise HTTPException(status_code=400, detail="run_id must look like 20261029T12Z (or 'latest')")
    run_dir = root / "runs" / run_id
    if not (run_dir / "manifest.json").is_file():
        raise HTTPException(status_code=404, detail="Unknown run")
    return run_dir


def _serving(manifest: dict) -> dict:
    now = datetime.now(timezone.utc)
    cycle = datetime.fromisoformat(manifest["cycle_time_utc"].replace("Z", "+00:00"))
    return {"stale": assemble.is_stale(manifest, now), "age_hours": round((now - cycle).total_seconds() / 3600, 2),
            "stale_after_hours": manifest.get("stale_after_hours", assemble.MAX_AGE_HOURS_STALE)}


@router.get("/status", dependencies=[Depends(require_public), Depends(status_limiter)])
def get_status():
    root = _root()
    status = _read_json(root / "status.json")
    if not status:
        return {"state": "never_run"}
    out = {k: status[k] for k in _PUBLIC_STATUS if k in status}
    coverage = status.get("source_coverage") or {}
    out["source_coverage"] = {k: coverage[k] for k in ("members_expected", "members_ready", "last_lead_hours",
                                                       "resolution", "product") if k in coverage}
    out["error_count"] = len(status.get("errors") or [])
    out["failed_stage"] = status["errors"][-1]["stage"] if status.get("errors") else None
    return out


@router.get("/runs", dependencies=[Depends(require_public), Depends(runs_limiter)])
def list_runs():
    root = _root()
    latest = _latest_id(root)
    runs = []
    runs_dir = root / "runs"
    if runs_dir.is_dir():
        for run_dir in sorted(runs_dir.iterdir(), reverse=True):
            manifest = _read_json(run_dir / "manifest.json") if RUN_ID_PATTERN.fullmatch(run_dir.name) else None
            if not manifest:
                continue
            runs.append({"run_id": manifest["run_id"], "cycle_time_utc": manifest["cycle_time_utc"],
                         "published_utc": manifest.get("published_utc"), "member_count": manifest["member_count"],
                         "windows": len(manifest["windows"]), "label": manifest["label"],
                         "conditioning_level": (manifest.get("conditioning") or {}).get("level"),
                         "is_latest": manifest["run_id"] == latest, **_serving(manifest)})
    return {"dataset": assemble.DATASET_ID, "latest": latest, "runs": runs}


@router.get("/runs/{run_id}/manifest", dependencies=[Depends(require_public), Depends(manifest_limiter)])
def get_manifest(run_id: str):
    manifest = _read_json(_resolve(run_id, _root()) / "manifest.json")
    if manifest is None:
        raise HTTPException(status_code=404, detail="Unknown run")
    return {**manifest, "serving": _serving(manifest)}


@router.get("/runs/{run_id}/point", dependencies=[Depends(require_public), Depends(point_limiter)])
def get_point(run_id: str, lat: float = Query(..., ge=-90, le=90), lon: float = Query(..., ge=-180, le=180)):
    run_dir = _resolve(run_id, _root())
    out = point.point_series(run_dir, lat, lon)
    manifest = _read_json(run_dir / "manifest.json") or {}
    return {**out, "serving": _serving(manifest)} if manifest else out


def file_response(run_id: str, path: str, private: bool = False):
    """One contour file of a run. Anything else under the run directory is not served. `private` (the admin copy) must not
    be cached by shared caches."""
    if not FILE_PATTERN.fullmatch(path):
        raise HTTPException(status_code=404, detail="Unknown file")
    run_dir = _resolve(run_id, _root())
    target = run_dir / path
    if not target.is_file():
        raise HTTPException(status_code=404, detail="Unknown file")
    # A run is immutable, so a concrete run id caches for good; 'latest' must be re-asked. The admin copy is private.
    if private:
        cache = "private, max-age=60"
    else:
        cache = "public, max-age=60" if run_id == "latest" else "public, max-age=31536000, immutable"
    return FileResponse(target, media_type="application/geo+json", headers={"Cache-Control": cache})


@router.get("/runs/{run_id}/files/{path:path}", dependencies=[Depends(require_public), Depends(file_limiter)])
def get_file(run_id: str, path: str):
    return file_response(run_id, path)
