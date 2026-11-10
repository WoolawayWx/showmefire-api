"""Staff-only hybrid data, including every chunk, raster and image."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import FileResponse
from core.security import require_admin
from core.rate_limit import RateLimiter
from services import fire_weather_hybrid_store as store

limiter = RateLimiter("fire_weather_hybrid_admin", 1200)
router = APIRouter(prefix="/api/admin/fire-weather-hybrid", tags=["fire-weather-hybrid-admin"],
                   dependencies=[Depends(require_admin), Depends(limiter)])
PRIVATE = {"Cache-Control": "private, no-store", "Vary": "Cookie, Authorization"}


def directory(run_id):
    try:
        return store.resolve_run(store.root_dir(), run_id)
    except (FileNotFoundError, ValueError):
        raise HTTPException(404, "Hybrid run not found")


@router.get("/status")
def status(response: Response):
    response.headers.update(PRIVATE)
    latest = store.read_json(store.root_dir()/"latest.json") or {}
    age = None
    if latest.get("cycle"):
        from services.fire_weather_hybrid.contracts import utc
        doc = store.read_json(directory(latest["cycle"])/"manifest.json")
        age = (datetime.now(timezone.utc)-utc(doc["issued_at_utc"])).total_seconds()/3600
    return {"state": "never_run", "updating": False, **(store.read_json(store.root_dir()/"status.json") or {}),
            "serving_cycle": latest.get("cycle"), "public": False, "advisory_only": True,
            "age_hours": age, "stale": age is not None and age > 3,
            "enabled": os.getenv("SMF_FIRE_WEATHER_HYBRID_ENABLED", "false").lower() == "true"}


@router.get("/verification")
def verification(response: Response):
    response.headers.update(PRIVATE)
    records = sorted((store.root_dir()/"verification").glob("*.json"),reverse=True)[:30]
    return {"status":"available" if records else "deferred","reports":[store.read_json(path) for path in records]}


@router.get("/runs")
def runs(response: Response):
    response.headers.update(PRIVATE)
    root = store.root_dir()/"runs"
    paths = sorted(root.iterdir(), reverse=True) if root.is_dir() else []
    return {"runs": [{"cycle": path.name, "complete": True} for path in paths
                     if store.RUN_ID.fullmatch(path.name) and (path/"manifest.json").is_file()]}


@router.get("/runs/{run_id}")
@router.get("/runs/{run_id}/manifest")
def manifest(run_id: str, response: Response):
    response.headers.update(PRIVATE)
    return store.read_json(directory(run_id)/"manifest.json")


@router.get("/runs/{run_id}/point")
def point(run_id: str, response: Response, lat: float = Query(..., ge=-90, le=90), lon: float = Query(..., ge=-180, le=180)):
    response.headers.update(PRIVATE)
    try:
        return store.point(directory(run_id), lat, lon)
    except ValueError as error:
        raise HTTPException(404, str(error))


@router.get("/runs/{run_id}/zarr/{path:path}")
def chunk(run_id: str, path: str):
    try:
        target = store.safe_asset(directory(run_id)/"forecast.zarr", path)
    except FileNotFoundError:
        raise HTTPException(404, "Chunk not found")
    return FileResponse(target, headers=PRIVATE, media_type="application/json" if target.name.startswith(".") else "application/octet-stream")


@router.get("/runs/{run_id}/files/{path:path}")
def asset(run_id: str, path: str):
    if not (path == "forecast.nc" or path.startswith("rasters/") or path.startswith("images/")):
        raise HTTPException(404, "Asset not found")
    try:
        target = store.safe_asset(directory(run_id), path)
    except FileNotFoundError:
        raise HTTPException(404, "Asset not found")
    return FileResponse(target, headers=PRIVATE)


@router.post("/run", status_code=202)
def queue_run(response: Response):
    response.headers.update(PRIVATE)
    if os.getenv("SMF_FIRE_WEATHER_HYBRID_ENABLED", "false").lower() != "true":
        raise HTTPException(409, "Enable the experimental worker first")
    if not (os.getenv("SMF_FIRE_WEATHER_HYBRID_INPUT") or os.getenv("SMF_FIRE_WEATHER_HYBRID_ACQUIRE_CONFIG")) or not os.getenv("SMF_FIRE_WEATHER_HYBRID_BUNDLE"):
        raise HTTPException(409, "Configure input and bundle first")
    root = store.root_dir()
    root.mkdir(parents=True, exist_ok=True)
    if (root/"worker.lock").exists() or (root/"pipeline.lock").exists():
        raise HTTPException(409, "Worker already running")
    script = Path(__file__).resolve().parents[1]/"scripts"/"run_fire_weather_hybrid.py"
    with (root/"worker.log").open("ab") as log:
        process = subprocess.Popen([sys.executable, str(script)], cwd=script.parents[1], stdout=log, stderr=log,
                                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    return {"queued": True, "pid": process.pid}

