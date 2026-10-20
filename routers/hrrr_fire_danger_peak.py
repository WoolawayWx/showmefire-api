"""Read-only access to the peak fire danger estimate (services/hrrr_hourly/peak.py): per local day, the worst of the 10 AM - 9 PM
Central hours of an HRRR run. Same shape as /hrrr-hourly (status, runs, manifest, Zarr store, point series), so the data
viewer reads it the same way; the CDN copy lives under datasets/hrrr-fire-danger-peak/. No /api prefix.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from core.rate_limit import RateLimiter
from routers.hrrr_hourly import CYCLE_PATTERN
from services.hrrr_hourly import job, live_store, peak, store
from services.hrrr_hourly import root_dir as hourly_root

chunk_limiter = RateLimiter("hrrr_peak_chunks", 600)
data_limiter = RateLimiter("hrrr_peak", 60)
point_limiter = RateLimiter("hrrr_peak_point", 30)
status_limiter = RateLimiter("hrrr_peak_status", 120)

router = APIRouter(prefix="/hrrr-fire-danger-peak", tags=["hrrr-fire-danger-peak"])


def _resolve_cycle(cycle: str) -> str:
    if cycle == "latest":
        try:
            cycle = json.loads((peak.root_dir() / "latest.json").read_text())["cycle"]
        except (OSError, ValueError, KeyError):
            raise HTTPException(status_code=404, detail="No peak fire danger has been built yet")
    if not CYCLE_PATTERN.fullmatch(cycle):
        raise HTTPException(status_code=400, detail="cycle must look like 20261016_12z (or 'latest')")
    return cycle


def _manifest(cycle: str) -> dict:
    manifest = store.read_manifest(cycle, peak.root_dir())
    if manifest is None:
        raise HTTPException(status_code=404, detail="Unknown cycle")
    return manifest


@router.get("/status", dependencies=[Depends(status_limiter)])
def get_status():
    """The hourly product's update state (this product is filled in by the same job), with this product's serving cycle."""
    status = job.read_status(hourly_root())
    serving = store._latest_cycle_id(peak.root_dir())
    if not status:
        return {"product": peak.PRODUCT, "state": "idle", "updating": False, "pending": False, "serving_cycle": serving,
                "updated_at": None}
    return {**status, "product": peak.PRODUCT, "serving_cycle": serving}


@router.get("/runs", dependencies=[Depends(data_limiter)])
def list_runs():
    runs = []
    for cycle in store.list_cycles(peak.root_dir()):
        manifest = store.read_manifest(cycle, peak.root_dir()) or {}
        runs.append({"cycle": cycle, "label": manifest.get("label"), "cycle_time_utc": manifest.get("cycle_time_utc"),
                     "issued_at_utc": manifest.get("issued_at_utc"), "forecast_hours": manifest.get("forecast_hours"),
                     "missing_hours": manifest.get("missing_hours"), "variables": len(manifest.get("variables", {})),
                     "complete": manifest.get("complete", True), "pending_hours": manifest.get("pending_hours", []),
                     "hours_expected": manifest.get("hours_expected"), "step_labels": manifest.get("step_labels")})
    return {"latest": store._latest_cycle_id(peak.root_dir()) or next((r["cycle"] for r in runs if r["complete"]), None), "runs": runs}


@router.get("/runs/{cycle}", dependencies=[Depends(data_limiter)])
def get_manifest(cycle: str):
    return _manifest(_resolve_cycle(cycle))


@router.get("/runs/{cycle}/zarr/{path:path}", dependencies=[Depends(chunk_limiter)])
def get_zarr_file(cycle: str, path: str):
    """One file of the Zarr store (.zmetadata, <array>/.zarray, <array>/<chunk>)."""
    cycle = _resolve_cycle(cycle)
    base = (peak.root_dir() / cycle / "forecast.zarr").resolve()
    target = (base / path).resolve()
    if base not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    finished = (store.read_manifest(cycle, peak.root_dir()) or {}).get("complete", True)
    return FileResponse(target, media_type="application/json" if target.name.startswith(".z") else "application/octet-stream",
                        headers={"Cache-Control": "public, max-age=300" if finished else "public, max-age=30"})


@router.get("/runs/{cycle}/point", dependencies=[Depends(point_limiter)])
def get_point(cycle: str, lat: float = Query(..., ge=-90, le=90), lon: float = Query(..., ge=-180, le=180)):
    cycle = _resolve_cycle(cycle)
    manifest = _manifest(cycle)
    zarr_dir = peak.root_dir() / cycle / "forecast.zarr"
    if not zarr_dir.is_dir():
        raise HTTPException(status_code=404, detail="Unknown cycle")
    result = live_store.point_series(zarr_dir, manifest, lat, lon, store.MAX_POINT_DISTANCE_KM)
    if result.get("error"):
        raise HTTPException(status_code=404, detail="Point is outside the forecast grid")
    result["label"] = manifest["label"]
    result["step_labels"] = manifest.get("step_labels")
    result["units"] = {name: spec["units"] for name, spec in manifest["variables"].items()}
    return result
