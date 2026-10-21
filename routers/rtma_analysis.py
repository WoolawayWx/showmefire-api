"""Read-only access to the "what happened" RTMA analysis days (services/rtma_analysis.py). Same shape as /hrrr-hourly
(status, runs, manifest, Zarr store, point series), so the data viewer reads it the same way; a run is a local day
(20261019). Finished days are also on the CDN under datasets/rtma-analysis/; today, while it fills in, is API-only.
No /api prefix.
"""
from __future__ import annotations

import re

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from core.rate_limit import RateLimiter
from services import rtma_analysis
from services.hrrr_hourly import live_store, store

DAY_PATTERN = re.compile(r"^\d{8}$")

chunk_limiter = RateLimiter("rtma_analysis_chunks", 600)
data_limiter = RateLimiter("rtma_analysis", 60)
point_limiter = RateLimiter("rtma_analysis_point", 30)
status_limiter = RateLimiter("rtma_analysis_status", 120)

router = APIRouter(prefix="/rtma-analysis", tags=["rtma-analysis"])


def _resolve(run: str) -> str:
    if run == "latest":
        run = store._latest_cycle_id(rtma_analysis.root_dir()) or ""
        if not run:
            raise HTTPException(status_code=404, detail="No analysis day has been built yet")
    if not DAY_PATTERN.fullmatch(run):
        raise HTTPException(status_code=400, detail="run must be a local date like 20261019 (or 'latest')")
    return run


def _manifest(run: str) -> dict:
    manifest = store.read_manifest(run, rtma_analysis.root_dir())
    if manifest is None:
        raise HTTPException(status_code=404, detail="No analysis for that day")
    return manifest


@router.get("/status", dependencies=[Depends(status_limiter)])
def get_status():
    """Built by an hourly job with no long-running state, so this only names the day being served."""
    serving = store._latest_cycle_id(rtma_analysis.root_dir())
    manifest = store.read_manifest(serving, rtma_analysis.root_dir()) if serving else None
    return {"product": rtma_analysis.PRODUCT, "state": "idle", "updating": False, "pending": False,
            "serving_cycle": serving, "updated_at": (manifest or {}).get("issued_at_utc")}


@router.get("/runs", dependencies=[Depends(data_limiter)])
def list_runs():
    root = rtma_analysis.root_dir()
    runs = []
    for run in store.list_cycles(root):
        manifest = store.read_manifest(run, root) or {}
        runs.append({"cycle": run, "label": manifest.get("label"), "cycle_time_utc": manifest.get("cycle_time_utc"),
                     "issued_at_utc": manifest.get("issued_at_utc"), "forecast_hours": manifest.get("forecast_hours"),
                     "missing_hours": manifest.get("missing_hours"), "variables": len(manifest.get("variables", {})),
                     "complete": manifest.get("complete", True), "pending_hours": manifest.get("pending_hours", []),
                     "hours_expected": manifest.get("hours_expected"), "step_labels": manifest.get("step_labels")})
    return {"latest": store._latest_cycle_id(root), "runs": runs}


@router.get("/runs/{run}", dependencies=[Depends(data_limiter)])
def get_manifest(run: str):
    return _manifest(_resolve(run))


@router.get("/runs/{run}/zarr/{path:path}", dependencies=[Depends(chunk_limiter)])
def get_zarr_file(run: str, path: str):
    """One file of the Zarr store (.zmetadata, <array>/.zarray, <array>/<chunk>)."""
    run = _resolve(run)
    base = (rtma_analysis.root_dir() / run / "forecast.zarr").resolve()
    target = (base / path).resolve()
    if base not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    finished = (store.read_manifest(run, rtma_analysis.root_dir()) or {}).get("complete", True)
    return FileResponse(target, media_type="application/json" if target.name.startswith(".z") else "application/octet-stream",
                        headers={"Cache-Control": "public, max-age=3600" if finished else "public, max-age=60"})


@router.get("/runs/{run}/point", dependencies=[Depends(point_limiter)])
def get_point(run: str, lat: float = Query(..., ge=-90, le=90), lon: float = Query(..., ge=-180, le=180)):
    run = _resolve(run)
    manifest = _manifest(run)
    zarr_dir = rtma_analysis.root_dir() / run / "forecast.zarr"
    if not zarr_dir.is_dir():
        raise HTTPException(status_code=404, detail="No analysis for that day")
    result = live_store.point_series(zarr_dir, manifest, lat, lon, store.MAX_POINT_DISTANCE_KM)
    if result.get("error"):
        raise HTTPException(status_code=404, detail="Point is outside the analysis grid")
    result["label"] = manifest["label"]
    result["step_labels"] = manifest.get("step_labels")
    result["units"] = {name: spec["units"] for name, spec in manifest["variables"].items()}
    return result
