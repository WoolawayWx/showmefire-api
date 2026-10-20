"""Read-only access to the hour-by-hour HRRR forecast product (12z run).

Update status, Zarr chunks, the NetCDF copy, per-hour images and a point time series. No /api prefix.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from core.rate_limit import RateLimiter
from services.hrrr_hourly import BY_NAME, FORECAST_HOURS, images_dir, root_dir
from services.hrrr_hourly import job, render, store

CYCLE_PATTERN = re.compile(r"^\d{8}_\d{2}z$")
# Opening a Zarr store is dozens of small requests, so this limit is higher than the JSON routes.
chunk_limiter = RateLimiter("hrrr_hourly_chunks", 600)
data_limiter = RateLimiter("hrrr_hourly", 60)
point_limiter = RateLimiter("hrrr_hourly_point", 30)
status_limiter = RateLimiter("hrrr_hourly_status", 120)  # front ends poll this while an update is in progress

router = APIRouter(prefix="/hrrr-hourly", tags=["hrrr-hourly"])


def _resolve_cycle(cycle: str) -> str:
    if cycle == "latest":
        try:
            cycle = json.loads((root_dir() / "latest.json").read_text())["cycle"]
        except (OSError, ValueError, KeyError):
            raise HTTPException(status_code=404, detail="No hourly forecast has been built yet")
    if not CYCLE_PATTERN.fullmatch(cycle):
        raise HTTPException(status_code=400, detail="cycle must look like 20261016_12z (or 'latest')")
    return cycle


def _manifest(cycle: str) -> dict:
    manifest = store.read_manifest(cycle)
    if manifest is None:
        raise HTTPException(status_code=404, detail="Unknown cycle")
    return manifest


@router.get("/status", dependencies=[Depends(status_limiter)])
def get_status():
    """Update state for the front end: waiting_for_hrrr | waiting_for_forecast | building | rendering |
    publishing | complete | partial | failed, with `updating`, `pending`, progress counts and `serving_cycle`."""
    status = job.read_status()
    if not status:
        return {"product": "hrrr_hourly", "state": "idle", "updating": False, "pending": False,
                "serving_cycle": None, "updated_at": None}
    return status


@router.get("/runs", dependencies=[Depends(data_limiter)])
def list_runs():
    runs = []
    for cycle in store.list_cycles():
        manifest = store.read_manifest(cycle) or {}
        runs.append({"cycle": cycle, "label": manifest.get("label"), "cycle_time_utc": manifest.get("cycle_time_utc"),
                     "issued_at_utc": manifest.get("issued_at_utc"), "forecast_hours": manifest.get("forecast_hours"),
                     "missing_hours": manifest.get("missing_hours"), "variables": len(manifest.get("variables", {})),
                     "complete": manifest.get("complete", True), "pending_hours": manifest.get("pending_hours", []),
                     "hours_expected": manifest.get("hours_expected")})
    # A run still filling in is listed but is not "latest" until enough of it is in (progressive.LATEST_AFTER_HOURS).
    return {"latest": store._latest_cycle_id(root_dir()) or next((r["cycle"] for r in runs if r["complete"]), None), "runs": runs}


@router.get("/runs/{cycle}", dependencies=[Depends(data_limiter)])
def get_manifest(cycle: str):
    return _manifest(_resolve_cycle(cycle))


@router.get("/runs/{cycle}/zarr/{path:path}", dependencies=[Depends(chunk_limiter)])
def get_zarr_file(cycle: str, path: str):
    """One file of the Zarr store (.zmetadata, .zattrs, <array>/.zarray, <array>/<chunk>)."""
    cycle = _resolve_cycle(cycle)
    base = (root_dir() / cycle / "forecast.zarr").resolve()
    target = (base / path).resolve()
    if base not in target.parents or not target.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    is_meta = target.name.startswith(".z")
    # A run that is still being filled in can rewrite a file (the peak product's days), so it is not cached for long.
    finished = (store.read_manifest(cycle) or {}).get("complete", True)
    return FileResponse(target, media_type="application/json" if is_meta else "application/octet-stream",
                        headers={"Cache-Control": "public, max-age=300" if finished else "public, max-age=30"})


@router.get("/runs/{cycle}/netcdf", dependencies=[Depends(data_limiter)])
def get_netcdf(cycle: str):
    cycle = _resolve_cycle(cycle)
    path = store.nc_path(cycle)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Unknown cycle")
    return FileResponse(path, media_type="application/x-netcdf", filename=path.name)


@router.get("/runs/{cycle}/point", dependencies=[Depends(point_limiter)])
def get_point(cycle: str, lat: float = Query(..., ge=-90, le=90), lon: float = Query(..., ge=-180, le=180)):
    cycle = _resolve_cycle(cycle)
    manifest = _manifest(cycle)
    result = store.point_series(cycle, lat, lon)
    if result is None:
        raise HTTPException(status_code=404, detail="Unknown cycle")
    if result.get("error"):
        raise HTTPException(status_code=404, detail="Point is outside the forecast grid")
    result["label"] = manifest["label"]
    result["units"] = {name: spec["units"] for name, spec in manifest["variables"].items()}
    return result


@router.get("/runs/{cycle}/image/{element}/{hour}", dependencies=[Depends(chunk_limiter)])
def get_image(cycle: str, element: str, hour: int):
    cycle = _resolve_cycle(cycle)
    if element not in BY_NAME or hour not in FORECAST_HOURS:
        raise HTTPException(status_code=404, detail="Unknown element or forecast hour")
    path = render.image_path(cycle, element, hour, images_dir())
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image not available")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "public, max-age=300"})
