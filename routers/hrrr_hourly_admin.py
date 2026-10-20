"""Admin tools for the hourly HRRR product: choose which HRRR runs are built (or pause it), see recent builds, make a
map image on demand and read build timings.

The build only writes the Zarr/NetCDF store; images are drawn here, from that stored data, when an admin asks.
"""
from __future__ import annotations

import logging
import os
import threading
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from core.rate_limit import RateLimiter
from core.security import require_admin
from routers.hrrr_hourly import _manifest, _resolve_cycle
from services.hrrr_hourly import BY_NAME, CYCLE_HOUR, ELEMENTS, FORECAST_HOURS, enabled_cycle_hours, images_dir, keep_days
from services.hrrr_hourly import controls, fetch, job, render, store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/hrrr-hourly", tags=["hrrr-hourly-admin"])
render_limiter = RateLimiter("hrrr_hourly_render", 30)
# matplotlib's pyplot state is not thread-safe and a render uses a core for a couple of seconds, so one at a time.
_render_lock = threading.Lock()
LOCK_WAIT_SECONDS = 60


class SettingsUpdate(BaseModel):
    """The whole settings document: `cycle_hours` (UTC hours to build, or null for the default of all 24) and `paused`."""
    cycle_hours: Optional[list[int]] = Field(default=None, max_length=24)
    paused: bool = False


def _settings_view() -> dict:
    return {
        "settings": controls.load_settings(),
        "effective_cycle_hours": list(enabled_cycle_hours()),
        "source": controls.cycle_hours_source(),
        "env_cycle_hours": os.environ.get("SMF_HRRR_HOURLY_CYCLE_HOURS") or None,
        "daily_forecast_cycle_hour": CYCLE_HOUR,
        "keep_days": keep_days(),
        "backfill_cycles": job.BACKFILL_CYCLES,
        "status": job.read_status(),
    }


@router.get("/settings")
def get_settings(_admin: str = Depends(require_admin)):
    """Which cycles are built, whether the product is paused, and where the cycle list comes from (the admin's saved
    settings, the SMF_HRRR_HOURLY_CYCLE_HOURS environment variable, or the default of all 24)."""
    return _settings_view()


@router.put("/settings")
def put_settings(body: SettingsUpdate, admin: str = Depends(require_admin)):
    """Save the settings. Takes effect on the next scheduler tick (within 10 minutes): a build that is already running
    finishes. `cycle_hours: null` clears the override. Pausing stops new builds from starting."""
    try:
        controls.save_settings(body.cycle_hours, body.paused, admin)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return _settings_view()


@router.get("/history")
def get_history(limit: int = Query(50, ge=1, le=500), _admin: str = Depends(require_admin)):
    """Recent build outcomes, newest first: cycle, complete/partial/failed, runtime and any error."""
    return {"events": controls.recent_events(limit)}


@router.get("/coverage")
def get_coverage(hours: int = Query(72, ge=6, le=168), _admin: str = Depends(require_admin)):
    """The last `hours` hourly cycles, newest first, each with whether it is enabled and whether it has been built.

    Starts at the newest cycle old enough to be fully published."""
    enabled = set(enabled_cycle_hours())
    cycles = []
    for cycle_time in fetch.cycle_candidates(datetime.now(timezone.utc), lookback=hours - 1, hours=range(24)):
        cycle = job.cycle_id(cycle_time)
        manifest = store.read_manifest(cycle)
        cycles.append({
            "cycle": cycle, "cycle_time_utc": cycle_time.strftime("%Y-%m-%dT%H:%M:%SZ"), "hour_utc": cycle_time.hour,
            "enabled": cycle_time.hour in enabled, "built": manifest is not None,
            "partial": bool(manifest and manifest.get("missing_hours")),
            "building": bool(manifest and not manifest.get("complete", True)),
            "runtime_sec": (manifest or {}).get("runtime_sec")})
    return {"cycles": cycles, "enabled_cycle_hours": sorted(enabled)}


@router.get("/elements")
def list_elements(_admin: str = Depends(require_admin)):
    """Elements that can be drawn as an image, for the admin picker."""
    return {"hours": list(FORECAST_HOURS),
            "elements": [{"name": e.name, "long_name": e.long_name, "units": e.units}
                         for e in ELEMENTS if e.image]}


@router.get("/runs/{cycle}/images")
def list_images(cycle: str, _admin: str = Depends(require_admin)):
    """Images already drawn for a cycle, as {element: [hours]}."""
    cycle = _resolve_cycle(cycle)
    _manifest(cycle)
    folder = images_dir() / cycle
    drawn: dict[str, list[int]] = {}
    for element in (e for e in ELEMENTS if e.image):
        hours = [h for h in FORECAST_HOURS if render.image_path(cycle, element.name, h).is_file()]
        if hours:
            drawn[element.name] = hours
    return {"cycle": cycle, "dir": str(folder), "images": drawn}


@router.post("/runs/{cycle}/image/{element}/{hour}", dependencies=[Depends(render_limiter)])
def make_image(cycle: str, element: str, hour: int, force: bool = Query(False, description="Draw again even if "
               "an image exists"), _admin: str = Depends(require_admin)):
    """Draw the PNG for one element and hour from the stored data and return it.

    An image that already exists is returned as is (X-Rendered: false) unless `force=true`. The public
    GET /hrrr-hourly/runs/{cycle}/image/{element}/{hour} serves the saved file afterwards."""
    cycle = _resolve_cycle(cycle)
    _manifest(cycle)
    spec = BY_NAME.get(element)
    if spec is None or not spec.image or hour not in FORECAST_HOURS:
        raise HTTPException(status_code=404, detail="Unknown element or forecast hour")
    if not _render_lock.acquire(timeout=LOCK_WAIT_SECONDS):
        raise HTTPException(status_code=503, detail="Another image is being drawn; try again shortly",
                            headers={"Retry-After": "10"})
    try:
        path, rendered, seconds = render.render_from_store(cycle, element, hour, force=force)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception:
        logger.exception("hrrr_hourly on-demand image %s %s f%02d failed", cycle, element, hour)
        raise HTTPException(status_code=500, detail="Could not draw the image")
    finally:
        _render_lock.release()
    return FileResponse(path, media_type="image/png",
                        headers={"X-Rendered": str(rendered).lower(), "X-Render-Seconds": str(seconds),
                                 "Cache-Control": "no-store"})


@router.get("/runs/{cycle}/timings")
def get_timings(cycle: str, _admin: str = Depends(require_admin)):
    """Where the build's time went: per-phase totals, per-hour fetch/derive times and per-field extras times."""
    cycle = _resolve_cycle(cycle)
    timings = store.read_timings(cycle)
    if timings is None:
        raise HTTPException(status_code=404, detail="No timings recorded for this cycle")
    return timings
