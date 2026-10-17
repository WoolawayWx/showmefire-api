"""Admin tools for the hourly HRRR product: make a map image on demand and read build timings.

The build only writes the Zarr/NetCDF store; images are drawn here, from that stored data, when an admin asks.
"""
from __future__ import annotations

import logging
import threading

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse

from core.rate_limit import RateLimiter
from core.security import require_admin
from routers.hrrr_hourly import _manifest, _resolve_cycle
from services.hrrr_hourly import BY_NAME, ELEMENTS, FORECAST_HOURS, images_dir
from services.hrrr_hourly import render, store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin/hrrr-hourly", tags=["hrrr-hourly-admin"])
render_limiter = RateLimiter("hrrr_hourly_render", 30)
# matplotlib's pyplot state is not thread-safe and a render uses a core for a couple of seconds, so one at a time.
_render_lock = threading.Lock()
LOCK_WAIT_SECONDS = 60


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
