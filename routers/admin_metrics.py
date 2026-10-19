"""Admin-only API request metrics, for the website's API metrics page (see core/metrics.py)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from core.metrics import metrics
from core.security import require_admin
from core.server_metrics import server_metrics

router = APIRouter(prefix="/admin/metrics", tags=["admin-metrics"])


@router.get("/api")
def api_metrics(window: int = Query(60, ge=5, le=1440, description="Minutes to cover: 5 to 1440 (24 hours)"),
                _admin: str = Depends(require_admin)):
    """Request volume, errors, latency percentiles and bytes for the last `window` minutes: an overview, a time series
    (1, 5 or 15 minute points depending on the window) and a table of routes by template. Counts only this API
    process, from its start (a snapshot is reloaded after a restart, up to 24 hours back)."""
    return metrics.snapshot(window)


@router.get("/server")
def server_resource_metrics(window: int = Query(60, ge=5, le=1440, description="Minutes to cover: 5 to 1440 (24 hours)"),
                            _admin: str = Depends(require_admin)):
    """CPU, load, memory, swap, disk and network rates, the API process, the container's limits, disk volumes, the size of
    the big data folders, the busiest processes and any warnings, sampled every 10 seconds (see core/server_metrics.py).
    In Docker the CPU, memory and disk figures are the host's; the process list is the API container's."""
    return server_metrics.snapshot(window)
