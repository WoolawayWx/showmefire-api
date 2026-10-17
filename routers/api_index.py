"""Index of the public read-only data API: what exists, who may call it, and the limits."""
from __future__ import annotations

import html
import os
from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse

from core.rate_limit import KEY_TIER_MULTIPLIER, WINDOW_SECONDS, RateLimiter
from routers.forecast_09z_metrics import router as metrics_09z_router
from routers.verification import station_limiter, verification_limiter

index_limiter = RateLimiter("api_index", 30)
router = APIRouter(tags=["api-index"], dependencies=[Depends(index_limiter)])

_STATION_LEVEL = "Station-level rows"

# (method, path, group, summary, params, access)
_ROUTES: List[tuple] = [
    ("GET", "/verification/history", "Station verification",
     "Daily MAE/bias per variable (station network) plus forecast-vs-RTMA MAE/bias for temp, RH, wind.",
     "limit (1-366, default 90)", "public"),
    ("GET", "/verification/report/{date}", "Station verification",
     "Full report for a date: metrics, directional bias, confusion matrices, neighborhood scores, raster links.",
     "date YYYY-MM-DD", "public"),
    ("GET", "/verification/report/{date}/comparisons", "Station verification",
     "Hourly forecast/observation pairs per station.",
     "date; station; limit (max 2000)", "station"),
    ("GET", "/verification/peak-history", "Peak class",
     "Daily statewide peak class: forecast vs station-observed vs RTMA, with bias for each.",
     "limit (1-366, default 90)", "public"),
    ("GET", "/verification/rtma/history", "RTMA verification",
     "Daily forecast-vs-RTMA and station-vs-RTMA MAE/bias, plus gridded peak-class agreement.",
     "limit (1-366, default 90)", "public"),
    ("GET", "/verification/rtma/{date}", "RTMA verification",
     "Station-point and gridded RTMA metrics for a date. Fuel moisture and fire danger are derived from RTMA RH.",
     "date YYYY-MM-DD", "public"),
    ("GET", "/verification/rainfall/diagnostics", "Station verification",
     "Active rainfall/fuel-model verification contract state.", "none", "public"),
    ("GET", "/forecast-09z/metrics/latest", "09z comparison",
     "Latest 09z-vs-12z comparison summary.", "none", "public"),
    ("GET", "/forecast-09z/metrics/history", "09z comparison",
     "History of 09z comparison summaries.", "none", "public"),
    ("GET", "/forecast-09z/metrics/rtma/history", "09z comparison",
     "09z forecast vs RTMA peak history.", "none", "public"),
    ("GET", "/forecast-09z/metrics/rtma/{date}", "09z comparison",
     "09z forecast vs RTMA peak for a date.", "date YYYY-MM-DD", "public"),
    ("GET", "/hrrr-hourly/status", "HRRR hourly forecast",
     "Update state (waiting_for_forecast, building, rendering, complete...) with progress and the cycle being served.",
     "none", "public"),
    ("GET", "/hrrr-hourly/runs", "HRRR hourly forecast",
     "Built cycles of the hour-by-hour HRRR forecast (12z run, f00-f18) and which is latest.", "none", "public"),
    ("GET", "/hrrr-hourly/runs/{cycle}", "HRRR hourly forecast",
     "Manifest: run/issue time, hours, variables with units, grid, statewide hourly series. cycle = 20261016_12z or latest.",
     "cycle", "public"),
    ("GET", "/hrrr-hourly/runs/{cycle}/zarr/{path}", "HRRR hourly forecast",
     "Files of the Zarr store (open the .../zarr URL with a Zarr reader; 600/min for chunk reads).", "cycle; path", "public"),
    ("GET", "/hrrr-hourly/runs/{cycle}/netcdf", "HRRR hourly forecast",
     "Download the NetCDF copy of the cycle.", "cycle", "public"),
    ("GET", "/hrrr-hourly/runs/{cycle}/point", "HRRR hourly forecast",
     "Hourly series of every element at the nearest grid cell (30/min).", "cycle; lat; lon", "public"),
    ("GET", "/hrrr-hourly/runs/{cycle}/image/{element}/{hour}", "HRRR hourly forecast",
     "Map image (PNG) of one element for one forecast hour.", "cycle; element; hour 0-18", "public"),
]


def _station_gate_on() -> bool:
    return os.getenv("VERIFICATION_STATION_DATA_REQUIRES_KEY", "false").lower() == "true"


def _registry() -> Dict[str, Any]:
    per_minute = verification_limiter.per_minute
    station_per_minute = station_limiter.per_minute
    gate = _station_gate_on()
    routes = []
    for method, path, group, summary, params, access in _ROUTES:
        is_station = access == "station"
        routes.append({
            "method": method,
            "path": path,
            "group": group,
            "summary": summary,
            "parameters": params,
            "access": "api key or admin" if is_station and gate else "public",
            "rate_limit_per_minute": station_per_minute if is_station else per_minute,
            "rate_limit_per_minute_with_key": (
                (station_per_minute if is_station else per_minute) * KEY_TIER_MULTIPLIER
            ),
        })
    return {
        "window_seconds": WINDOW_SECONDS,
        "authentication": {
            "public": "No credentials needed; limited per client IP.",
            "api_key": "Send 'Authorization: Bearer <key>' for a higher limit"
                       + (" and station-level rows." if gate else "."),
            "admin": "/api/admin/* routes require an administrator session and are not listed here.",
        },
        "rate_limit_response": "HTTP 429 with a Retry-After header (seconds).",
        "notes": [
            "Bias is forecast minus reference; positive means the forecast is higher.",
            "RTMA fuel moisture and fire danger are derived from RTMA RH, not observed.",
            "RTMA hourly files are only cached 7 days; metrics are computed at report time and stored.",
            "Reports appear after the 10:00-21:00 Central window closes.",
            "Every route also still answers at its older /api/... path (e.g. /api/forecast-09z/metrics/latest).",
        ],
        "routes": routes,
    }


@router.get("/routes")
def list_api_routes() -> Dict[str, Any]:
    return _registry()


@router.get("/routes/page", response_class=HTMLResponse)
def api_routes_page() -> HTMLResponse:
    data = _registry()
    e = html.escape
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for route in data["routes"]:
        groups.setdefault(route["group"], []).append(route)
    sections = []
    for group, items in groups.items():
        rows = "".join(
            f"<tr><td><code>{e(r['method'])} {e(r['path'])}</code></td><td>{e(r['summary'])}</td>"
            f"<td>{e(r['parameters'])}</td><td>{e(r['access'])}</td>"
            f"<td>{r['rate_limit_per_minute']}/min<br><small>{r['rate_limit_per_minute_with_key']}/min with key</small></td></tr>"
            for r in items
        )
        sections.append(
            f"<h2>{e(group)}</h2><table><thead><tr><th>Route</th><th>Returns</th><th>Parameters</th>"
            f"<th>Access</th><th>Limit</th></tr></thead><tbody>{rows}</tbody></table>"
        )
    auth = "".join(f"<li><b>{e(k)}</b>: {e(v)}</li>" for k, v in data["authentication"].items())
    notes = "".join(f"<li>{e(n)}</li>" for n in data["notes"])
    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Show Me Fire - API Routes</title>
<style>
body{{font-family:Arial,sans-serif;max-width:1100px;margin:0 auto;padding:20px;background:#f5f5f5;color:#222}}
table{{width:100%;border-collapse:collapse;background:#fff;box-shadow:0 2px 4px rgba(0,0,0,.1)}}
th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid #eee;vertical-align:top;font-size:14px}}
th{{background:#fafafa}} code{{font-size:13px}} small{{color:#666}}
.box{{background:#fff;padding:12px 20px;border-radius:8px;box-shadow:0 2px 4px rgba(0,0,0,.1);margin:16px 0}}
@media (max-width:700px){{table{{display:block;overflow-x:auto}}}}
</style></head><body>
<h1>Show Me Fire API</h1>
<p>Read-only verification data. JSON version: <a href="/routes">/routes</a></p>
<div class="box"><h3>Access</h3><ul>{auth}<li>Exceeding a limit returns HTTP 429 with a Retry-After header.</li></ul></div>
{''.join(sections)}
<div class="box"><h3>Notes</h3><ul>{notes}</ul></div>
</body></html>"""
    return HTMLResponse(page)
