"""Where the burn check gets its facts. Each function returns data or raises; the caller decides what a failure means (it never
treats a failure as "fine"). Heavy imports are inside the functions so the rules and geocoder stay light and testable."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

KT_TO_MPH = 1.15078
# The only hourly HRRR elements the burn check reads; asking for just these skips decompressing the other ~23.
OUTLOOK_ELEMENTS = ("temp_f", "rh_pct", "wind_speed_mph", "wind_gust_mph", "fire_danger_est")


def county(lat: float, lon: float) -> tuple[Optional[str], Optional[str]]:
    from services.county_lookup import county_for_point

    return county_for_point(lat, lon)


def current_conditions(lat: float, lon: float) -> Optional[dict]:
    """The latest RTMA analysis at the point: temp_f, rh_pct, wind_mph, fire_danger (0-4), fuel_moisture_pct, precip_mm, observed_at."""
    from routers import conditions

    values: dict = {}
    observed: Optional[str] = None
    for product in conditions.PRODUCTS:
        grid = conditions._grid(product)
        raw = conditions._sample(grid, lat, lon) if grid else None
        values[product.key] = raw
        if grid and grid.observed_at and (observed is None or grid.observed_at < observed):
            observed = grid.observed_at
    if all(values.get(key) is None for key in ("temperature", "rh", "wind", "fire_danger")):
        return None
    wind = values.get("wind")
    danger = values.get("fire_danger")
    return {
        "observed_at": observed,
        "temp_f": values.get("temperature"), "rh_pct": values.get("rh"),
        "wind_mph": None if wind is None else wind * KT_TO_MPH,
        "fire_danger": None if danger is None else int(round(danger)),
        "fuel_moisture_pct": values.get("fuel_moisture"), "precip_mm": values.get("precipitation"),
    }


def active_burn_ban(county_fips: str, now: datetime) -> Optional[dict]:
    """The confirmed, in-effect burn ban for the county, or None. Raises if the ban list cannot be read."""
    from core.database import list_active_burn_bans

    for ban in list_active_burn_bans(now=now.astimezone(timezone.utc).replace(tzinfo=None)):
        if ban.get("county_fips") == county_fips:
            return {"county_name": ban.get("county_name") or "this", "effective_at": ban.get("effective_at"),
                    "expires_at": ban.get("expires_at") or None, "note": ban.get("public_note") or None,
                    "proof_url": ban.get("proof_url") or None}
    return None


def fire_weather_alerts(county_fips: str, now: datetime, max_age_minutes: float,
                        path: Optional[Path] = None) -> tuple[list[dict], bool]:
    """(active Red Flag Warnings / Fire Weather Watches covering the county, whether the alert file is fresh).

    The alert file is refreshed every 5 minutes; an old or missing one is reported as not fresh instead of as "no alerts"."""
    from services import mobile_content

    path = path or mobile_content.ACTIVE_ALERTS_PATH
    try:
        age = time.time() - Path(path).stat().st_mtime
    except OSError:
        return [], False
    if age > max_age_minutes * 60:
        return [], False
    alerts = mobile_content.active_fire_weather_alerts(path, now)
    return [{"event": a["event"], "headline": a["headline"], "expires": a.get("expires"), "area": a["areaDescription"]}
            for a in alerts if county_fips in (a.get("countyFips") or [])], True


def hourly_outlook(lat: float, lon: float, now: datetime, max_age_hours: float, root: Optional[Path] = None) -> Optional[dict]:
    """The latest hourly HRRR forecast at the point, from this server's hourly product, or None if there is none or it is old."""
    from services.hrrr_hourly import root_dir, store

    root = Path(root or root_dir())
    cycle = store._latest_cycle_id(root)
    manifest = store.read_manifest(cycle, root) if cycle else None
    if not manifest:
        return None
    cycle_time = datetime.strptime(manifest["cycle_time_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    if (now - cycle_time).total_seconds() > max_age_hours * 3600:
        return None
    point = store.point_series(cycle, lat, lon, root, names=OUTLOOK_ELEMENTS)
    if not point or point.get("error"):
        return None
    series = point["series"]
    rows = []
    for index, stamp in enumerate(point["valid_times_utc"]):
        def at(name):
            values = series.get(name) or []
            return values[index] if index < len(values) else None
        rows.append({"time_utc": datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc),
                     "temp_f": at("temp_f"), "rh_pct": at("rh_pct"), "wind_mph": at("wind_speed_mph"),
                     "gust_mph": at("wind_gust_mph"), "fire_danger": at("fire_danger_est")})
    return {"cycle": cycle, "cycle_time_utc": manifest["cycle_time_utc"], "rows": rows}
