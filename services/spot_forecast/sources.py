"""Data for the spot forecast. County, alerts, burn ban and current conditions come from the burn check's sources; this adds the
hourly HRRR series with the extra elements a spot forecast needs. Raises or returns None on failure; the caller says so."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from services.burn_check.sources import active_burn_ban, county, fire_weather_alerts  # noqa: F401 - re-exported for the service

ELEMENTS = ("temp_f", "rh_pct", "wind_speed_mph", "wind_gust_mph", "wind_dir_deg", "mixing_height_ft", "transport_speed_mph",
            "ventilation_kt_ft", "haines", "hdw", "inversion_strength_f", "inversion_top_ft",
            "smoke_surface_ugm3")


def hourly(lat: float, lon: float, now: datetime, max_age_hours: float, root: Optional[Path] = None) -> Optional[dict]:
    """The latest HRRR hourly series at the point, or None if there is none or it is too old."""
    from services.hrrr_hourly import root_dir, store

    root = Path(root or root_dir())
    cycle = store._latest_cycle_id(root)
    manifest = store.read_manifest(cycle, root) if cycle else None
    if not manifest:
        return None
    cycle_time = datetime.strptime(manifest["cycle_time_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    if (now - cycle_time).total_seconds() > max_age_hours * 3600:
        return None
    point = store.point_series(cycle, lat, lon, root, names=ELEMENTS)
    if not point or point.get("error"):
        return None
    series = point["series"]
    rows = []
    for index, stamp in enumerate(point["valid_times_utc"]):
        def at(name):
            values = series.get(name) or []
            return values[index] if index < len(values) else None
        rows.append({"time_utc": datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc),
                     "temp_f": at("temp_f"), "rh_pct": at("rh_pct"), "wind_mph": at("wind_speed_mph"), "gust_mph": at("wind_gust_mph"),
                     "wind_dir_deg": at("wind_dir_deg"), "mixing_height_ft": at("mixing_height_ft"),
                     "transport_mph": at("transport_speed_mph"), "ventilation_kt_ft": at("ventilation_kt_ft"),
                     "haines": at("haines"), "hdw": at("hdw"),
                     "inversion_strength_f": at("inversion_strength_f"), "inversion_top_ft": at("inversion_top_ft"),
                     "smoke_ugm3": at("smoke_surface_ugm3")})
    return {"cycle": cycle, "cycle_time_utc": manifest["cycle_time_utc"], "rows": rows}
