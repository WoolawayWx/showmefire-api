"""Oklahoma-Mesonet-style Advisories: splits Missouri's combined active NWS
alerts feed (gis/active.json, refreshed every 5 minutes by
alerts/activemoalerts.py's run_active_mo_alerts) into one map per category -
Severe Weather, Flood, Fire, Wind, Heat, Winter Weather, Frost/Freeze,
Visibility - reading that one already-fetched file once rather than making 8
separate external calls.
"""
import json
import os
import sys
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_DIR))

from alerts.advisory_categories import CATEGORY_LABELS, filter_alerts_by_category
from maps import station_plot as sp

STATUS_DIR = PROJECT_DIR / "data" / "current_conditions" / "status"
IMAGES_DIR = PROJECT_DIR / "images"

# Mirrors services/graphic_renderer.py's _load_alert_bytes() candidate order:
# the Docker production path first, a project-root-relative fallback for
# local dev second.
ACTIVE_ALERTS_CANDIDATES = [Path("/app/gis/active.json"), PROJECT_DIR / "gis" / "active.json"]

CATEGORY_COLORS = {
    "severe_weather": "#DC2626",
    "flood": "#2563EB",
    "fire": "#F97316",
    "wind": "#0EA5E9",
    "heat": "#B91C1C",
    "winter_weather": "#7DD3FC",
    "frost_freeze": "#38BDF8",
    "visibility": "#9CA3AF",
}

FILENAME_SUFFIXES = {
    "severe_weather": "severe",
    "flood": "flood",
    "fire": "fire",
    "wind": "wind",
    "heat": "heat",
    "winter_weather": "winter",
    "frost_freeze": "frostfreeze",
    "visibility": "visibility",
}


def _load_active_alerts() -> dict:
    for candidate in ACTIVE_ALERTS_CANDIDATES:
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return {"type": "FeatureCollection", "features": []}


def render_advisory_category(category: str, alerts_geojson: dict, out_png: Path, status_path: Path) -> int:
    filtered = filter_alerts_by_category(alerts_geojson, category)
    features = [f for f in filtered["features"] if f.get("geometry")]

    fig, ax, data_crs, map_crs, mapdpi = sp.generate_basemap()
    color = CATEGORY_COLORS[category]
    event_names = []
    for feature in features:
        try:
            geometry = shape(feature["geometry"])
        except (KeyError, ValueError):
            continue
        ax.add_geometries(
            [geometry], crs=data_crs, facecolor=color, edgecolor=color,
            alpha=0.45, linewidth=1.5, zorder=7,
        )
        event_names.append((feature.get("properties") or {}).get("event", "Alert"))

    label = CATEGORY_LABELS[category]
    if event_names:
        unique_events = sorted(set(event_names))
        description = "Active alerts:\n" + "\n".join(f"- {name}" for name in unique_events[:8])
    else:
        description = "No active alerts in this category."

    sp.add_branding(
        fig, ax,
        title=f"Missouri {label} Advisories",
        subtitle=f"Valid Time: {pd.Timestamp.now(tz='America/Chicago').strftime('%Y-%m-%d %H:%M CT')}",
        description=(
            "Data Source: National Weather Service (api.weather.gov)\n\n"
            + description
        ),
    )

    sp.save_and_record_status(
        fig, out_png, status_path,
        {
            "product": f"advisory_{category}",
            "status": "updated",
            "active_alert_count": len(features),
            "last_update": pd.Timestamp.now(tz="America/Chicago").strftime("%Y-%m-%d %H:%M CT"),
        },
    )
    return len(features)


if __name__ == "__main__":
    start_time = time.time()
    alerts_geojson = _load_active_alerts()
    total_active = 0
    for category, suffix in FILENAME_SUFFIXES.items():
        count = render_advisory_category(
            category, alerts_geojson,
            IMAGES_DIR / f"mo-advisory-{suffix}.png",
            STATUS_DIR / f"advisory_{category}.json",
        )
        total_active += count
        print(f"Advisory category '{category}' updated ({count} active alerts)")
    runtime_sec = time.time() - start_time
    print(f"All advisory category maps updated ({total_active} total active alerts, {runtime_sec:.2f}s)")
