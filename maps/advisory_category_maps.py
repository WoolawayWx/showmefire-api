"""Oklahoma-Mesonet-style Advisories: splits Missouri's combined active NWS
alerts feed (gis/active.json, refreshed every 5 minutes by
alerts/activemoalerts.py's run_active_mo_alerts) into one map per category -
Severe Weather, Flood, Fire, Wind, Heat, Winter Weather, Frost/Freeze,
Visibility - reading that one already-fetched file once rather than making 8
separate external calls.
"""
import hashlib
import json
import logging
import os
import sys
import textwrap
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

logger = logging.getLogger(__name__)

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

# Official NWS hazard map colors (https://www.weather.gov/help-map), keyed by
# event name. Events not listed fall back to the category color.
NWS_EVENT_COLORS = {
    "Tornado Warning": "#FF0000",
    "Tornado Watch": "#FFFF00",
    "Severe Thunderstorm Warning": "#FFA500",
    "Severe Thunderstorm Watch": "#DB7093",
    "Special Weather Statement": "#FFE4B5",
    "Severe Weather Statement": "#00FFFF",
    "Flash Flood Warning": "#8B0000",
    "Flash Flood Watch": "#2E8B57",
    "Flash Flood Statement": "#8B0000",
    "Flood Warning": "#00FF00",
    "Flood Watch": "#2E8B57",
    "Flood Advisory": "#00FF7F",
    "Flood Statement": "#00FF00",
    "Red Flag Warning": "#FF1493",
    "Fire Weather Watch": "#FFDEAD",
    "High Wind Warning": "#DAA520",
    "High Wind Watch": "#B8860B",
    "Wind Advisory": "#D2B48C",
    "Lake Wind Advisory": "#D2B48C",
    "Extreme Wind Warning": "#FF8C00",
    "Excessive Heat Warning": "#C71585",
    "Excessive Heat Watch": "#800000",
    "Extreme Heat Warning": "#C71585",
    "Extreme Heat Watch": "#800000",
    "Heat Advisory": "#FF7F50",
    "Winter Storm Warning": "#FF69B4",
    "Winter Storm Watch": "#4682B4",
    "Winter Weather Advisory": "#7B68EE",
    "Blizzard Warning": "#FF4500",
    "Ice Storm Warning": "#8B008B",
    "Freezing Rain Advisory": "#DA70D6",
    "Wind Chill Warning": "#B0C4DE",
    "Wind Chill Watch": "#5F9EA0",
    "Wind Chill Advisory": "#AFEEEE",
    "Extreme Cold Warning": "#0000FF",
    "Extreme Cold Watch": "#0000FF",
    "Cold Weather Advisory": "#AFEEEE",
    "Freeze Warning": "#483D8B",
    "Freeze Watch": "#00FFFF",
    "Frost Advisory": "#6495ED",
    "Dense Fog Advisory": "#708090",
    "Dense Smoke Advisory": "#F0E68C",
    "Air Quality Alert": "#808080",
}

EVENT_DESCRIPTIONS = {
    "Tornado Warning": "A tornado is occurring or imminent. Take shelter now.",
    "Tornado Watch": "Conditions favor tornado development in and near the watch area.",
    "Severe Thunderstorm Warning": "Severe storm with damaging winds or large hail is occurring or imminent.",
    "Severe Thunderstorm Watch": "Conditions favor severe thunderstorms in and near the watch area.",
    "Special Weather Statement": "Significant weather that does not meet warning criteria.",
    "Flash Flood Warning": "Flash flooding is occurring or imminent. Move to higher ground.",
    "Flash Flood Watch": "Conditions favor flash flooding in and near the watch area.",
    "Flood Warning": "Flooding is occurring or expected soon along rivers or in low-lying areas.",
    "Flood Watch": "Conditions favor flooding. Be prepared to act if warnings are issued.",
    "Flood Advisory": "Minor flooding of low-lying areas and poor drainage is expected.",
    "Red Flag Warning": "Critical fire weather: strong wind, low humidity and dry fuels.",
    "Fire Weather Watch": "Critical fire weather conditions are possible within 12 to 72 hours.",
    "High Wind Warning": "Sustained winds of 40+ mph or gusts of 58+ mph are expected.",
    "High Wind Watch": "High winds are possible. Prepare for damaging gusts.",
    "Wind Advisory": "Sustained winds of 31-39 mph or gusts of 46-57 mph are expected.",
    "Excessive Heat Warning": "Dangerously hot conditions. Heat illness is likely.",
    "Extreme Heat Warning": "Dangerously hot conditions. Heat illness is likely.",
    "Heat Advisory": "Hot conditions may cause heat illness, especially for sensitive groups.",
    "Winter Storm Warning": "Heavy snow, sleet or ice will make travel dangerous.",
    "Winter Storm Watch": "Heavy winter precipitation is possible within 12 to 48 hours.",
    "Winter Weather Advisory": "Snow, sleet or freezing rain will cause travel difficulty.",
    "Blizzard Warning": "Heavy snow with strong wind and near-zero visibility.",
    "Ice Storm Warning": "Significant ice accumulation will damage trees and power lines.",
    "Freeze Warning": "Temperatures at or below 32\u00b0F will damage crops and sensitive plants.",
    "Freeze Watch": "A freeze is possible. Protect sensitive plants.",
    "Frost Advisory": "Frost is expected and may harm sensitive plants.",
    "Wind Chill Warning": "Dangerously low wind chills can cause frostbite quickly.",
    "Wind Chill Advisory": "Cold wind chills may cause discomfort and frostbite risk.",
    "Extreme Cold Warning": "Dangerously cold temperatures or wind chills are expected.",
    "Dense Fog Advisory": "Visibility of one quarter mile or less. Drive with caution.",
    "Dense Smoke Advisory": "Smoke is reducing visibility to one quarter mile or less.",
    "Air Quality Alert": "Air pollution may be unhealthy for sensitive groups.",
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
    fallback_color = CATEGORY_COLORS[category]
    event_names = []
    # Draw lower-priority (alphabetical) first; warnings sit on top of watches.
    def _draw_order(feature):
        event = (feature.get("properties") or {}).get("event", "")
        return (0 if "Warning" not in event else 1, event)
    for feature in sorted(features, key=_draw_order):
        try:
            geometry = shape(feature["geometry"])
        except (KeyError, ValueError):
            continue
        event = (feature.get("properties") or {}).get("event", "Alert")
        color = NWS_EVENT_COLORS.get(event, fallback_color)
        ax.add_geometries(
            [geometry], crs=data_crs, facecolor=color, edgecolor=color,
            alpha=0.6, linewidth=1.5, zorder=7,
        )
        event_names.append(event)

    label = CATEGORY_LABELS[category]
    if event_names:
        from matplotlib.patches import Patch
        handles = [
            Patch(
                facecolor=NWS_EVENT_COLORS.get(name, fallback_color), edgecolor="none",
                label=textwrap.fill(f"{name}: {EVENT_DESCRIPTIONS.get(name, 'Active NWS alert.')}", 44),
            )
            for name in sorted(set(event_names))
        ]
        legend = fig.legend(
            handles=handles, loc="upper right", bbox_to_anchor=(0.99, 0.50),
            bbox_transform=fig.transFigure, fontsize=11, frameon=False,
            labelspacing=1.1, handlelength=1.6, handleheight=1.4,
        )
        legend.set_zorder(20)
        description = "Active alerts"
    else:
        description = "No active alerts in this category."

    sp.add_branding(
        fig, ax,
        title=f"Missouri {label} Advisories",
        subtitle=f"Valid Time: {pd.Timestamp.now(tz='America/Chicago').strftime('%Y-%m-%d %H:%M CT')}",
        description=(
            "Data Source: National Weather Service\n\n"
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


def category_signature(alerts_geojson: dict, category: str) -> str:
    """Stable hash of what a category map would show. Changes when an alert is
    issued, cancelled, expires, is re-issued/updated, or its area changes."""
    now = pd.Timestamp.now(tz="UTC")
    rows = []
    for feature in filter_alerts_by_category(alerts_geojson, category)["features"]:
        props = feature.get("properties") or {}
        expires = pd.to_datetime(props.get("ends") or props.get("expires"), utc=True, errors="coerce")
        if pd.notna(expires) and expires <= now:
            continue  # the next poll drops it anyway; don't wait for NWS
        rows.append((
            str(props.get("id") or ""), str(props.get("event") or ""),
            str(props.get("sent") or ""), str(props.get("expires") or ""),
            bool(feature.get("geometry")),
        ))
    return hashlib.sha1(json.dumps(sorted(rows)).encode()).hexdigest()


def refresh_changed_advisory_maps(alerts_geojson: dict) -> list[str]:
    """Re-render (and push to the CDN) only the categories whose alerts changed
    since their map was last drawn. Called after each alert poll; maps.sh still
    re-renders everything every 15 minutes so Valid Time stays current."""
    changed = []
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    for category, suffix in FILENAME_SUFFIXES.items():
        signature = category_signature(alerts_geojson, category)
        sig_path = STATUS_DIR / f"advisory_{category}.signature"
        out_png = IMAGES_DIR / f"mo-advisory-{suffix}.png"
        try:
            if out_png.is_file() and sig_path.read_text() == signature:
                continue
        except OSError:
            pass
        count = render_advisory_category(
            category, alerts_geojson, out_png, STATUS_DIR / f"advisory_{category}.json",
        )
        sig_path.write_text(signature)
        logger.info("Advisory map '%s' refreshed on alert change (%s active)", category, count)
        print(f"Advisory map '{category}' refreshed on alert change ({count} active)")
        changed.append(out_png)
    if changed:
        try:
            from scripts.upload_cdn import upload_to_cdn
            upload_to_cdn(
                changed, [f"latest/{path.name}" for path in changed],
                ["image/png"] * len(changed), ["max-age=300"] * len(changed),
            )
        except Exception as exc:
            print(f"Advisory map CDN upload failed: {exc}")
    return [path.name for path in changed]


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
        (STATUS_DIR / f"advisory_{category}.signature").write_text(category_signature(alerts_geojson, category))
        print(f"Advisory category '{category}' updated ({count} active alerts)")
    runtime_sec = time.time() - start_time
    print(f"All advisory category maps updated ({total_active} total active alerts, {runtime_sec:.2f}s)")
