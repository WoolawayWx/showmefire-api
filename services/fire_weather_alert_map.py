"""Render a static Missouri fire-weather zone alert map (Red Flag Warning / Fire Weather Watch)."""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from zoneinfo import ZoneInfo

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.font_manager as font_manager
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Patch

from core.config import IMAGES_DIR
from services.gis_publisher import PUBLISH_ROOT, publish_vectors
from services.mobile_content import FIRE_ZONES_PATH, active_fire_weather_zone_status

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATE_SHP = PROJECT_ROOT / "maps/shapefiles/MO_State_Boundary/MO_State_Boundary.shp"
ALERT_MAP_PNG = Path(IMAGES_DIR) / "mo-firewx-alerts.png"
ALERT_MAP_GEOJSON = PUBLISH_ROOT / "fire_weather_alerts.geojson"
ALERT_MAP_GPKG = PUBLISH_ROOT / "fire_weather_alerts.gpkg"
STATUS_CACHE_PATH = Path(IMAGES_DIR) / "mo-firewx-alerts.status.json"

MAP_EXTENT = (-95.8, -89.1, 35.8, 40.8)
PIXEL_W = 2048
PIXEL_H = 1152
MAP_DPI = 144
BACKGROUND_COLOR = "#E8E8E8"
WARNING_COLOR = "#B91C1C"
WATCH_COLOR = "#F59E0B"
INACTIVE_COLOR = "#FFFFFF"
ZONE_EDGE_COLOR = "#B6B6B6"
STATE_EDGE_COLOR = "#000000"
CENTRAL_TZ = ZoneInfo("America/Chicago")

_STATUS_COLOR = {"Red Flag Warning": WARNING_COLOR, "Fire Weather Watch": WATCH_COLOR}


def fire_weather_alert_map_public_meta() -> dict:
    updated_at = None
    if ALERT_MAP_PNG.exists():
        updated_at = datetime.fromtimestamp(
            ALERT_MAP_PNG.stat().st_mtime, tz=timezone.utc,
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "image_path": "mo-firewx-alerts.png",
        "url": "/images/mo-firewx-alerts.png",
        "updated_at": updated_at,
    }


def ensure_fire_weather_alert_map() -> dict:
    """Generate the static map if it has never been built."""
    if not ALERT_MAP_PNG.is_file():
        generate_fire_weather_alert_map()
    return fire_weather_alert_map_public_meta()


def _atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    try:
        Path(temporary).write_bytes(source.read_bytes())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def fire_weather_alert_feature_collection(status: dict[str, str] | None = None) -> dict:
    """Return active fire-weather zone alerts joined to zone polygons in EPSG:4326."""
    status = active_fire_weather_zone_status() if status is None else status
    zones = gpd.read_file(FIRE_ZONES_PATH).to_crs("EPSG:4326")
    zones["zone_code"] = zones["ZONE"].astype(str).str.zfill(3)
    features = []
    for _, zone in zones[zones["zone_code"].isin(status)].iterrows():
        event = status[zone["zone_code"]]
        features.append({
            "type": "Feature",
            "geometry": zone.geometry.__geo_interface__,
            "properties": {
                "zone_code": f"MOZ{zone['zone_code']}",
                "zone_name": zone.get("NAME"),
                "event": event,
                "status": "active",
            },
        })
    return {
        "type": "FeatureCollection",
        "metadata": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "feature_count": len(features),
            "join_key": "zone_code",
            "crs": "EPSG:4326",
        },
        "features": features,
    }


def publish_fire_weather_alert_gis(status: dict[str, str] | None = None) -> dict:
    collection = fire_weather_alert_feature_collection(status)
    paths = publish_vectors(
        "fire_weather_alerts", collection["features"],
        generated_at=collection["metadata"]["generated_at"],
    )
    _atomic_copy(paths["geojson"], ALERT_MAP_GEOJSON)
    _atomic_copy(paths["geopackage"], ALERT_MAP_GPKG)
    return {
        "geojson": str(ALERT_MAP_GEOJSON),
        "geopackage": str(ALERT_MAP_GPKG),
        "feature_count": len(collection["features"]),
    }


def _format_central_timestamp(value: datetime | None = None) -> str:
    base = value.astimezone(CENTRAL_TZ) if value else datetime.now(CENTRAL_TZ)
    return base.strftime("%Y-%m-%d %H:%M CT")


def _load_fonts() -> None:
    font_paths = [
        PROJECT_ROOT / "assets/Montserrat/static/Montserrat-Regular.ttf",
        PROJECT_ROOT / "assets/Plus_Jakarta_Sans/static/PlusJakartaSans-Regular.ttf",
        PROJECT_ROOT / "assets/Plus_Jakarta_Sans/static/PlusJakartaSans-Bold.ttf",
    ]
    for font_path in font_paths:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
    plt.rcParams["font.family"] = "Montserrat"


def _create_base_figure():
    data_crs = ccrs.PlateCarree()
    map_crs = ccrs.LambertConformal(central_longitude=-92.45, central_latitude=38.3)

    fig = plt.figure(
        figsize=(PIXEL_W / MAP_DPI, PIXEL_H / MAP_DPI),
        dpi=MAP_DPI,
        facecolor=BACKGROUND_COLOR,
    )
    ax = plt.axes([0, 0, 1, 1], projection=map_crs)
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_extent(MAP_EXTENT, crs=data_crs)
    ax.set_anchor("W")
    plt.subplots_adjust(left=0.05)
    return fig, ax, data_crs


def _read_zones(data_crs) -> gpd.GeoDataFrame:
    zones = gpd.read_file(FIRE_ZONES_PATH)
    if zones.crs != data_crs.proj4_init:
        zones = zones.to_crs(data_crs.proj4_init)
    zones["zone_code"] = zones["ZONE"].astype(str).str.zfill(3)
    return zones


def _add_zone_fills(ax, zones: gpd.GeoDataFrame, status: dict[str, str], data_crs) -> None:
    inactive = zones[~zones["zone_code"].isin(status)]
    if not inactive.empty:
        ax.add_geometries(
            inactive.geometry,
            crs=data_crs,
            facecolor=INACTIVE_COLOR,
            edgecolor="none",
            zorder=6,
        )
    for event, color in (("Fire Weather Watch", WATCH_COLOR), ("Red Flag Warning", WARNING_COLOR)):
        codes = {code for code, active_event in status.items() if active_event == event}
        matched = zones[zones["zone_code"].isin(codes)]
        if not matched.empty:
            ax.add_geometries(
                matched.geometry,
                crs=data_crs,
                facecolor=color,
                edgecolor="none",
                zorder=7,
            )


def _add_boundaries(ax, zones: gpd.GeoDataFrame, data_crs) -> None:
    ax.add_geometries(
        zones.geometry,
        crs=data_crs,
        edgecolor=ZONE_EDGE_COLOR,
        facecolor="none",
        linewidth=1,
        zorder=9,
    )

    if STATE_SHP.exists():
        state = gpd.read_file(STATE_SHP)
        if state.crs != data_crs.proj4_init:
            state = state.to_crs(data_crs.proj4_init)
        ax.add_geometries(
            state.geometry,
            crs=data_crs,
            edgecolor=STATE_EDGE_COLOR,
            facecolor="none",
            linewidth=1.5,
            zorder=10,
        )


def _add_legend(fig) -> None:
    _load_fonts()
    legend_handles = [
        Patch(facecolor=WARNING_COLOR, edgecolor=STATE_EDGE_COLOR, linewidth=0.9, label="Red Flag Warning"),
        Patch(facecolor=WATCH_COLOR, edgecolor=STATE_EDGE_COLOR, linewidth=0.9, label="Fire Weather Watch"),
        Patch(facecolor=INACTIVE_COLOR, edgecolor=ZONE_EDGE_COLOR, linewidth=0.9, label="No active alert"),
    ]
    legend = fig.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.78),
        frameon=True,
        fancybox=False,
        edgecolor="#444444",
        facecolor="#FFFFFF",
        fontsize=12,
        handlelength=1.6,
        handleheight=1.2,
        borderpad=0.9,
        labelspacing=0.85,
        title="Fire Weather Alert Status",
    )
    legend.get_frame().set_linewidth(1.2)
    legend.get_frame().set_alpha(0.96)
    legend.get_title().set_fontsize(13)
    legend.get_title().set_fontweight("bold")


def _add_branding(fig, ax, status: dict[str, str], updated_at: datetime) -> None:
    _load_fonts()

    active_count = len(status)
    zone_label = "zone" if active_count == 1 else "zones"
    fig.text(
        0.99, 0.97, "Missouri Fire Weather Alerts",
        fontsize=26, fontweight="bold", ha="right", va="top", fontname="Plus Jakarta Sans",
    )
    fig.text(
        0.99, 0.90,
        f"{active_count} active {zone_label} | Updated: {_format_central_timestamp(updated_at)}",
        fontsize=16, ha="right", va="top", fontname="Montserrat",
    )
    fig.text(
        0.99, 0.48,
        "Active National Weather Service Red Flag Warnings and\n"
        "Fire Weather Watches by fire weather forecast zone.\n\n"
        "For more info, visit ShowMeFire.org",
        fontsize=10, ha="right", va="top", linespacing=1.6, fontname="Montserrat",
    )
    fig.text(
        0.02, 0.01, "ShowMeFire.org",
        fontsize=20, fontweight="bold", ha="left", va="bottom", fontname="Montserrat",
    )

    svg_path = PROJECT_ROOT / "assets/LightBackGroundLogo.svg"
    try:
        import cairosvg

        png_bytes = cairosvg.svg2png(url=str(svg_path))
        image = mpimg.imread(BytesIO(png_bytes), format="png")
        image_box = OffsetImage(image, zoom=0.03)
        logo = AnnotationBbox(
            image_box, (0.99, 0.01), frameon=False, xycoords="figure fraction", box_alignment=(1, 0),
        )
        ax.add_artist(logo)
    except Exception:
        pass


def generate_fire_weather_alert_map(status: dict[str, str] | None = None) -> dict:
    status = active_fire_weather_zone_status() if status is None else status
    updated_at = datetime.now(timezone.utc)

    fig, ax, data_crs = _create_base_figure()
    zones = _read_zones(data_crs)
    _add_zone_fills(ax, zones, status, data_crs)
    _add_boundaries(ax, zones, data_crs)
    _add_branding(fig, ax, status, updated_at)
    _add_legend(fig)

    ALERT_MAP_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ALERT_MAP_PNG, dpi=MAP_DPI, bbox_inches=None, pad_inches=0, facecolor=BACKGROUND_COLOR)
    plt.close(fig)

    gis = publish_fire_weather_alert_gis(status)
    STATUS_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_CACHE_PATH.write_text(json.dumps(status, sort_keys=True))

    updated_at_str = datetime.fromtimestamp(
        ALERT_MAP_PNG.stat().st_mtime, tz=timezone.utc,
    ).strftime("%Y-%m-%dT%H:%M:%SZ")
    warnings = sum(1 for event in status.values() if event == "Red Flag Warning")
    watches = sum(1 for event in status.values() if event == "Fire Weather Watch")
    logger.info(
        "Fire weather alert map generated: %s active zones (%s warnings, %s watches)",
        len(status), warnings, watches,
    )
    return {
        "active_zones": len(status),
        "warnings": warnings,
        "watches": watches,
        "image_path": "mo-firewx-alerts.png",
        "updated_at": updated_at_str,
        "gis": gis,
    }


def maybe_regenerate_fire_weather_alert_map(status: dict[str, str] | None = None) -> dict | None:
    """Regenerate the map/GIS assets only if the active zone set changed.

    Called from the 5-minute alert poll - skips the matplotlib render (and
    GeoJSON/GPKG republish) entirely when nothing changed since last time.
    """
    status = active_fire_weather_zone_status() if status is None else status
    try:
        cached = json.loads(STATUS_CACHE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        cached = None
    if status == cached:
        return None
    return generate_fire_weather_alert_map(status)
