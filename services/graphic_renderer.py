"""Process-safe GIS renderer for department static graphics."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
from PIL import Image
from matplotlib.patches import FancyBboxPatch
from matplotlib.patches import Patch
from dotenv import load_dotenv

load_dotenv()

WIDTH, HEIGHT, DPI = 1920, 1080, 120
RENDERER_VERSION = "graphics-gis-v8"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MISSOURI_STATE_BOUNDARY = PROJECT_ROOT / "maps/shapefiles/MO_State_Boundary/MO_State_Boundary.shp"
MISSOURI_COUNTY_BOUNDARIES = PROJECT_ROOT / "maps/shapefiles/MO_County_Boundaries/MO_County_Boundaries.shp"
SPC_BASE = "https://www.spc.noaa.gov/products/outlook"
PRODUCT_URLS = {
    "spc_cat": f"{SPC_BASE}/day1otlk_cat.nolyr.geojson",
    "spc_tor": f"{SPC_BASE}/day1otlk_torn.nolyr.geojson",
    "spc_wind": f"{SPC_BASE}/day1otlk_wind.nolyr.geojson",
    "spc_hail": f"{SPC_BASE}/day1otlk_hail.nolyr.geojson",
}
PRODUCT_IDS = set(PRODUCT_URLS) | {"spc_four_panel", "mo_alerts"}
CONUS_EXTENT = (-125.0, -66.5, 24.0, 50.0)
MISSOURI_EXTENT = (-95.9, -89.0, 35.8, 40.8)
FALLBACK_COLORS = {2: "#78c878", 5: "#8b4726", 10: "#ffc77d", 15: "#ffff00", 30: "#ff0000", 45: "#ff00ff", 60: "#912cee"}
WEB_MERCATOR_LIMIT = 85.05112878
WEB_MERCATOR_RADIUS = 6378137.0
BASEMAP_LAYERS = {
    "rastertiles/voyager": ("rastertiles/voyager_nolabels", "rastertiles/voyager_only_labels"),
    "light_all": ("light_nolabels", "light_only_labels"),
    "dark_all": ("dark_nolabels", "dark_only_labels"),
}


def _fetch(url: str) -> bytes:
    response = requests.get(url, timeout=float(os.getenv("SMF_GRAPHICS_FETCH_TIMEOUT", "20")), headers={
        "User-Agent": "ShowMeFire/1.0 graphics@showmefire.org", "Accept": "application/geo+json,application/json",
    })
    response.raise_for_status()
    return response.content


def _load_alert_bytes() -> tuple[bytes, str]:
    configured = os.getenv("SMF_ACTIVE_ALERTS_GEOJSON", "").strip()
    candidates = [Path(configured)] if configured else []
    candidates += [Path("/app/gis/active.json"), PROJECT_ROOT / "gis/active.json"]
    for path in candidates:
        if path.is_file():
            return path.read_bytes(), str(path)
    url = "https://api.weather.gov/alerts/active?area=MO"
    return _fetch(url), url


def _as_frame(payload: bytes) -> gpd.GeoDataFrame:
    collection = json.loads(payload)
    features = [feature for feature in collection.get("features", []) if feature.get("geometry")]
    if not features:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
    return gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")


def _property(row, *names, default=None):
    lookup = {str(key).lower(): value for key, value in row.items()}
    for name in names:
        value = lookup.get(name.lower())
        if value not in (None, ""):
            return value
    return default


def _color(row):
    event = str(_property(row, "event", default="")).lower()
    if "tornado warning" in event:
        return "#dc2626"
    if "severe thunderstorm warning" in event:
        return "#f59e0b"
    if "flash flood warning" in event or "flood warning" in event:
        return "#16a34a"
    if "watch" in event:
        return "#facc15"
    if "advisory" in event:
        return "#38bdf8"
    supplied = str(_property(row, "fill", "fillcolor", default="")).strip()
    if supplied.startswith("#") and len(supplied) in (4, 7, 9):
        return supplied[:7]
    try:
        return FALLBACK_COLORS.get(int(float(_property(row, "dn", default=0))), "#d1d5db")
    except (TypeError, ValueError):
        return "#d1d5db"


def _legend_entries(frames):
    entries = {}
    for frame in frames:
        for _, row in frame.iterrows():
            value = _property(row, "dn", default=0)
            label = _property(row, "label", "label2", "event")
            if not label:
                try:
                    label = f"{int(float(value))}%"
                except (TypeError, ValueError):
                    label = "Active area"
            entries[(str(label), _color(row))] = float(value or 0)
    return [(label, color) for (label, color), _ in sorted(entries.items(), key=lambda item: item[1])][:9]


def _add_powered_by(fig):
    logo_path = PROJECT_ROOT / "assets/LightBackGroundLogo.svg"
    if not logo_path.is_file():
        return
    try:
        import cairosvg
        logo_bytes = cairosvg.svg2png(url=str(logo_path), output_width=420)
        logo = Image.open(io.BytesIO(logo_bytes)).convert("RGBA")
        backing = FancyBboxPatch((0.012, 0.012), 0.12, 0.087, boxstyle="round,pad=0.004,rounding_size=0.007", transform=fig.transFigure, facecolor="#ffffff", edgecolor="#d1d5db", linewidth=0.7, alpha=0.94, zorder=70)
        fig.patches.append(backing)
        fig.text(0.072, 0.083, "Powered by", ha="center", va="center", fontsize=9, fontweight="bold", color="#374151", zorder=80)
        logo_ax = fig.add_axes((0.021, 0.02, 0.102, 0.05), zorder=80)
        logo_ax.imshow(logo)
        logo_ax.axis("off")
    except Exception:
        return


def _add_department_logo(fig, logo_path: str | None):
    if not logo_path or not Path(logo_path).is_file():
        return 0.0
    try:
        logo = Image.open(logo_path).convert("RGBA")
        # Keep a consistent height while sizing the header slot to the PNG's
        # actual aspect ratio. Figure coordinates must account for 16:9 pixels.
        logo.thumbnail((190, 76), Image.Resampling.LANCZOS)
        logo_height = 0.07
        logo_width = min(0.20, logo_height * (HEIGHT / WIDTH) * (logo.width / logo.height))
        logo_ax = fig.add_axes((0.032, 0.902, logo_width, logo_height), zorder=65)
        logo_ax.imshow(logo)
        logo_ax.axis("off")
        return logo_width
    except (OSError, ValueError):
        return 0.0


def _add_legend(fig, frames):
    entries = _legend_entries(frames)
    if not entries:
        entries = [("No active areas", "#d1d5db")]
    handles = [Patch(facecolor=color, edgecolor="#f9fafb", linewidth=0.7, label=label) for label, color in entries]
    legend = fig.legend(handles=handles, title="Legend", loc="upper right", bbox_to_anchor=(0.985, 0.975), frameon=True, ncol=1, fontsize=10, title_fontsize=11, borderpad=0.8, labelspacing=0.45)
    legend.set_zorder(90)
    legend.get_frame().set_facecolor("#111827")
    legend.get_frame().set_edgecolor("#374151")
    legend.get_frame().set_alpha(0.94)
    legend.get_title().set_color("#f9fafb")
    legend.get_title().set_fontweight("bold")
    for text in legend.get_texts():
        text.set_color("#f9fafb")


def _world_pixel(lon: float, lat: float, zoom: int):
    lat = max(-WEB_MERCATOR_LIMIT, min(WEB_MERCATOR_LIMIT, lat))
    scale = 256 * (2 ** zoom)
    x = (lon + 180.0) / 360.0 * scale
    sin_lat = math.sin(math.radians(lat))
    y = (0.5 - math.log((1 + sin_lat) / (1 - sin_lat)) / (4 * math.pi)) * scale
    return x, y


def _world_lonlat(x: float, y: float, zoom: float):
    scale = 256 * (2 ** zoom)
    lon = x / scale * 360.0 - 180.0
    mercator_y = math.pi - (2.0 * math.pi * y / scale)
    lat = math.degrees(math.atan(math.sinh(mercator_y)))
    return lon, lat


def viewport_extent(center, zoom: float, width: int, height: int):
    """MapLibre-compatible center/zoom viewport expressed as lon/lat bounds."""
    center_x, center_y = _world_pixel(float(center[0]), float(center[1]), zoom)
    west, north = _world_lonlat(center_x - width / 2, center_y - height / 2, zoom)
    east, south = _world_lonlat(center_x + width / 2, center_y + height / 2, zoom)
    return west, east, south, north


def center_zoom_for_bounds(bounds, width=WIDTH, height=HEIGHT, padding=0.12):
    """Return a Web Mercator center/zoom that contains GIS bounds."""
    west, south, east, north = [float(value) for value in bounds]
    left, top = _world_pixel(west, north, 0)
    right, bottom = _world_pixel(east, south, 0)
    usable_width = width * (1 - 2 * padding)
    usable_height = height * (1 - 2 * padding)
    zoom_x = math.log2(usable_width / max(right - left, 1e-9))
    zoom_y = math.log2(usable_height / max(bottom - top, 1e-9))
    zoom = max(1.0, min(14.0, zoom_x, zoom_y))
    center = _world_lonlat((left + right) / 2, (top + bottom) / 2, 0)
    return [round(center[0], 6), round(center[1], 6)], round(zoom, 2)


def _mercator(lon: float, lat: float):
    lat = max(-WEB_MERCATOR_LIMIT, min(WEB_MERCATOR_LIMIT, lat))
    return (
        WEB_MERCATOR_RADIUS * math.radians(lon),
        WEB_MERCATOR_RADIUS * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2)),
    )


def _tile_zoom(extent, width: int, height: int):
    for zoom in range(12, 0, -1):
        west, north = _world_pixel(extent[0], extent[3], zoom)
        east, south = _world_pixel(extent[1], extent[2], zoom)
        if east - west <= width * 1.35 and south - north <= height * 1.35:
            return zoom
    return 1


def _basemap(extent, width=1600, height=800, style="voyager", transparent=False):
    zoom = _tile_zoom(extent, width, height)
    left, top = _world_pixel(extent[0], extent[3], zoom)
    right, bottom = _world_pixel(extent[1], extent[2], zoom)
    min_x, max_x = math.floor(left / 256), math.floor((right - 1) / 256)
    min_y, max_y = math.floor(top / 256), math.floor((bottom - 1) / 256)
    mode = "RGBA" if transparent else "RGB"
    background = (0, 0, 0, 0) if transparent else "#e5e7eb"
    mosaic = Image.new(mode, ((max_x - min_x + 1) * 256, (max_y - min_y + 1) * 256), background)
    template = os.getenv("SMF_GRAPHICS_BASEMAP_URL", "https://a.basemaps.cartocdn.com/{style}/{z}/{x}/{y}.png")
    key = os.getenv("CARTO_API_KEY") or os.getenv("NUXT_CARTO_KEY") or os.getenv("NUXT_PUBLIC_CARTO_KEY")
    fetched = 0
    for tile_x in range(min_x, max_x + 1):
        for tile_y in range(min_y, max_y + 1):
            url = template.format(style=style, z=zoom, x=tile_x, y=tile_y)
            if key:
                url += ("&" if "?" in url else "?") + f"key={key}"
            try:
                response = requests.get(url, timeout=12, headers={"User-Agent": "ShowMeFire/1.0 graphics@showmefire.org"})
                response.raise_for_status()
                tile = Image.open(io.BytesIO(response.content)).convert(mode)
                mosaic.paste(tile, ((tile_x - min_x) * 256, (tile_y - min_y) * 256))
                fetched += 1
            except Exception:
                continue
    crop = (int(left - min_x * 256), int(top - min_y * 256), int(right - min_x * 256), int(bottom - min_y * 256))
    cropped = mosaic.crop(crop)
    if cropped.width < 1 or cropped.height < 1:
        raise ValueError("invalid basemap crop")
    return cropped.resize((width, height), Image.Resampling.LANCZOS), fetched


def _draw_frame(ax, frame: gpd.GeoDataFrame, title: str, extent, basemap, reference_overlay,
                reference_boundaries, jurisdiction_path: str | None = None):
    ax.set_facecolor("#e5e7eb")
    west, south = _mercator(extent[0], extent[2])
    east, north = _mercator(extent[1], extent[3])
    ax.imshow(basemap, extent=(west, east, south, north), origin="upper", zorder=0)
    ax.set_xlim(west, east); ax.set_ylim(south, north)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([])
    if not frame.empty:
        sortable = frame.to_crs("EPSG:3857").copy()
        sortable["_dn"] = sortable.apply(lambda row: float(_property(row, "dn", default=0) or 0), axis=1)
        for _, row in sortable.sort_values("_dn").iterrows():
            gpd.GeoSeries([row.geometry], crs="EPSG:3857").plot(
                ax=ax, facecolor=_color(row), edgecolor=str(_property(row, "stroke", default="#374151")), linewidth=0.8, alpha=0.72,
            )
    # Borders remain above weather polygons, while city/town labels are the
    # uppermost map-data layer so no boundary line obscures a place name.
    state_boundary, county_boundaries = reference_boundaries
    if not county_boundaries.empty:
        county_boundaries.boundary.plot(ax=ax, color="#ffffff", linewidth=0.55, alpha=0.72, zorder=16)
    if not state_boundary.empty:
        state_boundary.boundary.plot(ax=ax, color="#ffffff", linewidth=2.8, alpha=0.95, zorder=17)
        state_boundary.boundary.plot(ax=ax, color="#374151", linewidth=1.0, alpha=0.9, zorder=18)
    if jurisdiction_path and Path(jurisdiction_path).is_file():
        gpd.read_file(jurisdiction_path).to_crs("EPSG:3857").boundary.plot(ax=ax, color="#ffffff", linewidth=4.2, zorder=20)
        gpd.read_file(jurisdiction_path).to_crs("EPSG:3857").boundary.plot(ax=ax, color="#111827", linewidth=1.6, zorder=21)
    ax.imshow(reference_overlay, extent=(west, east, south, north), origin="upper", zorder=30)
    # The extent was derived in Web Mercator for this exact viewport ratio.
    # Equal axis scaling preserves Missouri's shape instead of stretching it.
    ax.set_aspect("equal", adjustable="box")
    if title:
        ax.set_title(title, fontsize=16, fontweight="bold", pad=8)


def _configured_extent(config: dict, width: int, height: int):
    if config.get("center") is not None and config.get("zoom") is not None:
        return viewport_extent(config["center"], float(config["zoom"]), width, height)
    if config.get("map_extent"):
        return tuple(config["map_extent"])
    jurisdiction_path = config.get("jurisdiction_path")
    if jurisdiction_path and Path(jurisdiction_path).is_file():
        bounds = gpd.read_file(jurisdiction_path).to_crs("EPSG:4326").total_bounds
        center, zoom = center_zoom_for_bounds(bounds, width, height)
        return viewport_extent(center, zoom, width, height)
    default_center = [-92.45, 38.343121]
    default_zoom = 7.5 if config["product_id"] == "mo_alerts" else 4.0
    return viewport_extent(default_center, default_zoom, width, height)


def _render_sources(config: dict):
    product = config["product_id"]
    product_ids = ("spc_cat", "spc_tor", "spc_wind", "spc_hail") if product == "spc_four_panel" else (product,)
    payloads, urls, frames = [], [], []
    for product_id in product_ids:
        if product_id == "mo_alerts":
            payload, source = _load_alert_bytes()
        else:
            source = PRODUCT_URLS[product_id]
            payload = _fetch(source)
        payloads.append(payload); urls.append(source); frames.append(_as_frame(payload))
    return product_ids, payloads, urls, frames


def _reference_boundaries():
    empty = gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:3857")
    try:
        state = gpd.read_file(MISSOURI_STATE_BOUNDARY).to_crs("EPSG:3857") if MISSOURI_STATE_BOUNDARY.is_file() else empty
        counties = gpd.read_file(MISSOURI_COUNTY_BOUNDARIES).to_crs("EPSG:3857") if MISSOURI_COUNTY_BOUNDARIES.is_file() else empty
        return state, counties
    except (OSError, ValueError):
        return empty, empty


def _render_fingerprint(config: dict, payloads: list[bytes]) -> str:
    """Include data, presentation settings, and local assets in change detection."""
    digest = hashlib.sha256()
    for payload in payloads:
        digest.update(payload)
    public_config = {key: value for key, value in config.items() if key not in {"jurisdiction_path", "department_logo_path"}}
    digest.update(json.dumps(public_config, sort_keys=True, separators=(",", ":"), default=str).encode())
    digest.update(RENDERER_VERSION.encode())
    for key in ("jurisdiction_path", "department_logo_path"):
        path = Path(config[key]) if config.get(key) else None
        if path and path.is_file():
            digest.update(path.read_bytes())
    return digest.hexdigest()


def render_graphic(config: dict) -> dict:
    """Render one configured product; safe to call in a ProcessPoolExecutor."""
    product_ids, payloads, urls, frames = _render_sources(config)
    source_hash = _render_fingerprint(config, payloads)
    map_width = 820 if config["product_id"] == "spc_four_panel" else WIDTH
    map_height = 350 if config["product_id"] == "spc_four_panel" else HEIGHT
    extent = _configured_extent(config, map_width, map_height)
    if len(extent) != 4 or extent[0] >= extent[1] or extent[2] >= extent[3]:
        raise ValueError("invalid map extent")
    configured_style = config.get("basemap_style", "rastertiles/voyager")
    base_style, reference_style = BASEMAP_LAYERS.get(configured_style, BASEMAP_LAYERS["rastertiles/voyager"])
    basemap, base_tile_count = _basemap(extent, map_width, map_height, base_style)
    reference_overlay, reference_tile_count = _basemap(extent, map_width, map_height, reference_style, transparent=True)
    tile_count = base_tile_count + reference_tile_count
    reference_boundaries = _reference_boundaries()
    fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI), dpi=DPI, facecolor=config.get("background_color", "#e8e8e8"))
    header_box = FancyBboxPatch((0.018, 0.89), 0.72, 0.09, boxstyle="round,pad=0.008,rounding_size=0.012", transform=fig.transFigure, facecolor="#111827", edgecolor="#374151", linewidth=1.2, alpha=0.94, zorder=50)
    fig.patches.append(header_box)
    department_logo_width = _add_department_logo(fig, config.get("department_logo_path"))
    header_left = 0.032 + department_logo_width + 0.018 if department_logo_width else 0.035
    fig.text(header_left, 0.956, config.get("header_text") or "Show Me Fire Weather Graphics", ha="left", va="top", fontsize=23, fontweight="bold", color="#f9fafb", zorder=60)
    if config.get("subtitle"):
        fig.text(header_left, 0.915, config["subtitle"], ha="left", va="top", fontsize=12, color="#d1d5db", zorder=60)
    labels = {"spc_cat": "Day 1 Categorical", "spc_tor": "Day 1 Tornado", "spc_wind": "Day 1 Wind", "spc_hail": "Day 1 Hail", "mo_alerts": "Missouri Weather Alerts"}
    positions = ([(0.025, 0.51, 0.46, 0.35), (0.515, 0.51, 0.46, 0.35), (0.025, 0.12, 0.46, 0.35), (0.515, 0.12, 0.46, 0.35)] if config["product_id"] == "spc_four_panel" else [(0, 0, 1, 1)])
    for product_id, frame, position in zip(product_ids, frames, positions):
        panel_title = labels[product_id] if config["product_id"] == "spc_four_panel" else ""
        _draw_frame(fig.add_axes(position), frame, panel_title, extent, basemap, reference_overlay,
                    reference_boundaries, config.get("jurisdiction_path"))
    fig.text(0.985, 0.018, "Sources: NOAA/NWS SPC, weather.gov, OpenStreetMap & CARTO", ha="right", va="bottom", fontsize=9, fontweight="bold", color="#374151", zorder=80, bbox={"boxstyle": "round,pad=0.4", "facecolor": "#ffffff", "edgecolor": "#d1d5db", "linewidth": 0.7, "alpha": 0.94})
    _add_powered_by(fig)
    _add_legend(fig, frames)
    output = io.BytesIO()
    fig.savefig(output, format="png", dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)
    data = output.getvalue()
    with Image.open(io.BytesIO(data)) as check:
        if check.size != (WIDTH, HEIGHT) or check.format != "PNG":
            raise ValueError("renderer produced an invalid PNG")
    return {"bytes": data, "source_fingerprint": source_hash, "source_urls": urls, "basemap_tiles": tile_count,
            "renderer_version": RENDERER_VERSION, "generated_at": datetime.now(timezone.utc).isoformat()}
