"""Render shareable incident cards for dense satellite-detection clusters."""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

from core.config import IMAGES_DIR
from core.database import list_fire_incidents, list_fire_incident_members, set_fire_incident_graphic

logger = logging.getLogger(__name__)
MIN_DETECTIONS = int(__import__("os").getenv("FIRE_INCIDENT_GRAPHIC_MIN_DETECTIONS", "5"))
# Incident cards cover a small local area. A high zoom keeps field patterns
# and road geometry legible instead of stretching a low-resolution tile.
INCIDENT_BASEMAP_ZOOM = int(__import__("os").getenv("FIRE_INCIDENT_BASEMAP_ZOOM", "15"))
PUBLIC_SITE_URL = os.getenv("PUBLIC_SITE_URL", "https://showmefire.org").rstrip("/")


@lru_cache(maxsize=64)
def _read_local_roads(path: str, bbox: tuple[float, float, float, float]):
    """Read only the roads in a map window and cache repeat requests."""
    import geopandas as gpd

    columns = ["NAME", "DESIGNATION", "geometry"]
    def fiona_fallback():
        try:
            return gpd.read_file(path, bbox=bbox, columns=columns)
        except TypeError:
            return gpd.read_file(path, bbox=bbox)

    try:
        import pyogrio
        frame = pyogrio.read_dataframe(path, bbox=bbox, columns=columns, use_arrow=True)
    except (ImportError, TypeError):
        # GeoPandas/Fiona fallback for older deployments without Pyogrio.
        frame = fiona_fallback()
    except Exception:
        # Some older Pyogrio builds do not support every optional argument.
        frame = fiona_fallback()
    return frame.to_crs("EPSG:4326")


def _display_time(value) -> str:
    """Show an incident timestamp in Central time and UTC/Zulu."""
    if not value:
        return "Unknown"
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        local = parsed.astimezone(ZoneInfo("America/Chicago"))
        utc = parsed.astimezone(timezone.utc)
        return f"{local:%b} {local.day}, {local:%Y} {local:%I:%M %p} CT\n{utc:%Y-%m-%d %H:%MZ} UTC"
    except (TypeError, ValueError):
        return str(value)


def _locator_extent(detail_extent):
    """Regional context, at least 50 km tall and four times the detail view."""
    from services.graphic_renderer import _mercator

    west, south = _mercator(detail_extent[0], detail_extent[2])
    east, north = _mercator(detail_extent[1], detail_extent[3])
    center_x, center_y = (west + east) / 2, (south + north) / 2
    # Mercator distances near Missouri are about 1.25 times ground distance.
    half_height = max(32000, (north - south) * 2, (east - west) * 2 / 1.25)
    half_width = half_height * 1.25
    import math
    from services.graphic_renderer import WEB_MERCATOR_RADIUS as radius

    def lon(x):
        return math.degrees(x / radius)

    def lat(y):
        return math.degrees(math.atan(math.sinh(y / radius)))

    return (lon(center_x - half_width), lon(center_x + half_width),
            lat(center_y - half_height), lat(center_y + half_height))


def _add_locator_map(ax, detail_extent, lons, lats):
    """Opaque labeled street map above the satellite imagery, in its corner."""
    from matplotlib.patches import Rectangle
    from services.graphic_renderer import _basemap, _mercator

    extent = _locator_extent(detail_extent)
    west, south = _mercator(extent[0], extent[2])
    east, north = _mercator(extent[1], extent[3])
    locator = ax.inset_axes([0.61, 0.665, 0.37, 0.315], zorder=30)
    locator.set_facecolor("#e5e7eb")
    available = False
    try:
        basemap, fetched = _basemap(
            extent, width=500, height=400,
            url_template="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
        )
        if fetched:
            locator.imshow(basemap, extent=(west, east, south, north), origin="upper", zorder=0)
            available = True
    except Exception:
        logger.info("Incident locator basemap unavailable", exc_info=True)
    detail_west, detail_south = _mercator(detail_extent[0], detail_extent[2])
    detail_east, detail_north = _mercator(detail_extent[1], detail_extent[3])
    locator.add_patch(Rectangle(
        (detail_west, detail_south), detail_east - detail_west, detail_north - detail_south,
        facecolor="#ef4444", edgecolor="#b91c1c", alpha=0.35, linewidth=1.5, zorder=3,
    ))
    points = [_mercator(lon, lat) for lon, lat in zip(lons, lats)]
    locator.scatter(*zip(*points), s=18, c="#ff3b20", edgecolors="white", linewidths=0.6, zorder=4)
    locator.set_xlim(west, east)
    locator.set_ylim(south, north)
    locator.set_aspect("equal", adjustable="box")
    locator.set_xticks([])
    locator.set_yticks([])
    for spine in locator.spines.values():
        spine.set_edgecolor("#ffffff")
        spine.set_linewidth(2)
    if not available:
        locator.text(0.5, 0.18, "Town / road basemap unavailable", transform=locator.transAxes,
                     ha="center", fontsize=6, color="#374151", zorder=6)
    else:
        locator.text(0.99, 0.01, "© OpenStreetMap contributors",
                     transform=locator.transAxes, ha="right", va="bottom", fontsize=4.5,
                     bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1), zorder=6)
    return locator


def render_incident_graphic(incident: dict, detections: Iterable[dict], output: Path) -> Path:
    """Create a map-left/info-right PNG. Basemap downloads are optional; the
    graphic remains useful in restricted/offline environments."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as patheffects
    from matplotlib.patches import Patch
    from PIL import Image

    rows = list(detections)
    lats = [float(row["latitude"]) for row in rows]
    lons = [float(row["longitude"]) for row in rows]
    margin = max(0.015, max(max(lats) - min(lats), max(lons) - min(lons)) * 0.35)
    detail_extent = (min(lons) - margin, max(lons) + margin, min(lats) - margin, max(lats) + margin)
    fig, (ax, info) = plt.subplots(
        1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [1.7, 1]},
        facecolor="#f5f7fa",
    )
    fig.subplots_adjust(left=0.018, right=0.985, top=0.965, bottom=0.04, wspace=0.035)
    ax.set_facecolor("#dbe7d0")
    ax.set_xlim(min(lons) - margin, max(lons) + margin)
    ax.set_ylim(min(lats) - margin, max(lats) + margin)
    try:
        import cartopy.crs as ccrs
        import cartopy.io.img_tiles as cimgt
        # Satellite imagery is the primary context. OSM raster tiles are
        # opaque in Cartopy and can completely cover the imagery, so roads
        # are added below as transparent vector data instead.
        # Cartopy caches tiles and failures fall through to the clean map.
        ax.remove()
        ax = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
        ax.set_extent([min(lons) - margin, max(lons) + margin, min(lats) - margin, max(lats) + margin])
        satellite = cimgt.GoogleTiles(style="satellite")
        ax.add_image(satellite, INCIDENT_BASEMAP_ZOOM)

        # Do not add the OSM raster layer here: its labels become faint and
        # unreadable over satellite imagery. The optional local vector layer
        # below can still provide road geometry without adding map text.
        road_root = Path(os.getenv("SMF_ROADS_DIR", str(Path(__file__).resolve().parents[1] / "maps" / "shapefiles")))
        road_path = road_root / "MO_MoDOT_Roads_Arcs" / "MO_MoDOT_Roads_Arcs.shp"
        if not road_path.is_file():
            road_path = road_root / "MO_TIGER_Primary_Roads" / "MO_TIGER_Primary_Roads.shp"
        if road_path.is_file():
            try:
                import geopandas as gpd
                # MoDOT is Web Mercator. Read only the incident bounding box
                # so the statewide road file does not need to be loaded in
                # full for every graphic.
                from pyproj import Transformer
                to_mercator = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
                west, south = to_mercator.transform(min(lons) - margin, min(lats) - margin)
                east, north = to_mercator.transform(max(lons) + margin, max(lats) + margin)
                if road_path.parent.name == "MO_MoDOT_Roads_Arcs":
                    road_frame = _read_local_roads(str(road_path), (west, south, east, north))
                else:
                    road_frame = gpd.read_file(road_path).to_crs("EPSG:4326")
                road_frame = road_frame[road_frame.geometry.notna() & ~road_frame.geometry.is_empty]
                for geometry in road_frame.geometry:
                    ax.add_geometries([geometry], ccrs.PlateCarree(), facecolor="none",
                                      edgecolor="#111827", linewidth=2.4, alpha=0.82, zorder=4)
                    ax.add_geometries([geometry], ccrs.PlateCarree(), facecolor="none",
                                      edgecolor="#f8fafc", linewidth=1.0, alpha=0.95, zorder=5)

                # Label only distinct named roads, centered on their geometry.
                # A white halo keeps dark text legible over both fields and
                # satellite shadows without adding an opaque map layer.
                seen_names = set()
                for _, road in road_frame.iterrows():
                    name = str(road.get("NAME") or "").strip()
                    if not name or name.upper() in {"NAN", "NONE"} or name.casefold() in seen_names:
                        continue
                    geometry = road.geometry
                    if geometry.length < 0.001:
                        continue
                    point = geometry.representative_point()
                    seen_names.add(name.casefold())
                    label = ax.text(point.x, point.y, name, transform=ccrs.PlateCarree(),
                                    fontsize=6.4, color="#111827", weight="bold",
                                    ha="center", va="center", zorder=6)
                    label.set_path_effects([patheffects.withStroke(linewidth=2.8, foreground="white", alpha=0.9)])
            except Exception:
                logger.debug("Local primary-road overlay unavailable", exc_info=True)
        ax.scatter(lons, lats, s=55, c="#ff3b20", edgecolors="white", linewidths=1, transform=ccrs.PlateCarree(), zorder=5)
    except Exception as exc:
        logger.info("Incident basemap unavailable: %s", exc)
        ax.scatter(lons, lats, s=55, c="#e53935", edgecolors="white", linewidths=1, zorder=5)
        ax.grid(True, alpha=0.25)
    _add_locator_map(ax, detail_extent, lons, lats)
    info.axis("off")
    info.set_facecolor("#ffffff")
    county_names = incident.get("county_names") or sorted({str(row.get("county_name")) for row in rows if row.get("county_name")})
    sources = sorted({str(row.get("source")).upper() for row in rows if row.get("source")})
    incident_url = f"{PUBLIC_SITE_URL}/fires/incident/{incident.get('public_slug', '')}"
    # Reuse the forecast-card branding so incident graphics feel like part of
    # the same product family. Failure to rasterize the optional SVG must not
    # prevent the map from being generated.
    try:
        import cairosvg
        from io import BytesIO
        logo_path = Path(__file__).resolve().parents[1] / "assets" / "LightBackGroundLogo.svg"
        if logo_path.is_file():
            logo = Image.open(BytesIO(cairosvg.svg2png(url=str(logo_path), output_width=330))).convert("RGBA")
            logo.thumbnail((145, 58), Image.Resampling.LANCZOS)
            # Keep branding in the footer so it never competes with the
            # incident title or summary text.
            logo_ax = fig.add_axes((0.885, 0.045, 0.085, 0.052), zorder=10)
            logo_ax.imshow(logo)
            logo_ax.axis("off")
    except Exception:
        logger.debug("Incident graphic logo unavailable", exc_info=True)

    # Put a scannable link beside the footer text.
    try:
        import qrcode
        qr = qrcode.make(incident_url, box_size=8, border=2)
        qr_ax = fig.add_axes((0.775, 0.035, 0.09, 0.09), zorder=20, facecolor="white")
        qr_ax.imshow(qr, cmap="gray", interpolation="nearest")
        qr_ax.axis("off")
        qr_ax.text(0.5, -0.06, "Scan for details", transform=qr_ax.transAxes,
                   ha="center", va="top", fontsize=6.2, color="#374151")
    except Exception:
        logger.warning("Incident graphic QR code unavailable", exc_info=True)

    center_lat = float(incident["centroid_latitude"])
    center_lon = float(incident["centroid_longitude"])
    graphic_updated = _display_time(datetime.now(timezone.utc))
    # Use short, spaced sections rather than densely packed report text. This
    # keeps the card readable when the image is viewed on a phone.
    info.axhline(0.985, color="#b91c1c", linewidth=4, clip_on=False)
    info.text(0, 0.95, "FIRE DETECTION CLUSTER", fontsize=15, weight="bold", color="#b91c1c")
    info.text(0, 0.875, f"{incident.get('detection_count', len(rows))} detections", fontsize=16, weight="bold", color="#111827")
    info.text(0, 0.83, "Automated satellite heat signatures", fontsize=8.5, color="#6b7280")

    # One unified information card keeps the incident details visually
    # together; the small headings preserve scanability inside the card.
    info.text(0.025, 0.755, "LOCATION", fontsize=9, weight="bold", color="#6b7280")
    info.text(0.025, 0.712, f"{', '.join(county_names) or 'Unknown'} County  ·  {', '.join(sources) or 'Unknown'}\n{center_lat:.5f}, {center_lon:.5f}", fontsize=9.5, color="#111827", linespacing=1.35)

    info.text(0.025, 0.635, "DETECTION WINDOW", fontsize=9, weight="bold", color="#6b7280")
    info.text(0.025, 0.595, "First detected", fontsize=9, weight="bold", color="#111827")
    info.text(0.025, 0.56, _display_time(incident.get('first_detected_at')), fontsize=8.8, color="#374151", linespacing=1.3)
    info.text(0.025, 0.485, "Last detected", fontsize=9, weight="bold", color="#111827")
    info.text(0.025, 0.45, _display_time(incident.get('last_detected_at')), fontsize=8.8, color="#374151", linespacing=1.3)

    info.text(0.025, 0.32, "INCIDENT DETAILS", fontsize=9, weight="bold", color="#6b7280")
    info.text(0.025, 0.285, "Open the incident page for updates and context:", fontsize=8.4, color="#374151")
    info.text(0.025, 0.24, incident_url, fontsize=8.2, color="#b91c1c", weight="bold", wrap=True)
    info.text(0.025, 0.14, "Verify this heat signature before treating it as a confirmed fire.", fontsize=8.2, color="#374151", wrap=True)
    info.text(0.025, 0.095, f"Graphic updated\n{graphic_updated}", fontsize=7.2, color="#6b7280", linespacing=1.25)
    info.legend(handles=[Patch(facecolor="#ef4444", label="Detection location")], loc="lower left", frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def refresh_incident_graphics(incident_id: int | None = None, force: bool = False) -> dict:
    """Render cards only for dense clusters and return a small job summary."""
    rendered = 0
    failed = 0
    failed_ids = []
    for incident in list_fire_incidents(limit=200):
        if incident_id is not None and int(incident["id"]) != incident_id:
            continue
        if (not force and int(incident.get("detection_count") or 0) < MIN_DETECTIONS) or not incident.get("public_slug"):
            continue
        rows = list_fire_incident_members(incident["id"])
        output = Path(IMAGES_DIR) / "fire-incidents" / f"{incident['public_slug']}.png"
        try:
            render_incident_graphic(incident, rows, output)
            set_fire_incident_graphic(incident["id"], output.name)
            rendered += 1
        except Exception:
            failed += 1
            failed_ids.append(int(incident["id"]))
            logger.exception("Could not render incident graphic %s", incident["id"])
    return {"rendered": rendered, "failed": failed, "failed_ids": failed_ids}
