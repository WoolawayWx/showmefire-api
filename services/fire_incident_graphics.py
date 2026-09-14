"""Render shareable incident cards for dense satellite-detection clusters."""
from __future__ import annotations

import logging
import os
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


def render_incident_graphic(incident: dict, detections: Iterable[dict], output: Path) -> Path:
    """Create a map-left/info-right PNG. Basemap downloads are optional; the
    graphic remains useful in restricted/offline environments."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from PIL import Image

    rows = list(detections)
    lats = [float(row["latitude"]) for row in rows]
    lons = [float(row["longitude"]) for row in rows]
    margin = max(0.015, max(max(lats) - min(lats), max(lons) - min(lons)) * 0.35)
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
        import numpy as np
        # Satellite imagery is the primary context. OSM raster tiles are
        # opaque in Cartopy and can completely cover the imagery, so roads
        # are added below as transparent vector data instead.
        # Cartopy caches tiles and failures fall through to the clean map.
        ax.remove()
        ax = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
        ax.set_extent([min(lons) - margin, max(lons) + margin, min(lats) - margin, max(lats) + margin])
        satellite = cimgt.GoogleTiles(style="satellite")
        ax.add_image(satellite, INCIDENT_BASEMAP_ZOOM)

        class TransparentOSM(cimgt.OSM):
            """Keep dark road/name pixels while removing OSM's opaque base."""

            def get_image(self, tile):
                image, extent, origin = super().get_image(tile)
                rgba = np.asarray(image.convert("RGBA")).copy()
                darkness = 255.0 - rgba[:, :, :3].mean(axis=2)
                rgba[:, :, 3] = np.clip(darkness * 2.8, 0, 215).astype(np.uint8)
                return Image.fromarray(rgba, mode="RGBA"), extent, origin

        # This is a transparent road/name layer, not a second opaque map.
        # It keeps the satellite detail visible while adding orientation cues.
        ax.add_image(TransparentOSM(), INCIDENT_BASEMAP_ZOOM, alpha=0.9)
        road_path = Path(__file__).resolve().parents[1] / "maps" / "shapefiles" / "MO_TIGER_Primary_Roads" / "MO_TIGER_Primary_Roads.shp"
        if road_path.is_file():
            try:
                import geopandas as gpd
                road_frame = gpd.read_file(road_path).to_crs("EPSG:4326")
                for geometry in road_frame.geometry:
                    if geometry is not None and not geometry.is_empty:
                        ax.add_geometries(
                            [geometry], ccrs.PlateCarree(), facecolor="none",
                            edgecolor="#f8fafc", linewidth=1.0, alpha=0.78, zorder=4,
                        )
            except Exception:
                logger.debug("Local primary-road overlay unavailable", exc_info=True)
        ax.scatter(lons, lats, s=55, c="#ff3b20", edgecolors="white", linewidths=1, transform=ccrs.PlateCarree(), zorder=5)
    except Exception as exc:
        logger.info("Incident basemap unavailable: %s", exc)
        ax.scatter(lons, lats, s=55, c="#e53935", edgecolors="white", linewidths=1, zorder=5)
        ax.grid(True, alpha=0.25)
    ax.set_title("Satellite detection cluster · satellite imagery and roads", loc="left", weight="bold", pad=8)
    info.axis("off")
    info.set_facecolor("#ffffff")
    county_names = incident.get("county_names") or sorted({str(row.get("county_name")) for row in rows if row.get("county_name")})
    sources = sorted({str(row.get("source")).upper() for row in rows if row.get("source")})
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

    center_lat = float(incident["centroid_latitude"])
    center_lon = float(incident["centroid_longitude"])
    incident_url = f"{PUBLIC_SITE_URL}/fires/incident/{incident.get('public_slug', '')}"
    info.axhline(0.985, color="#b91c1c", linewidth=4, clip_on=False)
    info.text(0, 0.95, "FIRE DETECTION CLUSTER", fontsize=15, weight="bold", color="#b91c1c")
    info.text(0, 0.875, f"{incident.get('detection_count', len(rows))} satellite detections", fontsize=14, weight="bold", color="#111827")
    info.text(0, 0.76, "SUMMARY", fontsize=10, weight="bold", color="#6b7280")
    info.text(0, 0.705, f"County: {', '.join(county_names) or 'Unknown'}\nSources: {', '.join(sources) or 'Unknown'}\nCenter: {center_lat:.5f}, {center_lon:.5f}", wrap=True)
    info.text(0, 0.545, "TIMELINE", fontsize=10, weight="bold", color="#6b7280")
    info.text(0, 0.49, f"First detected\n{_display_time(incident.get('first_detected_at'))}\n\nLast detected\n{_display_time(incident.get('last_detected_at'))}", wrap=True)
    info.text(0, 0.285, "LEARN MORE", fontsize=10, weight="bold", color="#6b7280")
    info.text(0, 0.235, f"Incident page:\n{incident_url}", fontsize=8.2, color="#b91c1c", weight="bold", wrap=True)
    info.text(0, 0.135, "Satellite imagery with roads shown.\nPlease verify this heat signature before treating it as a confirmed fire.", fontsize=8.2, color="#374151", wrap=True)
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
