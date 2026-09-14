"""Render shareable incident cards for dense satellite-detection clusters."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable

from core.config import IMAGES_DIR
from core.database import list_fire_incidents, list_fire_incident_members, set_fire_incident_graphic

logger = logging.getLogger(__name__)
MIN_DETECTIONS = int(__import__("os").getenv("FIRE_INCIDENT_GRAPHIC_MIN_DETECTIONS", "5"))
# Incident cards cover a small local area. Zoom 13 keeps roads and field
# patterns legible instead of stretching a low-resolution regional tile.
INCIDENT_BASEMAP_ZOOM = int(__import__("os").getenv("FIRE_INCIDENT_BASEMAP_ZOOM", "13"))
PUBLIC_SITE_URL = os.getenv("PUBLIC_SITE_URL", "https://showmefire.org").rstrip("/")


def render_incident_graphic(incident: dict, detections: Iterable[dict], output: Path) -> Path:
    """Create a map-left/info-right PNG. Basemap downloads are optional; the
    graphic remains useful in restricted/offline environments."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

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
        # Satellite imagery supplies the visual context; OSM overlays roads.
        # Cartopy caches tiles and failures fall through to the clean map.
        ax.remove()
        ax = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
        ax.set_extent([min(lons) - margin, max(lons) + margin, min(lats) - margin, max(lats) + margin])
        satellite = cimgt.GoogleTiles(style="satellite")
        roads = cimgt.OSM()
        ax.add_image(satellite, INCIDENT_BASEMAP_ZOOM)
        # OSM supplies the road network, town names, and other orientation
        # cues that are missing from the raw imagery. Keep it translucent so
        # the thermal-detection markers remain the visual priority.
        ax.add_image(roads, INCIDENT_BASEMAP_ZOOM, alpha=0.82)
        gridlines = ax.gridlines(
            draw_labels=True, linewidth=0.35, color="white", alpha=0.55,
            linestyle="--", x_inline=False, y_inline=False,
        )
        gridlines.top_labels = False
        gridlines.right_labels = False
        gridlines.xlabel_style = {"size": 7, "color": "#374151"}
        gridlines.ylabel_style = {"size": 7, "color": "#374151"}
        ax.scatter(lons, lats, s=55, c="#ff3b20", edgecolors="white", linewidths=1, transform=ccrs.PlateCarree(), zorder=5)
    except Exception as exc:
        logger.info("Incident basemap unavailable: %s", exc)
        ax.scatter(lons, lats, s=55, c="#e53935", edgecolors="white", linewidths=1, zorder=5)
        ax.grid(True, alpha=0.25)
    ax.set_title("Satellite detection cluster · roads and place labels", loc="left", weight="bold")
    info.axis("off")
    info.set_facecolor("#ffffff")
    county_names = incident.get("county_names") or sorted({str(row.get("county_name")) for row in rows if row.get("county_name")})
    sources = sorted({str(row.get("source")).upper() for row in rows if row.get("source")})
    # Reuse the forecast-card branding so incident graphics feel like part of
    # the same product family. Failure to rasterize the optional SVG must not
    # prevent the map from being generated.
    try:
        import cairosvg
        from PIL import Image
        from io import BytesIO
        logo_path = Path(__file__).resolve().parents[1] / "assets" / "LightBackGroundLogo.svg"
        if logo_path.is_file():
            logo = Image.open(BytesIO(cairosvg.svg2png(url=str(logo_path), output_width=330))).convert("RGBA")
            logo.thumbnail((190, 76), Image.Resampling.LANCZOS)
            logo_ax = fig.add_axes((0.765, 0.885, 0.18, 0.07), zorder=10)
            logo_ax.imshow(logo)
            logo_ax.axis("off")
    except Exception:
        logger.debug("Incident graphic logo unavailable", exc_info=True)

    center_lat = float(incident["centroid_latitude"])
    center_lon = float(incident["centroid_longitude"])
    incident_url = f"{PUBLIC_SITE_URL}/fires/incident/{incident.get('public_slug', '')}"
    osm_url = f"https://www.openstreetmap.org/?mlat={center_lat:.5f}&mlon={center_lon:.5f}#map={INCIDENT_BASEMAP_ZOOM}/{center_lat:.5f}/{center_lon:.5f}"
    firms_url = "https://firms.modaps.eosdis.nasa.gov/map/"
    info.text(0, 0.95, "FIRE DETECTION CLUSTER", fontsize=17, weight="bold", color="#b91c1c")
    info.text(0, 0.875, f"{incident.get('detection_count', len(rows))} satellite detections", fontsize=14, weight="bold", color="#111827")
    info.text(0, 0.77, f"County/counties: {', '.join(county_names) or 'Unknown'}", wrap=True)
    info.text(0, 0.69, f"First detected: {incident.get('first_detected_at', 'Unknown')}\nLast detected: {incident.get('last_detected_at', 'Unknown')}", wrap=True)
    info.text(0, 0.55, f"Sources: {', '.join(sources) or 'Unknown'}\nCenter: {center_lat:.5f}, {center_lon:.5f}\nBasemap: satellite imagery + OpenStreetMap roads", wrap=True)
    info.text(0, 0.37, "Incident information", fontsize=11, weight="bold", color="#111827")
    info.text(0, 0.32, incident_url, fontsize=8.5, color="#b91c1c", weight="bold", wrap=True)
    info.text(0, 0.24, f"Reference maps\nNASA FIRMS: {firms_url}\nOpenStreetMap: {osm_url}", fontsize=7.2, color="#2563eb", wrap=True)
    info.text(0, 0.10, "Is this a confirmed fire?\nTell Show Me Fire if this was a wildfire,\ncontrolled burn, or another heat source.", color="#374151")
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
