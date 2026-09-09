"""Render shareable incident cards for dense satellite-detection clusters."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from core.config import IMAGES_DIR
from core.database import list_fire_incidents, list_fire_incident_members, set_fire_incident_graphic

logger = logging.getLogger(__name__)
MIN_DETECTIONS = int(__import__("os").getenv("FIRE_INCIDENT_GRAPHIC_MIN_DETECTIONS", "5"))


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
    fig, (ax, info) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [1.7, 1]})
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
        ax.add_image(cimgt.GoogleTiles(style="satellite"), 10)
        ax.add_image(cimgt.OSM(), 10, alpha=0.65)
        ax.scatter(lons, lats, s=55, c="#ff3b20", edgecolors="white", linewidths=1, transform=ccrs.PlateCarree(), zorder=5)
    except Exception as exc:
        logger.info("Incident basemap unavailable: %s", exc)
        ax.scatter(lons, lats, s=55, c="#e53935", edgecolors="white", linewidths=1, zorder=5)
        ax.grid(True, alpha=0.25)
    ax.set_title("Satellite detection cluster", loc="left", weight="bold")
    info.axis("off")
    county_names = sorted({str(row.get("county_name")) for row in rows if row.get("county_name")})
    sources = sorted({str(row.get("source")).upper() for row in rows if row.get("source")})
    info.text(0, 0.95, "FIRE DETECTION CLUSTER", fontsize=16, weight="bold", color="#b91c1c")
    info.text(0, 0.87, f"{incident.get('detection_count', len(rows))} detections", fontsize=14, weight="bold")
    info.text(0, 0.77, f"County/counties: {', '.join(county_names) or 'Unknown'}", wrap=True)
    info.text(0, 0.69, f"First detected: {incident.get('first_detected_at', 'Unknown')}\nLast detected: {incident.get('last_detected_at', 'Unknown')}", wrap=True)
    info.text(0, 0.55, f"Sources: {', '.join(sources) or 'Unknown'}\nCenter: {incident['centroid_latitude']:.5f}, {incident['centroid_longitude']:.5f}", wrap=True)
    info.text(0, 0.30, "Is this a confirmed fire?\nTell Show Me Fire if this was a wildfire,\ncontrolled burn, or another heat source.", color="#374151")
    info.add_patch(Patch(facecolor="#ef4444", label="Detection location"))
    info.legend(loc="lower left", frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output


def refresh_incident_graphics(incident_id: int | None = None, force: bool = False) -> dict:
    """Render cards only for dense clusters and return a small job summary."""
    rendered = 0
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
            logger.exception("Could not render incident graphic %s", incident["id"])
    return {"rendered": rendered}
