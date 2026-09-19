"""Extracts a real, irregular shape for a fire incident from the GOES-19
NGFS Microphysics RGB composite - the same family of technique radar
storm-cell identification (SCIT/TITAN-style) uses on a reflectivity field:
threshold to a mask, connected-component-label contiguous regions, keep
only the ones anchored by a real return (here: a real satellite detection),
then trace the contour into a polygon.

This is unsupervised/algorithmic by design - there is no labeled fire-
perimeter dataset in this system to train against. The color threshold
below is a documented starting heuristic (env-tunable), not a calibrated
one: live probes of the tile server during development confirmed the
product's clear-sky/land background is a consistent, saturated cool cyan
(blue channel pinned ~255, green ~200-220, red low-to-mid), but no
currently-hot pixel was available to sample at that time. Tune
FIRE_SHAPE_BLUE_MAX/FIRE_SHAPE_RED_RATIO_MIN once a live fire is visible
in the composite.

The tile endpoint has no time parameter - it only ever serves the current
composite, so shapes can only be computed near real-time for currently
active incidents, never backfilled.
"""
from __future__ import annotations

import json
import logging
import math
import os

import numpy as np
from affine import Affine
from shapely.affinity import scale, translate
from shapely.geometry import Point
from shapely.geometry import shape as shapely_shape
from shapely.ops import unary_union

from core.database import (
    get_fire_incident,
    list_fire_incident_members,
    list_fire_incidents,
    set_fire_incident_shape,
)

logger = logging.getLogger(__name__)

MICROPHYSICS_TILE_URL = "https://re-ngfs.ssec.wisc.edu/api/image?products=G19-C-NGFSMicrophysics-TC&x={x}&y={y}&z={z}"

# Same MIN_DETECTIONS gate fire_incident_graphics.py uses - a lone detection
# isn't worth an imagery fetch/analysis pass.
MIN_DETECTIONS = int(os.getenv("FIRE_SHAPE_MIN_DETECTIONS", "5"))
IMAGE_SIZE = int(os.getenv("FIRE_SHAPE_IMAGE_SIZE", "300"))

# Fire/hot-signature threshold against the confirmed cool-cyan background -
# see module docstring. Deliberately env-tunable, not hardcoded.
BLUE_MAX = int(os.getenv("FIRE_SHAPE_BLUE_MAX", "200"))
RED_RATIO_MIN = float(os.getenv("FIRE_SHAPE_RED_RATIO_MIN", "1.1"))

# Approximate half-pixel-size buffer (km) for a detection with no stored
# footprint polygon - VIIRS (~375m nominal pixel) and MODIS (~1km nominal
# pixel) point detections don't carry a footprint_geojson the way NGFS
# detections do, so they're represented as a small buffered circle instead,
# letting them still contribute area to the merged incident shape rather
# than being ignored. Same degree-conversion approach fire_confidence.py's
# _circle() already uses.
POINT_BUFFER_KM = {"ngfs": 1.0, "viirs": 0.2, "modis": 0.5}
DEFAULT_POINT_BUFFER_KM = 0.3

# Buffer-out-then-in (morphological closing) distance, in degrees, applied
# to the final merged shape: rounds sharp pixel-corner joints into curves
# and bridges small gaps between nearby-but-not-touching detections into
# one contiguous outline. ~150m at Missouri's latitude. If that isn't
# enough to fully connect every member of the incident,
# _merge_to_single_polygon below keeps doubling this before falling back to
# a convex hull - one incident always renders as exactly one shape, never
# several disconnected ones.
SMOOTHING_DEGREES = float(os.getenv("FIRE_SHAPE_SMOOTHING_DEGREES", "0.0015"))
MAX_SMOOTHING_DOUBLINGS = 6


def extract_hot_mask(rgb: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels that break the product's cool-cyan background
    pattern - candidate fire/hot-signature pixels."""
    red = rgb[:, :, 0].astype(np.float32)
    blue = rgb[:, :, 2].astype(np.float32)
    return (blue < BLUE_MAX) & (red > blue * RED_RATIO_MIN)


def _pixel_transform(extent: tuple[float, float, float, float], width: int, height: int) -> Affine:
    """extent = (min_lon, max_lon, min_lat, max_lat), matching
    graphic_renderer._basemap's convention. Linear lon/lat mapping is
    accurate enough at incident scale (a few km across)."""
    min_lon, max_lon, min_lat, max_lat = extent
    x_res = (max_lon - min_lon) / width
    y_res = (max_lat - min_lat) / height
    return Affine(x_res, 0, min_lon, 0, -y_res, max_lat)


def _lonlat_to_pixel(lon: float, lat: float, extent, width: int, height: int) -> tuple[int, int]:
    min_lon, max_lon, min_lat, max_lat = extent
    px = int((lon - min_lon) / (max_lon - min_lon) * width)
    py = int((max_lat - lat) / (max_lat - min_lat) * height)
    return px, py


def _incident_extent(members: list[dict]) -> tuple[float, float, float, float]:
    lats = [float(m["latitude"]) for m in members]
    lons = [float(m["longitude"]) for m in members]
    margin = max(0.01, max(max(lats) - min(lats), max(lons) - min(lons)) * 0.6)
    return (min(lons) - margin, max(lons) + margin, min(lats) - margin, max(lats) + margin)


def _merge_to_single_polygon(combined):
    """An incident is one thing - it must always render as exactly one
    shape, never several disconnected polygons for detections that happen
    to sit further apart. Grows the same closing-buffer used for corner
    rounding until every piece connects into one contiguous outline; if the
    detections are so far apart that would balloon the shape unreasonably,
    falls back to the convex hull of everything so it's still a single
    polygon rather than a MultiPolygon."""
    radius = SMOOTHING_DEGREES
    merged = combined.buffer(radius).buffer(-radius)
    for _ in range(MAX_SMOOTHING_DOUBLINGS):
        if merged.geom_type != "MultiPolygon":
            break
        radius *= 2
        merged = combined.buffer(radius).buffer(-radius)
    if merged.geom_type == "MultiPolygon":
        merged = combined.convex_hull
    return merged if not merged.is_empty else combined


def _point_buffer_polygon(lat: float, lon: float, radius_km: float):
    """A small ellipse approximating a circle of radius_km around (lat, lon),
    in degrees - longitude is stretched by 1/cos(lat) so it reads as a
    circle on a real map despite the projection, same approach
    fire_confidence.py's _circle() uses for confidence-area rings."""
    lat_deg = radius_km / 110.574
    lon_deg = radius_km / max(0.0001, 111.32 * math.cos(math.radians(lat)))
    circle = Point(0, 0).buffer(1.0, quad_segs=12)
    return translate(scale(circle, lon_deg, lat_deg), lon, lat)


def _member_polygons(members: list[dict]) -> list:
    """One polygon per member detection: its real stored pixel footprint
    when available (currently NGFS only), otherwise a small buffered circle
    sized to that sensor's approximate pixel footprint - this is what lets
    VIIRS/MODIS point detections (which have no footprint_geojson) still
    contribute area to the merged incident shape instead of being dropped."""
    polygons = []
    for member in members:
        raw = member.get("footprint_geojson")
        if raw:
            try:
                polygons.append(shapely_shape(json.loads(raw)))
                continue
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        try:
            lat, lon = float(member["latitude"]), float(member["longitude"])
        except (TypeError, ValueError, KeyError):
            continue
        radius_km = POINT_BUFFER_KM.get(member.get("source"), DEFAULT_POINT_BUFFER_KM)
        polygons.append(_point_buffer_polygon(lat, lon, radius_km))
    return polygons


def compute_incident_shape(incident: dict, members: list[dict]) -> dict | None:
    """Returns a single GeoJSON Polygon geometry dict (always one contiguous
    polygon, never a MultiPolygon - see _merge_to_single_polygon) combining
    every member detection - real pixel footprints, buffered point
    detections (VIIRS/MODIS), and (when available) the NGFS Microphysics
    imagery analysis - into one rounded-corner shape for the whole incident,
    rather than a point/footprint/circle per individual detection. This is
    the incident's one "parent" shape: everything else (per-detection rows,
    the ML confidence score, the pixel footprints themselves) is still
    stored and queryable exactly as before, this just gives the incident a
    single easy-to-display outline that already reflects all of it.

    Returns None if there are no members at all or nothing could be
    combined into a valid shape. Never raises."""
    if not members:
        return None
    ngfs_members = [m for m in members if m.get("source") == "ngfs"]

    # The tile server only ever serves the *current* composite (see module
    # docstring), so a detection from even a few scans ago may no longer
    # show a hot signature by the time this runs - imagery is treated as an
    # enhancement, not a hard requirement. The stored per-pixel footprints
    # (always available for ngfs members) are the reliable fallback shape.
    imagery_polygons: list = []
    if ngfs_members:
        try:
            from scipy import ndimage
            from rasterio import features as rio_features
            from services.graphic_renderer import _basemap

            extent = _incident_extent(members)
            image, fetched = _basemap(extent, width=IMAGE_SIZE, height=IMAGE_SIZE, url_template=MICROPHYSICS_TILE_URL)
            if fetched:
                rgb = np.array(image.convert("RGB"))
                mask = extract_hot_mask(rgb)
                if mask.any():
                    labeled, _ = ndimage.label(mask)
                    seed_labels = set()
                    for member in ngfs_members:
                        px, py = _lonlat_to_pixel(float(member["longitude"]), float(member["latitude"]), extent, IMAGE_SIZE, IMAGE_SIZE)
                        if 0 <= px < IMAGE_SIZE and 0 <= py < IMAGE_SIZE:
                            label_id = labeled[py, px]
                            if label_id:
                                seed_labels.add(int(label_id))
                    if seed_labels:
                        seeded_mask = np.isin(labeled, list(seed_labels))
                        transform = _pixel_transform(extent, IMAGE_SIZE, IMAGE_SIZE)
                        imagery_polygons = [
                            shapely_shape(geom) for geom, value in rio_features.shapes(
                                seeded_mask.astype(np.uint8), mask=seeded_mask, transform=transform,
                            )
                        ]
        except Exception:
            logger.warning("incident_shape_extractor: imagery analysis failed for incident %s, falling back to footprints/points only", incident.get("id"), exc_info=True)

    try:
        combined = unary_union(imagery_polygons + _member_polygons(members))
        if combined.is_empty:
            return None
        # Round sharp pixel-corner joints into curves and merge every piece
        # into one contiguous outline - one incident is always exactly one
        # shape, never several disconnected polygons.
        rounded = _merge_to_single_polygon(combined)
        simplified = rounded.simplify(0.0004, preserve_topology=True)
        return json.loads(json.dumps(simplified.__geo_interface__))
    except Exception:
        logger.warning("incident_shape_extractor: failed for incident %s", incident.get("id"), exc_info=True)
        return None


def _iter_incidents(incident_id: int | None):
    if incident_id is not None:
        incident = get_fire_incident(incident_id)
        if incident is not None:
            yield incident
        return

    # list_fire_incidents caps its own page size at 200, so a single call
    # only ever sees the 200 most-recently-active incidents - with enough
    # incident churn, that page shifts fast enough to permanently skip
    # eligible incidents that are still well within an active fire's
    # lifespan. Page through offsets so every incident gets considered.
    offset = 0
    page_size = 200
    while True:
        page = list_fire_incidents(limit=page_size, offset=offset)
        if not page:
            return
        yield from page
        if len(page) < page_size:
            return
        offset += page_size


def refresh_incident_shapes(incident_id: int | None = None, force: bool = False) -> dict:
    """Recompute shapes only for incidents whose detection_count changed
    since the last shape computation (or on force) - avoids re-fetching and
    re-analyzing imagery for an incident that hasn't changed, same
    compute-cost-conscious design as the recurring-source suppression."""
    updated = 0
    skipped = 0
    failed = 0
    for incident in _iter_incidents(incident_id):
        if int(incident.get("detection_count") or 0) < MIN_DETECTIONS:
            continue
        if not force and incident.get("shape_detection_count") == incident.get("detection_count"):
            skipped += 1
            continue

        members = list_fire_incident_members(incident["id"])
        geometry = compute_incident_shape(incident, members)
        if geometry is None:
            failed += 1
            continue
        set_fire_incident_shape(incident["id"], json.dumps(geometry, separators=(",", ":")), incident["detection_count"])
        updated += 1

    summary = {"updated": updated, "skipped": skipped, "failed": failed}
    logger.info("incident_shape_extractor: %s", summary)
    return summary
