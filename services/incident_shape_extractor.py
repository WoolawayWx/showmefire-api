"""One outline per fire incident: where the incident's satellite detections cover.

It is the union of every member detection's real pixel footprint (NGFS/GOES keeps one) or, for detections that carry none
(VIIRS/MODIS points), a small circle sized to that sensor's pixel, smoothed so corners are rounded and nearby pieces join
into a single contiguous polygon. It is an outline of where the satellite pixels landed, NOT the fire's perimeter: pixel
positions are only good to a few hundred metres (GOES ~2 km).

There used to be a second step that thresholded the NGFS Microphysics satellite image to add hot pixels. It was never
calibrated, could not be backfilled (the tile server only serves the current composite) and, measured on production
incidents with real GOES footprints, changed the area in only about one case in ten (median ratio 1.00x), so it was
removed.

The shape is used by the app's map and incident screen, the incident graphic, the ground-cover sampling area and the
cropland-burn model.
"""
from __future__ import annotations

import json
import logging
import math
import os

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

# A lone detection is not worth an outline of its own (same gate fire_incident_graphics.py uses).
MIN_DETECTIONS = int(os.getenv("FIRE_SHAPE_MIN_DETECTIONS", "5"))

# Approximate half-pixel-size buffer (km) for a detection with no stored
# footprint polygon - VIIRS (~375m nominal pixel) and MODIS (~1km nominal
# pixel) point detections don't carry a footprint_geojson the way NGFS
# detections do, so they're represented as a small buffered circle instead,
# letting them still contribute area to the merged incident shape rather
# than being ignored.
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

# Running count of convex-hull fallbacks (see _merge_to_single_polygon); reset
# and reported by refresh_incident_shapes.
hull_fallbacks = 0


def _merge_to_single_polygon(combined, incident_id=None):
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
        # A convex hull is the blobbiest possible outline - it means this
        # incident's detections don't connect even after MAX_SMOOTHING_DOUBLINGS,
        # which usually points at grouping that is too loose. Counted so
        # refresh_incident_shapes can report how often it happens.
        global hull_fallbacks
        hull_fallbacks += 1
        logger.warning("incident_shape_extractor: convex-hull fallback for incident %s (detections do not connect)", incident_id)
        merged = combined.convex_hull
    return merged if not merged.is_empty else combined


def _point_buffer_polygon(lat: float, lon: float, radius_km: float):
    """A small ellipse approximating a circle of radius_km around (lat, lon),
    in degrees - longitude is stretched by 1/cos(lat) so it reads as a
    circle on a real map despite the projection, same approach
    the old confidence rings used."""
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
    """Returns a single GeoJSON Polygon geometry dict (always one contiguous polygon, never a MultiPolygon - see
    _merge_to_single_polygon) combining every member detection (real pixel footprints, and buffered points for
    VIIRS/MODIS) into one rounded-corner outline for the whole incident. Everything else (the per-detection rows, the
    confidence score, the pixel footprints themselves) is still stored and queryable exactly as before.

    Returns None if there are no members or nothing could be combined into a valid shape. Never raises."""
    if not members:
        return None
    try:
        combined = unary_union(_member_polygons(members))
        if combined.is_empty:
            return None
        # Round sharp pixel-corner joints into curves and merge every piece
        # into one contiguous outline - one incident is always exactly one
        # shape, never several disconnected polygons.
        rounded = _merge_to_single_polygon(combined, incident.get("id"))
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
    since the last shape computation (or on force)."""
    global hull_fallbacks
    hull_fallbacks = 0
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

    summary = {"updated": updated, "skipped": skipped, "failed": failed, "hull_fallbacks": hull_fallbacks}
    logger.info("incident_shape_extractor: %s", summary)
    return summary
