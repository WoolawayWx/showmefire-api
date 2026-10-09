"""A per-incident "join area": the incident's real outline (member pixel
footprints / buffered point detections, closed into one contiguous polygon)
grown outward by a buffer.

New detections that land inside an active incident's join area are attached
to that incident (see core.database.find_or_create_incident_for_detection)
instead of only being matched by distance to the incident's centroid, so a
fire that spreads or is long and thin keeps absorbing its own detections
instead of spawning a string of side-by-side incidents. The area follows the
shape of the fire rather than being a circle.

The area is maintained incrementally as detections join (grow_area) and is
rebuilt from every member - with the buffer applied to the merged outline -
whenever incidents are merged or split (core.database.rebuild_incident_area,
called from incident_merger._recompute_incident_stats).

Pure geometry only: no database access, so core.database and the merger can
both use it.
"""
from __future__ import annotations

import math
import os

from shapely.ops import transform, unary_union

# How far the join area extends beyond the fire's outline.
AREA_BUFFER_KM = float(os.getenv("FIRE_INCIDENT_AREA_BUFFER_KM", "1.0"))
# An incident whose area is wider than this stops attracting detections by
# shape (it falls back to the old centroid-radius rule) - the same guard the
# merger uses, so a ridge/road line of detections can't chain separate fires.
MAX_AREA_EXTENT_KM = float(os.getenv("FIRE_INCIDENT_AREA_MAX_EXTENT_KM", "12.0"))
SIMPLIFY_DEGREES = 0.0004


def _scales(reference_lat: float) -> tuple[float, float]:
    return 111.320 * max(0.0001, math.cos(math.radians(reference_lat))), 110.574


def buffer_km(geometry, km: float):
    """Buffer a lon/lat geometry outward by `km` kilometres, following its
    outline (not a circle), by buffering in a locally-scaled km space."""
    kx, ky = _scales(geometry.centroid.y)
    projected = transform(lambda x, y, z=None: (x * kx, y * ky), geometry)
    return transform(lambda x, y, z=None: (x / kx, y / ky), projected.buffer(km))


def extent_km(geometry) -> float:
    minx, miny, maxx, maxy = geometry.bounds
    kx, ky = _scales((miny + maxy) / 2)
    return math.hypot((maxx - minx) * kx, (maxy - miny) * ky)


def _one_polygon(geometry):
    if geometry.geom_type == "Polygon":
        return geometry
    return geometry.convex_hull if geometry.geom_type != "GeometryCollection" else unary_union(
        [g for g in geometry.geoms if g.geom_type in ("Polygon", "MultiPolygon")]
    ).convex_hull


def build_area(polygons: list, km: float = AREA_BUFFER_KM):
    """Join area for a full set of member polygons: close them into one
    contiguous outline, then add the buffer to that merged shape."""
    from services.incident_shape_extractor import _merge_to_single_polygon

    polygons = [p for p in polygons if p is not None and not p.is_empty]
    if not polygons:
        return None
    outline = _merge_to_single_polygon(unary_union(polygons))
    return _one_polygon(buffer_km(outline, km).simplify(SIMPLIFY_DEGREES, preserve_topology=True))


def grow_area(area, new_polygons: list, km: float = AREA_BUFFER_KM):
    """Extend an existing (already buffered) area with newly joined
    detections, each buffered the same way."""
    from services.incident_shape_extractor import _merge_to_single_polygon

    new_polygons = [p for p in new_polygons if p is not None and not p.is_empty]
    if not new_polygons:
        return area
    addition = buffer_km(unary_union(new_polygons), km)
    merged = _merge_to_single_polygon(unary_union([area, addition]))
    return _one_polygon(merged.simplify(SIMPLIFY_DEGREES, preserve_topology=True))
