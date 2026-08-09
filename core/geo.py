"""
Lightweight geospatial helpers for the fire-event store.

No R-Tree, no SpatiaLite - row volume is small enough that an indexed
bounding-box prefilter plus a Python haversine pass is simpler and has
no runtime-extension availability risk.
"""
from math import asin, cos, radians, sin, sqrt

EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 points, in kilometers."""
    p1, p2 = radians(lat1), radians(lat2)
    dp = p2 - p1
    dl = radians(lon2 - lon1)
    a = sin(dp / 2) ** 2 + cos(p1) * cos(p2) * sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * asin(sqrt(a))


def degree_box(lat: float, lon: float, radius_km: float) -> tuple:
    """
    Bounding box (min_lat, max_lat, min_lon, max_lon) that circumscribes a
    circle of the given radius around (lat, lon). Used as a SQL prefilter
    before an exact haversine_km distance check.
    """
    d_lat = radius_km / 110.574
    d_lon = radius_km / max(0.0001, 111.320 * cos(radians(lat)))
    return (lat - d_lat, lat + d_lat, lon - d_lon, lon + d_lon)
