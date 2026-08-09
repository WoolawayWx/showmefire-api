"""
Resolve a WGS84 point to its Missouri county FIPS/name.

api/services/mobile_content.py::county_catalog() gives the authoritative
29xxx FIPS <-> name mapping but only reads .dbf attribute rows - it never
touches geometry. The geometry lives in the same shapefile, but
MO_County_Boundaries.prj is EPSG:3857 (Web Mercator, meters). Comparing
raw lat/lon against those shapes matches nothing, silently. Every point
must be reprojected before the containment test.
"""
import logging
from functools import lru_cache
from typing import Optional, Tuple

import shapefile
from pyproj import Transformer
from shapely.geometry import Point, shape as shapely_shape
from shapely.prepared import prep

from services.mobile_content import COUNTIES_SHP_PATH, county_catalog

logger = logging.getLogger(__name__)

_TO_MERCATOR = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


@lru_cache(maxsize=1)
def _county_polygons():
    """Prepared county polygons in EPSG:3857 with their 29xxx FIPS/name."""
    reader = shapefile.Reader(str(COUNTIES_SHP_PATH))
    fields = [field[0] for field in reader.fields[1:]]
    out = []
    for record, shp in zip(reader.records(), reader.shapes()):
        row = dict(zip(fields, record))
        geom = shapely_shape(shp.__geo_interface__)
        fips = f"29{str(row.get('COUNTYFIPS') or '').zfill(3)}"
        name = str(row.get('COUNTYNAME') or '').strip()
        out.append((prep(geom), geom.bounds, fips, name))
    return out


def county_for_point(latitude: float, longitude: float) -> Tuple[Optional[str], Optional[str]]:
    """
    Resolve the Missouri county FIPS and name for a WGS84 point.

    Returns (None, None) if the point does not fall inside any county
    polygon (river boundaries, GPS drift, or a point outside Missouri).
    """
    x, y = _TO_MERCATOR.transform(longitude, latitude)
    probe = Point(x, y)
    for prepared, (minx, miny, maxx, maxy), fips, name in _county_polygons():
        if not (minx <= x <= maxx and miny <= y <= maxy):
            continue
        if prepared.contains(probe):
            return fips, name
    return None, None


def verify_county_catalog_coverage() -> None:
    """
    Cross-check that every FIPS the shapefile can return is a FIPS the
    catalog whitelist (used by routers/mobile.py) also knows about.
    Catches a shapefile swap that changes FIPS formatting.
    """
    known = {c["fips"] for c in county_catalog()}
    for _, _, fips, name in _county_polygons():
        if fips not in known:
            logger.warning("county_lookup: shapefile FIPS %s (%s) not in county_catalog()", fips, name)
