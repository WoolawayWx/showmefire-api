"""
Backfill/re-ingest the existing file-based fire detections into the
unified fire_events store.

Runs as a separate scheduled job from the fetch jobs that write these
files (see core/scheduler.py) - a SQLite lock or a bad row here degrades
only the store, never the GeoJSON files /fires/satdet and the shipped
mobile app depend on.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import GIS_DIR, MISSOURI_FIRES_GEOJSON
from core.database import upsert_detection_event
from services.county_lookup import county_for_point

logger = logging.getLogger(__name__)

SATDET_PATH = GIS_DIR / "satfiredetection.geojson"
NGFS_PATH = MISSOURI_FIRES_GEOJSON


def _load_geojson(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("fire_ingest: could not read %s: %s", path, exc)
        return []
    return data.get("features", []) if isinstance(data, dict) else []


def _extract_coordinates(feature: Dict[str, Any]) -> Optional[tuple]:
    geometry = feature.get("geometry") or {}
    coords = geometry.get("coordinates")
    if not coords or len(coords) < 2:
        return None
    lon, lat = coords[0], coords[1]
    if lon is None or lat is None:
        return None
    return float(lat), float(lon)


def _ingest_satdet_feature(feature: Dict[str, Any]) -> Optional[Dict]:
    coords = _extract_coordinates(feature)
    if coords is None:
        return None
    lat, lon = coords
    props = feature.get("properties") or {}
    typename = str(props.get("TYPENAME") or "").upper()
    source = "modis" if "MODIS" in typename else "viirs"
    external_id = props.get("SOURCE_ID") or f"{source}:{props.get('SATELLITE')}:{props.get('ACQ_DATE_TIME')}:{lat:.3f}:{lon:.3f}"
    occurred_at = props.get("ACQ_DATE_TIME")
    if not occurred_at:
        return None

    county_fips, county_name = county_for_point(lat, lon)
    return upsert_detection_event(
        source=source,
        external_id=str(external_id),
        latitude=lat,
        longitude=lon,
        occurred_at=occurred_at,
        county_fips=county_fips,
        county_name=county_name,
        frp=props.get("FRP"),
        confidence=props.get("CONFIDENCE"),
        satellite=props.get("SATELLITE"),
    )


def _ingest_ngfs_feature(feature: Dict[str, Any]) -> Optional[Dict]:
    """Maps a NOAA NESDIS NGFS OGC API detection pixel (flat properties -
    see api/tools/ngfs_ogc_firedetect.py) into the unified fire_events store.

    Unlike satdet (FIRMS), this feature's geometry is the pixel's Polygon
    footprint, not a Point - _extract_coordinates() would reject it (it
    expects a flat [lon, lat] pair). The centroid lat/lon is already given
    directly in properties, so read it from there instead.
    """
    props = feature.get("properties") or {}
    lat, lon = props.get("latitude"), props.get("longitude")
    if lat is None or lon is None:
        return None
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return None
    external_id = props.get("id")
    occurred_at = props.get("acq_date_time")
    if external_id is None or not occurred_at:
        return None

    county_fips, county_name = county_for_point(lat, lon)
    geometry = feature.get("geometry")
    footprint_geojson = json.dumps(geometry, separators=(",", ":")) if geometry and geometry.get("type") == "Polygon" else None

    return upsert_detection_event(
        source="ngfs",
        external_id=str(external_id),
        latitude=lat,
        longitude=lon,
        occurred_at=occurred_at,
        county_fips=county_fips,
        county_name=county_name or props.get("county"),
        frp=props.get("frp"),
        confidence=props.get("confidence"),
        satellite=props.get("satellite"),
        bright_t7=props.get("bright_t7"),
        bright_t13=props.get("bright_t13"),
        pixel_area=props.get("pixel_area"),
        quality_flag=props.get("quality_flag"),
        solar_zenith_angle=props.get("solar_zenith_angle"),
        satellite_zenith_angle=props.get("satellite_zenith_angle"),
        footprint_geojson=footprint_geojson,
        daynight=props.get("daynight"),
        land_cover=props.get("land_cover"),
    )


def _satdet_sort_key(feature: Dict[str, Any]) -> str:
    return (feature.get("properties") or {}).get("ACQ_DATE_TIME") or ""


def _ngfs_sort_key(feature: Dict[str, Any]) -> str:
    return (feature.get("properties") or {}).get("acq_date_time") or ""


def ingest_detection_files(paths: Optional[Dict[str, Path]] = None, dry_run: bool = False) -> Dict[str, Any]:
    """
    Ingest api/gis/satfiredetection.geojson and
    api/data/missouri_fires.geojson into fire_events.

    Features are processed oldest-first (GeoJSON feature order isn't
    guaranteed chronological) so incident clustering in
    upsert_detection_event/find_or_create_incident_for_detection sees
    detections in a stable, time-forward order.

    Returns {"inserted": n, "updated": n, "skipped": n, "errors": [...]}.
    """
    resolved = paths or {"satdet": SATDET_PATH, "ngfs": NGFS_PATH}
    inserted = updated = skipped = 0
    errors: List[str] = []

    satdet_features = sorted(_load_geojson(resolved.get("satdet", SATDET_PATH)), key=_satdet_sort_key)
    ngfs_features = sorted(_load_geojson(resolved.get("ngfs", NGFS_PATH)), key=_ngfs_sort_key)

    for satdet_feature in satdet_features:
        try:
            if dry_run:
                if _extract_coordinates(satdet_feature) is None:
                    skipped += 1
                continue
            result = _ingest_satdet_feature(satdet_feature)
            if result is None:
                skipped += 1
            elif result["inserted"]:
                inserted += 1
            else:
                updated += 1
        except Exception as exc:
            errors.append(f"satdet: {exc}")

    for ngfs_feature in ngfs_features:
        try:
            if dry_run:
                if _extract_coordinates(ngfs_feature) is None:
                    skipped += 1
                continue
            result = _ingest_ngfs_feature(ngfs_feature)
            if result is None:
                skipped += 1
            elif result["inserted"]:
                inserted += 1
            else:
                updated += 1
        except Exception as exc:
            errors.append(f"ngfs: {exc}")

    summary = {"inserted": inserted, "updated": updated, "skipped": skipped, "errors": errors, "dry_run": dry_run}
    logger.info("fire_ingest: %s", summary)
    return summary
