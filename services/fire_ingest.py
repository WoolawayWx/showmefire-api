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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import GIS_DIR, MISSOURI_FIRES_GEOJSON
from core.database import finish_ingest_run, start_ingest_run, upsert_detection_event
from services.county_lookup import county_for_point
from services.static_context import sample_static_context

logger = logging.getLogger(__name__)

SATDET_PATH = GIS_DIR / "satfiredetection.geojson"
NGFS_PATH = MISSOURI_FIRES_GEOJSON
# Feed names as they appear in the source log (fire_event_origins.feed / fire_ingest_runs.feed).
SATDET_FEED = "NASA FIRMS (VIIRS/MODIS)"
NGFS_FEED = "NOAA NESDIS NGFS"


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


def _skip(skips: Optional[Dict[str, int]], reason: str) -> None:
    """Count why an inbound feature was not stored, so nothing that arrives goes unaccounted for."""
    if skips is not None:
        skips[reason] = skips.get(reason, 0) + 1


def _origin(feed: str, path: Path, run_id: Optional[int], provider_ref: str, props: Dict[str, Any]) -> Dict[str, Any]:
    return {"kind": "satellite_feed", "feed": feed, "channel": path.name, "provider_ref": provider_ref,
            "run_id": run_id, "raw": props}


def _ingest_satdet_feature(feature: Dict[str, Any], run_id: Optional[int] = None, skips: Optional[Dict[str, int]] = None,
                           path: Path = SATDET_PATH) -> Optional[Dict]:
    coords = _extract_coordinates(feature)
    if coords is None:
        _skip(skips, "no_coordinates")
        return None
    lat, lon = coords
    props = feature.get("properties") or {}
    typename = str(props.get("TYPENAME") or "").upper()
    source = "modis" if "MODIS" in typename else "viirs"
    external_id = props.get("SOURCE_ID") or f"{source}:{props.get('SATELLITE')}:{props.get('ACQ_DATE_TIME')}:{lat:.3f}:{lon:.3f}"
    occurred_at = props.get("ACQ_DATE_TIME")
    if not occurred_at:
        _skip(skips, "no_acquisition_time")
        return None

    county_fips, county_name = county_for_point(lat, lon)
    if county_fips is None:
        # Inside the coarse bbox but not in any Missouri county polygon
        # (IL/KS/AR/etc. corners of the box) - not our incident.
        _skip(skips, "outside_missouri")
        return None
    fuel_model_fbfm40, canopy_cover_pct = sample_static_context(lat, lon)
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
        fuel_model_fbfm40=fuel_model_fbfm40,
        canopy_cover_pct=canopy_cover_pct,
        origin=_origin(SATDET_FEED, path, run_id, str(external_id), props),
    )


def _ingest_ngfs_feature(feature: Dict[str, Any], run_id: Optional[int] = None, skips: Optional[Dict[str, int]] = None,
                         path: Path = NGFS_PATH) -> Optional[Dict]:
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
        _skip(skips, "no_coordinates")
        return None
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        _skip(skips, "bad_coordinates")
        return None
    external_id = props.get("id")
    occurred_at = props.get("acq_date_time")
    if external_id is None or not occurred_at:
        _skip(skips, "no_id_or_acquisition_time")
        return None

    county_fips, county_name = county_for_point(lat, lon)
    if county_fips is None:
        # Same rule as satdet: the bbox is a loose filter, the county
        # polygons are the real Missouri boundary.
        _skip(skips, "outside_missouri")
        return None
    geometry = feature.get("geometry")
    footprint_geojson = json.dumps(geometry, separators=(",", ":")) if geometry and geometry.get("type") == "Polygon" else None
    fuel_model_fbfm40, canopy_cover_pct = sample_static_context(lat, lon)

    return upsert_detection_event(
        source="ngfs",
        external_id=str(external_id),
        latitude=lat,
        longitude=lon,
        occurred_at=occurred_at,
        county_fips=county_fips,
        county_name=county_name,
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
        fuel_model_fbfm40=fuel_model_fbfm40,
        canopy_cover_pct=canopy_cover_pct,
        origin=_origin(NGFS_FEED, path, run_id, str(external_id), props),
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
    skip_reasons: Dict[str, int] = {}

    satdet_path = resolved.get("satdet", SATDET_PATH)
    ngfs_path = resolved.get("ngfs", NGFS_PATH)
    satdet_features = sorted(_load_geojson(satdet_path), key=_satdet_sort_key)
    ngfs_features = sorted(_load_geojson(ngfs_path), key=_ngfs_sort_key)

    def run_feed(label, feed, path, features, ingest_one):
        """Ingest one feed file under its own run record: which file, how fresh it was, what became of each feature."""
        nonlocal inserted, updated, skipped
        feed_inserted = feed_updated = feed_skipped = 0
        feed_skips: Dict[str, int] = {}
        feed_errors: List[str] = []
        run_id = None
        if not dry_run:
            try:
                modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds") if path.exists() else None
                run_id = start_ingest_run(f"fire_ingest:{label}", feed, path.name, modified, len(features))
            except Exception as exc:  # the log must never stop the ingest
                logger.warning("fire_ingest: could not start run record for %s: %s", label, exc)
        for feature in features:
            try:
                if dry_run:
                    if _extract_coordinates(feature) is None:
                        feed_skipped += 1
                    continue
                result = ingest_one(feature, run_id, feed_skips, path)
                if result is None:
                    feed_skipped += 1
                elif result["inserted"]:
                    feed_inserted += 1
                else:
                    feed_updated += 1
            except Exception as exc:
                feed_errors.append(f"{label}: {exc}")
                _skip(feed_skips, "error")
        if run_id is not None:
            try:
                finish_ingest_run(run_id, feed_inserted, feed_updated, feed_skipped, feed_skips, feed_errors)
            except Exception as exc:
                logger.warning("fire_ingest: could not finish run record for %s: %s", label, exc)
        inserted += feed_inserted
        updated += feed_updated
        skipped += feed_skipped
        errors.extend(feed_errors)
        for reason, count in feed_skips.items():
            skip_reasons[f"{label}:{reason}"] = count

    run_feed("satdet", SATDET_FEED, satdet_path, satdet_features, _ingest_satdet_feature)
    run_feed("ngfs", NGFS_FEED, ngfs_path, ngfs_features, _ingest_ngfs_feature)

    summary = {"inserted": inserted, "updated": updated, "skipped": skipped, "errors": errors, "dry_run": dry_run,
               "skipped_reasons": skip_reasons}
    logger.info("fire_ingest: %s", summary)
    return summary
