"""Fetch NGFS (Next Generation Fire System) detections for Missouri from
NOAA NESDIS's public OGC Features API and write them in the same output
shape/paths the rest of the pipeline (fire_ingest.py, gis_vectors.py) already
expects from api/tools/nfgs_firedetect.py, which this module replaces.

Source: https://fire.data.nesdis.noaa.gov/api/ogc/detections/collections/
ngfs_schema.ngfs_detections_scene_east_conus - per-pixel GOES-East CONUS
detections, no auth required. Missouri sits entirely in the East CONUS scene.
"""
import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

import requests

# --- Path Configurations ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
LOGS_DIR = BASE_DIR / 'logs'
ARCHIVE_RAW_DATA_DIR = BASE_DIR / 'archive' / 'raw_data' / 'ngfs_ogc'

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOGS_DIR / 'ngfs_ogc_firedetect.log'
logger = logging.getLogger('ngfs_ogc_firedetect')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = RotatingFileHandler(LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=5)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

# Same Missouri bbox convention used by tools/firedetections.py (FIRMS).
MISSOURI_BBOX_PARAM = "-95.8,35.9,-89.0,40.7"

NESDIS_OGC_BASE = "https://fire.data.nesdis.noaa.gov/api/ogc/detections"
NESDIS_COLLECTION = "ngfs_schema.ngfs_detections_scene_east_conus"
NESDIS_ITEMS_URL = f"{NESDIS_OGC_BASE}/collections/{NESDIS_COLLECTION}/items"

# NOAA asks API consumers to identify themselves.
REQUEST_HEADERS = {"User-Agent": "ShowMeFire (contact@showmefire.org)"}

PAGE_LIMIT = 200
MAX_PAGES = 20  # guards against an unbounded loop if pagination ever misbehaves


def fetch_missouri_detections():
    """Fetch all NGFS detection pixels intersecting the Missouri bbox,
    following OGC 'next' pagination. Returns the raw list of GeoJSON
    features (bbox-filtered only; state filtering happens separately since
    the bbox can spill into neighboring states)."""
    features = []
    url = NESDIS_ITEMS_URL
    params = {"bbox": MISSOURI_BBOX_PARAM, "limit": PAGE_LIMIT, "f": "json"}

    for _ in range(MAX_PAGES):
        response = requests.get(url, params=params, headers=REQUEST_HEADERS, timeout=20)
        response.raise_for_status()
        payload = response.json()
        features.extend(payload.get("features", []))

        next_link = next(
            (link.get("href") for link in payload.get("links", []) if link.get("rel") == "next"),
            None,
        )
        if not next_link:
            break
        url, params = next_link, None  # next_link already carries all query params

    return features


def _missouri_only(features):
    return [f for f in features if str((f.get("properties") or {}).get("state") or "").upper() == "MO"]


def get_missouri_fires_with_coords():
    """Fetch, filter to Missouri, and return the raw feature list plus a
    fetch timestamp - mirrors the old function's name/shape so callers that
    still import it (if any) keep working."""
    fetch_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        all_features = fetch_missouri_detections()
    except requests.RequestException as exc:
        logger.error(f"Error fetching NGFS OGC detections: {exc}")
        return {"error": str(exc), "features": [], "fetched_at": fetch_time}

    mo_features = _missouri_only(all_features)
    logger.info(
        f"Fetched {len(all_features)} NGFS detections in bbox, {len(mo_features)} in Missouri"
    )
    return {"features": mo_features, "fetched_at": fetch_time}


def _archive_raw_capture(all_features, fetch_time):
    """Keep a dated raw capture of every poll for the 7-day retention
    window; api/services/archive_bundler.py sweeps subfolders here once
    they age past the retention cutoff."""
    try:
        fetch_dt = datetime.fromisoformat(fetch_time.replace("Z", "+00:00"))
    except ValueError:
        fetch_dt = datetime.now(timezone.utc)

    day_dir = ARCHIVE_RAW_DATA_DIR / fetch_dt.strftime("%Y%m%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    capture_path = day_dir / f"ngfs_ogc_{fetch_dt.strftime('%H%M%S')}.geojson"
    payload = {"type": "FeatureCollection", "features": all_features, "metadata": {"fetched_at": fetch_time}}
    capture_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def main():
    """Fetch NGFS detections and write missouri_fires.geojson /
    missouri_fires_coords.json in the paths fire_ingest.py already reads."""
    print(f"Starting NGFS OGC fire detection fetch. Logging to: {LOG_FILE}")

    fetch_time = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    try:
        all_features = fetch_missouri_detections()
    except requests.RequestException as exc:
        logger.error(f"Error fetching NGFS OGC detections: {exc}")
        print(f"Error fetching NGFS OGC detections: {exc}")
        return {"error": str(exc), "features": []}

    _archive_raw_capture(all_features, fetch_time)

    mo_features = _missouri_only(all_features)

    geojson = {
        "type": "FeatureCollection",
        "features": mo_features,
        "metadata": {"fetched_at": fetch_time, "source": "NOAA NESDIS NGFS OGC API"},
    }
    geojson_path = DATA_DIR / 'missouri_fires.geojson'
    with open(geojson_path, 'w') as f:
        json.dump(geojson, f, indent=2)

    coords_payload = {
        "fetched_at": fetch_time,
        "summary": {
            "total_detections_in_bbox": len(all_features),
            "missouri_detection_count": len(mo_features),
        },
        "detections": mo_features,
    }
    json_path = DATA_DIR / 'missouri_fires_coords.json'
    with open(json_path, 'w') as f:
        json.dump(coords_payload, f, indent=2)

    print(f"Success! Found {len(mo_features)} Missouri NGFS detections.")
    print(f"  - Files saved in: {DATA_DIR}")
    return geojson


if __name__ == "__main__":
    main()
