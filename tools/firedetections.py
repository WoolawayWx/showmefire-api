import csv
import requests
import json
import os
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# --- Path Configurations ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
GIS_DIR = BASE_DIR / 'gis'
LOGS_DIR = BASE_DIR / 'logs'

load_dotenv(BASE_DIR / '.env')

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
GIS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Set up rotating log handler
LOG_FILE = LOGS_DIR / 'ngfs_advanced_detections.log'
logger = logging.getLogger('ngfs_advanced_detections')
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = RotatingFileHandler(LOG_FILE, maxBytes=5*1024*1024, backupCount=5)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)

# API endpoint for advanced fire detections
FIRMS_CSV_BASE_TEMPLATE = "https://firms.modaps.eosdis.nasa.gov/usfs/api/area/csv/{key}/{source}/{bbox}/1/"
FIRMS_CSV_SOURCES = [
    "VIIRS_SNPP_NRT",
    "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21_NRT",
    "MODIS_NRT",
]

# Missouri-ish bounds for FIRMS CSV endpoint in lon_min,lat_min,lon_max,lat_max order
MISSOURI_BBOX_PARAM = "-95.8,35.9,-89.0,40.7"
MISSOURI_LON_MIN = -95.8
MISSOURI_LON_MAX = -89.0
MISSOURI_LAT_MIN = 35.9
MISSOURI_LAT_MAX = 40.7


def _to_iso_datetime(acq_date: Any, acq_time: Any) -> Optional[str]:
    if not acq_date:
        return None
    date_str = str(acq_date).strip()
    time_str = str(acq_time or '').strip()
    if not date_str:
        return None

    # Handle full datetime values like "YYYY-MM-DD HH:MM[:SS]" or ISO8601 strings.
    if 'T' in date_str or ' ' in date_str:
        dt_candidate = date_str.replace(' ', 'T').replace('Z', '')
        dt_candidate = dt_candidate.split('.', 1)[0]
        try:
            dt_obj = datetime.fromisoformat(dt_candidate)
            return dt_obj.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            pass

    digits = ''.join(ch for ch in time_str if ch.isdigit())
    if digits:
        # FIRMS often provides 3-digit times (e.g. 741 => 07:41).
        digits = digits.zfill(4)
        hh = digits[:2]
        mm = digits[2:4]
        ss = digits[4:6] if len(digits) >= 6 else '00'
        return f"{date_str}T{hh}:{mm}:{ss}Z"

    return f"{date_str}T00:00:00Z"


def _normalize_confidence(raw_conf: Any) -> str:
    if raw_conf is None:
        return "nominal"

    text = str(raw_conf).strip().lower()
    if text in {"h", "high", "90", "100"}:
        return "high"
    if text in {"l", "low", "0", "1", "2"}:
        return "low"
    if text in {"n", "nominal", "medium", "moderate"}:
        return "nominal"

    try:
        val = float(text)
        if val >= 80:
            return "high"
        if val <= 30:
            return "low"
    except ValueError:
        pass

    return "nominal"


def _parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _clean_csv_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _get_csv_value(row: Dict[str, Any], *keys: str) -> Optional[str]:
    for key in keys:
        value = row.get(key)
        cleaned = _clean_csv_value(value)
        if cleaned is not None:
            return cleaned
    return None


def _parse_csv_rows(csv_text: str) -> List[Dict[str, Any]]:
    text = (csv_text or '').lstrip('\ufeff').strip()
    if not text:
        return []

    reader = csv.DictReader(text.splitlines())
    rows: List[Dict[str, Any]] = []
    for row in reader:
        if not row:
            continue
        rows.append({str(k).strip(): v for k, v in row.items() if k is not None})
    return rows


def _source_to_satellite(source: str) -> str:
    mapping = {
        'VIIRS_SNPP_NRT': 'VIIRS SNPP',
        'VIIRS_NOAA20_NRT': 'VIIRS NOAA-20',
        'VIIRS_NOAA21_NRT': 'VIIRS NOAA-21',
        'MODIS_NRT': 'MODIS',
    }
    return mapping.get(source, source)


def _extract_coordinates(feature: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    geom = feature.get('geometry') or {}
    coords = geom.get('coordinates')
    if geom.get('type') == 'Point' and isinstance(coords, list) and len(coords) >= 2:
        try:
            return float(coords[0]), float(coords[1])
        except (TypeError, ValueError):
            pass

    props = feature.get('properties') or {}
    lon_candidates = ['LONGITUDE', 'longitude', 'lon', 'LON']
    lat_candidates = ['LATITUDE', 'latitude', 'lat', 'LAT']

    lon = None
    lat = None
    for key in lon_candidates:
        if props.get(key) is not None:
            try:
                lon = float(props.get(key))
            except (TypeError, ValueError):
                lon = None
            break
    for key in lat_candidates:
        if props.get(key) is not None:
            try:
                lat = float(props.get(key))
            except (TypeError, ValueError):
                lat = None
            break

    return lon, lat


def _is_missouri_feature(feature: Dict[str, Any]) -> bool:
    props = feature.get('properties') or {}
    state_candidates = [
        props.get('STATE'),
        props.get('state'),
        props.get('ADM1'),
        props.get('adm1'),
        props.get('state_name'),
    ]
    for state in state_candidates:
        if not state:
            continue
        state_text = str(state).strip().upper()
        if state_text in {'MO', 'MISSOURI'}:
            return True

    lon, lat = _extract_coordinates(feature)
    if lon is None or lat is None:
        return False
    return MISSOURI_LON_MIN <= lon <= MISSOURI_LON_MAX and MISSOURI_LAT_MIN <= lat <= MISSOURI_LAT_MAX


def _normalize_csv_row(row: Dict[str, Any], source: str) -> Optional[Dict[str, Any]]:
    lon = _parse_float(_get_csv_value(row, 'longitude', 'LONGITUDE', 'lon', 'LON'))
    lat = _parse_float(_get_csv_value(row, 'latitude', 'LATITUDE', 'lat', 'LAT'))
    if lon is None or lat is None:
        return None

    acq_dt_raw = _get_csv_value(row, 'acq_date_time', 'ACQ_DATE_TIME', 'datetime', 'DATETIME')
    acq_date = _get_csv_value(row, 'acq_date', 'ACQ_DATE', 'date', 'DATE')
    acq_time = _get_csv_value(row, 'acq_time', 'ACQ_TIME', 'time', 'TIME')
    acq_dt = _to_iso_datetime(acq_dt_raw or acq_date, acq_time)

    satellite = (
        _get_csv_value(row, 'satellite', 'SATELLITE', 'instrument', 'INSTRUMENT', 'sat_id', 'SAT_ID')
        or _source_to_satellite(source)
    )

    county = _get_csv_value(row, 'county', 'COUNTY', 'adm2', 'ADM2') or 'Unknown'
    state = _get_csv_value(row, 'state', 'STATE', 'adm1', 'ADM1') or 'Unknown'

    confidence_raw = _get_csv_value(row, 'confidence', 'CONFIDENCE')
    confidence = _normalize_confidence(confidence_raw)

    frp = _parse_float(_get_csv_value(row, 'frp', 'FRP', 'fire_radiative_power', 'FIRE_RADIATIVE_POWER'))
    bright_t7 = _parse_float(
        _get_csv_value(
            row,
            'bright_t7',
            'BRIGHT_T7',
            'bright_t31',
            'BRIGHT_T31',
            'bright_ti4',
            'BRIGHT_TI4',
            'brightness',
            'BRIGHTNESS',
        )
    )

    source_id = (
        _get_csv_value(row, 'id', 'ID')
        or f"{source}:{satellite}:{acq_dt}:{lat:.3f}:{lon:.3f}"
    )

    return {
        'type': 'Feature',
        'geometry': {
            'type': 'Point',
            'coordinates': [lon, lat]
        },
        'properties': {
            'SOURCE': 'FIRMS',
            'TYPENAME': source,
            'SOURCE_ID': source_id,
            'LATITUDE': lat,
            'LONGITUDE': lon,
            'COUNTY': county,
            'STATE': state,
            'COUNTRY': _get_csv_value(row, 'country', 'COUNTRY') or 'United States',
            'FRP': frp,
            'BRIGHT_T7': bright_t7,
            'CONFIDENCE': confidence,
            'CONFIDENCE_RAW': confidence_raw,
            'ACQ_DATE_TIME': acq_dt,
            'SATELLITE': str(satellite),
            'TYPE_DESCRIPTION': 'FIRMS NRT Thermal Anomaly',
            'LAND_COVER': 'Unknown',
            'FUEL': 'Unknown',
            'DAYNIGHT': _get_csv_value(row, 'daynight', 'DAYNIGHT'),
        }
    }


def _dedup_key(feature: Dict[str, Any]) -> str:
    props = feature.get('properties') or {}
    lon, lat = _extract_coordinates(feature)
    lon_key = round(lon or 0.0, 3)
    lat_key = round(lat or 0.0, 3)
    sat = str(props.get('SATELLITE') or 'UNK').upper()
    dt = str(props.get('ACQ_DATE_TIME') or '')
    return f"{sat}|{dt}|{lat_key}|{lon_key}"


def fetch_firms_24h_detections() -> Dict[str, Any]:
    modis_key = os.getenv('MODIS_KEY', '').strip().strip("'").strip('"')
    fetch_time = datetime.utcnow().isoformat() + 'Z'

    if not modis_key:
        err = 'MODIS_KEY is not set in environment'
        logger.error(err)
        return {
            'type': 'FeatureCollection',
            'features': [],
            'metadata': {
                'fetched_at': fetch_time,
                'error': err,
                'feature_count': 0,
            }
        }

    merged_features: List[Dict[str, Any]] = []
    per_source_counts: Dict[str, int] = {}

    logger.info('Fetching FIRMS detections from CSV area endpoint')
    for source in FIRMS_CSV_SOURCES:
        source_url = FIRMS_CSV_BASE_TEMPLATE.format(
            key=modis_key,
            source=source,
            bbox=MISSOURI_BBOX_PARAM,
        )

        try:
            resp = requests.get(source_url, timeout=30)
            resp.raise_for_status()
            rows = _parse_csv_rows(resp.text)
            per_source_counts[source] = len(rows)
            logger.info('FIRMS source %s returned %s CSV rows', source, len(rows))

            for row in rows:
                normalized = _normalize_csv_row(row, source)
                if normalized and _is_missouri_feature(normalized):
                    merged_features.append(normalized)

        except requests.exceptions.RequestException as exc:
            per_source_counts[source] = 0
            logger.error('FIRMS CSV request failed for %s: %s', source, exc)

    unique_features: List[Dict[str, Any]] = []
    seen = set()
    duplicates = 0
    for feature in merged_features:
        key = _dedup_key(feature)
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique_features.append(feature)

    logger.info(
        'FIRMS merge complete: raw=%s unique=%s duplicates=%s',
        len(merged_features),
        len(unique_features),
        duplicates,
    )

    return {
        'type': 'FeatureCollection',
        'features': unique_features,
        'metadata': {
            'fetched_at': fetch_time,
            'source': 'FIRMS CSV /usfs/api/area/csv',
            'products': FIRMS_CSV_SOURCES,
            'product_counts': per_source_counts,
            'feature_count': len(unique_features),
            'raw_merged_feature_count': len(merged_features),
            'duplicates_removed': duplicates,
            'bbox': MISSOURI_BBOX_PARAM,
        }
    }

def save_detections(data, suffix=''):
    """
    Save fire detections to GIS directory as GeoJSON
    
    Args:
        data: GeoJSON FeatureCollection
        suffix: Optional suffix for filename (e.g., '_missouri')
    """
    output_path = GIS_DIR / f'satfiredetection.geojson'
    
    try:
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        feature_count = len(data.get('features', []))
        logger.info(f"Saved {feature_count} fire detection features to {output_path}")
        return str(output_path)
    
    except Exception as e:
        logger.error(f"Failed to save detections: {e}")
        return None

def main():
    """Main execution function for scheduled runs"""
    logger.info("Starting FIRMS 24h fire detection fetch")
    
    mo_data = fetch_firms_24h_detections()
    save_detections(mo_data, suffix='_missouri')
    
    mo_count = len(mo_data.get('features', []))
    
    print(f"✓ Successfully fetched and saved fire detections")
    print(f"  - Missouri: {mo_count} detections")
    
    return mo_data

if __name__ == "__main__":
    main()
