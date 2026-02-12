import requests
import json
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# --- Path Configurations ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
GIS_DIR = BASE_DIR / 'gis'
LOGS_DIR = BASE_DIR / 'logs'

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
API_URL = "https://re-ngfs.ssec.wisc.edu/api/shapes?products=NGFS-SCENE-CONUS-EAST"

def filter_by_state(data, states=['MO']):
    """
    Filter fire detections by state(s).
    
    Args:
        data: GeoJSON FeatureCollection
        states: List of state abbreviations (e.g., ['MO', 'KS', 'AR'])
    
    Returns:
        Filtered GeoJSON FeatureCollection
    """
    if not isinstance(data, dict) or data.get('type') != 'FeatureCollection':
        return data
    
    # Convert states to uppercase for case-insensitive matching
    states = [s.upper() for s in states]
    
    # Filter features by STATE property (not 'state')
    filtered_features = [
        feature for feature in data.get('features', [])
        if feature.get('properties', {}).get('STATE', '').upper() in states
    ]
    
    # Update metadata
    original_count = data.get('metadata', {}).get('feature_count', 0)
    data['features'] = filtered_features
    data['metadata']['feature_count'] = len(filtered_features)
    data['metadata']['filtered_by_states'] = states
    data['metadata']['original_feature_count'] = original_count
    
    logger.info(f"Filtered from {original_count} to {len(filtered_features)} features for states: {', '.join(states)}")
    
    return data

def fetch_advanced_fire_detections(filter_states=None):
    """
    Fetch advanced fire detection data from NGFS API.
    
    Args:
        filter_states: List of state abbreviations to filter by (e.g., ['MO', 'KS'])
                      If None, returns all states
    
    Returns GeoJSON FeatureCollection with comprehensive fire detection data.
    """
    logger.info(f"Fetching advanced fire detections from {API_URL}")
    
    try:
        response = requests.get(API_URL, timeout=30)
        response.raise_for_status()
        data = response.json()
        fetch_time = datetime.utcnow().isoformat() + 'Z'
        
        # Add metadata to the GeoJSON
        if isinstance(data, dict) and data.get('type') == 'FeatureCollection':
            data['metadata'] = {
                'fetched_at': fetch_time,
                'source': 'NGFS CONUS-EAST',
                'product': 'NGFS-SCENE-CONUS-EAST',
                'feature_count': len(data.get('features', []))
            }
            
            logger.info(f"Successfully fetched {len(data.get('features', []))} fire detection features")
            
            # Apply state filter if specified
            if filter_states:
                data = filter_by_state(data, filter_states)
            
            return data
        else:
            logger.warning("Unexpected data format from API")
            return {
                'type': 'FeatureCollection',
                'features': [],
                'metadata': {
                    'fetched_at': fetch_time,
                    'error': 'Unexpected data format',
                    'feature_count': 0
                }
            }
    
    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP request failed: {e}")
        return {
            'type': 'FeatureCollection',
            'features': [],
            'metadata': {
                'fetched_at': datetime.utcnow().isoformat() + 'Z',
                'error': str(e),
                'feature_count': 0
            }
        }
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {
            'type': 'FeatureCollection',
            'features': [],
            'metadata': {
                'fetched_at': datetime.utcnow().isoformat() + 'Z',
                'error': str(e),
                'feature_count': 0
            }
        }

def save_detections(data, suffix=''):
    """
    Save fire detections to GIS directory as GeoJSON
    
    Args:
        data: GeoJSON FeatureCollection
        suffix: Optional suffix for filename (e.g., '_missouri')
    """
    output_path = GIS_DIR / f'ngfs_advanced_fire_detections{suffix}.geojson'
    
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
    logger.info("Starting advanced fire detection fetch")
    
    # Fetch Missouri detections only
    mo_data = fetch_advanced_fire_detections(filter_states=['MO'])
    save_detections(mo_data, suffix='_missouri')
    
    mo_count = len(mo_data.get('features', []))
    
    print(f"✓ Successfully fetched and saved fire detections")
    print(f"  - Missouri: {mo_count} detections")
    
    return mo_data

if __name__ == "__main__":
    main()
