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

def fetch_advanced_fire_detections():
    """
    Fetch advanced fire detection data from NGFS API.
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

def save_detections(data):
    """Save fire detections to GIS directory as GeoJSON"""
    output_path = GIS_DIR / 'ngfs_advanced_fire_detections.geojson'
    
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
    
    data = fetch_advanced_fire_detections()
    save_path = save_detections(data)
    
    if save_path:
        feature_count = len(data.get('features', []))
        print(f"✓ Successfully fetched and saved {feature_count} fire detections")
        print(f"  - Saved to: {save_path}")
        return data
    else:
        print("✗ Failed to save fire detections")
        return None

if __name__ == "__main__":
    main()
