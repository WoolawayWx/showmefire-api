import requests
import json
import re
from datetime import datetime
import logging
import os
from pathlib import Path

# --- Path Configurations ---
# Uses pathlib for robust path handling
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / 'data'
LOGS_DIR = BASE_DIR / 'logs'

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
LOG_FILE = LOGS_DIR / 'nfgs_firedetect.log'
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

def extract_coordinates(event):
    """Extract lat/lon from the satellite imagery link"""
    try:
        for link in event.get('events', [{}])[0].get('links', []):
            if link.get('text') == 'Satellite Imagery':
                href = link.get('href', '')
                match = re.search(r'/(-?\d+\.\d+),(-?\d+\.\d+)/', href)
                if match:
                    return {'lat': float(match.group(1)), 'lon': float(match.group(2))}
    except (KeyError, IndexError):
        pass
    return None

def extract_satellite_info(link_href):
    """Extract satellite type and scan info from the imagery URL"""
    try:
        parts = link_href.split('/')
        if len(parts) >= 6:
            return {
                'scan_time': parts[3].replace('%20', ' '),
                'satellite': parts[5] if len(parts) > 5 else None,
                'scan_type': parts[6] if len(parts) > 6 else None,
                'scan_id': parts[7] if len(parts) > 7 else None
            }
    except Exception:
        pass
    return None

def parse_event_id_datetime(event_id):
    """Extract datetime from event ID: 2025-12-30_18-41-54_1005988"""
    try:
        parts = event_id.split('_')
        if len(parts) >= 2:
            date_str = parts[0]
            time_str = parts[1].replace('-', ':')
            return f"{date_str}T{time_str}Z"
    except Exception:
        pass
    return None

def get_missouri_fires_with_coords():
    """Fetch fire events and extract data specifically for Missouri"""
    url = "https://cimss.ssec.wisc.edu/ngfs/alerts-dashboard/api/"
    logging.info(f"Fetching fire events from {url}")
    
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()
        fetch_time = datetime.utcnow().isoformat() + 'Z'
        
        config_info = {
            'last_updated': data.get('config', {}).get('last_updated'),
            'refresh_interval_ms': data.get('config', {}).get('refresh'),
            'latest_event_date_time': data.get('config', {}).get('latest_event_date_time')
        }

        mo_events = []
        for event in data.get('event_list', []):
            # Check for Missouri in location text
            if 'MO' in event.get('location', {}).get('text', ''):
                coords = extract_coordinates(event)
                event_datetime = parse_event_id_datetime(event['id'])
                
                detections = []
                for evt in event.get('events', []):
                    links = []
                    sat_info = None
                    for link in evt.get('links', []):
                        links.append({
                            'text': link.get('text'),
                            'href': link.get('href')
                        })
                        if link.get('text') == 'Satellite Imagery':
                            sat_info = extract_satellite_info(link.get('href', ''))
                    
                    detections.append({
                        'detection_id': evt.get('id'),
                        'detection_time': parse_event_id_datetime(evt.get('id')),
                        'age': evt.get('age'),
                        'frp': evt.get('frp'),
                        'satellite_info': sat_info,
                        'links': links
                    })

                detections.sort(key=lambda x: x.get('detection_time') or '', reverse=True)
                latest = detections[0] if detections else {}

                mo_events.append({
                    'event_id': event.get('id'),
                    'event_datetime': event_datetime,
                    'location': {
                        'county': event['location']['text'],
                        'state': 'Missouri',
                        'coordinates': coords
                    },
                    'wfo': event.get('wfo', {}).get('text'),
                    'fire_info': {
                        'frp': latest.get('frp'),
                        'type': latest.get('type')
                    },
                    'detections': {
                        'count': len(detections),
                        'latest_satellite': latest.get('satellite_info', {}).get('satellite') if latest.get('satellite_info') else None,
                        'all_detections': detections
                    },
                    'urls': {
                        'alert_detail': next((l['href'] for d in detections for l in d['links'] if l['text'] == 'Alert Detail'), None),
                        'satellite_imagery': latest.get('satellite_info', {}).get('href') if latest.get('satellite_info') else None
                    },
                    'metadata': {'fetched_at': fetch_time}
                })

        return {
            'fetched_at': fetch_time,
            'config': config_info,
            'summary': {
                'total_events_nationwide': len(data.get('event_list', [])),
                'missouri_event_count': len(mo_events)
            },
            'events': mo_events
        }

    except Exception as e:
        logging.error(f"Error in data collection: {e}")
        return {'error': str(e), 'events': [], 'summary': {'missouri_event_count': 0}}

def main():
    """Main execution logic"""
    print(f"Starting Missouri fire detection. Logging to: {LOG_FILE}")
    result = get_missouri_fires_with_coords()
    events = result.get('events', [])

    if not events:
        print("No active Missouri fire events found.")
        return result

    # 1. Save standard JSON
    json_path = DATA_DIR / 'missouri_fires_coords.json'
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)

    # 2. Save GeoJSON
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [e['location']['coordinates']['lon'], e['location']['coordinates']['lat']]
                },
                "properties": {**e, "coordinates": None} # Flatten for geojson properties
            } for e in events if e['location']['coordinates']
        ]
    }
    
    geojson_path = DATA_DIR / 'missouri_fires.geojson'
    with open(geojson_path, 'w') as f:
        json.dump(geojson, f, indent=2)

    print(f"✓ Success! Found {len(events)} events.")
    print(f"  - Files saved in: {DATA_DIR}")
    return result

if __name__ == "__main__":
    main()