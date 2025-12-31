import requests
import json
import re
from datetime import datetime

def extract_coordinates(event):
    """Extract lat/lon from the satellite imagery link"""
    try:
        # Get the first event's satellite imagery link
        for link in event['events'][0]['links']:
            if link['text'] == 'Satellite Imagery':
                href = link['href']
                # Extract coordinates from URL pattern: /lat,lon/
                match = re.search(r'/(-?\d+\.\d+),(-?\d+\.\d+)/', href)
                if match:
                    lat = float(match.group(1))
                    lon = float(match.group(2))
                    return {'lat': lat, 'lon': lon}
    except (KeyError, IndexError):
        pass
    
    return None

def extract_satellite_info(link_href):
    """Extract satellite type and scan info from the imagery URL"""
    try:
        # Example: /map/realtime/2025-12-31%2000:41:19/37.632221,-93.427223/GOES-19 ABI/CONUS/1006000
        parts = link_href.split('/')
        if len(parts) >= 6:
            scan_time = parts[3].replace('%20', ' ')  # Datetime of satellite scan
            satellite = parts[5] if len(parts) > 5 else None  # e.g., "GOES-19 ABI"
            scan_type = parts[6] if len(parts) > 6 else None  # e.g., "CONUS" or "Mesoscale2"
            scan_id = parts[7] if len(parts) > 7 else None
            
            return {
                'scan_time': scan_time,
                'satellite': satellite,
                'scan_type': scan_type,
                'scan_id': scan_id
            }
    except:
        pass
    return None

def parse_event_id_datetime(event_id):
    """Extract datetime from event ID format: 2025-12-30_18-41-54_1005988"""
    try:
        # Split by underscore and get date and time parts
        parts = event_id.split('_')
        if len(parts) >= 2:
            date_str = parts[0]  # 2025-12-30
            time_str = parts[1].replace('-', ':')  # 18:41:54
            datetime_str = f"{date_str}T{time_str}Z"
            return datetime_str
    except:
        pass
    return None

def get_missouri_fires_with_coords():
    """Fetch fire events and extract coordinates"""
    url = "https://cimss.ssec.wisc.edu/ngfs/alerts-dashboard/api/"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        fetch_time = datetime.utcnow().isoformat() + 'Z'
        
        # Extract config info
        config_info = {
            'last_updated': data.get('config', {}).get('last_updated'),
            'refresh_interval_ms': data.get('config', {}).get('refresh'),
            'pause_timeout_ms': data.get('config', {}).get('pause_timeout'),
            'latest_event_date_time': data.get('config', {}).get('latest_event_date_time')
        }
        
        # Filter for Missouri and add coordinates
        mo_events = []
        for event in data['event_list']:
            if 'MO' in event['location']['text']:
                coords = extract_coordinates(event)
                event_datetime = parse_event_id_datetime(event['id'])
                
                # Get all detection details with full information
                detections = []
                for evt in event['events']:
                    # Extract all links
                    links = []
                    satellite_info = None
                    
                    for link in evt.get('links', []):
                        link_detail = {
                            'text': link.get('text'),
                            'tooltip': link.get('tooltip'),
                            'link_type': link.get('link_type'),
                            'href': link.get('href')
                        }
                        links.append(link_detail)
                        
                        # Extract satellite info from imagery link
                        if link.get('text') == 'Satellite Imagery':
                            satellite_info = extract_satellite_info(link.get('href', ''))
                    
                    detection = {
                        'detection_id': evt.get('id'),
                        'detection_time': parse_event_id_datetime(evt.get('id')),
                        'age': evt.get('age'),
                        'type': evt.get('type'),
                        'frp': evt.get('frp'),
                        'frp_unit': 'MW',  # Fire Radiative Power in Megawatts
                        'importance': evt.get('importance'),
                        'satellite_info': satellite_info,
                        'links': links
                    }
                    detections.append(detection)
                
                # Sort detections by time (most recent first)
                detections.sort(key=lambda x: x.get('detection_time') or '', reverse=True)
                
                # Get the most recent detection info
                latest_detection = detections[0] if detections else {}
                
                # Create comprehensive event object
                simplified = {
                    # Event identification
                    'event_id': event.get('id'),
                    'event_datetime': event_datetime,
                    
                    # Location information
                    'location': {
                        'county': event['location']['text'],
                        'state': 'Missouri',
                        'state_abbr': 'MO',
                        'country': event.get('country', '').replace('Country:  ', ''),
                        'coordinates': coords
                    },
                    
                    # Weather Forecast Office
                    'wfo': {
                        'name': event['wfo']['text'],
                        'full_name': event['wfo']['text']
                    },
                    
                    # Event metadata
                    'age': event.get('age'),
                    'importance': event.get('importance'),
                    'highlight': event.get('highlight', False),
                    
                    # Fire characteristics (from most recent detection)
                    'fire_info': {
                        'frp': latest_detection.get('frp'),
                        'frp_unit': 'MW',
                        'type': latest_detection.get('type'),
                        'risk_level': 'Nominal' if 'Nominal' in latest_detection.get('type', '') else 'Unknown'
                    },
                    
                    # Detection information
                    'detections': {
                        'count': len(event['events']),
                        'latest_detection_time': latest_detection.get('detection_time'),
                        'latest_satellite': latest_detection.get('satellite_info', {}).get('satellite') if latest_detection.get('satellite_info') else None,
                        'latest_scan_type': latest_detection.get('satellite_info', {}).get('scan_type') if latest_detection.get('satellite_info') else None,
                        'all_detections': detections
                    },
                    
                    # URLs and links
                    'urls': {
                        'alert_detail': next((link['href'] for det in event['events'] 
                                            for link in det.get('links', [])
                                            if link.get('text') == 'Alert Detail'), None),
                        'satellite_imagery': next((link['href'] for det in event['events']
                                                 for link in det.get('links', [])
                                                 if link.get('text') == 'Satellite Imagery'), None),
                        'base_url': 'https://cimss.ssec.wisc.edu/ngfs'
                    },
                    
                    # Metadata
                    'metadata': {
                        'fetched_at': fetch_time,
                        'data_source': 'CIMSS NGFS Alerts Dashboard',
                        'api_url': url
                    }
                }
                mo_events.append(simplified)
        
        return {
            'fetched_at': fetch_time,
            'config': config_info,
            'summary': {
                'total_events_nationwide': len(data['event_list']),
                'missouri_event_count': len(mo_events),
                'data_source': 'CIMSS SSEC NGFS (Next Generation Fire System)',
                'api_version': 'alerts-dashboard/api'
            },
            'events': mo_events
        }
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return {
            'fetched_at': datetime.utcnow().isoformat() + 'Z',
            'error': str(e),
            'summary': {
                'missouri_event_count': 0
            },
            'events': []
        }
    
def main():
    result = get_missouri_fires_with_coords()
    events = result.get('events', [])
    
    if events:
        # Save comprehensive JSON data
        with open('data/missouri_fires_coords.json', 'w') as f:
            json.dump(result, f, indent=2)
        
        # Create detailed GeoJSON for mapping
        geojson = {
            "type": "FeatureCollection",
            "metadata": {
                "title": "Missouri Fire Detections",
                "description": "Real-time fire detections from CIMSS NGFS for Missouri",
                "fetched_at": result['fetched_at'],
                "data_source": "CIMSS SSEC Next Generation Fire System",
                "total_events_nationwide": result.get('summary', {}).get('total_events_nationwide', 0),
                "missouri_event_count": result.get('summary', {}).get('missouri_event_count', 0),
                "config": result.get('config', {})
            },
            "features": [
                {
                    "type": "Feature",
                    "id": e['event_id'],
                    "geometry": {
                        "type": "Point",
                        "coordinates": [
                            e['location']['coordinates']['lon'], 
                            e['location']['coordinates']['lat']
                        ]
                    },
                    "properties": {
                        # Identification
                        "event_id": e['event_id'],
                        "event_datetime": e['event_datetime'],
                        
                        # Location
                        "county": e['location']['county'],
                        "state": e['location']['state'],
                        "state_abbr": e['location']['state_abbr'],
                        "country": e['location']['country'],
                        "latitude": e['location']['coordinates']['lat'],
                        "longitude": e['location']['coordinates']['lon'],
                        
                        # WFO
                        "wfo": e['wfo']['name'],
                        
                        # Event details
                        "age": e['age'],
                        "importance": e['importance'],
                        "highlight": e['highlight'],
                        
                        # Fire characteristics
                        "frp": e['fire_info']['frp'],
                        "frp_unit": e['fire_info']['frp_unit'],
                        "fire_type": e['fire_info']['type'],
                        "risk_level": e['fire_info']['risk_level'],
                        
                        # Detection info
                        "detection_count": e['detections']['count'],
                        "latest_detection_time": e['detections']['latest_detection_time'],
                        "latest_satellite": e['detections']['latest_satellite'],
                        "latest_scan_type": e['detections']['latest_scan_type'],
                        
                        # URLs
                        "alert_url": e['urls']['alert_detail'],
                        "satellite_imagery_url": e['urls']['satellite_imagery'],
                        
                        # Metadata
                        "fetched_at": e['metadata']['fetched_at'],
                        "data_source": e['metadata']['data_source']
                    }
                }
                for e in events if e['location']['coordinates']
            ]
        }
        
        with open('data/missouri_fires.geojson', 'w') as f:
            json.dump(geojson, f, indent=2)
        
        print(f"✓ Saved {len(events)} Missouri fire events")
        print(f"  - JSON: data/missouri_fires_coords.json")
        print(f"  - GeoJSON: data/missouri_fires.geojson")
    else:
        print("No Missouri fire events found")
    
    return result


if __name__ == "__main__":
    result = main()
    
    if result.get('events'):
        print(f"\nSummary:")
        print(f"  Missouri events: {result['summary']['missouri_event_count']}")
        print(f"  Total nationwide: {result['summary']['total_events_nationwide']}")
        print(f"  Fetched at: {result['fetched_at']}")
    else:
        print("No Missouri fire events found")