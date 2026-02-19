import requests
import os
import json

# Define the API endpoint
API_URL = 'https://api.weather.gov/alerts/active?area=MO'
MO_FIRE_ZONES_PATH = '/app/gis/MOFireWxZones.geojson'
ACTIVE_JSON_PATH = '/app/gis/active.json'

def fetch_weather_data(api_url):
    try:
        response = requests.get(api_url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def load_zones(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading zones: {e}")
        return None

def enrich_alerts_with_zones(alerts, zones):
    # Build lookup: zone URL -> geometry/properties
    zone_lookup = {}
    for zone in zones.get('features', []):
        zone_id = f"https://api.weather.gov/zones/fire/MOZ{zone['properties']['ZONE']}"
        zone_lookup[zone_id] = zone

    enriched_features = []
    for alert in alerts.get('features', []):
        matched_coords = []
        for zone_url in alert.get('properties', {}).get('affectedZones', []):
            zone = zone_lookup.get(zone_url)
            if zone and zone.get('geometry'):
                geom = zone['geometry']
                # Support both Polygon and MultiPolygon
                if geom['type'] == 'Polygon':
                    matched_coords.append(geom['coordinates'])
                elif geom['type'] == 'MultiPolygon':
                    matched_coords.extend(geom['coordinates'])
        # Attach all matched geometries as a MultiPolygon
        if matched_coords:
            alert['geometry'] = {
                'type': 'MultiPolygon',
                'coordinates': matched_coords
            }
        else:
            alert['geometry'] = None
        enriched_features.append(alert)
    alerts['features'] = enriched_features
    return alerts

def save_json_to_file(data, file_path):
    try:
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        print(f"Data saved to {file_path}")
    except Exception as e:
        print(f"Error saving data: {e}")
        
def run_active_mo_alerts(api_url=API_URL, zones_path=MO_FIRE_ZONES_PATH, out_path=ACTIVE_JSON_PATH):
    alerts = fetch_weather_data(api_url)
    zones = load_zones(zones_path)
    if alerts and zones:
        enriched_alerts = enrich_alerts_with_zones(alerts, zones)
        save_json_to_file(enriched_alerts, out_path)
        return True
    return False

if __name__ == "__main__":
    # Print the current working directory
    print("Current Working Directory:", os.getcwd())

    # Change to the expected directory if necessary
    os.chdir('/app')

    run_active_mo_alerts()
