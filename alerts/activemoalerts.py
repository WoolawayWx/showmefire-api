import requests
import os
import json
from pathlib import Path

# Define the API endpoint
API_URL = 'https://api.weather.gov/alerts/active?area=MO'
MO_FIRE_ZONES_PATH = '/app/gis/MOFireWxZones.geojson'
# NWS public forecast zones: flood/heat/winter/etc. alerts reference
# zones/forecast/MOZxxx, not the fire weather zones above.
MO_PUBLIC_ZONES_PATH = '/app/gis/MOPublicZones.geojson'
ACTIVE_JSON_PATH = '/app/gis/active.json'
MO_COUNTY_SHP_PATH = str(
    Path(__file__).resolve().parent.parent / 'maps' / 'shapefiles' / 'MO_County_Boundaries' / 'MO_County_Boundaries.shp'
)
# NWS rejects/throttles requests without an identifying User-Agent.
NWS_HEADERS = {
    'User-Agent': 'ShowMeFire/1.0 (alerts@showmefire.org)',
    'Accept': 'application/geo+json',
}
FETCH_TIMEOUT_SECONDS = 20

def fetch_weather_data(api_url):
    try:
        response = requests.get(api_url, headers=NWS_HEADERS, timeout=FETCH_TIMEOUT_SECONDS)
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

_county_zone_cache = None

def load_county_zone_lookup():
    """Zone URL -> GeoJSON-like feature for Missouri counties (zones/county/MOC###).

    Severe Thunderstorm/Tornado *Watches* (and county-based warnings) reference
    county zones, which are in neither the fire nor public forecast zone files.
    The NWS county zone id is MOC + the 3-digit county FIPS.
    """
    global _county_zone_cache
    if _county_zone_cache is not None:
        return _county_zone_cache
    lookup = {}
    try:
        import geopandas as gpd
        counties = gpd.read_file(MO_COUNTY_SHP_PATH).to_crs('EPSG:4326')
        for _, row in counties.iterrows():
            fips = str(row.get('COUNTYFIPS') or '').zfill(3)
            if fips and row.geometry is not None:
                lookup[f"https://api.weather.gov/zones/county/MOC{fips}"] = {
                    'geometry': row.geometry.__geo_interface__,
                }
    except Exception as e:
        print(f"Error loading county zones: {e}")
        return {}  # don't cache a failure; retry next poll
    _county_zone_cache = lookup
    return lookup

def enrich_alerts_with_zones(alerts, zones, public_zones=None, county_zones=None):
    # Build lookup: zone URL -> geometry/properties
    zone_lookup = dict(county_zones or {})
    for zone in zones.get('features', []):
        zone_id = f"https://api.weather.gov/zones/fire/MOZ{zone['properties']['ZONE']}"
        zone_lookup[zone_id] = zone
    for zone in (public_zones or {}).get('features', []):
        zone_id = f"https://api.weather.gov/zones/forecast/MOZ{zone['properties']['ZONE']}"
        zone_lookup[zone_id] = zone

    enriched_features = []
    for alert in alerts.get('features', []):
        # Warnings such as Severe Thunderstorm/Tornado/Flash Flood arrive with
        # their own storm-based polygon. That is more precise than the zones,
        # so never replace it (this used to null it out when no zone matched).
        if alert.get('geometry'):
            enriched_features.append(alert)
            continue
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
        
def run_active_mo_alerts(api_url=API_URL, zones_path=MO_FIRE_ZONES_PATH, out_path=ACTIVE_JSON_PATH, public_zones_path=MO_PUBLIC_ZONES_PATH):
    alerts = fetch_weather_data(api_url)
    zones = load_zones(zones_path)
    public_zones = load_zones(public_zones_path)  # optional: without it only fire-zone alerts get geometry
    if alerts and zones:
        enriched_alerts = enrich_alerts_with_zones(
            alerts, zones, public_zones, load_county_zone_lookup(),
        )
        save_json_to_file(enriched_alerts, out_path)
        # Re-render only the advisory maps whose alerts actually changed, so a
        # new/updated/cancelled warning shows up within a poll rather than
        # waiting for the 15-minute map cron.
        try:
            from maps.advisory_category_maps import refresh_changed_advisory_maps
            refresh_changed_advisory_maps(enriched_alerts)
        except Exception as exc:
            print(f"Advisory map refresh failed: {exc}")
        try:
            from services.mobile_content import active_fire_weather_alerts, record_daily_fire_weather_alerts
            from services.mobile_push import process_fire_weather_alerts
            fire_weather_alerts = active_fire_weather_alerts(Path(out_path))
            process_fire_weather_alerts(fire_weather_alerts)
            record_daily_fire_weather_alerts(fire_weather_alerts)
        except Exception as exc:
            print(f"Mobile fire weather notification processing failed: {exc}")
        try:
            from services.fire_weather_alert_map import maybe_regenerate_fire_weather_alert_map
            maybe_regenerate_fire_weather_alert_map()
        except Exception as exc:
            print(f"Fire weather alert map regeneration failed: {exc}")
        # After the map regenerates, so the posted image shows the new alert.
        try:
            from services.discord_notifier import process_fire_weather_alerts_for_discord
            from services.mobile_content import active_fire_weather_alerts
            process_fire_weather_alerts_for_discord(active_fire_weather_alerts(Path(out_path)))
        except Exception as exc:
            print(f"Discord fire weather alert posting failed: {exc}")
        return True
    return False

if __name__ == "__main__":
    # Print the current working directory
    print("Current Working Directory:", os.getcwd())

    # Change to the expected directory if necessary
    os.chdir('/app')

    run_active_mo_alerts()
