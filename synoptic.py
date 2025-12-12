import aiohttp
from datetime import datetime, timezone
import os
from dotenv import load_dotenv
import logging
from broadcast import broadcast_update

logger = logging.getLogger(__name__)

station_data = {
    "stations": None,
    "last_updated": None,
    "error": None
}

load_dotenv()
SYNOPTIC_API_TOKEN = os.getenv("SYNOPTIC_API_TOKEN")

def flatten_station_data(weather_station, metadata_station):
    """Flatten and deduplicate station data for optimized storage"""
    
    station = {
        "id": weather_station.get("ID"),
        "stid": weather_station.get("STID"),
        "name": weather_station.get("NAME"),
        "state": weather_station.get("STATE"),
        "country": weather_station.get("COUNTRY"),
        "county": metadata_station.get("COUNTY") if metadata_station else None,
        "latitude": float(weather_station.get("LATITUDE", 0)),
        "longitude": float(weather_station.get("LONGITUDE", 0)),
        "elevation": float(weather_station.get("ELEVATION", 0)) if weather_station.get("ELEVATION") else None,
        "elevation_dem": float(weather_station.get("ELEV_DEM", 0)) if weather_station.get("ELEV_DEM") else None,
        "timezone": weather_station.get("TIMEZONE"),
        "status": weather_station.get("STATUS"),
        "qc_flagged": weather_station.get("QC_FLAGGED", False),
        "restricted": weather_station.get("RESTRICTED", False),
    }
    
    if metadata_station:
        station["nws_zone"] = metadata_station.get("NWSZONE")
        station["nws_fire_zone"] = metadata_station.get("NWSFIREZONE")
        station["gacc"] = metadata_station.get("GACC")
        station["cwa"] = metadata_station.get("CWA")
        station["wims_id"] = metadata_station.get("WIMS_ID")
        station["network"] = metadata_station.get("SHORTNAME")
        station["network_name"] = metadata_station.get("LONGNAME")
        
        providers = metadata_station.get("PROVIDERS", [])
        if providers:
            station["providers"] = [{"name": p.get("name"), "url": p.get("url")} for p in providers]
    
    por = weather_station.get("PERIOD_OF_RECORD", {})
    station["record_start"] = por.get("start")
    station["record_end"] = por.get("end")
    
    observations = {}
    raw_obs = weather_station.get("OBSERVATIONS", {})
    for key, data in raw_obs.items():
        if isinstance(data, dict) and "value" in data:
            clean_key = key.replace("_value_1d", "").replace("_value_1", "")
            if clean_key not in observations or not key.endswith("_value_1d"):
                observations[clean_key] = {
                    "value": data.get("value"),
                    "time": data.get("date_time"),
                    "derived": key.endswith("_value_1d")
                }
    
    station["observations"] = observations
    
    sensors = {}
    weather_sensor_vars = weather_station.get("SENSOR_VARIABLES", {})
    metadata_sensor_vars = metadata_station.get("SENSOR_VARIABLES", {}) if metadata_station else {}
    
    for sensor_type in weather_sensor_vars.keys():
        sensor_info = {"available": True}
        
        if sensor_type in metadata_sensor_vars:
            meta_sensor = metadata_sensor_vars[sensor_type]
            for variant_key, variant_data in meta_sensor.items():
                if isinstance(variant_data, dict):
                    if variant_data.get("position"):
                        sensor_info["height_m"] = float(variant_data["position"])
                    if variant_data.get("summary"):
                        sensor_info["summary"] = variant_data["summary"]
                    sensor_por = variant_data.get("PERIOD_OF_RECORD", {})
                    if sensor_por.get("start"):
                        sensor_info["record_start"] = sensor_por["start"]
                    if sensor_por.get("end"):
                        sensor_info["record_end"] = sensor_por["end"]
                    break
        
        sensors[sensor_type] = sensor_info
    
    station["sensors"] = sensors
    
    return station

async def fetch_synoptic_data():
    """Fetch weather data and metadata from Synoptic API and combine them"""
    global station_data
    
    try:
        weather_url = "https://api.synopticdata.com/v2/stations/latest"
        weather_params = {
            "token": SYNOPTIC_API_TOKEN,
            "state": "MO",
            "units": "english",
            "within": "60",
            "status": "active",
            "network": "1,2,156,65"
        }
        
        metadata_url = "https://api.synopticdata.com/v2/stations/metadata"
        metadata_params = {
            "token": SYNOPTIC_API_TOKEN,
            "state": "MO",
            "status": "active",
            "complete": "1",
            "sensorvars": "1",
            "network": "1,2,156,65"
        }
        
        async with aiohttp.ClientSession() as session:
            weather_response = await session.get(weather_url, params=weather_params, timeout=30)
            weather_response.raise_for_status()
            weather_json = await weather_response.json()
            
            metadata_response = await session.get(metadata_url, params=metadata_params, timeout=30)
            metadata_response.raise_for_status()
            metadata_json = await metadata_response.json()
        
        metadata_lookup = {}
        if metadata_json.get("STATION"):
            for station in metadata_json["STATION"]:
                stid = station.get("STID")
                if stid:
                    metadata_lookup[stid] = station
        
        combined_stations = []
        if weather_json.get("STATION"):
            for station in weather_json["STATION"]:
                stid = station.get("STID")
                flattened = flatten_station_data(station, metadata_lookup.get(stid, {}))
                combined_stations.append(flattened)
        
        station_data["stations"] = combined_stations
        station_data["count"] = len(combined_stations)
        station_data["last_updated"] = datetime.now(timezone.utc).isoformat()
        station_data["error"] = None
        
        logger.info(f"[{station_data['last_updated']}] Station data updated successfully ({len(combined_stations)} stations)")
        
        # Broadcast update to WebSocket clients
        await broadcast_update("synoptic", station_data)
        
    except Exception as e:
        station_data["error"] = str(e)
        logger.error(f"Error fetching station data: {e}")
        await broadcast_update("synoptic", station_data)

def get_station_data():
    """Return the combined station data"""
    return station_data
