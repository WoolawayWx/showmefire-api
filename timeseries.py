import requests
from datetime import datetime
import os
from dotenv import load_dotenv

timeseriesdata = {
    "stations": None,
    "last_updated": None,
    "error": None
}

load_dotenv()
SYNOPTIC_API_TOKEN = os.getenv("SYNOPTIC_API_TOKEN")

def flatten_timeseries_data(response_json):
    """Extract only STID, values, and timestamps to minimize file size"""
    flattened = []
    
    stations = response_json.get("STATION", [])
    for station in stations:
        stid = station.get("STID")
        observations = station.get("OBSERVATIONS", {})
        
        # Build a minimal station record with just what's needed
        station_record = {
            "stid": stid,
            "observations": {}
        }
        
        # Extract only values and timestamps
        for obs_key, obs_data in observations.items():
            if isinstance(obs_data, list):
                # Observations are typically arrays of [value, timestamp] pairs
                station_record["observations"][obs_key] = obs_data
            elif isinstance(obs_data, dict) and "value" in obs_data:
                # Sometimes they're dicts with value and date_time
                station_record["observations"][obs_key] = {
                    "value": obs_data.get("value"),
                    "time": obs_data.get("date_time")
                }
        
        flattened.append(station_record)
    
    return flattened

async def fetchtimeseriesdata():
    global timeseriesdata
    
    try:
        url = "https://api.synopticdata.com/v2/stations/timeseries"
        url_params = {
            "token": SYNOPTIC_API_TOKEN,
            "state": "MO",
            "status": "active",
            "recent": "300",
            "obtimezon": "local",
            "network": "1,2,156,65",
            "units": "english",
        }
        
        url_response = requests.get(url, params=url_params, timeout=30)
        url_response.raise_for_status()
        response_json = url_response.json()
        
        # Flatten the data to reduce size
        flattened = flatten_timeseries_data(response_json)
        timeseriesdata["stations"] = flattened
        timeseriesdata["last_updated"] = datetime.now().isoformat()
        timeseriesdata["error"] = None
        
        print(f"[{timeseriesdata['last_updated']}] Timeseries data updated successfully")
        
    except Exception as e:
        timeseriesdata["error"] = str(e)
        print(f"Error fetching timeseries data: {e}")
    
    
def get_timeseries_data():
    return timeseriesdata