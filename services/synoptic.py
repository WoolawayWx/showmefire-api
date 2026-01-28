import aiohttp
from datetime import datetime, timedelta, timezone
import os
from dotenv import load_dotenv
import logging
import pandas as pd
import json
from pathlib import Path
# from broadcast import broadcast_update

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
            "within": "70",
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
        # await broadcast_update("synoptic", station_data)
        
    except Exception as e:
        station_data["error"] = str(e)
        logger.error(f"Error fetching station data: {e}")
        # await broadcast_update("synoptic", station_data)

def get_station_data():
    """Return the combined station data"""
    return station_data

async def fetch_raws_stations_multi_state(states=None):
    """
    Fetch all RAWS stations in the specified states.
    Default states: MO, OK, AR, TN, KY, IL, IA, NE, KS.
    Returns a list of flattened station dicts.
    """
    if states is None:
        states = ["MO", "OK", "AR", "TN", "KY", "IL", "IA", "NE", "KS"]

    weather_url = "https://api.synopticdata.com/v2/stations/latest"
    weather_params = {
        "token": SYNOPTIC_API_TOKEN,
        "state": ",".join(states),
        "units": "english",
        "within": "70",
        "status": "active",
        "network": "2"  # Includes RAWS (network 1)
    }

    metadata_url = "https://api.synopticdata.com/v2/stations/metadata"
    metadata_params = {
        "token": SYNOPTIC_API_TOKEN,
        "state": ",".join(states),
        "status": "active",
        "complete": "1",
        "sensorvars": "1",
        "network": "2"
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

    raws_stations = []
    if weather_json.get("STATION"):
        for station in weather_json["STATION"]:
            stid = station.get("STID")
            meta = metadata_lookup.get(stid, {})
            # Only include RAWS stations (network SHORTNAME == 'RAWS')
            if meta.get("SHORTNAME") == "RAWS":
                flattened = flatten_station_data(station, meta)
                raws_stations.append(flattened)

    return raws_stations


async def fetch_historical_station_data(states=None, days_back=1, networks=None, start_time=None, end_time=None):
    """
    Fetch historical data from multiple networks.
    
    Args:
        states: List of state abbreviations (default: MO and surrounding states)
        days_back: Number of days back to fetch (if start_time/end_time not provided)
        networks: List of network IDs:
            1 = RAWS
            2 = Other (includes ASOS)
            156 = Missouri Mesonet
            65 = CWOP
        start_time: Optional datetime object for custom start time
        end_time: Optional datetime object for custom end time
    
    Returns: Raw Synoptic API response
    """
    if states is None:
        states = ["MO", "OK", "AR", "TN", "KY", "IL", "IA", "NE", "KS"]
    
    if networks is None:
        # Only use networks that work with timeseries API
        # 1 = RAWS (works)
        # 153 = ASOS/AWOS might work
        # 156 = Missouri Mesonet might work
        # Test these one at a time!
        networks = [2]  # Start with just RAWS since we know it works
    
    # Use custom times or calculate from days_back
    if end_time is None:
        end_time = datetime.utcnow()
    if start_time is None:
        start_time = end_time - timedelta(days=days_back)
    
    # Format times for Synoptic API (YYYYMMDDhhmm)
    start_str = start_time.strftime("%Y%m%d%H%M")
    end_str = end_time.strftime("%Y%m%d%H%M")
    
    url = "https://api.synopticdata.com/v2/stations/timeseries"
    params = {
        "token": SYNOPTIC_API_TOKEN,
        "state": ",".join(states),
        "start": start_str,
        "end": end_str,
        "units": "english",
        "status": "active",
        "network": ",".join(map(str, networks)),
        "obtimezone": "UTC",
        "vars": "fuel_moisture,relative_humidity,air_temp,wind_speed,wind_gust,solar_radiation,precip_accum"
    }
    
    logger.info(f"Fetching data from {start_str} to {end_str} ({len(states)} states, networks: {networks})")
    
    async with aiohttp.ClientSession() as session:
        response = await session.get(url, params=params, timeout=120)
        response.raise_for_status()
        data = await response.json()
    
    # Add metadata about the request
    data['_metadata'] = {
        'start_time': start_time.isoformat(),
        'end_time': end_time.isoformat(),
        'states': states,
        'networks': networks,
        'fetched_at': datetime.utcnow().isoformat()
    }
    
    return data


def save_raw_data_to_archive(days_back=1, archive_dir="archive/raw_data", states=None, networks=None, obs_start_utc=None, obs_end_utc=None):
    """
    Fetch and save raw API response to archive.
    Saves as JSON for later processing.
    
    Args:
        days_back: Number of days to fetch
        archive_dir: Directory to save raw data
        states: List of state abbreviations
        networks: List of network IDs
        obs_start_utc: (optional) UTC datetime for filtering obs window
        obs_end_utc: (optional) UTC datetime for filtering obs window
    
    Returns: Path to saved file
    """
    archive_path = Path(archive_dir)
    archive_path.mkdir(parents=True, exist_ok=True)
    
    # Fetch data
    logger.info(f"Fetching {days_back} days of raw data...")
    import asyncio
    try:
        api_response = asyncio.run(fetch_historical_station_data(
            days_back=days_back,
            states=states,
            networks=networks
        ))
    except RuntimeError:
        # If already in an event loop (e.g. Jupyter), use create_task
        import nest_asyncio
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        api_response = loop.run_until_complete(fetch_historical_station_data(
            days_back=days_back,
            states=states,
            networks=networks
        ))

    # If obs_start_utc and obs_end_utc are provided, filter observations
    if obs_start_utc is not None and obs_end_utc is not None:
        for station in api_response["STATION"]:
            obs = station.get("OBSERVATIONS", {})
            times = obs.get("date_time", [])
            keep_indices = []
            for idx, t in enumerate(times):
                try:
                    ts = pd.Timestamp(t)
                    if ts.tzinfo is None:
                        ts = ts.tz_localize('UTC')
                    else:
                        ts = ts.tz_convert('UTC')
                    if obs_start_utc <= ts <= obs_end_utc:
                        keep_indices.append(idx)
                except Exception:
                    continue
            # Filter all observation arrays to only keep these indices
            for key in list(obs.keys()):
                if isinstance(obs[key], list) and len(obs[key]) == len(times):
                    obs[key] = [obs[key][i] for i in keep_indices]
            obs["date_time"] = [times[i] for i in keep_indices]
            station["OBSERVATIONS"] = obs
    
    if not api_response.get("STATION"):
        logger.warning("No data received from API")
        return None

    # Ensure all date_time are in UTC ISO8601 format with Z
    for station in api_response["STATION"]:
        obs = station.get("OBSERVATIONS", {})
        times = obs.get("date_time", [])
        new_times = []
        for t in times:
            if isinstance(t, str):
                # Parse the time, assuming it's UTC
                dt = datetime.fromisoformat(t.replace('Z', '+00:00'))
                new_times.append(dt.strftime('%Y-%m-%dT%H:%M:%SZ'))
            else:
                new_times.append(t)
        obs["date_time"] = new_times

    # Always use US/Central time for filename and time fields
    import pytz
    central = pytz.timezone('US/Central')
    now_central = datetime.now(central)
    filename = f"raw_data_{now_central.strftime('%Y%m%d')}.json"
    filepath = archive_path / filename

    # Optionally, add a top-level field to indicate the archive time in Central
    api_response['_archive_time_central'] = now_central.strftime('%Y-%m-%dT%H:%M:%S')

    # Save to JSON
    with open(filepath, 'w') as f:
        json.dump(api_response, f, indent=2)

    station_count = len(api_response.get("STATION", []))
    logger.info(f"Saved raw data ({station_count} stations) to {filepath}")

    return filepath


async def fetch_fuel_moisture_at_time(target_time=None, states=None, networks=None):
    """
    Fetch fuel moisture observations at a specific time from RAWS stations.
    
    Args:
        target_time: datetime object for the target observation time (UTC). 
                    If None, defaults to 7 AM Central Time today.
        states: List of state abbreviations. If None, defaults to MO and surrounding states.
        networks: List of network IDs. If None, defaults to [2] (RAWS).
    
    Returns: Dict with fuel moisture observations from stations near the target time
    """
    import pytz
    
    # Set defaults
    if target_time is None:
        # Default to 7 AM Central Time today
        central = pytz.timezone('US/Central')
        now_central = datetime.now(central)
        target_time = central.localize(datetime(now_central.year, now_central.month, now_central.day, 7, 0, 0))
        target_time = target_time.astimezone(pytz.UTC)
    
    if states is None:
        states = ["MO", "OK", "AR", "TN", "KY", "IL", "IA", "NE", "KS"]
    
    if networks is None:
        networks = [2]  # RAWS network
    
    # Fetch data for a 2-hour window around target time
    start_time = target_time - timedelta(hours=1)
    end_time = target_time + timedelta(hours=1)
    
    logger.info(f"Fetching fuel moisture data around {target_time} from {len(states)} states, networks: {networks}")
    
    # Fetch historical data
    data = await fetch_historical_station_data(
        states=states,
        networks=networks,
        start_time=start_time,
        end_time=end_time
    )
    
    # Log the response for debugging
    logger.info(f"API Response: SUMMARY={data.get('SUMMARY', {})}")
    if data.get("STATION"):
        logger.info(f"Received {len(data['STATION'])} stations from API")
        # Log first station structure for debugging
        if len(data['STATION']) > 0:
            first_station = data['STATION'][0]
            logger.info(f"Sample station STID={first_station.get('STID')}, "
                       f"OBS keys={list(first_station.get('OBSERVATIONS', {}).keys())}")
    else:
        logger.warning(f"No STATION data in response. Keys: {list(data.keys())}")
    
    # Process and flatten the response
    stations_with_fm = []
    if data.get("STATION"):
        for station in data["STATION"]:
            obs = station.get("OBSERVATIONS", {})
            
            # Extract fuel moisture values - check for both possible key formats
            fm_value = None
            obs_time = None
            
            # Try different key formats that Synoptic API might use
            for key in ["fuel_moisture_value_1", "fuel_moisture_set_1", "fuel_moisture"]:
                if key in obs:
                    fm_data = obs[key]
                    if isinstance(fm_data, dict):
                        fm_value = fm_data.get("value")
                    elif isinstance(fm_data, list) and len(fm_data) > 0:
                        fm_value = fm_data[0]
                    elif isinstance(fm_data, (int, float)):
                        fm_value = fm_data
                    
                    if fm_value is not None:
                        break
            
            # Get observation time
            if obs.get("date_time"):
                date_time = obs["date_time"]
                if isinstance(date_time, list) and len(date_time) > 0:
                    obs_time = date_time[0]
                else:
                    obs_time = date_time
            
            # Only include stations that have fuel moisture data
            if fm_value is not None and fm_value > 0:
                station_info = {
                    "stid": station.get("STID"),
                    "name": station.get("NAME"),
                    "state": station.get("STATE"),
                    "latitude": float(station.get("LATITUDE", 0)),
                    "longitude": float(station.get("LONGITUDE", 0)),
                    "elevation": float(station.get("ELEVATION", 0)) if station.get("ELEVATION") else None,
                    "network": station.get("MNET_SHORTNAME"),
                    "observation_time": obs_time,
                    "observations": {
                        "fuel_moisture": {
                            "value": fm_value
                        }
                    }
                }
                stations_with_fm.append(station_info)
    
    result = {
        "target_time": target_time.isoformat(),
        "target_time_formatted": target_time.strftime("%Y-%m-%d %H:%M UTC"),
        "station_count": len(stations_with_fm),
        "stations": stations_with_fm,
        "fetched_at": datetime.utcnow().isoformat()
    }
    
    logger.info(f"Found {len(stations_with_fm)} stations with fuel moisture data")
    
    return result



