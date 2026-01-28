import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import get_unprocessed_snapshots, save_hrrr_features, get_all_stations, mark_snapshot_processed, set_hrrr_filename

logger = logging.getLogger(__name__)

HRRR_DIR = Path("cache/hrrr")

def get_nearest_indices(ds, lat, lon):
    """Finds the x, y indices for a given lat/lon with 0-360 normalization."""
    # HRRR longitude is usually 0-360. 
    # If the file shows values > 180, convert our negative lon to positive.
    ds_lon_max = float(ds.longitude.max())
    search_lon = lon + 360 if (ds_lon_max > 180 and lon < 0) else lon
    
    # Standard distance math
    dist = (ds.latitude - lat)**2 + (ds.longitude - search_lon)**2
    obj = dist.argmin(dim=['y', 'x'])
    return int(obj['x']), int(obj['y'])

def extract_at_indices(ds, x, y):
    # Use isel(step=0) to ignore the clock entirely and just grab the first available slice
    point = ds.isel(x=x, y=y, step=0)
    
    u = float(point['u10'].values)
    v = float(point['v10'].values)
    
    # Extract precipitation if available
    precip_mm = 0.0
    try:
        if 'apcp' in ds:
            precip_mm = float(point['apcp'].values)
        elif 'APCP' in ds:
            precip_mm = float(point['APCP'].values)
        elif 'tp' in ds:
            precip_mm = float(point['tp'].values)
    except Exception as e:
        logger.warning(f"Could not extract precipitation: {e}")
    
    return {
        "temp_c": float(point['t2m'].values) - 273.15,
        "rel_humidity": float(point['r2'].values),
        "wind_speed_ms": (u**2 + v**2)**0.5,
        "precip_mm": precip_mm
    }

def run_miner():
    stations = get_all_stations()
    to_process = get_unprocessed_snapshots()
    
    logger.info(f"Found {len(to_process)} unprocessed snapshots.")
    if not to_process:
        logger.info("No new snapshots to process.")
        return

    station_indices = None

    for row in to_process:
        logger.info(f"ROW: {dict(row)}")
        clean_date = row['snapshot_date'].replace('-', '')
        logger.info(f"Looking for files with: {clean_date}")
        matching_files = list(HRRR_DIR.glob(f"*{clean_date}*.nc"))
        logger.info(f"Found files: {[str(f) for f in matching_files]}")
        if not matching_files:
            logger.warning(f"No HRRR file found for {row['snapshot_date']} (id={row['id']})")
            continue

        nc_path = matching_files[0]
        logger.info(f"Using HRRR file: {nc_path}")

        if not nc_path.exists():
            logger.warning(f"HRRR file does not exist on disk: {nc_path}")
            continue

        try:
            with xr.open_dataset(nc_path, engine='netcdf4', drop_variables=['step']) as ds:
                if station_indices is None:
                    logger.info("Calibrating station grid indices...")
                    station_indices = {
                        s['id']: get_nearest_indices(ds, s['lat'], s['lon']) 
                        for s in stations
                    }
                    logger.info(f"Station indices: {station_indices}")

                for station in stations:
                    x, y = station_indices[station['id']]
                    logger.info(f"Extracting for station {station['id']} at indices x={x}, y={y}")

                    extracted_dict = extract_at_indices(ds, x, y)
                    logger.info(f"Extracted data for station {station['id']}: {extracted_dict}")

                    if extracted_dict:
                        save_hrrr_features(row['id'], extracted_dict, station['id'])
                        logger.info(f"Saved features for snapshot {row['id']} and station {station['id']}")

            set_hrrr_filename(row['id'], nc_path.name)  # <-- Add this line
            mark_snapshot_processed(row['id'])
            logger.info(f"✅ Successfully finished all stations for: {row['snapshot_date']} (snapshot id: {row['id']})")

        except Exception as e:
            logger.error(f"Error processing {row['hrrr_filename']}: {e}", exc_info=True)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_miner()