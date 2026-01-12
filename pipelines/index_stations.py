import xarray as xr
import pandas as pd
import sqlite3
from core.database import get_db_path

def index_stations(sample_nc_path):
    conn = sqlite3.connect(get_db_path())
    # 1. Get all unique stations from your observations
    stations_df = pd.read_sql("SELECT DISTINCT station_id, latitude, longitude FROM observations", conn)
    
    # 2. Open a single HRRR file to use as a "Map"
    ds = xr.open_dataset(sample_nc_path, drop_variables=['step'])
    
    print(f"Mapping {len(stations_df)} stations to HRRR grid...")
    
    indices = []
    for _, row in stations_df.iterrows():
        # Find the pixel for this specific station
        dist = (ds.latitude - row['latitude'])**2 + (ds.longitude - row['longitude'])**2
        obj = dist.argmin(dim=['y', 'x'])
        
        indices.append((
            row['station_id'], 
            int(obj['x']), 
            int(obj['y']), 
            row['latitude'], 
            row['longitude']
        ))
    
    # 3. Save to the cache table
    cursor = conn.cursor()
    cursor.executemany(
        "INSERT OR REPLACE INTO station_grid_indices VALUES (?, ?, ?, ?, ?)", 
        indices
    )
    conn.commit()
    conn.close()
    print("✅ All stations indexed. Miner can now run at 100x speed.")

if __name__ == "__main__":
    # Use one of your verified files as the map
    index_stations("cache/hrrr/hrrr_20260103_12z_f04-15.nc")