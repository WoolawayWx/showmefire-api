import xarray as xr
import pandas as pd
import sqlite3
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import get_db_path

def index_stations(sample_nc_path=None):
    """
    Index stations to HRRR grid coordinates.
    
    Args:
        sample_nc_path: Path to HRRR file to use as reference. If None, uses most recent file.
    """
    # If no path provided, find the most recent HRRR file
    if sample_nc_path is None:
        hrrr_dir = Path("cache/hrrr")
        if not hrrr_dir.exists():
            print("❌ HRRR cache directory not found")
            return
            
        nc_files = list(hrrr_dir.glob("*.nc"))
        if not nc_files:
            print("❌ No HRRR files found in cache/hrrr/")
            return
            
        # Use the most recent file
        sample_nc_path = max(nc_files, key=lambda x: x.stat().st_mtime)
        print(f"📍 Using most recent HRRR file: {sample_nc_path}")
    
    conn = sqlite3.connect(get_db_path())
    # 1. Get all unique stations from your observations
    stations_df = pd.read_sql("SELECT DISTINCT station_id, latitude, longitude FROM observations", conn)
    
    if len(stations_df) == 0:
        print("❌ No stations found in database")
        conn.close()
        return
    
    # 2. Open a single HRRR file to use as a "Map"
    ds = xr.open_dataset(sample_nc_path, drop_variables=['step'])
    
    print(f"🗺️ Mapping {len(stations_df)} stations to HRRR grid...")
    
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
    
    # Ensure table exists
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS station_grid_indices (
            station_id TEXT PRIMARY KEY,
            grid_x INTEGER,
            grid_y INTEGER,
            original_lat REAL,
            original_lon REAL
        )
    ''')

    cursor.executemany(
        "INSERT OR REPLACE INTO station_grid_indices VALUES (?, ?, ?, ?, ?)", 
        indices
    )
    conn.commit()
    conn.close()
    print("✅ All stations indexed. Miner can now run at 100x speed.")

if __name__ == "__main__":
    # Use the most recent HRRR file automatically
    index_stations()