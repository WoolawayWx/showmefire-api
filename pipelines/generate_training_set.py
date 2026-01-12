import sqlite3
import pandas as pd
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import get_db_path

def generate_training_set():
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    
    # We join observations to weather_features.
    # We match the DATE of the observation to the DATE in snapshots.
    # Note: If your weather_features table doesn't have an 'hour' column, 
    # we match on the date and use the latitude/longitude for spatial alignment.
    query = """
    SELECT 
        o.station_id,
        o.observation_date as obs_time,
        o.fuel_moisture_percentage as target_fm,
        wf.temp_c,
        wf.rel_humidity,
        wf.wind_speed_ms,
        wf.precip_mm,
        s.lat,
        s.lon
    FROM observations o
    JOIN stations s ON o.station_id = s.id
    JOIN weather_features wf ON o.station_id = wf.station_id
    JOIN snapshots snap ON wf.snapshot_id = snap.id
    WHERE 
        DATE(o.observation_date) = snap.snapshot_date
    """
    
    df = pd.read_sql(query, conn)
    
    # Optional: If the join produces duplicates (multiple weather rows for one obs),
    # we group by station and time and take the mean.
    df = df.groupby(['station_id', 'obs_time']).mean().reset_index()
    
    conn.close()
    
    if df.empty:
        print("⚠️ No matching rows found.")
    else:
        df.to_csv("data/training_set_mo.csv", index=False)
        print(f"✅ Created training set with {len(df)} rows at data/training_set_mo.csv")

if __name__ == "__main__":
    generate_training_set()