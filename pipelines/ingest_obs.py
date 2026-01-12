import json
import sqlite3
from pathlib import Path

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.database import get_db_path

def clear_tables():
    """Clears existing data to start fresh with MO-only stations."""
    conn = sqlite3.connect(get_db_path())
    cur = conn.cursor()
    cur.execute("DELETE FROM observations")
    cur.execute("DELETE FROM stations")
    # Optional: Reset auto-increment counters
    cur.execute("DELETE FROM sqlite_sequence WHERE name IN ('observations', 'stations')")
    conn.commit()
    conn.close()
    print("🗑️ Database cleared of previous station and observation data.")

def ingest_archive():
    db_path = get_db_path()
    raw_data_path = Path("archive/raw_data")
    
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    json_files = list(raw_data_path.glob("*.json"))
    print(f"Found {len(json_files)} files to ingest.")

    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            stations = data.get('STATION', []) if isinstance(data, dict) else data
            if isinstance(data, dict) and not stations: 
                stations = [data]

            for st in stations:
                # --- FILTER: Only allow Missouri stations ---
                if st.get('STATE') != 'MO':
                    continue

                stid = st['STID']
                lat, lon = float(st['LATITUDE']), float(st['LONGITUDE'])
                
                cur.execute('''
                    INSERT OR IGNORE INTO stations (id, name, lat, lon, state)
                    VALUES (?, ?, ?, ?, ?)
                ''', (stid, st['NAME'], lat, lon, st.get('STATE')))

                obs = st.get('OBSERVATIONS', {})
                times = obs.get('date_time', [])
                fm_values = obs.get('fuel_moisture_set_1', [])

                for ts, val in zip(times, fm_values):
                    if val is not None:
                        cur.execute('''
                            INSERT OR IGNORE INTO observations 
                            (station_id, observation_date, fuel_moisture_percentage, latitude, longitude)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (stid, ts, val, lat, lon))
            
            print(f"✅ Ingested MO stations from: {file_path.name}")
        except Exception as e:
            print(f"❌ Failed to process {file_path.name}: {e}")

    conn.commit()
    conn.close()

if __name__ == "__main__":
    clear_tables() # Run the clear function first
    ingest_archive()