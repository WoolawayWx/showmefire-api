import sqlite3
import json
import pandas as pd
from pathlib import Path

DB_PATH = "data/showmefire.db"

def test_registry_health():
    print("--- 1. Testing SQLite Connection ---")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all snapshots
    cursor.execute("SELECT snapshot_date, obs_path, forecast_path FROM snapshots")
    rows = cursor.fetchall()
    print(f"Found {len(rows)} snapshots in database.")

    for date_str, obs_p, fc_p in rows:
        print(f"\nChecking Snapshot: {date_str}")
        
        # Check if files exist on disk
        obs_exists = Path(obs_p).exists()
        fc_exists = Path(fc_p).exists()
        
        print(f"  - Obs file exists: {obs_exists}")
        print(f"  - Forecast file exists: {fc_exists}")

        if obs_exists:
            # Test if JSON is valid and can be loaded into Pandas
            try:
                with open(obs_p) as f:
                    data = json.load(f)
                    # Verify it has the STATION key from your raw_data format
                    if "STATION" in data:
                        print(f"  - Data Integrity: OK ({len(data['STATION'])} stations found)")
            except Exception as e:
                print(f"  - ❌ ERROR parsing JSON: {e}")

    conn.close()

if __name__ == "__main__":
    test_registry_health()