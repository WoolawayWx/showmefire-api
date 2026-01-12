import shutil
import re
import sqlite3
from pathlib import Path

# Paths based on your setup
ARCHIVE_FORECASTS = Path("archive/forecasts")
ARCHIVE_RAW = Path("archive/raw_data")
SNAPSHOT_BASE = Path("data/snapshots")
DB_PATH = "data/showmefire.db"

def get_date_from_name(filename):
    """Extracts YYYYMMDD and returns YYYY-MM-DD."""
    match = re.search(r"(\d{8})", filename)
    if match:
        d = match.group(1)
        return f"{d[:4]}-{d[4:6]}-{d[6:]}"
    return None

def run_backfill():
    SNAPSHOT_BASE.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    
    conn.execute("""
                 CREATE TABLE IF NOT EXISTS snapshots (
                     id INTEGER PRIMARY KEY AUTOINCREMENT,
                     snapshot_date TEXT UNIQUE,
                     model_run TEXT,
                     obs_path TEXT,
                     forecast_path TEXT,
                     hrrr_filename TEXT,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                 );
                 """)
    print("Database table 'snapshots' verified/created")
    
    # 1. Process Forecasts
    print("Processing Forecasts...")
    for f_path in ARCHIVE_FORECASTS.glob("*.json"):
        date_str = get_date_from_name(f_path.name)
        if not date_str: continue
        
        day_dir = SNAPSHOT_BASE / date_str
        day_dir.mkdir(exist_ok=True)
        
        dest = day_dir / f_path.name
        shutil.copy(f_path, dest)
        
        conn.execute("""
            INSERT INTO snapshots (snapshot_date, forecast_path) 
            VALUES (?, ?) 
            ON CONFLICT(snapshot_date) DO UPDATE SET forecast_path=excluded.forecast_path
        """, (date_str, str(dest)))

    # 2. Process Raw Observations
    print("Processing Raw Observations...")
    for r_path in ARCHIVE_RAW.glob("*.json"):
        date_str = get_date_from_name(r_path.name)
        if not date_str: continue
        
        day_dir = SNAPSHOT_BASE / date_str
        day_dir.mkdir(exist_ok=True)
        
        dest = day_dir / r_path.name
        shutil.copy(r_path, dest)
        
        conn.execute("""
            UPDATE snapshots SET obs_path = ? WHERE snapshot_date = ?
        """, (str(dest), date_str))

    conn.commit()
    conn.close()
    print("Done! Check your 'data/snapshots' folder and 'showmefire.db'.")

if __name__ == "__main__":
    run_backfill()