"""
SQLite Database - core/database.py
"""
import sqlite3
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

def get_db_path():
    # Keep your existing logic for path detection
    if os.path.exists('/home/ubuntu'):
        return Path('/home/ubuntu/showmefire/showmefire-api/data/showmefire.db')
    else:
        # Standardize on one DB name for the whole project
        return Path(__file__).resolve().parent.parent / 'data' / 'showmefire.db'
    
def init_database():
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    
    # FORCED MIGRATION: These will run once and fail silently if already there
    try: conn.execute('ALTER TABLE snapshots ADD COLUMN is_processed INTEGER DEFAULT 0')
    except: pass
    try: conn.execute('ALTER TABLE snapshots ADD COLUMN hrrr_filename TEXT')
    except: pass

    cursor = conn.cursor()
    
    # 1. Your existing forecasts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            valid_time TIMESTAMP NOT NULL UNIQUE,
            title TEXT NOT NULL,
            discussion TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # 2. Snapshots table (Tracks your Golden Rows)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_date TEXT NOT NULL UNIQUE,
            obs_path TEXT,
            hrrr_filename TEXT,
            is_processed INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 3. Weather Features table (Stores extracted HRRR data)
    # Using snapshot_id as a foreign key creates a 1-to-many link (one snapshot -> many stations)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS weather_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id INTEGER,
            station_id TEXT,
            temp_c REAL,
            rel_humidity REAL,
            wind_speed_ms REAL,
            precip_mm REAL,
            extraction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (snapshot_id) REFERENCES snapshots (id)
        )
    ''')
    
    # 4. Stations table (Stores station metadata)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stations (
            id TEXT PRIMARY KEY,
            name TEXT,
            lat REAL,
            lon REAL,
            elevation REAL,
            state TEXT
        )
    ''')
    
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_valid_time ON forecasts(valid_time)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_snapshot_date ON snapshots(snapshot_date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_wf_snapshot ON weather_features(snapshot_id)')
    
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {db_path}")

def get_latest_forecast():
    """
    Retrieves the most recent forecast from the database.
    """
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM forecasts 
        ORDER BY valid_time DESC 
        LIMIT 1
    ''')
    
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None

def get_forecast_by_time(valid_time):
    """
    Retrieves a forecast by its valid_time.
    """
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT * FROM forecasts WHERE valid_time = ?', (valid_time,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None

def get_recent_forecasts(limit=5):
    """
    Retrieves the most recent forecasts from the database.
    """
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT * FROM forecasts 
        ORDER BY valid_time DESC 
        LIMIT ?
    ''', (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [dict(row) for row in rows]

def get_forecast_count():
    """
    Returns the total number of forecasts in the database.
    """
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM forecasts')
    count = cursor.fetchone()[0]
    
    conn.close()
    return count

# --- NEW HELPERS FOR THE HRRR MINER ---

def get_all_stations():
    """Returns all stations from the database."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT id, lat, lon FROM stations')
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_unprocessed_snapshots():
    """Returns all snapshots that haven't been mined yet."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, hrrr_filename, snapshot_date 
        FROM snapshots 
        WHERE is_processed = 0
    ''')
    rows = cursor.fetchall()
    conn.close()
    return rows

def save_hrrr_features(snapshot_id, features, station_id):
    db_path = get_db_path()
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO weather_features 
                (snapshot_id, station_id, temp_c, rel_humidity, wind_speed_ms, precip_mm)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                snapshot_id, 
                station_id,
                features['temp_c'], 
                features['rel_humidity'], 
                features['wind_speed_ms'], 
                features['precip_mm']
            ))
            conn.commit()
    except Exception as e:
        logger.error(f"Error saving HRRR features for {station_id}: {e}")

def mark_snapshot_processed(snapshot_id: int):
    """Marks a snapshot as fully processed."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('UPDATE snapshots SET is_processed = 1 WHERE id = ?', (snapshot_id,))
        conn.commit()
    except Exception as e:
        logger.error(f"Error marking snapshot {snapshot_id} as processed: {e}")
    finally:
        conn.close()

def set_hrrr_filename(snapshot_id: int, filename: str):
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE snapshots SET hrrr_filename = ? WHERE id = ?",
        (filename, snapshot_id)
    )
    conn.commit()
    conn.close()