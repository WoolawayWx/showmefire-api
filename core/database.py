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
    # 1. Prioritize the Docker container path if it exists
    if os.path.exists('/app/data/showmefire.db'):
        return Path('/app/data/showmefire.db')

    # 2. Fallback: Calculate path relative to this file (works for local dev)
    # core/database.py -> parent=core -> parent=root -> data/showmefire.db
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

    # 5. Station Forecasts (Stores point forecasts for verification)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS station_forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id TEXT,
            valid_time TIMESTAMP,
            forecast_run_time TIMESTAMP,
            temp_c REAL,
            rel_humidity REAL,
            wind_speed_ms REAL,
            precip_mm REAL,
            fuel_moisture REAL,
            UNIQUE(station_id, valid_time, forecast_run_time)
        )
    ''')
    
    # 6. Observations (Stores actuals)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_id TEXT,
            observation_date TEXT,
            fuel_moisture_percentage REAL,
            temp_c REAL,
            rel_humidity REAL,
            wind_speed_ms REAL,
            precip_accum_1h_mm REAL,
            latitude REAL,
            longitude REAL,
            UNIQUE(station_id, observation_date)
        )
    ''') 
    
    # Try to add columns if they don't exist (migrations)
    try: cursor.execute('ALTER TABLE observations ADD COLUMN temp_c REAL')
    except: pass
    try: cursor.execute('ALTER TABLE observations ADD COLUMN rel_humidity REAL')
    except: pass
    try: cursor.execute('ALTER TABLE observations ADD COLUMN wind_speed_ms REAL')
    except: pass
    try: cursor.execute('ALTER TABLE observations ADD COLUMN precip_accum_1h_mm REAL')
    except: pass

    # 7. Banner Configuration (Operational settings)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS banner_config (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            enabled INTEGER DEFAULT 0,
            type TEXT DEFAULT 'info',
            message TEXT DEFAULT '',
            link TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Initialize default banner row if it doesn't exist
    cursor.execute('''
        INSERT OR IGNORE INTO banner_config (id, enabled, type, message, link)
        VALUES (1, 0, 'info', 'Welcome to Show Me Fire', NULL)
    ''')

    # 8. Ignored stations table (IDs of stations to exclude from processing)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ignored_stations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            stid TEXT UNIQUE NOT NULL,
            reason TEXT,
            added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Seed with any known ignored stations (keeps existing behavior)
    try:
        cursor.execute("INSERT OR IGNORE INTO ignored_stations (stid, reason) VALUES (?, ?)", ('MBGM7', 'legacy default'))
    except Exception:
        pass

    # 9. Website info (stores website version and metadata)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS website_info (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            version TEXT DEFAULT '1',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Ensure a single row exists with default version
    try:
        cursor.execute("INSERT OR IGNORE INTO website_info (id, version) VALUES (1, '1')")
    except Exception:
        pass

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_valid_time ON forecasts(valid_time)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_snapshot_date ON snapshots(snapshot_date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_wf_snapshot ON weather_features(snapshot_id)')

    # 10. Development projects (tracks roadmap items for the website)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS dev_projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            description TEXT,
            timeline TEXT,
            status TEXT DEFAULT 'planned',
            sort_order INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_dev_projects_sort ON dev_projects(sort_order)')
    
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
    
    try:
        cursor.execute('''
            SELECT * FROM forecasts 
            ORDER BY id DESC 
            LIMIT 1
        ''')
        
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None
    finally:
        conn.close()

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

def get_website_version():
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT version, updated_at FROM website_info WHERE id = 1')
        row = cursor.fetchone()
        if row:
            return dict(row)
        return {"version": "1", "updated_at": None}
    finally:
        conn.close()

def set_website_version(version: str):
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('UPDATE website_info SET version = ?, updated_at = CURRENT_TIMESTAMP WHERE id = 1', (version,))
        conn.commit()
        return True
    except Exception as e:
        logger.error(f"Error updating website version: {e}")
        return False
    finally:
        conn.close()

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

def insert_forecast(valid_time, title, discussion):
    """
    Inserts a new forecast into the database.
    
    Args:
        valid_time (datetime): The valid time of the forecast.
        title (str): The headline/title of the forecast.
        discussion (str): The detailed discussion text.
        
    Returns:
        int: The ID of the inserted forecast.
    """
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
            INSERT INTO forecasts (valid_time, title, discussion)
            VALUES (?, ?, ?)
        ''', (valid_time, title, discussion))
        
        forecast_id = cursor.lastrowid
        conn.commit()
        return forecast_id
        
    except sqlite3.IntegrityError:
        # Forecast for this time already exists - update it instead
        cursor.execute('''
            UPDATE forecasts 
            SET title = ?, discussion = ?, updated_at = CURRENT_TIMESTAMP
            WHERE valid_time = ?
        ''', (title, discussion, valid_time))
        conn.commit()
        
        # Get the ID of the updated row
        cursor.execute('SELECT id FROM forecasts WHERE valid_time = ?', (valid_time,))
        row = cursor.fetchone()
        return row[0] if row else None
        
    finally:
        conn.close()


# --- Development projects helpers ---

def list_dev_projects() -> List[Dict]:
    """Return all development projects ordered by sort_order then id."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, name, description, timeline, status, sort_order, created_at, updated_at
        FROM dev_projects
        ORDER BY sort_order ASC, id ASC
    ''')
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def create_dev_project(
    name: str,
    description: Optional[str] = None,
    timeline: Optional[str] = None,
    status: str = 'planned',
    sort_order: Optional[int] = None
) -> Dict:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        if sort_order is None:
            cursor.execute('SELECT COALESCE(MAX(sort_order), 0) + 1 FROM dev_projects')
            sort_order = cursor.fetchone()[0]

        cursor.execute('''
            INSERT INTO dev_projects (name, description, timeline, status, sort_order)
            VALUES (?, ?, ?, ?, ?)
        ''', (name, description, timeline, status, sort_order))
        conn.commit()
        project_id = cursor.lastrowid
        cursor.execute('''
            SELECT id, name, description, timeline, status, sort_order, created_at, updated_at
            FROM dev_projects WHERE id = ?
        ''', (project_id,))
        row = cursor.fetchone()
        return dict(row) if row else {}
    finally:
        conn.close()


def update_dev_project(
    project_id: int,
    name: Optional[str] = None,
    description: Optional[str] = None,
    timeline: Optional[str] = None,
    status: Optional[str] = None,
    sort_order: Optional[int] = None
) -> bool:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            UPDATE dev_projects
            SET name = COALESCE(?, name),
                description = COALESCE(?, description),
                timeline = COALESCE(?, timeline),
                status = COALESCE(?, status),
                sort_order = COALESCE(?, sort_order),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (name, description, timeline, status, sort_order, project_id))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def delete_dev_project(project_id: int) -> bool:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('DELETE FROM dev_projects WHERE id = ?', (project_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()