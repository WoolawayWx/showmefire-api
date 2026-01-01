"""
SQLite Database
"""
import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

def get_db_path():
    import os
    if os.path.exists('/home/ubuntu'):
        return Path('/home/ubuntu/showmefire/showmefire-api/data/daily_forecast.db')
    else:
        return Path(__file__).resolve().parent / 'data' / 'fire_danger.db'
    
def init_database():
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
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
    
    cursor.execute('''
                   CREATE INDEX IF NOT EXISTS idx_valid_time ON forecasts(valid_time)''')
    
    conn.commit()
    conn.close()
    
    logger.info(f"Database initialized at {db_path}")
    
def insert_forecast(valid_time: datetime, title: str, discussion: str) -> int:
    """
    Docstring for insert_forecast
    
    :param valid_time: Description
    :type valid_time: datetime
    :param title: Description
    :type title: str
    :param discussion: Description
    :type discussion: str
    :return: Description
    :rtype: int
    """
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
                       INSERT INTO forecasts (valid_time, title, discussion)
                       VALUES (?,?,?)
                       ON CONFLICT(valid_time)
                       DO UPDATE SET
                            title = excluded.title,
                            discussion = excluded.discussion,
                            updated_at = CURRENT_TIMESTAMP
                        ''', (valid_time, title, discussion))
        forecast_id = cursor.lastrowid
        if forecast_id == 0:
            cursor.execute('SELECT id FROM forecasts WHERE valid_time = ?', (valid_time,))
            forecast_id = cursor.fetchone()[0]
            
        conn.commit()
        logger.info(f"Saved forecast {forecast_id} for {valid_time}")
        return forecast_id
    except Exception as e:
        conn.rollback()
        logger.error(f"Error inserting forecast: {e}")
        raise
    finally:
        conn.close()
        
def get_latest_forecast() -> Optional[Dict]:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    cursor = conn.cursor()
    
    cursor.execute(
        '''
        SELECT * FROM forecasts
        ORDER BY valid_time DESC
        LIMIT 1
        '''
    )
    row = cursor.fetchone()
    conn.close()
    
    if row:
        return dict(row)
    return None

def get_forecast_by_time(valid_time: datetime) -> Optional[dict]:
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

def get_recent_forecasts(limit: int = 10) -> List[Dict]:
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

def delete_old_forecasts(days_old: int = 30) -> int:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
                       DELETE FROM forecasts
                       WHERE valid_time < datetime('now','-' || ? || ' days)
                       ''', (days_old,))
        deleted_count = cursor.rowcount
        conn.commit()
        logger.info(f"Deleted {deleted_count} old forecasts")
        return deleted_count
    except Exception as e:
        conn.rollback()
        logger.error(f"Error deleting forecasts: {e}")
        raise
    finally:
        conn.close()
        
def get_forecast_count() -> int:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    
    cursor = conn.cursor()
    
    cursor.execute('SELECT COUNT(*) FROM forecasts')
    count = cursor.fetchone()[0]
    
    conn.close()
    return count

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_database()
    
    count = get_forecast_count()
    print(f"\nTotal forecasts: {count}")
    
    latest = get_latest_forecast()
    if latest:
        print(f"\nLatest forecast:")
        print(f"  Valid: {latest['valid_time']}")
        print(f"  Title: {latest['title']}")
        print(f"  Discussion: {latest['discussion'][:100]}...")