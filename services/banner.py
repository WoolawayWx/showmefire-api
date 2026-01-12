import sqlite3
import logging
from typing import Optional
from pydantic import BaseModel
from core.database import get_db_path

logger = logging.getLogger(__name__)

class BannerData(BaseModel):
    enabled: bool = False
    type: str = "info"  # info, warning, danger, success
    message: str = ""
    link: Optional[str] = None

def load_banner_config() -> BannerData:
    try:
        conn = sqlite3.connect(get_db_path())
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT enabled, type, message, link FROM banner_config WHERE id = 1")
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return BannerData(
                enabled=bool(row['enabled']),
                type=row['type'],
                message=row['message'],
                link=row['link']
            )
    except Exception as e:
        logger.error(f"Failed to load banner config from DB: {e}")
        
    return BannerData()

def save_banner_config(banner: BannerData):
    try:
        conn = sqlite3.connect(get_db_path())
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO banner_config (id, enabled, type, message, link, updated_at)
            VALUES (1, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(id) DO UPDATE SET
                enabled=excluded.enabled,
                type=excluded.type,
                message=excluded.message,
                link=excluded.link,
                updated_at=CURRENT_TIMESTAMP
        """, (int(banner.enabled), banner.type, banner.message, banner.link))
        
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to save banner config to DB: {e}")
