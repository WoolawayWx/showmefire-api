"""
SQLite Database - core/database.py
"""
import sqlite3
import logging
import os
import re
import unicodedata
from pathlib import Path
from datetime import datetime
from typing import Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


def _ensure_discord_settings_table(cursor: sqlite3.Cursor) -> None:
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS discord_admin_settings (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            channel_id TEXT DEFAULT '',
            channel_name TEXT DEFAULT '',
            forecast_channel_id TEXT DEFAULT '',
            forecast_channel_name TEXT DEFAULT '',
            outlook_channel_id TEXT DEFAULT '',
            outlook_channel_name TEXT DEFAULT '',
            forecast_role_ids TEXT DEFAULT '',
            outlook_role_ids TEXT DEFAULT '',
            event_url_override TEXT DEFAULT '',
            event_secret_override TEXT DEFAULT '',
            image_fetch_retries INTEGER DEFAULT 3,
            image_fetch_timeout_ms INTEGER DEFAULT 5000,
            dedupe_ttl_ms INTEGER DEFAULT 21600000,
            updated_by TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        INSERT OR IGNORE INTO discord_admin_settings (
            id,
            channel_id,
            channel_name,
            forecast_channel_id,
            forecast_channel_name,
            outlook_channel_id,
            outlook_channel_name,
            forecast_role_ids,
            outlook_role_ids,
            event_url_override,
            event_secret_override,
            image_fetch_retries,
            image_fetch_timeout_ms,
            dedupe_ttl_ms,
            updated_by
        ) VALUES (1, '', '', '', '', '', '', '', '', '', '', 3, 5000, 21600000, NULL)
    ''')

    cursor.execute("PRAGMA table_info(discord_admin_settings)")
    columns = {row[1] for row in cursor.fetchall()}
    if "forecast_channel_id" not in columns:
        cursor.execute("ALTER TABLE discord_admin_settings ADD COLUMN forecast_channel_id TEXT DEFAULT ''")
    if "forecast_channel_name" not in columns:
        cursor.execute("ALTER TABLE discord_admin_settings ADD COLUMN forecast_channel_name TEXT DEFAULT ''")
    if "outlook_channel_id" not in columns:
        cursor.execute("ALTER TABLE discord_admin_settings ADD COLUMN outlook_channel_id TEXT DEFAULT ''")
    if "outlook_channel_name" not in columns:
        cursor.execute("ALTER TABLE discord_admin_settings ADD COLUMN outlook_channel_name TEXT DEFAULT ''")
    if "forecast_role_ids" not in columns:
        cursor.execute("ALTER TABLE discord_admin_settings ADD COLUMN forecast_role_ids TEXT DEFAULT ''")
    if "outlook_role_ids" not in columns:
        cursor.execute("ALTER TABLE discord_admin_settings ADD COLUMN outlook_role_ids TEXT DEFAULT ''")
    if "event_url_override" not in columns:
        cursor.execute("ALTER TABLE discord_admin_settings ADD COLUMN event_url_override TEXT DEFAULT ''")
    if "event_secret_override" not in columns:
        cursor.execute("ALTER TABLE discord_admin_settings ADD COLUMN event_secret_override TEXT DEFAULT ''")

def _ensure_fire_event_tables(cursor: sqlite3.Cursor) -> None:
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fire_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            external_id TEXT,
            status TEXT NOT NULL DEFAULT 'pending',
            verification_tier TEXT NOT NULL DEFAULT 'unverified',
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            county_fips TEXT,
            county_name TEXT,
            occurred_at TEXT NOT NULL,
            occurred_at_precision TEXT NOT NULL DEFAULT 'minute',
            occurred_at_tz_offset_minutes INTEGER,
            acres REAL,
            acres_is_estimate INTEGER NOT NULL DEFAULT 1,
            cause_category TEXT NOT NULL DEFAULT 'unknown',
            description TEXT NOT NULL DEFAULT '',
            out_of_ordinary TEXT NOT NULL DEFAULT '',
            frp REAL,
            confidence TEXT,
            satellite TEXT,
            official_source_system TEXT NOT NULL DEFAULT '',
            official_source_ref TEXT NOT NULL DEFAULT '',
            label_revision INTEGER NOT NULL DEFAULT 1,
            revised_at TIMESTAMP,
            parent_event_id INTEGER,
            reporter_contact TEXT NOT NULL DEFAULT '',
            reporter_name TEXT NOT NULL DEFAULT '',
            reporter_org TEXT NOT NULL DEFAULT '',
            address_text TEXT NOT NULL DEFAULT '',
            submitter_ip_hash TEXT NOT NULL DEFAULT '',
            upload_token_hash TEXT NOT NULL DEFAULT '',
            consent_version TEXT NOT NULL DEFAULT '',
            captcha_verdict TEXT NOT NULL DEFAULT '',
            moderated_by TEXT NOT NULL DEFAULT '',
            moderated_at TIMESTAMP,
            pii_purged_at TIMESTAMP,
            first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_fire_events_source_external ON fire_events(source, external_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_fire_events_status_occurred ON fire_events(status, occurred_at DESC)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_fire_events_bbox ON fire_events(latitude, longitude)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_fire_events_tier_occurred ON fire_events(verification_tier, occurred_at)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_fire_events_county ON fire_events(county_fips)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_fire_events_source_status ON fire_events(source, status)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_fire_events_purge ON fire_events(moderated_at)')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fire_event_fuels (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER NOT NULL,
            fuel_type TEXT NOT NULL,
            FOREIGN KEY (event_id) REFERENCES fire_events(id)
        )
    ''')
    cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_fire_event_fuels_unique ON fire_event_fuels(event_id, fuel_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_fire_event_fuels_type ON fire_event_fuels(fuel_type)')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fire_event_moderation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            actor TEXT NOT NULL DEFAULT '',
            from_status TEXT NOT NULL DEFAULT '',
            to_status TEXT NOT NULL DEFAULT '',
            from_tier TEXT NOT NULL DEFAULT '',
            to_tier TEXT NOT NULL DEFAULT '',
            reason TEXT NOT NULL DEFAULT '',
            changed_fields_json TEXT NOT NULL DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (event_id) REFERENCES fire_events(id)
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_fire_moderation_event ON fire_event_moderation(event_id, created_at)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_fire_moderation_actor ON fire_event_moderation(actor, created_at)')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fire_event_media (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id INTEGER NOT NULL,
            stored_filename TEXT NOT NULL UNIQUE,
            original_filename TEXT NOT NULL DEFAULT '',
            content_type TEXT NOT NULL DEFAULT '',
            size_bytes INTEGER NOT NULL DEFAULT 0,
            sha256 TEXT NOT NULL DEFAULT '',
            review_state TEXT NOT NULL DEFAULT 'pending',
            kind TEXT NOT NULL DEFAULT 'photo',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (event_id) REFERENCES fire_events(id)
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_fire_event_media_event ON fire_event_media(event_id)')

    # Migrate databases created before the fire-report enhancement.
    cursor.execute("PRAGMA table_info(fire_events)")
    columns = {row[1] for row in cursor.fetchall()}
    for name, definition in (
        ("reporter_name", "TEXT NOT NULL DEFAULT ''"),
        ("reporter_org", "TEXT NOT NULL DEFAULT ''"),
        ("address_text", "TEXT NOT NULL DEFAULT ''"),
        ("upload_token_hash", "TEXT NOT NULL DEFAULT ''"),
    ):
        if name not in columns:
            cursor.execute(f"ALTER TABLE fire_events ADD COLUMN {name} {definition}")

    # Migrate fire_event_media tables created before department-report uploads.
    cursor.execute("PRAGMA table_info(fire_event_media)")
    media_columns = {row[1] for row in cursor.fetchall()}
    if "kind" not in media_columns:
        cursor.execute("ALTER TABLE fire_event_media ADD COLUMN kind TEXT NOT NULL DEFAULT 'photo'")


def _ensure_fire_abuse_tables(cursor: sqlite3.Cursor) -> None:
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fire_submission_throttle (
            bucket_key TEXT NOT NULL,
            window_kind TEXT NOT NULL,
            window_start TEXT NOT NULL,
            hits INTEGER NOT NULL DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (bucket_key, window_kind, window_start)
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_fire_throttle_updated ON fire_submission_throttle(updated_at)')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fire_submission_blocklist (
            ip_hash TEXT PRIMARY KEY,
            reason TEXT NOT NULL DEFAULT '',
            created_by TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')


def _ensure_feedback_tables(cursor: sqlite3.Cursor) -> None:
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL DEFAULT '',
            email TEXT NOT NULL DEFAULT '',
            category TEXT NOT NULL,
            details TEXT NOT NULL DEFAULT '{}',
            message TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'new',
            submitter_ip_hash TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_status_created ON feedback(status, created_at DESC)')

    # A submission throttle table of its own - kept separate from
    # fire_submission_throttle (core/database.py's _ensure_fire_abuse_tables)
    # so feedback and fire-report rate limits never share the same buckets.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback_submission_throttle (
            bucket_key TEXT NOT NULL,
            window_kind TEXT NOT NULL,
            window_start TEXT NOT NULL,
            hits INTEGER NOT NULL DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (bucket_key, window_kind, window_start)
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_feedback_throttle_updated ON feedback_submission_throttle(updated_at)')


def get_db_path():
    # Honor the documented container/local override even before the database
    # file exists. This keeps first-start initialization on the mounted volume.
    configured_data_dir = os.getenv("DATA_DIR", "").strip()
    if configured_data_dir:
        return Path(configured_data_dir).expanduser().resolve() / "showmefire.db"

    # Preserve compatibility with older containers that did not set DATA_DIR.
    if os.path.isdir('/app/data'):
        return Path('/app/data/showmefire.db')

    # 2. Fallback: Calculate path relative to this file (works for local dev)
    # core/database.py -> parent=core -> parent=root -> data/showmefire.db
    return Path(__file__).resolve().parent.parent / 'data' / 'showmefire.db'
    
def init_database():
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    
    # FORCED MIGRATION: These will run once and fail silently if already there
    try: conn.execute('ALTER TABLE snapshots ADD COLUMN is_processed INTEGER DEFAULT 0')
    except: pass
    try: conn.execute('ALTER TABLE snapshots ADD COLUMN hrrr_filename TEXT')
    except: pass
    try: conn.execute("ALTER TABLE forecasts ADD COLUMN title TEXT NOT NULL DEFAULT ''")
    except: pass
    try: conn.execute('ALTER TABLE forecasts ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP')
    except: pass
    try: conn.execute('ALTER TABLE forecasts ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP')
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
            precip_interval_mm REAL,
            precip_interval_hours REAL,
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
            precip_interval_mm REAL,
            precip_interval_hours REAL,
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
    try: cursor.execute('ALTER TABLE station_forecasts ADD COLUMN precip_interval_mm REAL')
    except: pass
    try: cursor.execute('ALTER TABLE station_forecasts ADD COLUMN precip_interval_hours REAL')
    except: pass
    try: cursor.execute('ALTER TABLE weather_features ADD COLUMN precip_interval_mm REAL')
    except: pass
    try: cursor.execute('ALTER TABLE weather_features ADD COLUMN precip_interval_hours REAL')
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

    # 10. Discord admin settings (singleton config row for website control panel)
    _ensure_discord_settings_table(cursor)

    cursor.execute('CREATE INDEX IF NOT EXISTS idx_valid_time ON forecasts(valid_time)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_snapshot_date ON snapshots(snapshot_date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_wf_snapshot ON weather_features(snapshot_id)')

    # 11. Development projects (tracks roadmap items for the website)
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

    # 12. Briefings (singleton config row: id must be 1)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS briefings (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            title TEXT,
            file_path TEXT NOT NULL,
            is_active INTEGER DEFAULT 1,
            expires_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_briefings_active ON briefings(is_active)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_briefings_expires ON briefings(expires_at)')
    # Ensure singleton row exists
    cursor.execute('''
        INSERT OR IGNORE INTO briefings (id, title, file_path, is_active, expires_at)
        VALUES (1, NULL, '', 0, NULL)
    ''')
    # Cleanup safety in case older schema/data allowed multiple rows
    cursor.execute('DELETE FROM briefings WHERE id != 1')

    # 13. NWS Area Forecast Discussions (AFDs)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS afds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            office TEXT NOT NULL,
            product_id TEXT NOT NULL UNIQUE,
            issued_at TIMESTAMP NOT NULL,
            raw_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # Keep only one row per office (the latest by issued_at, then id).
    cursor.execute('''
        DELETE FROM afds
        WHERE EXISTS (
            SELECT 1
            FROM afds newer
            WHERE newer.office = afds.office
              AND (
                  newer.issued_at > afds.issued_at
                  OR (newer.issued_at = afds.issued_at AND newer.id > afds.id)
              )
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_afds_office_issued_at ON afds(office, issued_at DESC)')
    cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_afds_unique_office ON afds(office)')

    # 14. Anonymous mobile push subscriptions and delivery bookkeeping
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS mobile_push_subscriptions (
            installation_id TEXT PRIMARY KEY,
            expo_push_token TEXT NOT NULL UNIQUE,
            platform TEXT NOT NULL,
            app_version TEXT NOT NULL DEFAULT '',
            forecast_enabled INTEGER NOT NULL DEFAULT 0,
            sitrep_enabled INTEGER NOT NULL DEFAULT 0,
            fire_weather_enabled INTEGER NOT NULL DEFAULT 0,
            county_fips_json TEXT NOT NULL DEFAULT '[]',
            enabled INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_mobile_push_enabled ON mobile_push_subscriptions(enabled)')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS mobile_push_events (
            event_key TEXT PRIMARY KEY,
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS mobile_push_tickets (
            ticket_id TEXT PRIMARY KEY,
            installation_id TEXT NOT NULL,
            event_key TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (installation_id) REFERENCES mobile_push_subscriptions(installation_id)
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_mobile_ticket_created ON mobile_push_tickets(created_at)')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS mobile_push_receipts (
            ticket_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            error TEXT,
            checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (ticket_id) REFERENCES mobile_push_tickets(ticket_id)
        )
    ''')

    # 15. Staff discussion posts and comments
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS posts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            author_name TEXT NOT NULL,
            slug TEXT,
            excerpt TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'published',
            category TEXT NOT NULL DEFAULT 'Field Notes',
            cover_image TEXT,
            seo_title TEXT,
            seo_description TEXT,
            body_format TEXT NOT NULL DEFAULT 'plain',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute("PRAGMA table_info(posts)")
    post_columns = {row[1] for row in cursor.fetchall()}
    post_migrations = {
        "slug": "TEXT",
        "excerpt": "TEXT NOT NULL DEFAULT ''",
        "status": "TEXT NOT NULL DEFAULT 'published'",
        "category": "TEXT NOT NULL DEFAULT 'Field Notes'",
        "cover_image": "TEXT",
        "seo_title": "TEXT",
        "seo_description": "TEXT",
        "body_format": "TEXT NOT NULL DEFAULT 'plain'",
    }
    for column, definition in post_migrations.items():
        if column not in post_columns:
            cursor.execute(f"ALTER TABLE posts ADD COLUMN {column} {definition}")
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS post_categories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute("INSERT OR IGNORE INTO post_categories (name) VALUES ('Field Notes')")
    cursor.execute("SELECT id, title, slug FROM posts WHERE slug IS NULL OR slug = ''")
    for post_id, title, existing_slug in cursor.fetchall():
        base = re.sub(r"[^a-z0-9]+", "-", unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode().lower()).strip("-") or f"post-{post_id}"
        slug = base
        suffix = 2
        while cursor.execute("SELECT 1 FROM posts WHERE slug = ? AND id != ?", (slug, post_id)).fetchone():
            slug = f"{base}-{suffix}"
            suffix += 1
        cursor.execute("UPDATE posts SET slug = ?, excerpt = COALESCE(NULLIF(excerpt, ''), substr(body, 1, 220)) WHERE id = ?", (slug, post_id))
    cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_posts_slug ON posts(slug)")
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_posts_created ON posts(created_at)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_posts_status_category ON posts(status, category, created_at DESC)')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS post_tags (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id INTEGER NOT NULL,
            tag TEXT NOT NULL,
            FOREIGN KEY (post_id) REFERENCES posts(id)
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_post_tags_post_id ON post_tags(post_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_post_tags_tag ON post_tags(tag)')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS post_comments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            post_id INTEGER NOT NULL,
            author_name TEXT NOT NULL,
            body TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (post_id) REFERENCES posts(id)
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_post_comments_post_id ON post_comments(post_id)')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS post_media (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL UNIQUE,
            original_name TEXT NOT NULL,
            content_type TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            public_url TEXT NOT NULL,
            cdn_url TEXT,
            sha256 TEXT NOT NULL,
            uploaded_by TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_post_media_created ON post_media(created_at DESC)')

    # 16. Unified fire-event store (user submissions + satellite/NGFS/official detections)
    _ensure_fire_event_tables(cursor)

    # 17. Anonymous fire-report abuse controls (per-IP throttle + blocklist)
    _ensure_fire_abuse_tables(cursor)

    # 18. Public feedback form + its own submission throttle
    _ensure_feedback_tables(cursor)

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


def get_discord_admin_settings() -> Dict:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        _ensure_discord_settings_table(cursor)
        conn.commit()
        cursor.execute('''
            SELECT
                channel_id,
                channel_name,
                forecast_channel_id,
                forecast_channel_name,
                outlook_channel_id,
                outlook_channel_name,
                forecast_role_ids,
                outlook_role_ids,
                event_url_override,
                event_secret_override,
                image_fetch_retries,
                image_fetch_timeout_ms,
                dedupe_ttl_ms,
                updated_by,
                updated_at
            FROM discord_admin_settings
            WHERE id = 1
        ''')
        row = cursor.fetchone()
        return dict(row) if row else {
            "channel_id": "",
            "channel_name": "",
            "forecast_channel_id": "",
            "forecast_channel_name": "",
            "outlook_channel_id": "",
            "outlook_channel_name": "",
            "forecast_role_ids": "",
            "outlook_role_ids": "",
            "event_url_override": "",
            "event_secret_override": "",
            "image_fetch_retries": 3,
            "image_fetch_timeout_ms": 5000,
            "dedupe_ttl_ms": 21600000,
            "updated_by": None,
            "updated_at": None,
        }
    finally:
        conn.close()


def update_discord_admin_settings(
    *,
    channel_id: Optional[str] = None,
    channel_name: Optional[str] = None,
    forecast_channel_id: Optional[str] = None,
    forecast_channel_name: Optional[str] = None,
    outlook_channel_id: Optional[str] = None,
    outlook_channel_name: Optional[str] = None,
    forecast_role_ids: Optional[str] = None,
    outlook_role_ids: Optional[str] = None,
    event_url_override: Optional[str] = None,
    event_secret_override: Optional[str] = None,
    image_fetch_retries: Optional[int] = None,
    image_fetch_timeout_ms: Optional[int] = None,
    dedupe_ttl_ms: Optional[int] = None,
    updated_by: Optional[str] = None,
) -> Dict:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        _ensure_discord_settings_table(cursor)
        cursor.execute('''
            UPDATE discord_admin_settings
            SET channel_id = COALESCE(?, channel_id),
                channel_name = COALESCE(?, channel_name),
                forecast_channel_id = COALESCE(?, forecast_channel_id),
                forecast_channel_name = COALESCE(?, forecast_channel_name),
                outlook_channel_id = COALESCE(?, outlook_channel_id),
                outlook_channel_name = COALESCE(?, outlook_channel_name),
                forecast_role_ids = COALESCE(?, forecast_role_ids),
                outlook_role_ids = COALESCE(?, outlook_role_ids),
                event_url_override = COALESCE(?, event_url_override),
                event_secret_override = COALESCE(?, event_secret_override),
                image_fetch_retries = COALESCE(?, image_fetch_retries),
                image_fetch_timeout_ms = COALESCE(?, image_fetch_timeout_ms),
                dedupe_ttl_ms = COALESCE(?, dedupe_ttl_ms),
                updated_by = COALESCE(?, updated_by),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
        ''', (
            channel_id,
            channel_name,
            forecast_channel_id,
            forecast_channel_name,
            outlook_channel_id,
            outlook_channel_name,
            forecast_role_ids,
            outlook_role_ids,
            event_url_override,
            event_secret_override,
            image_fetch_retries,
            image_fetch_timeout_ms,
            dedupe_ttl_ms,
            updated_by,
        ))
        conn.commit()
        cursor.execute('''
            SELECT
                channel_id,
                channel_name,
                forecast_channel_id,
                forecast_channel_name,
                outlook_channel_id,
                outlook_channel_name,
                forecast_role_ids,
                outlook_role_ids,
                event_url_override,
                event_secret_override,
                image_fetch_retries,
                image_fetch_timeout_ms,
                dedupe_ttl_ms,
                updated_by,
                updated_at
            FROM discord_admin_settings
            WHERE id = 1
        ''')
        row = cursor.fetchone()
        return dict(row) if row else {}
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
                (snapshot_id, station_id, temp_c, rel_humidity, wind_speed_ms, precip_mm,
                 precip_interval_mm, precip_interval_hours)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                snapshot_id, 
                station_id,
                features['temp_c'], 
                features['rel_humidity'], 
                features['wind_speed_ms'], 
                features['precip_mm'],
                features.get('precip_interval_mm'),
                features.get('precip_interval_hours')
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


def _ensure_forecasts_schema(cursor: sqlite3.Cursor) -> None:
    """Ensure legacy databases have the columns required by forecast writes."""
    cursor.execute("PRAGMA table_info(forecasts)")
    columns = {row[1] for row in cursor.fetchall()}

    if "title" not in columns:
        cursor.execute("ALTER TABLE forecasts ADD COLUMN title TEXT NOT NULL DEFAULT ''")
    if "created_at" not in columns:
        cursor.execute("ALTER TABLE forecasts ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
    if "updated_at" not in columns:
        cursor.execute("ALTER TABLE forecasts ADD COLUMN updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")

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
        _ensure_forecasts_schema(cursor)
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


# --- Briefings helpers ---

def list_briefings() -> List[Dict]:
    """Return all briefings ordered by newest first."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, title, file_path, is_active, expires_at, created_at, updated_at
        FROM briefings
        ORDER BY id DESC
    ''')
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_briefing_config() -> Dict:
    """Return the singleton briefing configuration row (id=1)."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, title, file_path, is_active, expires_at, created_at, updated_at
            FROM briefings
            WHERE id = 1
        ''')
        row = cursor.fetchone()
        return dict(row) if row else {}
    finally:
        conn.close()


def create_briefing(
    file_path: str,
    title: Optional[str] = None,
    is_active: bool = True,
    expires_at: Optional[str] = None
) -> Dict:
    """
    Create a briefing record.
    expires_at should be an ISO timestamp string (or None).
    """
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO briefings (id, title, file_path, is_active, expires_at)
            VALUES (1, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                title = excluded.title,
                file_path = excluded.file_path,
                is_active = excluded.is_active,
                expires_at = excluded.expires_at,
                updated_at = CURRENT_TIMESTAMP
        ''', (title, file_path, 1 if is_active else 0, expires_at))
        conn.commit()
        cursor.execute('''
            SELECT id, title, file_path, is_active, expires_at, created_at, updated_at
            FROM briefings WHERE id = 1
        ''')
        row = cursor.fetchone()
        return dict(row) if row else {}
    finally:
        conn.close()


def update_briefing(
    briefing_id: int = 1,
    title: Optional[str] = None,
    file_path: Optional[str] = None,
    is_active: Optional[bool] = None,
    expires_at: Optional[str] = None
) -> bool:
    """Update one briefing row. Pass only fields you want to change."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        is_active_db = None if is_active is None else (1 if is_active else 0)
        cursor.execute('''
            UPDATE briefings
            SET title = COALESCE(?, title),
                file_path = COALESCE(?, file_path),
                is_active = COALESCE(?, is_active),
                expires_at = COALESCE(?, expires_at),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (title, file_path, is_active_db, expires_at, 1))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def get_active_briefings() -> List[Dict]:
    """
    Return active briefings where expiration is unset or in the future.
    Uses UTC CURRENT_TIMESTAMP from SQLite.
    """
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, title, file_path, is_active, expires_at, created_at, updated_at
        FROM briefings
        WHERE is_active = 1
          AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
        ORDER BY id DESC
    ''')
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]


# --- AFD helpers ---

def get_known_afd_product_ids(office: Optional[str] = None) -> set:
    """Return known AFD product IDs, optionally filtered by office."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        if office:
            cursor.execute('SELECT product_id FROM afds WHERE office = ?', (office.upper(),))
        else:
            cursor.execute('SELECT product_id FROM afds')
        rows = cursor.fetchall()
        return {row[0] for row in rows if row and row[0]}
    finally:
        conn.close()


def insert_afd_records(records: Iterable[Dict]) -> int:
    """Insert or update latest AFD per office. Returns changed row count."""
    payload = []
    for record in records:
        office = (record.get('office') or '').upper()
        product_id = record.get('product_id')
        if not office or not product_id:
            continue

        issued_at = record.get('issued_at')
        if isinstance(issued_at, datetime):
            issued_at_value = issued_at.isoformat()
        else:
            issued_at_value = str(issued_at)

        payload.append((
            office,
            product_id,
            issued_at_value,
            record.get('raw_text', ''),
        ))

    if not payload:
        return 0

    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        before = conn.total_changes
        cursor.executemany('''
            INSERT INTO afds (office, product_id, issued_at, raw_text)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(office) DO UPDATE SET
                product_id = excluded.product_id,
                issued_at = excluded.issued_at,
                raw_text = excluded.raw_text,
                created_at = CURRENT_TIMESTAMP
            WHERE excluded.issued_at > afds.issued_at
               OR (excluded.issued_at = afds.issued_at AND excluded.product_id != afds.product_id)
        ''', payload)
        conn.commit()
        return conn.total_changes - before
    finally:
        conn.close()


def get_afds_by_office(office: str, limit: int = 10, since: Optional[str] = None) -> List[Dict]:
    """Return most recent AFDs for an office, newest first."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    office_code = office.upper()
    safe_limit = max(1, min(limit, 100))

    try:
        if since:
            cursor.execute('''
                SELECT office, product_id, issued_at, raw_text, created_at
                FROM afds
                WHERE office = ? AND issued_at >= ?
                ORDER BY issued_at DESC
                LIMIT ?
            ''', (office_code, since, safe_limit))
        else:
            cursor.execute('''
                SELECT office, product_id, issued_at, raw_text, created_at
                FROM afds
                WHERE office = ?
                ORDER BY issued_at DESC
                LIMIT ?
            ''', (office_code, safe_limit))

        rows = cursor.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


# --- Discussion posts helpers ---

def _normalize_post_tags(tags: Iterable[str]) -> List[str]:
    normalized = []
    for raw in tags or []:
        tag = re.sub(r"\s+", "-", str(raw or "").strip().lower())
        tag = re.sub(r"[^a-z0-9_-]", "", tag)
        if tag and tag not in normalized:
            normalized.append(tag)
    return normalized


def _post_row_extras(cursor: sqlite3.Cursor, post_id: int) -> Dict:
    cursor.execute('SELECT tag FROM post_tags WHERE post_id = ? ORDER BY tag', (post_id,))
    tags = [row[0] for row in cursor.fetchall()]
    cursor.execute('SELECT COUNT(*) FROM post_comments WHERE post_id = ?', (post_id,))
    comment_count = cursor.fetchone()[0]
    return {"tags": tags, "comment_count": comment_count}


def _post_slug(title: str, post_id: int, cursor: sqlite3.Cursor) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode().lower()).strip("-") or f"post-{post_id}"
    slug, suffix = base, 2
    while cursor.execute("SELECT 1 FROM posts WHERE slug = ? AND id != ?", (slug, post_id)).fetchone():
        slug = f"{base}-{suffix}"
        suffix += 1
    return slug


def create_post(title: str, body: str, author_name: str, tags: List[str], excerpt: str = "",
                status: str = "published", category: str = "Field Notes", cover_image: Optional[str] = None,
                seo_title: Optional[str] = None, seo_description: Optional[str] = None,
                body_format: str = "plain") -> Dict:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO posts (title, body, author_name, excerpt, status, category, cover_image,
                              seo_title, seo_description, body_format)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (title, body, author_name, excerpt or body[:220], status, category or "Field Notes",
              cover_image, seo_title, seo_description, body_format))
        post_id = cursor.lastrowid
        cursor.execute("UPDATE posts SET slug = ? WHERE id = ?", (_post_slug(title, post_id, cursor), post_id))
        cursor.execute("INSERT OR IGNORE INTO post_categories (name) VALUES (?)", (category or "Field Notes",))
        for tag in _normalize_post_tags(tags):
            cursor.execute('INSERT INTO post_tags (post_id, tag) VALUES (?, ?)', (post_id, tag))
        conn.commit()

        cursor.execute('''
            SELECT id, title, body, author_name, slug, excerpt, status, category, cover_image,
                   seo_title, seo_description, body_format, created_at, updated_at
            FROM posts WHERE id = ?
        ''', (post_id,))
        row = cursor.fetchone()
        post = dict(row)
        post.update(_post_row_extras(cursor, post_id))
        return post
    finally:
        conn.close()


def list_posts(tag: Optional[str] = None, limit: int = 50, offset: int = 0,
               category: Optional[str] = None, status: Optional[str] = "published") -> List[Dict]:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        safe_limit = max(1, min(limit, 200))
        safe_offset = max(0, offset)

        clauses, params = [], []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if category:
            clauses.append("category = ?")
            params.append(category)
        if tag:
            clauses.append("id IN (SELECT post_id FROM post_tags WHERE tag = ?)")
            params.append(tag)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cursor.execute(f'''
                SELECT id, title, body, author_name, slug, excerpt, status, category, cover_image,
                       seo_title, seo_description, body_format, created_at, updated_at
                FROM posts
                {where}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            ''', (*params, safe_limit, safe_offset))
        rows = cursor.fetchall()
        posts = []
        for row in rows:
            post = dict(row)
            post.update(_post_row_extras(cursor, post["id"]))
            posts.append(post)
        return posts
    finally:
        conn.close()


def list_post_tags() -> List[str]:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT DISTINCT tag FROM post_tags ORDER BY tag')
        return [row[0] for row in cursor.fetchall()]
    finally:
        conn.close()


def list_post_categories() -> List[str]:
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM post_categories ORDER BY name")
        return [row[0] for row in cursor.fetchall()]


def create_post_category(name: str) -> str:
    value = str(name or "").strip()
    if not value:
        raise ValueError("category name is required")
    with sqlite3.connect(get_db_path()) as conn:
        conn.execute("INSERT OR IGNORE INTO post_categories (name) VALUES (?)", (value,))
        conn.commit()
    return value


def get_post(post_id: int, slug: Optional[str] = None) -> Optional[Dict]:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        if slug is not None:
            cursor.execute('''
                SELECT id, title, body, author_name, slug, excerpt, status, category, cover_image,
                       seo_title, seo_description, body_format, created_at, updated_at
                FROM posts WHERE slug = ?
            ''', (slug,))
        else:
            cursor.execute('''
                SELECT id, title, body, author_name, slug, excerpt, status, category, cover_image,
                       seo_title, seo_description, body_format, created_at, updated_at
                FROM posts WHERE id = ?
            ''', (post_id,))
        row = cursor.fetchone()
        if not row:
            return None

        post = dict(row)
        post.update(_post_row_extras(cursor, post_id))
        cursor.execute('''
            SELECT id, post_id, author_name, body, created_at
            FROM post_comments
            WHERE post_id = ?
            ORDER BY created_at ASC
        ''', (post_id,))
        post["comments"] = [dict(comment_row) for comment_row in cursor.fetchall()]
        return post
    finally:
        conn.close()


def update_post(
    post_id: int,
    title: Optional[str] = None,
    body: Optional[str] = None,
    tags: Optional[List[str]] = None,
    author_name: Optional[str] = None,
    excerpt: Optional[str] = None, status: Optional[str] = None, category: Optional[str] = None,
    cover_image: Optional[str] = None, seo_title: Optional[str] = None,
    seo_description: Optional[str] = None, body_format: Optional[str] = None,
    slug: Optional[str] = None
) -> Optional[Dict]:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT id FROM posts WHERE id = ?', (post_id,))
        if not cursor.fetchone():
            return None

        cursor.execute('''
            UPDATE posts
            SET title = COALESCE(?, title),
                body = COALESCE(?, body),
                author_name = COALESCE(?, author_name),
                excerpt = COALESCE(?, excerpt),
                status = COALESCE(?, status),
                category = COALESCE(?, category),
                cover_image = COALESCE(?, cover_image),
                seo_title = COALESCE(?, seo_title),
                seo_description = COALESCE(?, seo_description),
                body_format = COALESCE(?, body_format),
                slug = COALESCE(?, slug),
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (title, body, author_name, excerpt, status, category, cover_image, seo_title, seo_description, body_format, slug, post_id))

        if tags is not None:
            cursor.execute('DELETE FROM post_tags WHERE post_id = ?', (post_id,))
            for tag in _normalize_post_tags(tags):
                cursor.execute('INSERT INTO post_tags (post_id, tag) VALUES (?, ?)', (post_id, tag))
        if category:
            cursor.execute("INSERT OR IGNORE INTO post_categories (name) VALUES (?)", (category,))

        conn.commit()
    finally:
        conn.close()

    return get_post(post_id)


def delete_post(post_id: int) -> bool:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('DELETE FROM post_comments WHERE post_id = ?', (post_id,))
        cursor.execute('DELETE FROM post_tags WHERE post_id = ?', (post_id,))
        cursor.execute('DELETE FROM posts WHERE id = ?', (post_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        return deleted
    finally:
        conn.close()


def create_comment(post_id: int, author_name: str, body: str) -> Optional[Dict]:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT id FROM posts WHERE id = ?', (post_id,))
        if not cursor.fetchone():
            return None

        cursor.execute('''
            INSERT INTO post_comments (post_id, author_name, body)
            VALUES (?, ?, ?)
        ''', (post_id, author_name, body))
        comment_id = cursor.lastrowid
        conn.commit()

        cursor.execute('''
            SELECT id, post_id, author_name, body, created_at
            FROM post_comments WHERE id = ?
        ''', (comment_id,))
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def delete_comment(post_id: int, comment_id: int) -> bool:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('DELETE FROM post_comments WHERE id = ? AND post_id = ?', (comment_id, post_id))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def create_post_media_record(
    filename: str,
    original_name: str,
    content_type: str,
    size_bytes: int,
    public_url: str,
    cdn_url: Optional[str],
    sha256: str,
    uploaded_by: Optional[str] = None,
) -> Dict:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO post_media (filename, original_name, content_type, size_bytes, public_url, cdn_url, sha256, uploaded_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (filename, original_name, content_type, size_bytes, public_url, cdn_url, sha256, uploaded_by))
        media_id = cursor.lastrowid
        conn.commit()
        cursor.execute('SELECT * FROM post_media WHERE id = ?', (media_id,))
        return dict(cursor.fetchone())
    finally:
        conn.close()


def list_post_media_records(limit: int = 100, offset: int = 0) -> List[Dict]:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        safe_limit = max(1, min(limit, 200))
        safe_offset = max(0, offset)
        cursor.execute('''
            SELECT id, filename, original_name, content_type, size_bytes, public_url, cdn_url, uploaded_by, created_at
            FROM post_media
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
        ''', (safe_limit, safe_offset))
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


# --- Fire event store helpers ---

_PUBLIC_EVENT_COLUMNS = (
    "id", "source", "external_id", "status", "verification_tier",
    "latitude", "longitude", "county_fips", "county_name",
    "occurred_at", "occurred_at_precision",
    "acres", "acres_is_estimate", "cause_category",
    "description", "out_of_ordinary",
    "frp", "confidence", "satellite",
    "official_source_ref",
    "created_at", "updated_at",
)

_ADMIN_EVENT_COLUMNS = _PUBLIC_EVENT_COLUMNS + (
    "occurred_at_tz_offset_minutes",
    "official_source_system",
    "label_revision", "revised_at", "parent_event_id",
    "reporter_contact", "submitter_ip_hash",
    "reporter_name", "reporter_org", "address_text",
    "consent_version", "captcha_verdict",
    "moderated_by", "moderated_at", "pii_purged_at",
    "first_seen_at", "last_seen_at",
)


def _fire_event_fuels(cursor: sqlite3.Cursor, event_id: int) -> List[str]:
    cursor.execute('SELECT fuel_type FROM fire_event_fuels WHERE event_id = ? ORDER BY fuel_type', (event_id,))
    return [row[0] for row in cursor.fetchall()]


def _set_fire_event_fuels(cursor: sqlite3.Cursor, event_id: int, fuel_types: Iterable[str]) -> None:
    cursor.execute('DELETE FROM fire_event_fuels WHERE event_id = ?', (event_id,))
    for fuel_type in fuel_types:
        cursor.execute(
            'INSERT OR IGNORE INTO fire_event_fuels (event_id, fuel_type) VALUES (?, ?)',
            (event_id, fuel_type),
        )


def _fetch_fire_event_row(cursor: sqlite3.Cursor, event_id: int, columns) -> Optional[Dict]:
    cursor.execute(f'SELECT {", ".join(columns)} FROM fire_events WHERE id = ?', (event_id,))
    row = cursor.fetchone()
    if not row:
        return None
    event = dict(row)
    event["fuel_types"] = _fire_event_fuels(cursor, event_id)
    return event


def record_fire_moderation(
    cursor: sqlite3.Cursor,
    event_id: int,
    action: str,
    actor: str = "",
    from_status: str = "",
    to_status: str = "",
    from_tier: str = "",
    to_tier: str = "",
    reason: str = "",
    changed_fields: Optional[Dict] = None,
) -> None:
    """Append-only audit row. Caller owns the transaction/commit."""
    import json as _json
    cursor.execute('''
        INSERT INTO fire_event_moderation
            (event_id, action, actor, from_status, to_status, from_tier, to_tier, reason, changed_fields_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (event_id, action, actor, from_status, to_status, from_tier, to_tier, reason,
          _json.dumps(changed_fields or {})))


def create_fire_report(
    latitude: float,
    longitude: float,
    occurred_at: str,
    occurred_at_precision: str,
    acres: float,
    acres_is_estimate: bool,
    fuel_types: List[str],
    description: str,
    out_of_ordinary: str,
    reporter_contact: str,
    submitter_ip_hash: str,
    consent_version: str,
    captcha_verdict: str,
    reporter_name: str = "",
    reporter_org: str = "",
    address_text: str = "",
    upload_token_hash: str = "",
    county_fips: Optional[str] = None,
    county_name: Optional[str] = None,
    occurred_at_tz_offset_minutes: Optional[int] = None,
) -> Dict:
    """Insert a public, anonymous fire report as status='pending'."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO fire_events (
                source, status, verification_tier,
                latitude, longitude, county_fips, county_name,
                occurred_at, occurred_at_precision, occurred_at_tz_offset_minutes,
                acres, acres_is_estimate, description, out_of_ordinary,
                reporter_contact, reporter_name, reporter_org, address_text,
                submitter_ip_hash, upload_token_hash, consent_version, captcha_verdict
            ) VALUES ('user_submission', 'pending', 'unverified', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            latitude, longitude, county_fips, county_name,
            occurred_at, occurred_at_precision, occurred_at_tz_offset_minutes,
            acres, 1 if acres_is_estimate else 0, description, out_of_ordinary,
            reporter_contact, reporter_name, reporter_org, address_text,
            submitter_ip_hash, upload_token_hash, consent_version, captcha_verdict,
        ))
        event_id = cursor.lastrowid
        _set_fire_event_fuels(cursor, event_id, fuel_types)
        record_fire_moderation(cursor, event_id, action="submitted", to_status="pending", to_tier="unverified")
        conn.commit()
        return _fetch_fire_event_row(cursor, event_id, _ADMIN_EVENT_COLUMNS)
    finally:
        conn.close()


def upsert_detection_event(
    source: str,
    external_id: str,
    latitude: float,
    longitude: float,
    occurred_at: str,
    county_fips: Optional[str] = None,
    county_name: Optional[str] = None,
    frp: Optional[float] = None,
    confidence: Optional[str] = None,
    satellite: Optional[str] = None,
    occurred_at_precision: str = "minute",
    verification_tier: str = "unverified",
    cause_category: Optional[str] = None,
    acres: Optional[float] = None,
    official_source_system: Optional[str] = None,
    official_source_ref: Optional[str] = None,
) -> Dict:
    """
    Idempotent upsert for a non-submission fire record (satellite/NGFS
    detections at verification_tier='unverified', or an already-vetted
    official dataset like USFS FPA-FOD at verification_tier=
    'official_source_confirmed'). Always lands as status='approved'.

    The ON CONFLICT clause deliberately never touches latitude/longitude/
    occurred_at/verification_tier/cause_category/acres, so an admin
    correction (or, for an official import, a source data correction on
    re-ingest) survives the next ingest cycle rather than being silently
    overwritten - same guarantee the satellite/NGFS callers already rely on.
    """
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT id FROM fire_events WHERE source = ? AND external_id = ?', (source, external_id))
        existing = cursor.fetchone()
        is_new = existing is None

        cursor.execute('''
            INSERT INTO fire_events (
                source, external_id, status, verification_tier,
                latitude, longitude, county_fips, county_name,
                occurred_at, occurred_at_precision, frp, confidence, satellite,
                cause_category, acres, official_source_system, official_source_ref,
                first_seen_at, last_seen_at
            ) VALUES (?, ?, 'approved', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(source, external_id) DO UPDATE SET
                last_seen_at = CURRENT_TIMESTAMP,
                frp = COALESCE(excluded.frp, fire_events.frp),
                confidence = COALESCE(excluded.confidence, fire_events.confidence),
                satellite = COALESCE(excluded.satellite, fire_events.satellite),
                updated_at = CURRENT_TIMESTAMP
        ''', (
            source, external_id, verification_tier, latitude, longitude, county_fips, county_name,
            occurred_at, occurred_at_precision, frp, confidence, satellite,
            cause_category or "unknown", acres, official_source_system or "", official_source_ref or "",
        ))
        cursor.execute('SELECT id FROM fire_events WHERE source = ? AND external_id = ?', (source, external_id))
        event_id = cursor.fetchone()[0]
        if is_new:
            record_fire_moderation(cursor, event_id, action="ingested", actor=f"system:{source}_ingest",
                                    to_status="approved", to_tier=verification_tier)
        conn.commit()
        return {"event_id": event_id, "inserted": is_new, "updated": not is_new}
    finally:
        conn.close()


def get_fire_event(event_id: int, admin: bool = False) -> Optional[Dict]:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        columns = _ADMIN_EVENT_COLUMNS if admin else _PUBLIC_EVENT_COLUMNS
        event = _fetch_fire_event_row(cursor, event_id, columns)
        if event and admin:
            event["media"] = get_fire_event_media(event_id)
            cursor.execute('''
                SELECT id, event_id, action, actor, from_status, to_status, from_tier, to_tier,
                       reason, changed_fields_json, created_at
                FROM fire_event_moderation WHERE event_id = ? ORDER BY created_at ASC
            ''', (event_id,))
            event["moderation"] = [dict(row) for row in cursor.fetchall()]
        return event
    finally:
        conn.close()


def count_fire_event_media(event_id: int, kind: Optional[str] = None) -> int:
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        if kind is None:
            row = conn.execute(
                "SELECT COUNT(*) FROM fire_event_media WHERE event_id = ?", (event_id,)
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT COUNT(*) FROM fire_event_media WHERE event_id = ? AND kind = ?", (event_id, kind)
            ).fetchone()
        return int(row[0] if row else 0)


def get_fire_upload_token_hash(event_id: int) -> Optional[str]:
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT upload_token_hash FROM fire_events WHERE id = ?", (event_id,)
        ).fetchone()
        return row[0] if row else None


def add_fire_event_media(
    event_id: int,
    stored_filename: str,
    original_filename: str,
    content_type: str,
    size_bytes: int,
    sha256: str,
    kind: str = "photo",
) -> Dict:
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO fire_event_media
                (event_id, stored_filename, original_filename, content_type, size_bytes, sha256, kind)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (event_id, stored_filename, original_filename, content_type, size_bytes, sha256, kind),
        )
        conn.commit()
        row = cursor.execute(
            """
            SELECT id, event_id, stored_filename, original_filename, content_type,
                   size_bytes, sha256, review_state, kind, created_at
            FROM fire_event_media WHERE id = ?
            """,
            (cursor.lastrowid,),
        ).fetchone()
        return dict(row)


def get_fire_event_media(event_id: int) -> List[Dict]:
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, event_id, stored_filename, original_filename, content_type,
                   size_bytes, sha256, review_state, kind, created_at
            FROM fire_event_media WHERE event_id = ? ORDER BY created_at, id
            """,
            (event_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def list_fire_events(
    status: Optional[str] = None,
    source: Optional[str] = None,
    verification_tier: Optional[str] = None,
    county_fips: Optional[str] = None,
    since: Optional[str] = None,
    until: Optional[str] = None,
    bbox: Optional[tuple] = None,
    limit: int = 200,
    offset: int = 0,
    admin: bool = False,
) -> List[Dict]:
    """
    List fire events. Public callers must pass status='approved' (the
    router enforces this); admin callers may omit it to see everything.
    bbox is (min_lon, min_lat, max_lon, max_lat).
    """
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        safe_limit = max(1, min(limit, 200))
        safe_offset = max(0, offset)
        columns = _ADMIN_EVENT_COLUMNS if admin else _PUBLIC_EVENT_COLUMNS

        clauses = []
        params: List = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if source:
            clauses.append("source = ?")
            params.append(source)
        if verification_tier:
            clauses.append("verification_tier = ?")
            params.append(verification_tier)
        if county_fips:
            clauses.append("county_fips = ?")
            params.append(county_fips)
        if since:
            clauses.append("occurred_at >= ?")
            params.append(since)
        if until:
            clauses.append("occurred_at <= ?")
            params.append(until)
        if bbox:
            min_lon, min_lat, max_lon, max_lat = bbox
            clauses.append("latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ?")
            params.extend([min_lat, max_lat, min_lon, max_lon])

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cursor.execute(f'''
            SELECT {", ".join(columns)} FROM fire_events
            {where}
            ORDER BY occurred_at DESC
            LIMIT ? OFFSET ?
        ''', (*params, safe_limit, safe_offset))

        events = []
        for row in cursor.fetchall():
            event = dict(row)
            event["fuel_types"] = _fire_event_fuels(cursor, event["id"])
            events.append(event)
        return events
    finally:
        conn.close()


def list_nearby_fire_events(latitude: float, longitude: float, radius_km: float, hours: float) -> List[Dict]:
    """Duplicate-report hint for the admin detail page: other reports near this point/time."""
    from core.geo import degree_box, haversine_km

    min_lat, max_lat, min_lon, max_lon = degree_box(latitude, longitude, radius_km)
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('''
            SELECT id, latitude, longitude, occurred_at, status, verification_tier, source
            FROM fire_events
            WHERE latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ?
              AND status != 'deleted'
        ''', (min_lat, max_lat, min_lon, max_lon))
        nearby = []
        for row in cursor.fetchall():
            distance = haversine_km(latitude, longitude, row["latitude"], row["longitude"])
            if distance <= radius_km:
                event = dict(row)
                event["distance_km"] = round(distance, 3)
                nearby.append(event)
        return nearby
    finally:
        conn.close()


def set_fire_event_status(
    event_id: int,
    to_status: str,
    actor: str,
    to_tier: Optional[str] = None,
    official_source_ref: Optional[str] = None,
    reason: str = "",
) -> Optional[Dict]:
    """Approve/reject a pending report. Refuses (returns a sentinel) if not currently pending."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT status, verification_tier FROM fire_events WHERE id = ?', (event_id,))
        row = cursor.fetchone()
        if not row:
            return None
        if row["status"] != "pending":
            return {"already_moderated": True, "status": row["status"]}

        from_status, from_tier = row["status"], row["verification_tier"]
        new_tier = to_tier or from_tier
        cursor.execute('''
            UPDATE fire_events
            SET status = ?, verification_tier = ?, official_source_ref = COALESCE(?, official_source_ref),
                moderated_by = ?, moderated_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (to_status, new_tier, official_source_ref, actor, event_id))
        action = "approved" if to_status == "approved" else "rejected"
        record_fire_moderation(cursor, event_id, action=action, actor=actor,
                                from_status=from_status, to_status=to_status,
                                from_tier=from_tier, to_tier=new_tier, reason=reason)
        conn.commit()
        return _fetch_fire_event_row(cursor, event_id, _ADMIN_EVENT_COLUMNS)
    finally:
        conn.close()


def update_fire_event(event_id: int, actor: str, edit_reason: str, **fields) -> Optional[Dict]:
    """
    Edit an event. `fields` may include latitude, longitude, acres,
    fuel_types, description, out_of_ordinary, verification_tier,
    official_source_ref, redact_reporter_contact, cause_category.
    None values leave the column untouched (COALESCE), matching update_post.
    """
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT status, verification_tier FROM fire_events WHERE id = ?', (event_id,))
        row = cursor.fetchone()
        if not row:
            return None
        from_tier = row["verification_tier"]

        fuel_types = fields.pop("fuel_types", None)
        redact_contact = fields.pop("redact_reporter_contact", False)
        to_tier = fields.get("verification_tier")

        cursor.execute('''
            UPDATE fire_events
            SET latitude = COALESCE(?, latitude),
                longitude = COALESCE(?, longitude),
                acres = COALESCE(?, acres),
                description = COALESCE(?, description),
                out_of_ordinary = COALESCE(?, out_of_ordinary),
                verification_tier = COALESCE(?, verification_tier),
                official_source_ref = COALESCE(?, official_source_ref),
                cause_category = COALESCE(?, cause_category),
                reporter_name = COALESCE(?, reporter_name),
                reporter_org = COALESCE(?, reporter_org),
                address_text = COALESCE(?, address_text),
                revised_at = CURRENT_TIMESTAMP,
                label_revision = label_revision + 1,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (
            fields.get("latitude"), fields.get("longitude"), fields.get("acres"),
            fields.get("description"), fields.get("out_of_ordinary"),
            fields.get("verification_tier"), fields.get("official_source_ref"),
            fields.get("cause_category"), fields.get("reporter_name"),
            fields.get("reporter_org"), fields.get("address_text"), event_id,
        ))
        if fuel_types is not None:
            _set_fire_event_fuels(cursor, event_id, fuel_types)
        if redact_contact:
            cursor.execute("UPDATE fire_events SET reporter_contact = '' WHERE id = ?", (event_id,))

        changed = {k: v for k, v in {**fields, "fuel_types": fuel_types}.items() if v is not None}
        record_fire_moderation(cursor, event_id, action="edited", actor=actor,
                                from_tier=from_tier, to_tier=to_tier or from_tier,
                                reason=edit_reason, changed_fields=changed)
        conn.commit()
        return _fetch_fire_event_row(cursor, event_id, _ADMIN_EVENT_COLUMNS)
    finally:
        conn.close()


def delete_fire_event(event_id: int, actor: str, reason: str = "") -> bool:
    """
    Soft delete: status='deleted'. fire_event_moderation is retained on
    purpose - it is the audit trail, and the resulting orphan reference is
    intentional (there is no FK enforcement in SQLite here, so nothing
    breaks, but document this in the runbook).
    """
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT status FROM fire_events WHERE id = ?', (event_id,))
        row = cursor.fetchone()
        if not row:
            return False
        from_status = row[0]
        cursor.execute("UPDATE fire_events SET status = 'deleted', updated_at = CURRENT_TIMESTAMP WHERE id = ?", (event_id,))
        cursor.execute('DELETE FROM fire_event_fuels WHERE event_id = ?', (event_id,))
        record_fire_moderation(cursor, event_id, action="deleted", actor=actor,
                                from_status=from_status, to_status="deleted", reason=reason)
        conn.commit()
        return True
    finally:
        conn.close()


def export_fire_labels(
    min_tier: str = "admin_reviewed",
    since: Optional[str] = None,
    until: Optional[str] = None,
    limit: int = 100000,
) -> List[Dict]:
    """
    Fire events eligible as model labels, newest first. No PII columns in
    the select list by construction - this export is safe to ship off-box.
    """
    from core.fire_events import TIER_RANK

    allowed = [tier for tier, rank in TIER_RANK.items() if rank >= TIER_RANK[min_tier]]
    placeholders = ",".join("?" for _ in allowed)

    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        params: List = list(allowed)
        clauses = [f"e.verification_tier IN ({placeholders})"]
        if since:
            clauses.append("e.occurred_at >= ?")
            params.append(since)
        if until:
            clauses.append("e.occurred_at <= ?")
            params.append(until)
        where = " AND ".join(clauses)

        cursor.execute(f'''
            SELECT e.id AS event_id, e.source, e.verification_tier,
                   e.latitude, e.longitude, e.county_fips, e.county_name,
                   e.occurred_at, e.occurred_at_precision, e.occurred_at_tz_offset_minutes,
                   e.cause_category, e.official_source_system,
                   e.acres, e.acres_is_estimate,
                   e.frp, e.confidence, e.satellite,
                   e.label_revision, e.revised_at,
                   GROUP_CONCAT(f.fuel_type) AS fuel_types,
                   e.created_at, e.updated_at
            FROM fire_events e
            LEFT JOIN fire_event_fuels f ON f.event_id = e.id
            WHERE e.status = 'approved' AND {where}
            GROUP BY e.id
            ORDER BY e.occurred_at DESC
            LIMIT ?
        ''', (*params, max(1, limit)))

        rows = []
        for row in cursor.fetchall():
            event = dict(row)
            event["fuel_types"] = event["fuel_types"].split(",") if event["fuel_types"] else []
            rows.append(event)
        return rows
    finally:
        conn.close()


def consume_fire_submission_quota(bucket_key: str, now: datetime, per_hour_limit: int, per_day_limit: int) -> Dict:
    """
    Atomically charge one submission against the hour and day windows for a
    bucket. The only function in this codebase where two concurrent
    requests race on the same row, hence the manual BEGIN IMMEDIATE
    transaction rather than the usual autocommit pattern.
    """
    db_path = get_db_path()
    conn = sqlite3.connect(db_path, timeout=10.0, isolation_level=None)
    cursor = conn.cursor()
    try:
        cursor.execute('BEGIN IMMEDIATE')
        hour_key = now.strftime('%Y-%m-%dT%H')
        day_key = now.strftime('%Y-%m-%d')
        windows = (('hour', hour_key, per_hour_limit), ('day', day_key, per_day_limit))

        for kind, window_start, limit in windows:
            cursor.execute('''
                SELECT hits FROM fire_submission_throttle
                WHERE bucket_key = ? AND window_kind = ? AND window_start = ?
            ''', (bucket_key, kind, window_start))
            row = cursor.fetchone()
            if row and row[0] >= limit:
                cursor.execute('ROLLBACK')
                if kind == 'hour':
                    retry_after = 3600 - (now.minute * 60 + now.second)
                else:
                    retry_after = 86400 - (now.hour * 3600 + now.minute * 60 + now.second)
                return {"allowed": False, "window": kind, "retry_after": max(1, retry_after)}

        for kind, window_start, _limit in windows:
            cursor.execute('''
                INSERT INTO fire_submission_throttle (bucket_key, window_kind, window_start, hits, updated_at)
                VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP)
                ON CONFLICT(bucket_key, window_kind, window_start)
                DO UPDATE SET hits = hits + 1, updated_at = CURRENT_TIMESTAMP
            ''', (bucket_key, kind, window_start))
        cursor.execute('COMMIT')
        return {"allowed": True, "window": "", "retry_after": 0}
    except Exception:
        try:
            cursor.execute('ROLLBACK')
        except Exception:
            pass
        raise
    finally:
        conn.close()


def is_ip_blocked(ip_hash: str) -> bool:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT 1 FROM fire_submission_blocklist WHERE ip_hash = ?', (ip_hash,))
        return cursor.fetchone() is not None
    finally:
        conn.close()


def add_ip_to_blocklist(ip_hash: str, reason: str, created_by: str) -> None:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO fire_submission_blocklist (ip_hash, reason, created_by)
            VALUES (?, ?, ?)
            ON CONFLICT(ip_hash) DO UPDATE SET reason = excluded.reason, created_by = excluded.created_by
        ''', (ip_hash, reason, created_by))
        conn.commit()
    finally:
        conn.close()


def purge_fire_submission_pii(older_than_days: int = 90) -> int:
    """Clear reporter_contact/submitter_ip_hash on reports moderated more than N days ago."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            UPDATE fire_events
            SET reporter_contact = '', reporter_name = '', reporter_org = '',
                address_text = '', submitter_ip_hash = '', upload_token_hash = '',
                pii_purged_at = CURRENT_TIMESTAMP
            WHERE moderated_at IS NOT NULL
              AND moderated_at <= datetime('now', ? || ' days')
              AND pii_purged_at IS NULL
              AND (reporter_contact != '' OR reporter_name != '' OR reporter_org != ''
                   OR address_text != '' OR submitter_ip_hash != '' OR upload_token_hash != '')
        ''', (f"-{max(0, older_than_days)}",))
        purged = cursor.rowcount
        conn.commit()
        return purged
    finally:
        conn.close()


def expire_unmoderated_fire_reports(older_than_days: int = 30) -> int:
    """Auto-reject reports still pending after N days - an unbounded pending queue is an unbounded PII store."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            SELECT id FROM fire_events
            WHERE status = 'pending' AND created_at <= datetime('now', ? || ' days')
        ''', (f"-{max(0, older_than_days)}",))
        ids = [row[0] for row in cursor.fetchall()]
        for event_id in ids:
            cursor.execute('''
                UPDATE fire_events SET status = 'rejected', moderated_by = 'system:auto-expire',
                       moderated_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (event_id,))
            record_fire_moderation(cursor, event_id, action="rejected", actor="system:auto-expire",
                                    from_status="pending", to_status="rejected",
                                    reason="expired-unmoderated")
        conn.commit()
        return len(ids)
    finally:
        conn.close()


def purge_fire_throttle_rows(older_than_hours: int = 48) -> int:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            DELETE FROM fire_submission_throttle WHERE updated_at <= datetime('now', ? || ' hours')
        ''', (f"-{max(0, older_than_hours)}",))
        purged = cursor.rowcount
        conn.commit()
        return purged
    finally:
        conn.close()


def _feedback_row_to_dict(row: sqlite3.Row) -> Dict:
    import json as _json
    data = dict(row)
    try:
        data["details"] = _json.loads(data.get("details") or "{}")
    except (TypeError, ValueError):
        data["details"] = {}
    return data


def create_feedback_submission(*, name: str, email: str, category: str, details: Dict, message: str,
                               submitter_ip_hash: str) -> Dict:
    """Insert a public feedback submission as status='new'."""
    import json as _json
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO feedback (name, email, category, details, message, status, submitter_ip_hash)
            VALUES (?, ?, ?, ?, ?, 'new', ?)
        ''', (name, email, category, _json.dumps(details), message, submitter_ip_hash))
        conn.commit()
        cursor.execute('SELECT * FROM feedback WHERE id = ?', (cursor.lastrowid,))
        return _feedback_row_to_dict(cursor.fetchone())
    finally:
        conn.close()


def list_feedback(status: Optional[str] = None, limit: int = 50, offset: int = 0) -> List[Dict]:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        if status:
            cursor.execute('''
                SELECT * FROM feedback WHERE status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?
            ''', (status, limit, offset))
        else:
            cursor.execute('SELECT * FROM feedback ORDER BY created_at DESC LIMIT ? OFFSET ?', (limit, offset))
        return [_feedback_row_to_dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def update_feedback_status(feedback_id: int, status: str) -> Optional[Dict]:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('''
            UPDATE feedback SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?
        ''', (status, feedback_id))
        if cursor.rowcount == 0:
            conn.commit()
            return None
        conn.commit()
        cursor.execute('SELECT * FROM feedback WHERE id = ?', (feedback_id,))
        row = cursor.fetchone()
        return _feedback_row_to_dict(row) if row else None
    finally:
        conn.close()


def consume_feedback_submission_quota(bucket_key: str, now: datetime, per_hour_limit: int, per_day_limit: int) -> Dict:
    """Same atomic per-hour/per-day charge as consume_fire_submission_quota, against feedback's own
    throttle table - see _ensure_feedback_tables for why these aren't shared."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path, timeout=10.0, isolation_level=None)
    cursor = conn.cursor()
    try:
        cursor.execute('BEGIN IMMEDIATE')
        hour_key = now.strftime('%Y-%m-%dT%H')
        day_key = now.strftime('%Y-%m-%d')
        windows = (('hour', hour_key, per_hour_limit), ('day', day_key, per_day_limit))

        for kind, window_start, limit in windows:
            cursor.execute('''
                SELECT hits FROM feedback_submission_throttle
                WHERE bucket_key = ? AND window_kind = ? AND window_start = ?
            ''', (bucket_key, kind, window_start))
            row = cursor.fetchone()
            if row and row[0] >= limit:
                cursor.execute('ROLLBACK')
                if kind == 'hour':
                    retry_after = 3600 - (now.minute * 60 + now.second)
                else:
                    retry_after = 86400 - (now.hour * 3600 + now.minute * 60 + now.second)
                return {"allowed": False, "window": kind, "retry_after": max(1, retry_after)}

        for kind, window_start, _limit in windows:
            cursor.execute('''
                INSERT INTO feedback_submission_throttle (bucket_key, window_kind, window_start, hits, updated_at)
                VALUES (?, ?, ?, 1, CURRENT_TIMESTAMP)
                ON CONFLICT(bucket_key, window_kind, window_start)
                DO UPDATE SET hits = hits + 1, updated_at = CURRENT_TIMESTAMP
            ''', (bucket_key, kind, window_start))
        cursor.execute('COMMIT')
        return {"allowed": True, "window": "", "retry_after": 0}
    except Exception:
        try:
            cursor.execute('ROLLBACK')
        except Exception:
            pass
        raise
    finally:
        conn.close()


def purge_feedback_throttle_rows(older_than_hours: int = 48) -> int:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            DELETE FROM feedback_submission_throttle WHERE updated_at <= datetime('now', ? || ' hours')
        ''', (f"-{max(0, older_than_hours)}",))
        purged = cursor.rowcount
        conn.commit()
        return purged
    finally:
        conn.close()
