"""
SQLite Database - core/database.py
"""
import sqlite3
import logging
import os
import json
import re
import secrets
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
    fire_events_columns = {row[1] for row in cursor.execute("PRAGMA table_info(fire_events)").fetchall()}
    for column_name, column_type in (
        ("bright_t7", "REAL"),
        ("bright_t13", "REAL"),
        ("pixel_area", "REAL"),
        ("quality_flag", "INTEGER"),
        ("solar_zenith_angle", "REAL"),
        ("satellite_zenith_angle", "REAL"),
        ("daynight", "TEXT"),
        ("land_cover", "TEXT"),
        ("detection_confidence_pct", "REAL"),
        ("footprint_geojson", "TEXT"),
        ("recurring_source_id", "INTEGER"),
        ("exclusion_zone_id", "INTEGER"),
        ("fuel_model_fbfm40", "INTEGER"),
        ("canopy_cover_pct", "REAL"),
        ("weather_danger_category", "TEXT"),
        ("weather_danger_prob", "REAL"),
        ("reporter_relationship", "TEXT"),
    ):
        if column_name not in fire_events_columns:
            cursor.execute(f"ALTER TABLE fire_events ADD COLUMN {column_name} {column_type}")

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
        ("incident_id", "INTEGER"),
    ):
        if name not in columns:
            cursor.execute(f"ALTER TABLE fire_events ADD COLUMN {name} {definition}")
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_fire_events_incident ON fire_events(incident_id)')

    # Migrate fire_event_media tables created before department-report uploads.
    cursor.execute("PRAGMA table_info(fire_event_media)")
    media_columns = {row[1] for row in cursor.fetchall()}
    if "kind" not in media_columns:
        cursor.execute("ALTER TABLE fire_event_media ADD COLUMN kind TEXT NOT NULL DEFAULT 'photo'")


def _ensure_fire_incident_tables(cursor: sqlite3.Cursor) -> None:
    """Persistent grouping of nearby/recent satellite fire detections into one incident."""
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fire_incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            centroid_latitude REAL NOT NULL,
            centroid_longitude REAL NOT NULL,
            first_detected_at TEXT NOT NULL,
            last_detected_at TEXT NOT NULL,
            detection_count INTEGER NOT NULL DEFAULT 0,
            county_fips TEXT,
            county_name TEXT,
            status TEXT NOT NULL DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute("PRAGMA table_info(fire_incidents)")
    incident_columns = {row[1] for row in cursor.fetchall()}
    for name, definition in (
        ("public_slug", "TEXT"),
        ("graphic_filename", "TEXT"),
        ("shape_geojson", "TEXT"),
        ("shape_detection_count", "INTEGER"),
    ):
        if name not in incident_columns:
            cursor.execute(f"ALTER TABLE fire_incidents ADD COLUMN {name} {definition}")
    cursor.execute("SELECT id FROM fire_incidents WHERE public_slug IS NULL OR public_slug = ''")
    for (incident_id,) in cursor.fetchall():
        cursor.execute("UPDATE fire_incidents SET public_slug = ? WHERE id = ?", (secrets.token_urlsafe(9), incident_id))
    cursor.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_fire_incidents_public_slug ON fire_incidents(public_slug)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_fire_incidents_last_detected ON fire_incidents(last_detected_at DESC)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_fire_incidents_centroid ON fire_incidents(centroid_latitude, centroid_longitude)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_fire_incidents_county ON fire_incidents(county_fips)')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fire_incident_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            incident_id INTEGER NOT NULL,
            classification TEXT NOT NULL,
            note TEXT NOT NULL DEFAULT '',
            contact TEXT NOT NULL DEFAULT '',
            submitter_ip_hash TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'pending',
            reviewed_by TEXT NOT NULL DEFAULT '',
            reviewed_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (incident_id) REFERENCES fire_incidents(id)
        )
    ''')
    cursor.execute("PRAGMA table_info(fire_incident_feedback)")
    feedback_columns = {row[1] for row in cursor.fetchall()}
    for name, definition in (
        ("status", "TEXT NOT NULL DEFAULT 'pending'"),
        ("reviewed_by", "TEXT NOT NULL DEFAULT ''"),
        ("reviewed_at", "TIMESTAMP"),
    ):
        if name not in feedback_columns:
            cursor.execute(f"ALTER TABLE fire_incident_feedback ADD COLUMN {name} {definition}")
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_fire_incident_feedback_incident ON fire_incident_feedback(incident_id, created_at DESC)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_fire_incident_feedback_status ON fire_incident_feedback(status, created_at DESC)')


def _ensure_recurring_fire_source_tables(cursor: sqlite3.Cursor) -> None:
    """Fixed industrial heat sources (mills, flares, kilns, plants) that
    repeatedly trigger satellite fire detections at the same location.
    Detected as 'candidate' rows by services/recurring_source_detector.py;
    a human must promote a row to 'confirmed' before it suppresses future
    detections (see find_confirmed_recurring_source/upsert_detection_event) -
    never suppressed automatically."""
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS recurring_fire_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label TEXT NOT NULL DEFAULT '',
            source_type TEXT NOT NULL DEFAULT 'unknown',
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            radius_km REAL NOT NULL DEFAULT 1.5,
            status TEXT NOT NULL DEFAULT 'candidate',
            detection_count INTEGER NOT NULL DEFAULT 0,
            distinct_day_count INTEGER NOT NULL DEFAULT 0,
            first_detected_at TEXT,
            last_detected_at TEXT,
            reviewed_by TEXT NOT NULL DEFAULT '',
            reviewed_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_recurring_fire_sources_status ON recurring_fire_sources(status)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_recurring_fire_sources_centroid ON recurring_fire_sources(latitude, longitude)')


def _ensure_fire_exclusion_zone_tables(cursor: sqlite3.Cursor) -> None:
    """Admin-drawn polygons marking areas where satellite fire detections are
    known false positives (e.g. a flare stack or quarry that keeps lighting
    up NGFS/VIIRS). Unlike recurring_fire_sources (a point+radius, one
    location at a time), a zone is an arbitrary shape. Any 'active' zone
    suppresses new detections at ingest (see _find_matching_exclusion_zone/
    upsert_detection_event) the moment it's created - no review step,
    since a staff member drew it deliberately."""
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fire_exclusion_zones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            notes TEXT NOT NULL DEFAULT '',
            geometry_geojson TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active',
            created_by TEXT NOT NULL DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_fire_exclusion_zones_status ON fire_exclusion_zones(status)')


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


def _ensure_fuel_moisture_sensor_tables(cursor: sqlite3.Cursor) -> None:
    """Field-deployed dowel fuel-moisture sensors (probe-in-dowel design,
    see SMF_FuelMoistureSensor). Distinct from the observations/station_forecasts
    tables, which hold RAWS network data, not our own hardware.
    """
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fuel_moisture_sensor_readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            site_id TEXT NOT NULL,
            device_id TEXT NOT NULL,
            recorded_at TIMESTAMP NOT NULL,
            air_temp_c REAL,
            relative_humidity_pct REAL,
            fuel_moisture_pct REAL,
            battery_v REAL,
            rssi_dbm INTEGER,
            uptime_s INTEGER,
            firmware_version TEXT,
            enclosure_state TEXT,
            received_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(device_id, recorded_at)
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_fuel_sensor_site_recorded ON fuel_moisture_sensor_readings(site_id, recorded_at DESC)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_fuel_sensor_device_recorded ON fuel_moisture_sensor_readings(device_id, recorded_at DESC)')

    # Migration: enclosure_state didn't exist in the first version of this table.
    cursor.execute("PRAGMA table_info(fuel_moisture_sensor_readings)")
    columns = {row[1] for row in cursor.fetchall()}
    if "enclosure_state" not in columns:
        cursor.execute("ALTER TABLE fuel_moisture_sensor_readings ADD COLUMN enclosure_state TEXT")


# Live, no-restart enable/disable state for the shadow/advisory model
# families (fire_weather_index, fire_weather_ml, risk_fusion_glm, v4, v5).
# Deliberately minimal - one boolean per family, not a multi-column config
# like forecast_source_models (status enum, schedule_minutes, blend
# weights) - this only ever needs to answer "should this family's
# _requested() return True right now." Seeded from each family's CURRENT
# env-var-derived state so a fresh deploy never silently disables
# something already running via .env.
SHADOW_MODEL_SETTINGS_DEFAULTS = {
    "fire_weather_index": os.getenv("FIRE_WEATHER_INDEX_SHADOW_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"},
    "fire_weather_ml": os.getenv("FIRE_WEATHER_ML_SHADOW_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"},
    "risk_fusion_glm": os.getenv("RISK_FUSION_GLM_SHADOW_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"},
    "v4": True,  # v4 has no existing env-var gate - defaults enabled to preserve today's always-on behavior
    "v5": os.getenv("V5_SHADOW_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"},
}


def _ensure_shadow_model_settings_table(cursor: sqlite3.Cursor) -> None:
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS shadow_model_settings (
            family TEXT PRIMARY KEY,
            enabled INTEGER NOT NULL DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_by TEXT
        )
    ''')
    for family, default_enabled in SHADOW_MODEL_SETTINGS_DEFAULTS.items():
        cursor.execute(
            'INSERT OR IGNORE INTO shadow_model_settings (family, enabled, updated_by) VALUES (?, ?, ?)',
            (family, int(default_enabled), None),
        )


def get_shadow_model_setting(family: str) -> Optional[Dict]:
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute(
            'SELECT family, enabled, updated_at, updated_by FROM shadow_model_settings WHERE family = ?', (family,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        return {"family": row["family"], "enabled": bool(row["enabled"]),
                "updated_at": row["updated_at"], "updated_by": row["updated_by"]}
    finally:
        conn.close()


def set_shadow_model_setting(family: str, enabled: bool, updated_by: Optional[str]) -> Dict:
    conn = sqlite3.connect(get_db_path())
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO shadow_model_settings (family, enabled, updated_by, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(family) DO UPDATE SET
                enabled = excluded.enabled, updated_by = excluded.updated_by, updated_at = CURRENT_TIMESTAMP
        ''', (family, int(enabled), updated_by))
        conn.commit()
    finally:
        conn.close()
    return get_shadow_model_setting(family)


def list_shadow_model_settings() -> Dict[str, Dict]:
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT family, enabled, updated_at, updated_by FROM shadow_model_settings')
        rows = cursor.fetchall()
        return {row["family"]: {"enabled": bool(row["enabled"]), "updated_at": row["updated_at"],
                                "updated_by": row["updated_by"]} for row in rows}
    finally:
        conn.close()


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
    
def _migrate_forecasts_off_legacy_valid_time_unique(conn: sqlite3.Connection) -> None:
    """Rebuild `forecasts` if it still carries the original single-column
    UNIQUE(valid_time) constraint. That constraint predates the `cycle` column
    and would otherwise block a 9z row from coexisting with a 12z row for the
    same valid_time even after the new composite (valid_time, cycle) index is
    added below. Only runs (and only once) when the legacy constraint is
    actually still present, so it's a no-op on fresh installs and on databases
    already migrated.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='forecasts'")
    if not cursor.fetchone():
        return
    cursor.execute("PRAGMA index_list('forecasts')")
    legacy_unique = False
    for _, index_name, is_unique, *_rest in cursor.fetchall():
        if not is_unique:
            continue
        cursor.execute(f"PRAGMA index_info('{index_name}')")
        cols = [row[2] for row in cursor.fetchall()]
        if cols == ['valid_time']:
            legacy_unique = True
            break
    if not legacy_unique:
        return
    logger.info("Migrating forecasts table off legacy UNIQUE(valid_time) constraint")
    cursor.execute('''
        CREATE TABLE forecasts_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            valid_time TIMESTAMP NOT NULL,
            title TEXT NOT NULL,
            discussion TEXT NOT NULL,
            cycle INTEGER NOT NULL DEFAULT 12,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        INSERT INTO forecasts_new (id, valid_time, title, discussion, cycle, created_at, updated_at)
        SELECT id, valid_time, title, discussion, cycle, created_at, updated_at FROM forecasts
    ''')
    cursor.execute('DROP TABLE forecasts')
    cursor.execute('ALTER TABLE forecasts_new RENAME TO forecasts')
    conn.commit()


def init_database():
    db_path = get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)

    # Persistent, database-level setting (stored in the file header) so every
    # future connection from any process uses WAL: readers no longer block
    # writers and vice versa, which matters since live requests and
    # APScheduler jobs hit this same file concurrently.
    conn.execute("PRAGMA journal_mode=WAL")

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
    # HRRR cycle this forecast was generated from (12 = operational, 9 = early secondary run).
    # Existing rows default to 12 so they keep meaning "the 12z forecast" with no data change.
    try: conn.execute('ALTER TABLE forecasts ADD COLUMN cycle INTEGER NOT NULL DEFAULT 12')
    except: pass
    _migrate_forecasts_off_legacy_valid_time_unique(conn)

    cursor = conn.cursor()

    # 1. Your existing forecasts table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            valid_time TIMESTAMP NOT NULL,
            title TEXT NOT NULL,
            discussion TEXT NOT NULL,
            cycle INTEGER NOT NULL DEFAULT 12,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # valid_time was UNIQUE on its own; now unique per (valid_time, cycle) so a
    # 9z row and a 12z row for the same day can coexist without colliding.
    cursor.execute(
        'CREATE UNIQUE INDEX IF NOT EXISTS idx_forecasts_valid_time_cycle '
        'ON forecasts(valid_time, cycle)'
    )

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS forecast_discussions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            author_name TEXT,
            issued_at TIMESTAMP,
            status TEXT NOT NULL DEFAULT 'draft'
                CHECK (status IN ('draft', 'published', 'archived')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute(
        'CREATE INDEX IF NOT EXISTS idx_forecast_discussions_status_issued '
        'ON forecast_discussions(status, issued_at DESC)'
    )
    
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
    _ensure_fire_incident_tables(cursor)
    _ensure_recurring_fire_source_tables(cursor)
    _ensure_fire_exclusion_zone_tables(cursor)

    # 17. Anonymous fire-report abuse controls (per-IP throttle + blocklist)
    _ensure_fire_abuse_tables(cursor)

    # 18. Public feedback form + its own submission throttle
    _ensure_feedback_tables(cursor)

    # 19. County burn-ban submissions and moderation
    _ensure_burn_ban_tables(cursor)

    # 20. Daily fire-weather-zone alert history, keyed by county for later
    # verification (e.g. does the model's predicted danger align with NWS
    # Red Flag Warning/Fire Weather Watch issuance).
    _ensure_fire_weather_alert_history_table(cursor)

    # 21. Department static graphics API and publication control plane.
    _ensure_graphics_tables(cursor)

    # 22. Field-deployed dowel fuel-moisture sensor readings (own hardware,
    # not RAWS). See SMF_FuelMoistureSensor/.
    _ensure_fuel_moisture_sensor_tables(cursor)

    # 23. Live, no-restart enable/disable state for the shadow/advisory
    # model families (fire_weather_index, fire_weather_ml, risk_fusion_glm,
    # v4, v5) - replaces requiring a .env edit + server restart to flip one.
    _ensure_shadow_model_settings_table(cursor)

    # 24. FireWx bulletin content, email preferences, county forecasts, and
    # delivery idempotency. Resend remains authoritative for contact status
    # and unsubscribe state; these tables only store local preferences and
    # operational bookkeeping.
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bulletins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT NOT NULL,
            html_body TEXT NOT NULL,
            text_body TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'draft'
                CHECK (status IN ('draft', 'sending', 'sent')),
            resend_broadcast_id TEXT,
            sent_at TIMESTAMP,
            last_error TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_bulletins_status_created ON bulletins(status, created_at DESC)')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS newsletter_subscribers (
            email TEXT PRIMARY KEY,
            resend_contact_id TEXT,
            name TEXT NOT NULL DEFAULT '',
            affiliation TEXT NOT NULL DEFAULT '',
            manage_token TEXT UNIQUE,
            unsubscribed_at TIMESTAMP,
            subscription_types_json TEXT NOT NULL DEFAULT '["fire-weather-forecasts"]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    subscriber_columns = {row[1] for row in cursor.execute("PRAGMA table_info(newsletter_subscribers)").fetchall()}
    if "name" not in subscriber_columns:
        cursor.execute("ALTER TABLE newsletter_subscribers ADD COLUMN name TEXT NOT NULL DEFAULT ''")
    if "affiliation" not in subscriber_columns:
        cursor.execute("ALTER TABLE newsletter_subscribers ADD COLUMN affiliation TEXT NOT NULL DEFAULT ''")
    if "manage_token" not in subscriber_columns:
        cursor.execute("ALTER TABLE newsletter_subscribers ADD COLUMN manage_token TEXT")
    if "unsubscribed_at" not in subscriber_columns:
        cursor.execute("ALTER TABLE newsletter_subscribers ADD COLUMN unsubscribed_at TIMESTAMP")
    if "subscription_types_json" not in subscriber_columns:
        cursor.execute(
            "ALTER TABLE newsletter_subscribers ADD COLUMN subscription_types_json TEXT NOT NULL DEFAULT "
            "'[\"fire-weather-forecasts\"]'"
        )
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS newsletter_preferences (
            email TEXT NOT NULL,
            county_fips TEXT NOT NULL,
            min_danger_level INTEGER NOT NULL CHECK (min_danger_level BETWEEN 0 AND 4),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (email, county_fips),
            FOREIGN KEY (email) REFERENCES newsletter_subscribers(email)
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_newsletter_preferences_county ON newsletter_preferences(county_fips, min_danger_level)')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS county_forecast_days (
            forecast_date TEXT NOT NULL,
            county_fips TEXT NOT NULL,
            danger_level INTEGER NOT NULL CHECK (danger_level BETWEEN 0 AND 4),
            summary TEXT NOT NULL DEFAULT '',
            forecast_run_id TEXT NOT NULL DEFAULT '',
            published_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (forecast_date, county_fips)
        )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_county_forecast_days_date ON county_forecast_days(forecast_date, danger_level)')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS newsletter_deliveries (
            email TEXT NOT NULL,
            county_fips TEXT NOT NULL,
            forecast_date TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            provider_message_id TEXT,
            error TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            sent_at TIMESTAMP,
            PRIMARY KEY (email, county_fips, forecast_date)
        )
    ''')
    delivery_columns = {row[1] for row in cursor.execute("PRAGMA table_info(newsletter_deliveries)").fetchall()}
    if "claimed_at" not in delivery_columns:
        cursor.execute("ALTER TABLE newsletter_deliveries ADD COLUMN claimed_at TIMESTAMP")
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_newsletter_deliveries_status ON newsletter_deliveries(status, created_at)')

    conn.commit()
    conn.close()
    logger.info(f"Database initialized at {db_path}")

def get_latest_forecast(cycle: int = 12):
    """
    Retrieves the most recent forecast from the database for the given HRRR
    cycle (default 12 = the operational forecast). This default means every
    existing caller keeps seeing only the 12z forecast, regardless of whether
    or when a secondary-cycle (e.g. 9z) forecast has also been written.
    """
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    try:
        cursor.execute('''
            SELECT * FROM forecasts
            WHERE cycle = ?
            ORDER BY id DESC
            LIMIT 1
        ''', (cycle,))

        row = cursor.fetchone()

        if row:
            return dict(row)
        return None
    finally:
        conn.close()

def get_forecast_by_time(valid_time, cycle: int = 12):
    """
    Retrieves a forecast by its valid_time and HRRR cycle (default 12).
    """
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute('SELECT * FROM forecasts WHERE valid_time = ? AND cycle = ?', (valid_time, cycle))
    row = cursor.fetchone()
    conn.close()

    if row:
        return dict(row)
    return None

def get_recent_forecasts(limit=5, cycle: int = 12):
    """
    Retrieves the most recent forecasts from the database for the given HRRR
    cycle (default 12 = the operational forecast).
    """
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute('''
        SELECT * FROM forecasts
        WHERE cycle = ?
        ORDER BY valid_time DESC
        LIMIT ?
    ''', (cycle, limit))

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
    if "cycle" not in columns:
        cursor.execute("ALTER TABLE forecasts ADD COLUMN cycle INTEGER NOT NULL DEFAULT 12")

def insert_forecast(valid_time, title, discussion, cycle: int = 12):
    """
    Inserts a new forecast into the database.

    Args:
        valid_time (datetime): The valid time of the forecast.
        title (str): The headline/title of the forecast.
        discussion (str): The detailed discussion text.
        cycle (int): The HRRR cycle hour this forecast was generated from
            (12 = operational run, 9 = early secondary run). Defaults to 12
            so existing callers keep writing the operational forecast.

    Returns:
        int: The ID of the inserted forecast.
    """
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        _ensure_forecasts_schema(cursor)
        cursor.execute('''
            INSERT INTO forecasts (valid_time, title, discussion, cycle)
            VALUES (?, ?, ?, ?)
        ''', (valid_time, title, discussion, cycle))

        forecast_id = cursor.lastrowid
        conn.commit()
        return forecast_id

    except sqlite3.IntegrityError:
        # Forecast for this (time, cycle) already exists - update it instead
        cursor.execute('''
            UPDATE forecasts
            SET title = ?, discussion = ?, updated_at = CURRENT_TIMESTAMP
            WHERE valid_time = ? AND cycle = ?
        ''', (title, discussion, valid_time, cycle))
        conn.commit()

        # Get the ID of the updated row
        cursor.execute('SELECT id FROM forecasts WHERE valid_time = ? AND cycle = ?', (valid_time, cycle))
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
    "bright_t7", "bright_t13", "pixel_area", "quality_flag",
    "solar_zenith_angle", "satellite_zenith_angle", "daynight", "land_cover",
    "detection_confidence_pct",
    "official_source_ref",
    "created_at", "updated_at",
)

_ADMIN_EVENT_COLUMNS = _PUBLIC_EVENT_COLUMNS + (
    "incident_id",
    "recurring_source_id",
    "exclusion_zone_id",
    "occurred_at_tz_offset_minutes",
    "official_source_system",
    "label_revision", "revised_at", "parent_event_id",
    "reporter_contact", "submitter_ip_hash",
    "reporter_name", "reporter_org", "address_text", "reporter_relationship",
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
    reporter_relationship: Optional[str] = None,
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
                reporter_contact, reporter_name, reporter_org, address_text, reporter_relationship,
                submitter_ip_hash, upload_token_hash, consent_version, captcha_verdict
            ) VALUES ('user_submission', 'pending', 'unverified', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            latitude, longitude, county_fips, county_name,
            occurred_at, occurred_at_precision, occurred_at_tz_offset_minutes,
            acres, 1 if acres_is_estimate else 0, description, out_of_ordinary,
            reporter_contact, reporter_name, reporter_org, address_text, reporter_relationship,
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
    bright_t7: Optional[float] = None,
    bright_t13: Optional[float] = None,
    pixel_area: Optional[float] = None,
    quality_flag: Optional[int] = None,
    solar_zenith_angle: Optional[float] = None,
    satellite_zenith_angle: Optional[float] = None,
    daynight: Optional[str] = None,
    land_cover: Optional[str] = None,
    footprint_geojson: Optional[str] = None,
    fuel_model_fbfm40: Optional[int] = None,
    canopy_cover_pct: Optional[float] = None,
) -> Dict:
    """
    Idempotent upsert for a non-submission fire record (satellite/NGFS
    detections at verification_tier='unverified', or an already-vetted
    official dataset like USFS FPA-FOD at verification_tier=
    'official_source_confirmed'). Raw satellite/NGFS detections
    (source in modis/viirs/ngfs) land as status='pending' and require the
    same admin moderation as user-submitted reports; already-vetted sources
    (e.g. official FPA-FOD records) still land as status='approved'.

    The ON CONFLICT clause deliberately never touches latitude/longitude/
    occurred_at/verification_tier/cause_category/acres/status, so an admin
    correction (or, for an official import, a source data correction on
    re-ingest) survives the next ingest cycle rather than being silently
    overwritten - same guarantee the satellite/NGFS callers already rely on.
    """
    initial_status = "pending" if source in ("modis", "viirs", "ngfs") else "approved"
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
                bright_t7, bright_t13, pixel_area, quality_flag,
                solar_zenith_angle, satellite_zenith_angle, daynight, land_cover, footprint_geojson,
                fuel_model_fbfm40, canopy_cover_pct,
                cause_category, acres, official_source_system, official_source_ref,
                first_seen_at, last_seen_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT(source, external_id) DO UPDATE SET
                last_seen_at = CURRENT_TIMESTAMP,
                frp = COALESCE(excluded.frp, fire_events.frp),
                confidence = COALESCE(excluded.confidence, fire_events.confidence),
                satellite = COALESCE(excluded.satellite, fire_events.satellite),
                bright_t7 = COALESCE(excluded.bright_t7, fire_events.bright_t7),
                bright_t13 = COALESCE(excluded.bright_t13, fire_events.bright_t13),
                pixel_area = COALESCE(excluded.pixel_area, fire_events.pixel_area),
                quality_flag = COALESCE(excluded.quality_flag, fire_events.quality_flag),
                solar_zenith_angle = COALESCE(excluded.solar_zenith_angle, fire_events.solar_zenith_angle),
                satellite_zenith_angle = COALESCE(excluded.satellite_zenith_angle, fire_events.satellite_zenith_angle),
                daynight = COALESCE(excluded.daynight, fire_events.daynight),
                land_cover = COALESCE(excluded.land_cover, fire_events.land_cover),
                footprint_geojson = COALESCE(excluded.footprint_geojson, fire_events.footprint_geojson),
                fuel_model_fbfm40 = COALESCE(excluded.fuel_model_fbfm40, fire_events.fuel_model_fbfm40),
                canopy_cover_pct = COALESCE(excluded.canopy_cover_pct, fire_events.canopy_cover_pct),
                updated_at = CURRENT_TIMESTAMP
        ''', (
            source, external_id, initial_status, verification_tier, latitude, longitude, county_fips, county_name,
            occurred_at, occurred_at_precision, frp, confidence, satellite,
            bright_t7, bright_t13, pixel_area, quality_flag,
            solar_zenith_angle, satellite_zenith_angle, daynight, land_cover, footprint_geojson,
            fuel_model_fbfm40, canopy_cover_pct,
            cause_category or "unknown", acres, official_source_system or "", official_source_ref or "",
        ))
        cursor.execute('SELECT id FROM fire_events WHERE source = ? AND external_id = ?', (source, external_id))
        event_id = cursor.fetchone()[0]
        if is_new:
            if source in ("modis", "viirs", "ngfs"):
                exclusion_zone = _find_matching_exclusion_zone(cursor, latitude, longitude)
                recurring_source = _find_confirmed_recurring_source(cursor, latitude, longitude) if exclusion_zone is None else None
                if exclusion_zone is not None:
                    # Staff drew a zone over a known false-positive area - store
                    # the raw read, but skip incident clustering entirely so it
                    # never becomes a published incident anywhere.
                    cursor.execute(
                        'UPDATE fire_events SET exclusion_zone_id = ? WHERE id = ?',
                        (exclusion_zone["id"], event_id),
                    )
                elif recurring_source is not None:
                    # A known non-fire source (mill/flare/kiln/...) - store the
                    # raw read, but skip incident clustering entirely so it
                    # never triggers ML scoring or incident-graphic regeneration.
                    cursor.execute(
                        'UPDATE fire_events SET recurring_source_id = ? WHERE id = ?',
                        (recurring_source["id"], event_id),
                    )
                    _bump_recurring_source_detection(cursor, recurring_source["id"], occurred_at)
                else:
                    incident_id = find_or_create_incident_for_detection(
                        cursor, latitude, longitude, occurred_at, county_fips, county_name
                    )
                    cursor.execute('UPDATE fire_events SET incident_id = ? WHERE id = ?', (incident_id, event_id))
            record_fire_moderation(cursor, event_id, action="ingested", actor=f"system:{source}_ingest",
                                    to_status=initial_status, to_tier=verification_tier)
        conn.commit()
        return {"event_id": event_id, "inserted": is_new, "updated": not is_new}
    finally:
        conn.close()


def update_detection_confidence(
    event_id: int,
    confidence_pct: float,
    weather_danger_category: Optional[str] = None,
    weather_danger_prob: Optional[float] = None,
) -> None:
    """Write the per-detection ML confidence score (0-100), plus the
    weather-danger context (services/weather_context.py) computed alongside
    it. Separate from upsert_detection_event because scoring runs as its own
    pass after ingest, once a detection's full feature set is committed."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            '''UPDATE fire_events
               SET detection_confidence_pct = ?,
                   weather_danger_category = COALESCE(?, weather_danger_category),
                   weather_danger_prob = COALESCE(?, weather_danger_prob)
               WHERE id = ?''',
            (confidence_pct, weather_danger_category, weather_danger_prob, event_id),
        )
        conn.commit()
    finally:
        conn.close()


def list_detection_footprints(limit: int = 500) -> List[Dict]:
    """Recent satellite detections that carry a stored pixel-footprint
    polygon (currently only NGFS - see fire_ingest.py's _ingest_ngfs_feature),
    newest first, for the map's optional 'Pixel Footprints' layer."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute(
            '''SELECT id, incident_id, source, frp, confidence, satellite, occurred_at,
                      bright_t7, detection_confidence_pct, footprint_geojson
               FROM fire_events
               WHERE footprint_geojson IS NOT NULL
               ORDER BY occurred_at DESC
               LIMIT ?''',
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def list_detection_events_for_scoring(limit: int = 2000) -> List[Dict]:
    """Rows the per-detection confidence model can score: satellite-sourced
    detections that haven't been scored yet, newest first. Excludes
    detections already attributed to a confirmed recurring non-fire source -
    scoring a known mill's "confidence" would be wasted compute."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute(
            '''SELECT id, source, frp, confidence, satellite, bright_t7, bright_t13,
                      pixel_area, quality_flag, solar_zenith_angle, satellite_zenith_angle,
                      daynight, land_cover, fuel_model_fbfm40, canopy_cover_pct, county_fips
               FROM fire_events
               WHERE source IN ('modis', 'viirs', 'ngfs') AND detection_confidence_pct IS NULL
                 AND recurring_source_id IS NULL
               ORDER BY id DESC
               LIMIT ?''',
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def list_labeled_detection_events(limit: int = 5000) -> List[Dict]:
    """Reviewed fire_events rows usable as training labels for the
    detection-confidence model - same bootstrap set fire_confidence.py's
    incident-level scorer already draws on."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute(
            '''SELECT id, source, frp, confidence, satellite, bright_t7, bright_t13,
                      pixel_area, quality_flag, solar_zenith_angle, satellite_zenith_angle,
                      daynight, land_cover, fuel_model_fbfm40, canopy_cover_pct, county_fips,
                      cause_category
               FROM fire_events
               WHERE source IN ('modis', 'viirs', 'ngfs')
                 AND verification_tier IN ('admin_reviewed', 'official_source_confirmed')
                 AND cause_category != 'unknown'
               ORDER BY id DESC
               LIMIT ?''',
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]
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
        safe_limit = max(1, min(limit, 500))
        safe_offset = max(0, offset)
        columns = _ADMIN_EVENT_COLUMNS if admin else _PUBLIC_EVENT_COLUMNS

        clauses = []
        params: List = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if source:
            # Accept a comma-separated list (e.g. "modis,viirs,ngfs") so
            # callers can fetch several sources - like "all satellite
            # detections" - in a single request.
            sources = [value.strip() for value in source.split(",") if value.strip()]
            if len(sources) > 1:
                clauses.append(f"source IN ({', '.join('?' for _ in sources)})")
                params.extend(sources)
            elif sources:
                clauses.append("source = ?")
                params.append(sources[0])
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


INCIDENT_CLUSTER_RADIUS_KM = float(os.getenv("FIRE_INCIDENT_CLUSTER_RADIUS_KM", "2.0"))
INCIDENT_CLUSTER_WINDOW_HOURS = float(os.getenv("FIRE_INCIDENT_CLUSTER_WINDOW_HOURS", "48.0"))


def _parse_occurred_at(occurred_at: str) -> datetime:
    return datetime.fromisoformat(occurred_at.replace("Z", "+00:00"))


def find_or_create_incident_for_detection(
    cursor: sqlite3.Cursor,
    latitude: float,
    longitude: float,
    occurred_at: str,
    county_fips: Optional[str] = None,
    county_name: Optional[str] = None,
    radius_km: float = INCIDENT_CLUSTER_RADIUS_KM,
    window_hours: float = INCIDENT_CLUSTER_WINDOW_HOURS,
) -> int:
    """
    Find an existing fire_incidents row within radius_km/window_hours of this
    satellite detection and attach to it (updating its running-average
    centroid, detection_count, and time span), or create a new incident.

    Caller owns the transaction/commit (same convention as
    record_fire_moderation). Originally satellite-ingest-only; also called
    by correlate_report_with_incident() below when an admin approves a user
    report, so an approved report joins an existing detection cluster
    instead of only ever being listed as a "nearby" sibling for a human to
    notice via list_nearby_fire_events().
    """
    from datetime import timedelta
    from core.geo import degree_box, haversine_km

    occurred_dt = _parse_occurred_at(occurred_at)
    window_start = (occurred_dt - timedelta(hours=window_hours)).isoformat()
    window_end = (occurred_dt + timedelta(hours=window_hours)).isoformat()
    min_lat, max_lat, min_lon, max_lon = degree_box(latitude, longitude, radius_km)

    cursor.execute('''
        SELECT id, centroid_latitude, centroid_longitude, detection_count,
               first_detected_at, last_detected_at
        FROM fire_incidents
        WHERE centroid_latitude BETWEEN ? AND ? AND centroid_longitude BETWEEN ? AND ?
          AND last_detected_at >= ? AND first_detected_at <= ?
    ''', (min_lat, max_lat, min_lon, max_lon, window_start, window_end))

    best_row, best_distance = None, None
    for row in cursor.fetchall():
        distance = haversine_km(latitude, longitude, row["centroid_latitude"], row["centroid_longitude"])
        if distance <= radius_km and (best_distance is None or distance < best_distance):
            best_row, best_distance = row, distance

    if best_row is not None:
        n = best_row["detection_count"]
        new_lat = (best_row["centroid_latitude"] * n + latitude) / (n + 1)
        new_lon = (best_row["centroid_longitude"] * n + longitude) / (n + 1)
        new_first = min(best_row["first_detected_at"], occurred_at)
        new_last = max(best_row["last_detected_at"], occurred_at)
        cursor.execute('''
            UPDATE fire_incidents
            SET centroid_latitude = ?, centroid_longitude = ?, detection_count = detection_count + 1,
                first_detected_at = ?, last_detected_at = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ?
        ''', (new_lat, new_lon, new_first, new_last, best_row["id"]))
        cursor.execute(
            "UPDATE fire_incidents SET public_slug = COALESCE(public_slug, ?) WHERE id = ?",
            (secrets.token_urlsafe(9), best_row["id"]),
        )
        return best_row["id"]

    cursor.execute('''
        INSERT INTO fire_incidents
            (centroid_latitude, centroid_longitude, first_detected_at, last_detected_at,
             detection_count, county_fips, county_name, public_slug)
        VALUES (?, ?, ?, ?, 1, ?, ?, ?)
    ''', (latitude, longitude, occurred_at, occurred_at, county_fips, county_name, secrets.token_urlsafe(9)))
    return cursor.lastrowid


def _find_matching_exclusion_zone(cursor: sqlite3.Cursor, latitude: float, longitude: float) -> Optional[Dict]:
    """Active admin-drawn exclusion zone containing this point, or None. Only
    'active' rows suppress anything - a 'deleted' zone stops affecting new
    detections immediately but (like recurring sources) doesn't retroactively
    restore anything. Takes the caller's own cursor (same transaction as
    upsert_detection_event) rather than a separate connection."""
    import json as _json
    from shapely.geometry import Point, shape

    cursor.execute("SELECT id, name, geometry_geojson FROM fire_exclusion_zones WHERE status = 'active'")
    point = Point(longitude, latitude)
    for row in cursor.fetchall():
        try:
            polygon = shape(_json.loads(row["geometry_geojson"]))
        except (TypeError, ValueError):
            continue
        if polygon.contains(point):
            return dict(row)
    return None


def _find_confirmed_recurring_source(cursor: sqlite3.Cursor, latitude: float, longitude: float) -> Optional[Dict]:
    """Nearest CONFIRMED recurring non-fire source within its own radius_km
    of this point, or None. Only 'confirmed' rows suppress anything -
    'candidate'/'dismissed' rows never affect ingestion. Takes the caller's
    own cursor (same transaction as upsert_detection_event) rather than a
    separate connection, to avoid lock contention with the write in progress."""
    from core.geo import haversine_km

    cursor.execute(
        "SELECT id, latitude, longitude, radius_km FROM recurring_fire_sources WHERE status = 'confirmed'"
    )
    best_row, best_distance = None, None
    for row in cursor.fetchall():
        distance = haversine_km(latitude, longitude, row["latitude"], row["longitude"])
        if distance <= row["radius_km"] and (best_distance is None or distance < best_distance):
            best_row, best_distance = row, distance
    return dict(best_row) if best_row is not None else None


def find_confirmed_recurring_source(latitude: float, longitude: float) -> Optional[Dict]:
    """Standalone-connection variant of _find_confirmed_recurring_source for
    read-only callers outside an existing transaction (e.g. tests, other
    services)."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        return _find_confirmed_recurring_source(cursor, latitude, longitude)
    finally:
        conn.close()


def _bump_recurring_source_detection(cursor: sqlite3.Cursor, source_id: int, occurred_at: str) -> None:
    """Caller-owned-transaction stats bump for a suppressed detection - same
    convention as the incident centroid update in
    find_or_create_incident_for_detection above."""
    cursor.execute(
        '''UPDATE recurring_fire_sources
           SET detection_count = detection_count + 1,
               last_detected_at = MAX(COALESCE(last_detected_at, ?), ?),
               updated_at = CURRENT_TIMESTAMP
           WHERE id = ?''',
        (occurred_at, occurred_at, source_id),
    )


def list_recurring_sources(status: Optional[str] = None) -> List[Dict]:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        if status:
            cursor.execute('SELECT * FROM recurring_fire_sources WHERE status = ? ORDER BY detection_count DESC', (status,))
        else:
            cursor.execute('SELECT * FROM recurring_fire_sources ORDER BY status, detection_count DESC')
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def find_recurring_source_near(latitude: float, longitude: float, radius_km: float) -> Optional[Dict]:
    """Any existing recurring_fire_sources row (any status) within radius_km -
    used by the detector job to dedupe candidates across grid-cell edges
    rather than creating a near-duplicate row next to one that already exists."""
    from core.geo import haversine_km

    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT id, latitude, longitude FROM recurring_fire_sources')
        best_row, best_distance = None, None
        for row in cursor.fetchall():
            distance = haversine_km(latitude, longitude, row["latitude"], row["longitude"])
            if distance <= radius_km and (best_distance is None or distance < best_distance):
                best_row, best_distance = row, distance
        return dict(best_row) if best_row is not None else None
    finally:
        conn.close()


def create_recurring_source_candidate(
    latitude: float, longitude: float, detection_count: int, distinct_day_count: int,
    first_detected_at: str, last_detected_at: str, radius_km: float = 1.5,
) -> int:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.execute(
            '''INSERT INTO recurring_fire_sources
                   (latitude, longitude, radius_km, detection_count, distinct_day_count,
                    first_detected_at, last_detected_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            (latitude, longitude, radius_km, detection_count, distinct_day_count, first_detected_at, last_detected_at),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def update_recurring_source_stats(
    source_id: int, detection_count: int, distinct_day_count: int, first_detected_at: str, last_detected_at: str,
) -> None:
    """Refresh a candidate/confirmed source's rolling stats without
    touching its status - the detector job never promotes/demotes a row,
    only a human via set_recurring_source_status can."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            '''UPDATE recurring_fire_sources
               SET detection_count = ?, distinct_day_count = ?,
                   first_detected_at = MIN(first_detected_at, ?),
                   last_detected_at = MAX(last_detected_at, ?),
                   updated_at = CURRENT_TIMESTAMP
               WHERE id = ?''',
            (detection_count, distinct_day_count, first_detected_at, last_detected_at, source_id),
        )
        conn.commit()
    finally:
        conn.close()


def set_recurring_source_status(source_id: int, status: str, reviewed_by: str, label: Optional[str] = None, source_type: Optional[str] = None) -> Optional[Dict]:
    if status not in ("candidate", "confirmed", "dismissed"):
        raise ValueError(f"invalid recurring source status: {status}")
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(
            '''UPDATE recurring_fire_sources
               SET status = ?, reviewed_by = ?, reviewed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP,
                   label = COALESCE(?, label), source_type = COALESCE(?, source_type)
               WHERE id = ?''',
            (status, reviewed_by, label, source_type, source_id),
        )
        conn.commit()
        row = conn.execute('SELECT * FROM recurring_fire_sources WHERE id = ?', (source_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_fire_exclusion_zones(status: Optional[str] = "active") -> List[Dict]:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        if status:
            cursor.execute('SELECT * FROM fire_exclusion_zones WHERE status = ? ORDER BY created_at DESC', (status,))
        else:
            cursor.execute('SELECT * FROM fire_exclusion_zones ORDER BY status, created_at DESC')
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def create_fire_exclusion_zone(name: str, geometry_geojson: str, created_by: str, notes: str = "") -> Dict:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(
            '''INSERT INTO fire_exclusion_zones (name, notes, geometry_geojson, created_by)
               VALUES (?, ?, ?, ?)''',
            (name, notes, geometry_geojson, created_by),
        )
        conn.commit()
        row = conn.execute('SELECT * FROM fire_exclusion_zones WHERE id = ?', (cursor.lastrowid,)).fetchone()
        return dict(row)
    finally:
        conn.close()


def set_fire_exclusion_zone_status(zone_id: int, status: str) -> Optional[Dict]:
    if status not in ("active", "deleted"):
        raise ValueError(f"invalid exclusion zone status: {status}")
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(
            "UPDATE fire_exclusion_zones SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (status, zone_id),
        )
        conn.commit()
        row = conn.execute('SELECT * FROM fire_exclusion_zones WHERE id = ?', (zone_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_fire_incidents_in_geometry(geometry_geojson: str) -> List[Dict]:
    """Active (non-deleted) incidents whose centroid falls inside the given
    GeoJSON polygon/multipolygon - used to retroactively hide existing
    incidents the moment a new exclusion zone is drawn over them."""
    import json as _json
    from shapely.geometry import Point, shape

    polygon = shape(_json.loads(geometry_geojson))
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT id, centroid_latitude, centroid_longitude FROM fire_incidents WHERE status != 'deleted'")
        return [
            dict(row) for row in cursor.fetchall()
            if polygon.contains(Point(row["centroid_longitude"], row["centroid_latitude"]))
        ]
    finally:
        conn.close()


def list_unclustered_satellite_events() -> List[Dict]:
    """Satellite-sourced fire_events rows with no incident_id yet, oldest first."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('''
            SELECT id, latitude, longitude, occurred_at, county_fips, county_name
            FROM fire_events
            WHERE source IN ('modis', 'viirs', 'ngfs') AND incident_id IS NULL
            ORDER BY occurred_at ASC
        ''')
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def _fire_incident_sources(cursor: sqlite3.Cursor, incident_id: int) -> Dict[str, int]:
    cursor.execute('SELECT source, COUNT(*) as n FROM fire_events WHERE incident_id = ? GROUP BY source', (incident_id,))
    return {row["source"]: row["n"] for row in cursor.fetchall()}


def list_fire_incidents(
    since: Optional[str] = None,
    until: Optional[str] = None,
    source: Optional[str] = None,
    bbox: Optional[tuple] = None,
    has_feedback: Optional[bool] = None,
    confirmed_only: Optional[bool] = None,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict]:
    """bbox is (min_lon, min_lat, max_lon, max_lat), matching list_fire_events.
    'Confirmed' means at least one public feedback submission classified
    'confirmed_fire' (see fire_incident_feedback/create_fire_incident_feedback) -
    that feedback is unmoderated, so this is a public signal, not a review gate."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        safe_limit = max(1, min(limit, 500))
        safe_offset = max(0, offset)

        clauses = []
        params: List = []
        if since:
            clauses.append("last_detected_at >= ?")
            params.append(since)
        if until:
            clauses.append("first_detected_at <= ?")
            params.append(until)
        if bbox:
            min_lon, min_lat, max_lon, max_lat = bbox
            clauses.append("centroid_latitude BETWEEN ? AND ? AND centroid_longitude BETWEEN ? AND ?")
            params.extend([min_lat, max_lat, min_lon, max_lon])
        if source:
            clauses.append('''EXISTS (
                SELECT 1 FROM fire_events fe WHERE fe.incident_id = fire_incidents.id AND fe.source = ?
            )''')
            params.append(source)
        if has_feedback:
            clauses.append('EXISTS (SELECT 1 FROM fire_incident_feedback fb WHERE fb.incident_id = fire_incidents.id)')
        if confirmed_only:
            clauses.append('''EXISTS (
                SELECT 1 FROM fire_incident_feedback fb
                WHERE fb.incident_id = fire_incidents.id AND fb.classification = 'confirmed_fire' AND fb.status = 'approved'
            )''')

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        cursor.execute(f'''
            SELECT * FROM fire_incidents
            {where}
            ORDER BY last_detected_at DESC
            LIMIT ? OFFSET ?
        ''', (*params, safe_limit, safe_offset))

        incidents = []
        for row in cursor.fetchall():
            incident = dict(row)
            incident["sources"] = _fire_incident_sources(cursor, incident["id"])
            counties = cursor.execute(
                "SELECT DISTINCT county_name FROM fire_events WHERE incident_id = ? AND county_name IS NOT NULL AND county_name != '' ORDER BY county_name",
                (incident["id"],),
            ).fetchall()
            incident["county_names"] = [item[0] for item in counties]
            incident["county_name"] = ", ".join(incident["county_names"]) or incident.get("county_name")
            feedback_rows = cursor.execute(
                "SELECT classification, status, COUNT(*) as n FROM fire_incident_feedback WHERE incident_id = ? GROUP BY classification, status",
                (incident["id"],),
            ).fetchall()
            feedback_counts: Dict[str, int] = {}
            approved_counts: Dict[str, int] = {}
            pending_count = 0
            for fb_row in feedback_rows:
                feedback_counts[fb_row["classification"]] = feedback_counts.get(fb_row["classification"], 0) + fb_row["n"]
                if fb_row["status"] == "approved":
                    approved_counts[fb_row["classification"]] = approved_counts.get(fb_row["classification"], 0) + fb_row["n"]
                elif fb_row["status"] == "pending":
                    pending_count += fb_row["n"]
            incident["feedback_counts"] = feedback_counts
            incident["feedback_count"] = sum(feedback_counts.values())
            incident["pending_feedback_count"] = pending_count
            # 'Confirmed' requires an admin-approved 'confirmed_fire' submission -
            # a pending or rejected one doesn't count (see set_fire_incident_feedback_status).
            incident["confirmed"] = approved_counts.get("confirmed_fire", 0) > 0
            incidents.append(incident)
        return incidents
    finally:
        conn.close()


def get_fire_incident(incident_id: int) -> Optional[Dict]:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT * FROM fire_incidents WHERE id = ?', (incident_id,))
        row = cursor.fetchone()
        if not row:
            return None
        incident = dict(row)
        incident["sources"] = _fire_incident_sources(cursor, incident_id)
        counties = cursor.execute(
            "SELECT DISTINCT county_name FROM fire_events WHERE incident_id = ? AND county_name IS NOT NULL AND county_name != '' ORDER BY county_name",
            (incident_id,),
        ).fetchall()
        incident["county_names"] = [item[0] for item in counties]
        incident["county_name"] = ", ".join(incident["county_names"]) or incident.get("county_name")
        return incident
    finally:
        conn.close()


def list_fire_incident_members(incident_id: int) -> List[Dict]:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('''
            SELECT id, latitude, longitude, occurred_at, satellite, confidence, frp, source,
                   bright_t7, land_cover, detection_confidence_pct, footprint_geojson
            FROM fire_events
            WHERE incident_id = ?
            ORDER BY occurred_at ASC
        ''', (incident_id,))
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_public_fire_incident(slug: str) -> Optional[Dict]:
    """Return a public incident and its non-sensitive detection summary."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM fire_incidents WHERE public_slug = ? AND status != 'deleted'",
            (slug,),
        ).fetchone()
        if not row:
            return None
        incident = dict(row)
        incident["sources"] = _fire_incident_sources(conn.cursor(), incident["id"])
        incident["detections"] = list_fire_incident_members(incident["id"])
        return incident
    finally:
        conn.close()


def create_fire_incident_feedback(
    incident_id: int, classification: str, note: str, contact: str, ip_hash: str,
) -> Dict:
    db_path = get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO fire_incident_feedback
               (incident_id, classification, note, contact, submitter_ip_hash)
               VALUES (?, ?, ?, ?, ?)""",
            (incident_id, classification, note, contact, ip_hash),
        )
        conn.commit()
        return {"id": cursor.lastrowid, "incident_id": incident_id, "classification": classification}


def set_fire_incident_graphic(incident_id: int, filename: str) -> None:
    with sqlite3.connect(get_db_path()) as conn:
        conn.execute("UPDATE fire_incidents SET graphic_filename = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (filename, incident_id))
        conn.commit()


def set_fire_incident_shape(incident_id: int, shape_geojson: str, detection_count: int) -> None:
    """Store the ML-extracted irregular shape for an incident (see
    services/incident_shape_extractor.py) along with the detection_count it
    was computed at, so refresh_incident_shapes() can skip recomputing an
    incident that hasn't changed."""
    with sqlite3.connect(get_db_path()) as conn:
        conn.execute(
            "UPDATE fire_incidents SET shape_geojson = ?, shape_detection_count = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (shape_geojson, detection_count, incident_id),
        )
        conn.commit()


def list_fire_incident_feedback(incident_id: int) -> List[Dict]:
    with sqlite3.connect(get_db_path()) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT id, incident_id, classification, note, contact, status, reviewed_by, reviewed_at, created_at "
            "FROM fire_incident_feedback WHERE incident_id = ? ORDER BY created_at DESC",
            (incident_id,),
        ).fetchall()
        return [dict(row) for row in rows]


def list_pending_fire_incident_feedback(limit: int = 100) -> List[Dict]:
    """Feedback awaiting admin review, most recent first, joined with just
    enough incident context (slug/county) for the moderation queue to link
    back to the incident without a second round-trip per row."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            '''SELECT fb.id, fb.incident_id, fb.classification, fb.note, fb.contact, fb.created_at,
                      fi.public_slug, fi.county_name, fi.detection_count
               FROM fire_incident_feedback fb
               JOIN fire_incidents fi ON fi.id = fb.incident_id
               WHERE fb.status = 'pending'
               ORDER BY fb.created_at DESC
               LIMIT ?''',
            (max(1, min(limit, 500)),),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def set_fire_incident_feedback_status(feedback_id: int, status: str, reviewed_by: str) -> Optional[Dict]:
    if status not in ("pending", "approved", "rejected"):
        raise ValueError(f"invalid feedback status: {status}")
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(
            "UPDATE fire_incident_feedback SET status = ?, reviewed_by = ?, reviewed_at = CURRENT_TIMESTAMP WHERE id = ?",
            (status, reviewed_by, feedback_id),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM fire_incident_feedback WHERE id = ?", (feedback_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_nearby_fire_events(
    latitude: float,
    longitude: float,
    radius_km: float,
    hours: float,
    occurred_at: Optional[str] = None,
    exclude_event_id: Optional[int] = None,
) -> List[Dict]:
    """Duplicate-report hint for the admin detail page, and the primitive
    report-detection correlation (see fires_v2/services.fire_labeling) uses
    to decide whether a report should join an existing detection incident:
    other events near this point *and* time. `hours` was previously accepted
    but never applied - pass `occurred_at` (the reference event's own
    occurred_at) to actually filter to that window; omit it to fall back to
    the old space-only behavior for any caller that hasn't been updated."""
    from core.geo import degree_box, haversine_km

    min_lat, max_lat, min_lon, max_lon = degree_box(latitude, longitude, radius_km)
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        query = '''
            SELECT id, latitude, longitude, occurred_at, status, verification_tier, source
            FROM fire_events
            WHERE latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ?
              AND status != 'deleted'
        '''
        params: List = [min_lat, max_lat, min_lon, max_lon]
        if occurred_at:
            query += " AND ABS(julianday(occurred_at) - julianday(?)) <= (? / 24.0)"
            params += [occurred_at, hours]
        if exclude_event_id is not None:
            query += " AND id != ?"
            params.append(exclude_event_id)
        cursor.execute(query, params)
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


def correlate_report_with_incident(event_id: int) -> Optional[int]:
    """Called after a user report is approved: join it to an existing
    nearby/recent detection incident cluster (same clustering radius/window
    satellite ingest uses) if one exists, so it groups with corroborating
    satellite detections instead of only surfacing as a "nearby" hint on the
    admin single-report view. Idempotent - a no-op if the event already has
    an incident_id. Returns the (possibly newly created) incident_id, or
    None if the event doesn't exist."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        row = cursor.execute(
            "SELECT latitude, longitude, occurred_at, county_fips, county_name, incident_id "
            "FROM fire_events WHERE id = ?",
            (event_id,),
        ).fetchone()
        if not row:
            return None
        if row["incident_id"] is not None:
            return row["incident_id"]
        incident_id = find_or_create_incident_for_detection(
            cursor, row["latitude"], row["longitude"], row["occurred_at"],
            row["county_fips"], row["county_name"],
        )
        cursor.execute("UPDATE fire_events SET incident_id = ? WHERE id = ?", (incident_id, event_id))
        conn.commit()
        return incident_id
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


def delete_fire_incident(incident_id: int, reason: str = "") -> bool:
    """Soft delete: status='deleted'. Member fire_events rows are left as-is
    (their incident_id reference becomes orphaned, same tolerated pattern as
    delete_fire_event/fire_event_moderation) - there is no FK enforcement in
    SQLite here. reason is currently just documentation at the call site
    (e.g. "inside exclusion zone <id>"); there's no incident-level moderation
    log table the way fire_events has, so it isn't persisted."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT id FROM fire_incidents WHERE id = ?', (incident_id,))
        if not cursor.fetchone():
            return False
        cursor.execute("UPDATE fire_incidents SET status = 'deleted', updated_at = CURRENT_TIMESTAMP WHERE id = ?", (incident_id,))
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


def _ensure_burn_ban_tables(cursor: sqlite3.Cursor) -> None:
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS burn_ban_submissions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            status TEXT NOT NULL DEFAULT 'pending',
            county_fips TEXT NOT NULL,
            county_name TEXT NOT NULL,
            submitter_name TEXT NOT NULL DEFAULT '',
            submitter_contact TEXT NOT NULL DEFAULT '',
            proof_url TEXT NOT NULL DEFAULT '',
            proof_stored_filename TEXT NOT NULL DEFAULT '',
            proof_original_filename TEXT NOT NULL DEFAULT '',
            proof_content_type TEXT NOT NULL DEFAULT '',
            request_type TEXT NOT NULL DEFAULT 'issue',
            effective_at TEXT NOT NULL,
            expires_at TEXT NOT NULL DEFAULT '',
            submitter_ip_hash TEXT NOT NULL DEFAULT '',
            upload_token_hash TEXT NOT NULL DEFAULT '',
            captcha_verdict TEXT NOT NULL DEFAULT '',
            consent_version TEXT NOT NULL DEFAULT '',
            moderator_note TEXT NOT NULL DEFAULT '',
            moderated_by TEXT NOT NULL DEFAULT '',
            moderated_at TIMESTAMP,
            published_at TIMESTAMP,
            pii_purged_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute(
        'CREATE INDEX IF NOT EXISTS idx_burn_ban_status_created '
        'ON burn_ban_submissions(status, created_at DESC)'
    )
    cursor.execute(
        'CREATE INDEX IF NOT EXISTS idx_burn_ban_county_active '
        'ON burn_ban_submissions(county_fips, status, effective_at, expires_at)'
    )

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS burn_ban_moderation (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            submission_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            actor TEXT NOT NULL DEFAULT '',
            from_status TEXT NOT NULL DEFAULT '',
            to_status TEXT NOT NULL DEFAULT '',
            reason TEXT NOT NULL DEFAULT '',
            changed_fields_json TEXT NOT NULL DEFAULT '{}',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (submission_id) REFERENCES burn_ban_submissions(id)
        )
    ''')
    cursor.execute(
        'CREATE INDEX IF NOT EXISTS idx_burn_ban_moderation_submission '
        'ON burn_ban_moderation(submission_id, created_at)'
    )

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS burn_ban_submission_throttle (
            bucket_key TEXT NOT NULL,
            window_kind TEXT NOT NULL,
            window_start TEXT NOT NULL,
            hits INTEGER NOT NULL DEFAULT 0,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (bucket_key, window_kind, window_start)
        )
    ''')
    cursor.execute(
        'CREATE INDEX IF NOT EXISTS idx_burn_ban_throttle_updated '
        'ON burn_ban_submission_throttle(updated_at)'
    )
    cursor.execute("PRAGMA table_info(burn_ban_submissions)")
    burn_ban_columns = {row[1] for row in cursor.fetchall()}
    if "request_type" not in burn_ban_columns:
        cursor.execute(
            "ALTER TABLE burn_ban_submissions ADD COLUMN request_type TEXT NOT NULL DEFAULT 'issue'"
        )


def _ensure_fire_weather_alert_history_table(cursor: sqlite3.Cursor) -> None:
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS fire_weather_alert_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            alert_date TEXT NOT NULL,
            county_fips TEXT NOT NULL,
            county_name TEXT NOT NULL DEFAULT '',
            event TEXT NOT NULL,
            alert_id TEXT NOT NULL DEFAULT '',
            onset TEXT NOT NULL DEFAULT '',
            expires TEXT NOT NULL DEFAULT '',
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(alert_date, county_fips, event)
        )
    ''')
    cursor.execute(
        'CREATE INDEX IF NOT EXISTS idx_fire_weather_alert_history_county_date '
        'ON fire_weather_alert_history(county_fips, alert_date)'
    )


def _ensure_graphics_tables(cursor: sqlite3.Cursor) -> None:
    """Control-plane tables for department-owned static graphics."""
    cursor.executescript('''
        CREATE TABLE IF NOT EXISTS graphic_departments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            slug TEXT NOT NULL UNIQUE,
            contact_email TEXT,
            daily_limit INTEGER NOT NULL DEFAULT 100,
            monthly_limit INTEGER NOT NULL DEFAULT 2000,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS graphic_api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            department_id INTEGER NOT NULL,
            key_prefix TEXT NOT NULL UNIQUE,
            key_hash TEXT NOT NULL UNIQUE,
            scopes_json TEXT NOT NULL DEFAULT '["graphics:write","graphics:read"]',
            revoked_at TIMESTAMP,
            last_used_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (department_id) REFERENCES graphic_departments(id)
        );
        CREATE TABLE IF NOT EXISTS graphic_bundles (
            id TEXT PRIMARY KEY,
            department_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            config_json TEXT NOT NULL,
            version INTEGER NOT NULL DEFAULT 1,
            active INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (department_id) REFERENCES graphic_departments(id)
        );
        CREATE TABLE IF NOT EXISTS graphic_assets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            department_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            content_type TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            path TEXT NOT NULL,
            cdn_key TEXT,
            cdn_url TEXT,
            version INTEGER NOT NULL DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (department_id) REFERENCES graphic_departments(id)
        );
        CREATE TABLE IF NOT EXISTS graphic_jobs (
            id TEXT PRIMARY KEY,
            bundle_id TEXT NOT NULL,
            department_id INTEGER NOT NULL,
            status TEXT NOT NULL,
            config_json TEXT NOT NULL,
            source_fingerprint TEXT,
            manifest_json TEXT NOT NULL DEFAULT '{}',
            image_url TEXT,
            error TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            started_at TIMESTAMP,
            finished_at TIMESTAMP,
            FOREIGN KEY (bundle_id) REFERENCES graphic_bundles(id),
            FOREIGN KEY (department_id) REFERENCES graphic_departments(id)
        );
        CREATE TABLE IF NOT EXISTS graphic_usage_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            department_id INTEGER NOT NULL,
            api_key_id INTEGER,
            job_id TEXT,
            status TEXT NOT NULL,
            product_id TEXT NOT NULL,
            latency_ms INTEGER,
            bytes INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (department_id) REFERENCES graphic_departments(id)
        );
        CREATE INDEX IF NOT EXISTS idx_graphic_jobs_department ON graphic_jobs(department_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_graphic_usage_department ON graphic_usage_events(department_id, created_at DESC);
        CREATE TABLE IF NOT EXISTS graphic_department_users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            department_id INTEGER NOT NULL,
            api_key_id INTEGER NOT NULL UNIQUE,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT,
            invited_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            password_set_at TIMESTAMP,
            last_login_at TIMESTAMP,
            terms_version INTEGER NOT NULL DEFAULT 0,
            terms_accepted_at TIMESTAMP,
            FOREIGN KEY (department_id) REFERENCES graphic_departments(id),
            FOREIGN KEY (api_key_id) REFERENCES graphic_api_keys(id)
        );
        CREATE TABLE IF NOT EXISTS graphic_source_state (
            product_id TEXT PRIMARY KEY,
            source_url TEXT NOT NULL,
            source_fingerprint TEXT NOT NULL,
            observed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS graphic_login_codes (
            id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            code_hash TEXT NOT NULL,
            requested_ip_hash TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            expires_at TIMESTAMP NOT NULL,
            consumed_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES graphic_department_users(id)
        );
        CREATE INDEX IF NOT EXISTS idx_graphic_login_codes_user
            ON graphic_login_codes(user_id, created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_graphic_login_codes_ip
            ON graphic_login_codes(requested_ip_hash, created_at DESC);
        CREATE TABLE IF NOT EXISTS graphic_terms_events (
            id TEXT PRIMARY KEY,
            user_id INTEGER,
            department_id INTEGER NOT NULL,
            email TEXT NOT NULL,
            terms_version INTEGER NOT NULL,
            decision TEXT NOT NULL CHECK(decision IN ('accepted','declined')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS forecast_source_models (
            key TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            adapter_key TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'disabled' CHECK(status IN ('disabled','shadow','active')),
            weight_profile_json TEXT,
            schedule_minutes INTEGER,
            last_acquired_at TIMESTAMP,
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            promoted_at TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS forecast_source_model_events (
            id TEXT PRIMARY KEY,
            model_key TEXT NOT NULL,
            action TEXT NOT NULL CHECK(action IN ('added','promoted','demoted','disabled','updated')),
            detail TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS forecast_source_model_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_key TEXT NOT NULL,
            cycle_time TIMESTAMP NOT NULL,
            variable TEXT NOT NULL,
            lead_hour INTEGER NOT NULL,
            available INTEGER NOT NULL,
            mean_abs_diff_from_blend REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_forecast_source_model_metrics_key
            ON forecast_source_model_metrics(model_key, cycle_time DESC);
    ''')

    # Seed the forecast-source-model registry once. hrrr/rrfs/refs/gefs are
    # marked 'active' with no weight_profile_json override (they keep using
    # contracts.BLEND_WEIGHTS as before - this table only carries an *overlay*
    # for models beyond those four). fv3hires starts 'shadow': acquired every
    # run and measured against the live hrrr/rrfs blend (forecast_source_model_metrics),
    # but contributes zero weight to what's actually served until someone
    # reviews that performance and explicitly promotes it to 'active' with a
    # weight through the admin UI - so this seed alone changes no live output.
    cursor.execute("SELECT COUNT(*) FROM forecast_source_models")
    if cursor.fetchone()[0] == 0:
        cursor.executemany(
            "INSERT INTO forecast_source_models(key,display_name,adapter_key,status,promoted_at) VALUES (?,?,?,?,CURRENT_TIMESTAMP)",
            [
                ("hrrr", "HRRR", "hrrr", "active"),
                ("rrfs", "RRFS", "rrfs", "active"),
                ("refs", "REFS (RRFS ensemble)", "refs", "active"),
                ("gefs", "GEFS (coarse fallback)", "gefs", "active"),
            ],
        )
        cursor.execute(
            "INSERT INTO forecast_source_models(key,display_name,adapter_key,status,notes) VALUES (?,?,?,?,?)",
            ("fv3hires", "FV3-HIRES (HiResW FV3)", "fv3hires", "shadow",
             "No native 2m temperature (only 80m TMP + 2m TMAX/TMIN) and no gust field - those "
             "variables fall back to whichever other active model covers the hour. NOMADS-only, "
             "short retention, forward-only capture. Beta: acquired + measured against the live "
             "blend every run, zero weight in production output until promoted."),
        )

    # Future billing hook: these columns are unused today (subscription_status
    # stays 'none' for every department) but leave room to wire a low-cost
    # Stripe subscription onto a department without another schema change.
    cursor.execute("PRAGMA table_info(graphic_departments)")
    department_columns = {row[1] for row in cursor.fetchall()}
    if "stripe_customer_id" not in department_columns:
        cursor.execute("ALTER TABLE graphic_departments ADD COLUMN stripe_customer_id TEXT")
    if "subscription_status" not in department_columns:
        cursor.execute("ALTER TABLE graphic_departments ADD COLUMN subscription_status TEXT NOT NULL DEFAULT 'none'")
    if "plan" not in department_columns:
        cursor.execute("ALTER TABLE graphic_departments ADD COLUMN plan TEXT")
    if "contact_email" not in department_columns:
        cursor.execute("ALTER TABLE graphic_departments ADD COLUMN contact_email TEXT")

    cursor.execute("PRAGMA table_info(graphic_assets)")
    asset_columns = {row[1] for row in cursor.fetchall()}
    if "cdn_key" not in asset_columns:
        cursor.execute("ALTER TABLE graphic_assets ADD COLUMN cdn_key TEXT")
    if "cdn_url" not in asset_columns:
        cursor.execute("ALTER TABLE graphic_assets ADD COLUMN cdn_url TEXT")

    cursor.execute("PRAGMA table_info(graphic_department_users)")
    user_columns = {row[1] for row in cursor.fetchall()}
    if "terms_version" not in user_columns:
        cursor.execute(
            "ALTER TABLE graphic_department_users ADD COLUMN terms_version INTEGER NOT NULL DEFAULT 0"
        )
    if "terms_accepted_at" not in user_columns:
        cursor.execute("ALTER TABLE graphic_department_users ADD COLUMN terms_accepted_at TIMESTAMP")


def record_fire_weather_alert_day(
    alert_date: str,
    county_fips: str,
    county_name: str,
    event: str,
    alert_id: str = "",
    onset: str = "",
    expires: str = "",
) -> bool:
    """Record that a county was under a fire-weather alert on a given date.

    Idempotent per (alert_date, county_fips, event) so a 5-minute alert poll
    doesn't create duplicate rows - only the first sighting of the day for
    that county/event pair is kept.
    """
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT OR IGNORE INTO fire_weather_alert_history
                (alert_date, county_fips, county_name, event, alert_id, onset, expires)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (alert_date, county_fips, county_name, event, alert_id, onset, expires))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def list_fire_weather_alert_history(
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    county_fips: Optional[str] = None,
) -> List[Dict]:
    """Query recorded fire-weather-alert days, optionally filtered by date range/county."""
    conditions = []
    params: List[str] = []
    if start_date:
        conditions.append("alert_date >= ?")
        params.append(start_date)
    if end_date:
        conditions.append("alert_date <= ?")
        params.append(end_date)
    if county_fips:
        conditions.append("county_fips = ?")
        params.append(county_fips)
    where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            f"SELECT * FROM fire_weather_alert_history {where_clause} "
            "ORDER BY alert_date ASC, county_fips ASC",
            params,
        ).fetchall()
        return [dict(row) for row in rows]
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


# --- Staff forecast discussion helpers ---

def _forecast_discussion_row(row: Optional[sqlite3.Row]) -> Optional[Dict]:
    return dict(row) if row else None


def list_forecast_discussions(status: Optional[str] = None, limit: int = 50, offset: int = 0, public_only: bool = False) -> List[Dict]:
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        if public_only:
            rows = conn.execute(
                "SELECT * FROM forecast_discussions WHERE status IN ('published', 'archived') "
                "ORDER BY COALESCE(issued_at, created_at) DESC, id DESC LIMIT ? OFFSET ?",
                (min(max(limit, 1), 100), max(offset, 0)),
            ).fetchall()
        elif status:
            rows = conn.execute(
                "SELECT * FROM forecast_discussions WHERE status = ? "
                "ORDER BY COALESCE(issued_at, created_at) DESC, id DESC LIMIT ? OFFSET ?",
                (status, min(max(limit, 1), 100), max(offset, 0)),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM forecast_discussions ORDER BY "
                "COALESCE(issued_at, created_at) DESC, id DESC LIMIT ? OFFSET ?",
                (min(max(limit, 1), 100), max(offset, 0)),
            ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def get_forecast_discussion(discussion_id: int) -> Optional[Dict]:
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        return _forecast_discussion_row(conn.execute(
            "SELECT * FROM forecast_discussions WHERE id = ?", (discussion_id,)
        ).fetchone())
    finally:
        conn.close()


def get_latest_forecast_discussion() -> Optional[Dict]:
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        return _forecast_discussion_row(conn.execute(
            "SELECT * FROM forecast_discussions WHERE status = 'published' "
            "ORDER BY issued_at DESC, id DESC LIMIT 1"
        ).fetchone())
    finally:
        conn.close()


def create_forecast_discussion(title: str, body: str, author_name: Optional[str] = None,
                               status: str = "draft") -> Dict:
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        issued_at = "CURRENT_TIMESTAMP" if status == "published" else None
        if issued_at:
            cursor = conn.execute(
                "INSERT INTO forecast_discussions "
                "(title, body, author_name, issued_at, status) VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?)",
                (title.strip(), body, (author_name or "").strip() or None, status),
            )
        else:
            cursor = conn.execute(
                "INSERT INTO forecast_discussions (title, body, author_name, status) VALUES (?, ?, ?, ?)",
                (title.strip(), body, (author_name or "").strip() or None, status),
            )
        conn.commit()
        return dict(conn.execute(
            "SELECT * FROM forecast_discussions WHERE id = ?", (cursor.lastrowid,)
        ).fetchone())
    finally:
        conn.close()


def update_forecast_discussion(discussion_id: int, title: Optional[str] = None, body: Optional[str] = None,
                               author_name: Optional[str] = None, status: Optional[str] = None) -> Optional[Dict]:
    current = get_forecast_discussion(discussion_id)
    if not current:
        return None
    values = {
        "title": current["title"] if title is None else title.strip(),
        "body": current["body"] if body is None else body,
        "author_name": current["author_name"] if author_name is None else (author_name.strip() or None),
        "status": current["status"] if status is None else status,
    }
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(
            "UPDATE forecast_discussions SET title = ?, body = ?, author_name = ?, status = ?, "
            "issued_at = CASE WHEN ? = 'published' AND issued_at IS NULL THEN CURRENT_TIMESTAMP ELSE issued_at END, "
            "updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (values["title"], values["body"], values["author_name"], values["status"],
             values["status"], discussion_id),
        )
        conn.commit()
        return dict(conn.execute(
            "SELECT * FROM forecast_discussions WHERE id = ?", (discussion_id,)
        ).fetchone())
    finally:
        conn.close()


def publish_forecast_discussion(discussion_id: int) -> Optional[Dict]:
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("UPDATE forecast_discussions SET status = 'archived', updated_at = CURRENT_TIMESTAMP "
                     "WHERE status = 'published' AND id != ?", (discussion_id,))
        conn.execute(
            "UPDATE forecast_discussions SET status = 'published', "
            "issued_at = COALESCE(issued_at, CURRENT_TIMESTAMP), updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (discussion_id,),
        )
        conn.commit()
        return dict(conn.execute(
            "SELECT * FROM forecast_discussions WHERE id = ?", (discussion_id,)
        ).fetchone())
    finally:
        conn.close()


def archive_forecast_discussion(discussion_id: int) -> Optional[Dict]:
    return update_forecast_discussion(discussion_id, status="archived")


def delete_forecast_discussion(discussion_id: int) -> bool:
    conn = sqlite3.connect(get_db_path())
    try:
        cursor = conn.execute("DELETE FROM forecast_discussions WHERE id = ?", (discussion_id,))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


# --- County burn-ban helpers ---

BURN_BAN_STATUSES = {"pending", "confirmed", "denied", "expired"}
BURN_BAN_REQUEST_TYPES = {"issue", "lift"}


def _burn_ban_row_to_dict(row: Optional[sqlite3.Row], *, admin: bool = False) -> Optional[Dict]:
    if not row:
        return None
    data = dict(row)
    if not admin:
        for key in (
            "submitter_name", "submitter_contact", "submitter_ip_hash",
            "upload_token_hash", "captcha_verdict", "consent_version",
            "proof_stored_filename", "proof_original_filename", "proof_content_type",
            "moderator_note", "moderated_by", "pii_purged_at",
        ):
            data.pop(key, None)
    return data


def _record_burn_ban_moderation(
    cursor: sqlite3.Cursor,
    submission_id: int,
    *,
    action: str,
    actor: str,
    from_status: str,
    to_status: str,
    reason: str = "",
    changed_fields: Optional[Dict] = None,
) -> None:
    import json as _json
    cursor.execute('''
        INSERT INTO burn_ban_moderation (
            submission_id, action, actor, from_status, to_status, reason, changed_fields_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        submission_id, action, actor, from_status, to_status, reason,
        _json.dumps(changed_fields or {}),
    ))


def create_burn_ban_submission(
    *,
    county_fips: str,
    county_name: str,
    submitter_name: str,
    submitter_contact: str,
    proof_url: str,
    effective_at: str,
    expires_at: str = "",
    submitter_ip_hash: str,
    upload_token_hash: str,
    captcha_verdict: str,
    consent_version: str,
    request_type: str = "issue",
) -> Dict:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        kind = request_type if request_type in BURN_BAN_REQUEST_TYPES else "issue"
        cursor.execute('''
            INSERT INTO burn_ban_submissions (
                status, county_fips, county_name, submitter_name, submitter_contact,
                proof_url, request_type, effective_at, expires_at, submitter_ip_hash,
                upload_token_hash, captcha_verdict, consent_version
            ) VALUES ('pending', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            county_fips, county_name, submitter_name, submitter_contact,
            proof_url, kind, effective_at, expires_at or "", submitter_ip_hash,
            upload_token_hash, captcha_verdict, consent_version,
        ))
        submission_id = cursor.lastrowid
        conn.commit()
        cursor.execute('SELECT * FROM burn_ban_submissions WHERE id = ?', (submission_id,))
        return dict(cursor.fetchone())
    finally:
        conn.close()


def get_burn_ban_submission(submission_id: int, *, admin: bool = False) -> Optional[Dict]:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT * FROM burn_ban_submissions WHERE id = ?', (submission_id,))
        row = cursor.fetchone()
        if not row:
            return None
        data = _burn_ban_row_to_dict(row, admin=admin)
        if admin:
            cursor.execute('''
                SELECT * FROM burn_ban_moderation WHERE submission_id = ?
                ORDER BY created_at ASC
            ''', (submission_id,))
            data["moderation_history"] = [dict(r) for r in cursor.fetchall()]
        return data
    finally:
        conn.close()


def get_burn_ban_upload_token_hash(submission_id: int) -> Optional[str]:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(
            'SELECT upload_token_hash FROM burn_ban_submissions WHERE id = ? AND status = ?',
            (submission_id, "pending"),
        )
        row = cursor.fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def list_burn_ban_submissions(
    *,
    status: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    admin: bool = False,
) -> List[Dict]:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        if status:
            cursor.execute('''
                SELECT * FROM burn_ban_submissions WHERE status = ?
                ORDER BY created_at DESC LIMIT ? OFFSET ?
            ''', (status, limit, offset))
        else:
            cursor.execute('''
                SELECT * FROM burn_ban_submissions
                ORDER BY created_at DESC LIMIT ? OFFSET ?
            ''', (limit, offset))
        return [
            _burn_ban_row_to_dict(row, admin=admin)
            for row in cursor.fetchall()
        ]
    finally:
        conn.close()


def _burn_ban_is_publicly_active(row: Dict, now_iso: str) -> bool:
    if row.get("status") != "confirmed":
        return False
    effective_at = str(row.get("effective_at") or "")
    expires_at = str(row.get("expires_at") or "")
    if effective_at and effective_at > now_iso:
        return False
    if expires_at and expires_at <= now_iso:
        return False
    return True


def list_active_burn_bans(*, now: Optional[datetime] = None) -> List[Dict]:
    now = now or datetime.utcnow()
    now_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('''
            SELECT * FROM burn_ban_submissions
            WHERE status = 'confirmed'
              AND COALESCE(request_type, 'issue') = 'issue'
              AND effective_at <= ?
              AND (expires_at IS NULL OR expires_at = '' OR expires_at > ?)
            ORDER BY county_name ASC
        ''', (now_iso, now_iso))
        return [
            _burn_ban_row_to_dict(row, admin=False)
            for row in cursor.fetchall()
        ]
    finally:
        conn.close()


def set_burn_ban_proof_file(
    submission_id: int,
    stored_filename: str,
    original_filename: str,
    content_type: str,
) -> Optional[Dict]:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('''
            UPDATE burn_ban_submissions
            SET proof_stored_filename = ?, proof_original_filename = ?,
                proof_content_type = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND status = 'pending'
        ''', (stored_filename, original_filename, content_type, submission_id))
        if cursor.rowcount == 0:
            conn.commit()
            return None
        conn.commit()
        cursor.execute('SELECT * FROM burn_ban_submissions WHERE id = ?', (submission_id,))
        return dict(cursor.fetchone())
    finally:
        conn.close()


def moderate_burn_ban_submission(
    submission_id: int,
    *,
    to_status: str,
    actor: str,
    reason: str = "",
    effective_at: Optional[str] = None,
    expires_at: Optional[str] = None,
) -> Optional[Dict]:
    if to_status not in BURN_BAN_STATUSES:
        raise ValueError(f"invalid status: {to_status}")
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT * FROM burn_ban_submissions WHERE id = ?', (submission_id,))
        row = cursor.fetchone()
        if not row:
            return None
        current = dict(row)
        from_status = current["status"]
        if from_status != "pending" and to_status in {"confirmed", "denied"}:
            return {"already_moderated": True, **current}

        changed: Dict = {}
        updates = ["status = ?", "moderated_by = ?", "moderator_note = ?",
                   "moderated_at = CURRENT_TIMESTAMP", "updated_at = CURRENT_TIMESTAMP"]
        params: List = [to_status, actor, reason]
        if effective_at is not None:
            updates.append("effective_at = ?")
            params.append(effective_at)
            changed["effective_at"] = effective_at
        if expires_at is not None:
            updates.append("expires_at = ?")
            params.append(expires_at)
            changed["expires_at"] = expires_at
        if to_status == "confirmed":
            updates.append("published_at = CURRENT_TIMESTAMP")
        params.append(submission_id)
        cursor.execute(
            f"UPDATE burn_ban_submissions SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        _record_burn_ban_moderation(
            cursor, submission_id, action=to_status, actor=actor,
            from_status=from_status, to_status=to_status, reason=reason,
            changed_fields=changed,
        )
        conn.commit()
        cursor.execute('SELECT * FROM burn_ban_submissions WHERE id = ?', (submission_id,))
        return dict(cursor.fetchone())
    finally:
        conn.close()


def update_burn_ban_submission(
    submission_id: int,
    *,
    actor: str,
    edit_reason: str,
    effective_at: Optional[str] = None,
    expires_at: Optional[str] = None,
    proof_url: Optional[str] = None,
    county_fips: Optional[str] = None,
    county_name: Optional[str] = None,
) -> Optional[Dict]:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT * FROM burn_ban_submissions WHERE id = ?', (submission_id,))
        row = cursor.fetchone()
        if not row:
            return None
        current = dict(row)
        if current["status"] not in {"confirmed", "pending"}:
            return None

        changed: Dict = {}
        clauses = ["updated_at = CURRENT_TIMESTAMP"]
        params: List = []
        for field, value in (
            ("effective_at", effective_at),
            ("expires_at", expires_at),
            ("proof_url", proof_url),
            ("county_fips", county_fips),
            ("county_name", county_name),
        ):
            if value is not None:
                clauses.append(f"{field} = ?")
                params.append(value)
                changed[field] = value
        if not changed:
            return current
        params.append(submission_id)
        cursor.execute(
            f"UPDATE burn_ban_submissions SET {', '.join(clauses)} WHERE id = ?",
            params,
        )
        _record_burn_ban_moderation(
            cursor, submission_id, action="edit", actor=actor,
            from_status=current["status"], to_status=current["status"],
            reason=edit_reason, changed_fields=changed,
        )
        conn.commit()
        cursor.execute('SELECT * FROM burn_ban_submissions WHERE id = ?', (submission_id,))
        return dict(cursor.fetchone())
    finally:
        conn.close()


def delete_burn_ban_submission(submission_id: int, *, actor: str, reason: str = "") -> bool:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    try:
        cursor.execute('SELECT status FROM burn_ban_submissions WHERE id = ?', (submission_id,))
        row = cursor.fetchone()
        if not row:
            return False
        from_status = row["status"]
        _record_burn_ban_moderation(
            cursor, submission_id, action="delete", actor=actor,
            from_status=from_status, to_status="deleted", reason=reason,
        )
        cursor.execute('DELETE FROM burn_ban_submissions WHERE id = ?', (submission_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        return deleted
    finally:
        conn.close()


def expire_stale_burn_bans(*, now: Optional[datetime] = None) -> int:
    now = now or datetime.utcnow()
    now_iso = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            SELECT id FROM burn_ban_submissions
            WHERE status = 'confirmed'
              AND expires_at IS NOT NULL
              AND expires_at != ''
              AND expires_at <= ?
        ''', (now_iso,))
        ids = [row[0] for row in cursor.fetchall()]
        for submission_id in ids:
            cursor.execute('''
                UPDATE burn_ban_submissions
                SET status = 'expired', updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (submission_id,))
            _record_burn_ban_moderation(
                cursor, submission_id, action="expired", actor="system:auto-expire",
                from_status="confirmed", to_status="expired",
                reason="expiration date reached",
            )
        conn.commit()
        return len(ids)
    finally:
        conn.close()


def expire_confirmed_burn_bans_for_county(
    county_fips: str,
    *,
    actor: str,
    reason: str = "",
    exclude_id: Optional[int] = None,
) -> int:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            SELECT id FROM burn_ban_submissions
            WHERE status = 'confirmed'
              AND county_fips = ?
              AND COALESCE(request_type, 'issue') = 'issue'
              AND (? IS NULL OR id != ?)
        ''', (county_fips, exclude_id, exclude_id))
        ids = [row[0] for row in cursor.fetchall()]
        for submission_id in ids:
            cursor.execute('''
                UPDATE burn_ban_submissions
                SET status = 'expired', updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            ''', (submission_id,))
            _record_burn_ban_moderation(
                cursor, submission_id, action="expired", actor=actor,
                from_status="confirmed", to_status="expired",
                reason=reason or "county burn ban lifted",
            )
        conn.commit()
        return len(ids)
    finally:
        conn.close()


def purge_burn_ban_submission_pii(older_than_days: int = 90) -> int:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            UPDATE burn_ban_submissions
            SET submitter_name = '', submitter_contact = '', submitter_ip_hash = '',
                upload_token_hash = '', pii_purged_at = CURRENT_TIMESTAMP
            WHERE moderated_at IS NOT NULL
              AND moderated_at <= datetime('now', ? || ' days')
              AND pii_purged_at IS NULL
              AND (submitter_name != '' OR submitter_contact != '' OR submitter_ip_hash != '')
        ''', (f"-{max(0, older_than_days)}",))
        purged = cursor.rowcount
        conn.commit()
        return purged
    finally:
        conn.close()


def consume_burn_ban_submission_quota(
    bucket_key: str, now: datetime, per_hour_limit: int, per_day_limit: int,
) -> Dict:
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
                SELECT hits FROM burn_ban_submission_throttle
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
                INSERT INTO burn_ban_submission_throttle (bucket_key, window_kind, window_start, hits, updated_at)
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


def purge_burn_ban_throttle_rows(older_than_hours: int = 48) -> int:
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('''
            DELETE FROM burn_ban_submission_throttle
            WHERE updated_at <= datetime('now', ? || ' hours')
        ''', (f"-{max(0, older_than_hours)}",))
        purged = cursor.rowcount
        conn.commit()
        return purged
    finally:
        conn.close()


def upsert_newsletter_subscriber(
    email: str,
    resend_contact_id: Optional[str] = None,
    subscription_types: Optional[List[str]] = None,
    name: str = "",
    affiliation: str = "",
) -> Dict:
    normalized = email.strip().lower()
    subscription_json = json.dumps(subscription_types or ["fire-weather-forecasts"])
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        existing = conn.execute(
            "SELECT manage_token FROM newsletter_subscribers WHERE email = ?",
            (normalized,),
        ).fetchone()
        manage_token = existing["manage_token"] if existing and existing["manage_token"] else secrets.token_urlsafe(32)
        conn.execute(
            '''INSERT INTO newsletter_subscribers
               (email, resend_contact_id, name, affiliation, manage_token, subscription_types_json)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(email) DO UPDATE SET
                 resend_contact_id = COALESCE(excluded.resend_contact_id, newsletter_subscribers.resend_contact_id),
                 name = excluded.name,
                 affiliation = excluded.affiliation,
                 subscription_types_json = excluded.subscription_types_json,
                 unsubscribed_at = NULL,
                 updated_at = CURRENT_TIMESTAMP''',
            (normalized, resend_contact_id, name.strip(), affiliation.strip(), manage_token, subscription_json),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM newsletter_subscribers WHERE email = ?", (normalized,)).fetchone()
        result = dict(row)
        try:
            result["subscription_types"] = json.loads(result.pop("subscription_types_json") or "[]")
        except json.JSONDecodeError:
            result["subscription_types"] = []
        return result
    finally:
        conn.close()


def list_newsletter_subscribers() -> List[Dict]:
    """Return subscriber identity fields needed for provider synchronization."""
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        return [
            dict(row) for row in conn.execute(
                """SELECT email, resend_contact_id, manage_token
                   FROM newsletter_subscribers
                   WHERE manage_token IS NOT NULL AND manage_token != ''
                   ORDER BY email"""
            ).fetchall()
        ]
    finally:
        conn.close()


def replace_newsletter_preferences(email: str, preferences: List[Dict]) -> List[Dict]:
    normalized = email.strip().lower()
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("DELETE FROM newsletter_preferences WHERE email = ?", (normalized,))
        for preference in preferences:
            conn.execute(
                '''INSERT INTO newsletter_preferences (email, county_fips, min_danger_level)
                   VALUES (?, ?, ?)''',
                (normalized, preference["county_fips"], preference["min_danger_level"]),
            )
        conn.commit()
        return [
            dict(row) for row in conn.execute(
                "SELECT county_fips, min_danger_level FROM newsletter_preferences WHERE email = ? ORDER BY county_fips",
                (normalized,),
            ).fetchall()
        ]
    finally:
        conn.close()


def get_newsletter_preferences(email: Optional[str] = None) -> List[Dict]:
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        query = """
            SELECT p.email, p.county_fips, p.min_danger_level
            FROM newsletter_preferences p
            JOIN newsletter_subscribers s ON s.email = p.email
        """
        params: tuple = ()
        if email:
            query += " WHERE p.email = ?"
            params = (email.strip().lower(),)
        query += " ORDER BY p.email, p.county_fips"
        return [dict(row) for row in conn.execute(query, params).fetchall()]
    finally:
        conn.close()


def get_newsletter_account_by_token(token: str) -> Optional[Dict]:
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        subscriber = conn.execute(
            "SELECT * FROM newsletter_subscribers WHERE manage_token = ?",
            (token.strip(),),
        ).fetchone()
        if not subscriber:
            return None
        result = dict(subscriber)
        result["subscription_types"] = json.loads(result.pop("subscription_types_json") or "[]")
        result["counties"] = [
            dict(row) for row in conn.execute(
                "SELECT county_fips, min_danger_level FROM newsletter_preferences WHERE email = ? ORDER BY county_fips",
                (result["email"],),
            ).fetchall()
        ]
        return result
    finally:
        conn.close()


def update_newsletter_account(
    token: str,
    name: str,
    affiliation: str,
    subscription_types: List[str],
    preferences: List[Dict],
) -> Optional[Dict]:
    account = get_newsletter_account_by_token(token)
    if not account:
        return None
    conn = sqlite3.connect(get_db_path())
    try:
        conn.execute(
            '''UPDATE newsletter_subscribers
               SET name = ?, affiliation = ?, subscription_types_json = ?,
                   unsubscribed_at = NULL, updated_at = CURRENT_TIMESTAMP
               WHERE manage_token = ?''',
            (name.strip(), affiliation.strip(), json.dumps(subscription_types), token.strip()),
        )
        conn.execute("DELETE FROM newsletter_preferences WHERE email = ?", (account["email"],))
        for preference in preferences:
            conn.execute(
                '''INSERT INTO newsletter_preferences (email, county_fips, min_danger_level)
                   VALUES (?, ?, ?)''',
                (account["email"], preference["county_fips"], preference["min_danger_level"]),
            )
        conn.commit()
        return get_newsletter_account_by_token(token)
    finally:
        conn.close()


def unsubscribe_newsletter(token: str) -> bool:
    conn = sqlite3.connect(get_db_path())
    try:
        cursor = conn.execute(
            '''UPDATE newsletter_subscribers
               SET unsubscribed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
               WHERE manage_token = ?''',
            (token.strip(),),
        )
        conn.commit()
        return cursor.rowcount == 1
    finally:
        conn.close()


def create_bulletin(subject: str, html_body: str, text_body: str) -> Dict:
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(
            "INSERT INTO bulletins (subject, html_body, text_body) VALUES (?, ?, ?)",
            (subject.strip(), html_body, text_body),
        )
        conn.commit()
        return dict(conn.execute("SELECT * FROM bulletins WHERE id = ?", (cursor.lastrowid,)).fetchone())
    finally:
        conn.close()


def get_bulletin(bulletin_id: int) -> Optional[Dict]:
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT * FROM bulletins WHERE id = ?", (bulletin_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_bulletins() -> List[Dict]:
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute("SELECT * FROM bulletins ORDER BY created_at DESC, id DESC").fetchall()]
    finally:
        conn.close()


def update_bulletin(bulletin_id: int, subject: str, html_body: str, text_body: str) -> Optional[Dict]:
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        cursor = conn.execute(
            '''UPDATE bulletins
               SET subject = ?, html_body = ?, text_body = ?, updated_at = CURRENT_TIMESTAMP
               WHERE id = ? AND status = 'draft' ''',
            (subject.strip(), html_body, text_body, bulletin_id),
        )
        conn.commit()
        if cursor.rowcount != 1:
            return None
        row = conn.execute("SELECT * FROM bulletins WHERE id = ?", (bulletin_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def mark_bulletin_sent(bulletin_id: int, resend_broadcast_id: str) -> Optional[Dict]:
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(
            '''UPDATE bulletins
               SET status = 'sent', resend_broadcast_id = ?, sent_at = CURRENT_TIMESTAMP,
                   last_error = NULL, updated_at = CURRENT_TIMESTAMP
               WHERE id = ? AND status = 'sending' ''',
            (resend_broadcast_id, bulletin_id),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM bulletins WHERE id = ?", (bulletin_id,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def set_bulletin_error(bulletin_id: int, error: str) -> None:
    conn = sqlite3.connect(get_db_path())
    try:
        conn.execute(
            "UPDATE bulletins SET status = 'draft', last_error = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
            (error[:2000], bulletin_id),
        )
        conn.commit()
    finally:
        conn.close()


def claim_bulletin_send(bulletin_id: int) -> bool:
    conn = sqlite3.connect(get_db_path())
    try:
        cursor = conn.execute(
            "UPDATE bulletins SET status = 'sending', last_error = NULL, updated_at = CURRENT_TIMESTAMP "
            "WHERE id = ? AND status = 'draft'",
            (bulletin_id,),
        )
        conn.commit()
        return cursor.rowcount == 1
    finally:
        conn.close()


def upsert_county_forecast_day(
    forecast_date: str, county_fips: str, danger_level: int,
    summary: str = "", forecast_run_id: str = "",
) -> Dict:
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(
            '''INSERT INTO county_forecast_days
               (forecast_date, county_fips, danger_level, summary, forecast_run_id)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(forecast_date, county_fips) DO UPDATE SET
                 danger_level = excluded.danger_level,
                 summary = excluded.summary,
                 forecast_run_id = excluded.forecast_run_id,
                 published_at = CURRENT_TIMESTAMP''',
            (forecast_date, county_fips, danger_level, summary, forecast_run_id),
        )
        conn.commit()
        return dict(conn.execute(
            "SELECT * FROM county_forecast_days WHERE forecast_date = ? AND county_fips = ?",
            (forecast_date, county_fips),
        ).fetchone())
    finally:
        conn.close()


def list_matching_newsletter_forecasts(forecast_date: str) -> List[Dict]:
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute(
            '''SELECT p.email, s.manage_token, p.county_fips, p.min_danger_level,
                      f.danger_level, f.summary, f.forecast_run_id
               FROM newsletter_preferences p
               JOIN newsletter_subscribers s ON s.email = p.email
               JOIN county_forecast_days f ON f.county_fips = p.county_fips
                                           AND f.forecast_date = ?
                                           AND f.danger_level >= p.min_danger_level
               WHERE EXISTS (
                   SELECT 1
                   FROM json_each(
                       CASE WHEN json_valid(s.subscription_types_json)
                            THEN s.subscription_types_json ELSE '[]' END
                   )
                   WHERE value = 'fire-weather-forecasts'
               )
                 AND s.unsubscribed_at IS NULL
               ORDER BY p.email, p.county_fips''',
            (forecast_date,),
        ).fetchall()]
    finally:
        conn.close()


def claim_newsletter_delivery(email: str, county_fips: str, forecast_date: str) -> bool:
    conn = sqlite3.connect(get_db_path(), isolation_level=None)
    try:
        cursor = conn.execute(
            '''INSERT OR IGNORE INTO newsletter_deliveries
               (email, county_fips, forecast_date, status, claimed_at)
               VALUES (?, ?, ?, 'pending', CURRENT_TIMESTAMP)''',
            (email, county_fips, forecast_date),
        )
        if cursor.rowcount == 1:
            return True
        stale = conn.execute(
            '''UPDATE newsletter_deliveries
               SET status = 'pending', error = NULL, claimed_at = CURRENT_TIMESTAMP
               WHERE email = ? AND county_fips = ? AND forecast_date = ?
                 AND status = 'pending'
                 AND claimed_at <= datetime('now', '-30 minutes')''',
            (email, county_fips, forecast_date),
        )
        if stale.rowcount == 1:
            return True
        retry = conn.execute(
            '''UPDATE newsletter_deliveries
               SET status = 'pending', error = NULL, claimed_at = CURRENT_TIMESTAMP
               WHERE email = ? AND county_fips = ? AND forecast_date = ? AND status = 'failed' ''',
            (email, county_fips, forecast_date),
        )
        return retry.rowcount == 1
    finally:
        conn.close()


def complete_newsletter_delivery(
    email: str, county_fips: str, forecast_date: str,
    status: str, provider_message_id: str = "", error: str = "",
) -> None:
    conn = sqlite3.connect(get_db_path())
    try:
        conn.execute(
            '''UPDATE newsletter_deliveries
               SET status = ?, provider_message_id = ?, error = ?, claimed_at = NULL, sent_at =
                   CASE WHEN ? = 'sent' THEN CURRENT_TIMESTAMP ELSE sent_at END
               WHERE email = ? AND county_fips = ? AND forecast_date = ?''',
            (status, provider_message_id, error[:2000], status, email, county_fips, forecast_date),
        )
        conn.commit()
    finally:
        conn.close()
