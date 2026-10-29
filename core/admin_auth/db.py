"""Connection helper and tables for admin auth. Tables are created on first use of each database file, so tests and the
break-glass CLI work without going through init_database()."""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from typing import Iterator

SCHEMA = [
    """CREATE TABLE IF NOT EXISTS admin_users (
           id INTEGER PRIMARY KEY AUTOINCREMENT,
           email TEXT NOT NULL UNIQUE,
           display_name TEXT NOT NULL DEFAULT '',
           password_hash TEXT NOT NULL,
           role TEXT NOT NULL DEFAULT 'admin',
           disabled_at REAL,
           must_change_password INTEGER NOT NULL DEFAULT 0,
           created_at REAL NOT NULL,
           last_login_at REAL
       )""",
    """CREATE TABLE IF NOT EXISTS admin_sessions (
           id TEXT PRIMARY KEY,
           admin_id INTEGER NOT NULL,
           refresh_hash TEXT NOT NULL UNIQUE,
           created_at REAL NOT NULL,
           last_seen_at REAL NOT NULL,
           expires_at REAL NOT NULL,
           revoked_at REAL,
           ip TEXT,
           user_agent TEXT,
           method TEXT
       )""",
    "CREATE INDEX IF NOT EXISTS idx_admin_sessions_admin ON admin_sessions(admin_id)",
    """CREATE TABLE IF NOT EXISTS admin_login_failures (
           scope TEXT NOT NULL,
           key TEXT NOT NULL,
           ts REAL NOT NULL
       )""",
    "CREATE INDEX IF NOT EXISTS idx_admin_login_failures ON admin_login_failures(scope, key, ts)",
    """CREATE TABLE IF NOT EXISTS admin_auth_events (
           id INTEGER PRIMARY KEY AUTOINCREMENT,
           admin_id INTEGER,
           kind TEXT NOT NULL,
           ip TEXT,
           detail TEXT,
           ts REAL NOT NULL
       )""",
    "CREATE INDEX IF NOT EXISTS idx_admin_auth_events_admin ON admin_auth_events(admin_id, id)",
    """CREATE TABLE IF NOT EXISTS admin_passkeys (
           id INTEGER PRIMARY KEY AUTOINCREMENT,
           admin_id INTEGER NOT NULL,
           credential_id BLOB NOT NULL UNIQUE,
           public_key BLOB NOT NULL,
           sign_count INTEGER NOT NULL DEFAULT 0,
           name TEXT NOT NULL DEFAULT '',
           transports TEXT NOT NULL DEFAULT '[]',
           created_at REAL NOT NULL,
           last_used_at REAL
       )""",
    """CREATE TABLE IF NOT EXISTS admin_totp (
           admin_id INTEGER PRIMARY KEY,
           secret_enc TEXT NOT NULL,
           confirmed_at REAL,
           last_step INTEGER NOT NULL DEFAULT 0
       )""",
    """CREATE TABLE IF NOT EXISTS admin_recovery_codes (
           id INTEGER PRIMARY KEY AUTOINCREMENT,
           admin_id INTEGER NOT NULL,
           code_hash TEXT NOT NULL,
           used_at REAL
       )""",
    "CREATE INDEX IF NOT EXISTS idx_admin_recovery_codes ON admin_recovery_codes(admin_id, code_hash)",
    """CREATE TABLE IF NOT EXISTS admin_challenges (
           id TEXT PRIMARY KEY,
           kind TEXT NOT NULL,
           admin_id INTEGER,
           challenge BLOB NOT NULL,
           expires_at REAL NOT NULL
       )""",
]

# Columns added after the first release. The CREATE TABLE statements above are IF NOT EXISTS, so an existing database
# gets these through ALTER TABLE the first time it is opened.
MIGRATIONS = (
    ("admin_users", "expires_at", "REAL"),                  # reviewer accounts stop working at this time
    ("admin_users", "note", "TEXT NOT NULL DEFAULT ''"),    # who a reviewer account was made for
    ("admin_users", "created_by", "TEXT"),
    ("admin_sessions", "prev_refresh_hash", "TEXT"),        # mobile refresh rotation: the token just replaced
    ("admin_sessions", "rotated_at", "REAL"),
)


def _migrate(conn: sqlite3.Connection) -> None:
    for table, column, declaration in MIGRATIONS:
        if column not in {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")


_ready: set[str] = set()
_ready_lock = threading.Lock()


def _path() -> str:
    # Accounts and sessions describe real people: always the real database, even inside a sandbox scope.
    from core.database import real_db_path

    return str(real_db_path())


@contextmanager
def tx() -> Iterator[sqlite3.Connection]:
    """A connection that commits on success, rolls back on error and always closes."""
    path = _path()
    conn = sqlite3.connect(path, timeout=15)
    conn.row_factory = sqlite3.Row
    try:
        if path not in _ready:
            with _ready_lock:
                if path not in _ready:
                    for statement in SCHEMA:
                        conn.execute(statement)
                    _migrate(conn)
                    conn.commit()
                    _ready.add(path)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
