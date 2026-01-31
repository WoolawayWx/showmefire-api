"""DB-backed list of station IDs to ignore across services.

Provides a cached getter to avoid repeated DB hits. Falls back to a
small in-code set if the DB/table isn't available yet.
"""
from typing import Set
import sqlite3
import logging
from core import database

logger = logging.getLogger(__name__)

# Fallback default while DB isn't initialized
_FALLBACK = {"MBGM7"}

# Cached value and a simple flag
_CACHE: Set[str] = set()
_CACHED = False

def _load_from_db() -> Set[str]:
	db_path = database.get_db_path()
	try:
		conn = sqlite3.connect(db_path)
		cursor = conn.cursor()
		cursor.execute('SELECT stid FROM ignored_stations')
		rows = cursor.fetchall()
		conn.close()
		return {r[0] for r in rows if r and r[0]}
	except Exception as e:
		logger.debug(f"Could not load ignored_stations from DB: {e}")
		return set()

def get_ignored_stations(force_reload: bool = False) -> Set[str]:
	"""Return a set of ignored station STIDs.

	If the DB table is not present or fails, returns a fallback set.
	Set `force_reload=True` to refresh the cache from disk.
	"""
	global _CACHE, _CACHED
	if not _CACHED or force_reload:
		loaded = _load_from_db()
		if loaded:
			_CACHE = loaded
			_CACHED = True
		else:
			# Keep fallback to avoid changing behavior if DB absent
			_CACHE = _FALLBACK.copy()
			_CACHED = True
	return _CACHE

def refresh_ignored_stations() -> Set[str]:
	"""Force reload the ignored stations from the DB and return the set."""
	return get_ignored_stations(force_reload=True)
