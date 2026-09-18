"""Finds candidate recurring non-fire satellite detection sources - fixed
industrial heat sources (mills, gas/landfill flares, kilns, plants) that
trigger a "fire" detection over and over at the same spot, wasting incident
clustering / ML scoring / PNG regeneration compute on something that was
never a wildfire.

This module only ever creates or updates 'candidate' rows (or refreshes
stats on already-reviewed rows) - it never sets a row to 'confirmed' or
changes ingest behavior. A human must promote a candidate via
core.database.set_recurring_source_status before core.database.
upsert_detection_event will start suppressing detections at that location.
See the "Detect and suppress recurring non-fire detection sources" plan.
"""
import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone

from core.database import (
    create_recurring_source_candidate,
    find_recurring_source_near,
    get_db_path,
    update_recurring_source_stats,
)

logger = logging.getLogger(__name__)

LOOKBACK_DAYS = int(os.getenv("FIRE_RECURRING_SOURCE_LOOKBACK_DAYS", "60"))
MIN_DISTINCT_DAYS = int(os.getenv("FIRE_RECURRING_SOURCE_MIN_DAYS", "8"))
GRID_DEGREES = 0.01  # ~1km - coarse enough to merge nearby pixels of the same source
DEDUPE_RADIUS_KM = float(os.getenv("FIRE_RECURRING_SOURCE_RADIUS_KM", "1.5"))


def _grid_cells(cursor: sqlite3.Cursor, cutoff_iso: str) -> list:
    """Group not-yet-attributed satellite detections into coarse lat/lon
    grid cells, counting total detections and distinct calendar days per
    cell. SQLite has no ROUND-to-grid built-in shortcut we'd trust across
    versions, so the rounding happens in SQL via arithmetic instead."""
    cursor.execute(
        f'''
        SELECT
            ROUND(latitude / {GRID_DEGREES}) * {GRID_DEGREES} AS cell_lat,
            ROUND(longitude / {GRID_DEGREES}) * {GRID_DEGREES} AS cell_lon,
            COUNT(*) AS detection_count,
            COUNT(DISTINCT substr(occurred_at, 1, 10)) AS distinct_day_count,
            MIN(occurred_at) AS first_detected_at,
            MAX(occurred_at) AS last_detected_at,
            AVG(latitude) AS avg_lat,
            AVG(longitude) AS avg_lon
        FROM fire_events
        WHERE source IN ('modis', 'viirs', 'ngfs')
          AND recurring_source_id IS NULL
          AND occurred_at >= ?
        GROUP BY cell_lat, cell_lon
        HAVING distinct_day_count >= ?
        ''',
        (cutoff_iso, MIN_DISTINCT_DAYS),
    )
    return [dict(row) for row in cursor.fetchall()]


def run_recurring_source_scan() -> dict:
    """Scan the lookback window for grid cells that fired on enough
    distinct days to look like a fixed non-fire source, and upsert them as
    'candidate' rows (or refresh stats on an existing nearby row of any
    status) for admin review. Returns a small job summary."""
    cutoff_iso = (datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)).isoformat()

    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        cells = _grid_cells(conn.cursor(), cutoff_iso)
    finally:
        conn.close()

    created, updated = 0, 0
    for cell in cells:
        latitude, longitude = cell["avg_lat"], cell["avg_lon"]
        existing = find_recurring_source_near(latitude, longitude, DEDUPE_RADIUS_KM)
        if existing is not None:
            update_recurring_source_stats(
                existing["id"], cell["detection_count"], cell["distinct_day_count"],
                cell["first_detected_at"], cell["last_detected_at"],
            )
            updated += 1
        else:
            create_recurring_source_candidate(
                latitude, longitude, cell["detection_count"], cell["distinct_day_count"],
                cell["first_detected_at"], cell["last_detected_at"], radius_km=DEDUPE_RADIUS_KM,
            )
            created += 1

    summary = {"cells_scanned": len(cells), "created": created, "updated": updated}
    logger.info("recurring_source_detector: %s", summary)
    return summary
