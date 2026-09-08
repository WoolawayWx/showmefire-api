"""Authenticated health and recent-reading views for SMF-FMS devices."""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, HTTPException

from core.database import get_db_path
from core.security import verify_token


router = APIRouter(prefix="/api/admin/fuel-sensors", tags=["fuel-sensor-admin"])


def _require_admin(token: Optional[str] = None) -> str:
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return email


def _as_utc(value: object) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _health_for(last_upload_at: object, *, stale_after_minutes: int) -> tuple[str, int | None]:
    uploaded = _as_utc(last_upload_at)
    if uploaded is None:
        return "unknown", None
    age_seconds = max(0, int((datetime.now(timezone.utc) - uploaded).total_seconds()))
    if age_seconds <= stale_after_minutes * 60:
        return "online", age_seconds
    if age_seconds <= stale_after_minutes * 3 * 60:
        return "delayed", age_seconds
    return "offline", age_seconds


@router.get("/status")
def fuel_sensor_status(token: Optional[str] = None, recent_limit: int = 100):
    """Return SMF-FMS device health and newest uploads for the admin console.

    Health is based on ``received_at`` (the server receipt time), rather than
    ``recorded_at``. That matters while bench firmware reports its uptime as
    the device timestamp and remains the safer upload-health signal in field
    deployments too.
    """
    _require_admin(token)
    recent_limit = max(1, min(recent_limit, 500))
    stale_after_minutes = max(5, int(os.getenv("SMF_FUEL_SENSOR_STALE_MINUTES", "20")))
    try:
        connection = sqlite3.connect(get_db_path())
        connection.row_factory = sqlite3.Row
        latest_rows = connection.execute(
            """
            SELECT latest.*, summary.reading_count, summary.first_upload_at,
                   summary.last_upload_at, summary.readings_last_24h
            FROM fuel_moisture_sensor_readings AS latest
            INNER JOIN (
                SELECT device_id, MAX(id) AS latest_id, COUNT(*) AS reading_count,
                       MIN(received_at) AS first_upload_at, MAX(received_at) AS last_upload_at,
                       SUM(CASE WHEN received_at >= datetime('now', '-24 hours') THEN 1 ELSE 0 END) AS readings_last_24h
                FROM fuel_moisture_sensor_readings
                GROUP BY device_id
            ) AS summary ON latest.id = summary.latest_id
            ORDER BY latest.received_at DESC, latest.id DESC
            """
        ).fetchall()
        recent_rows = connection.execute(
            """
            SELECT * FROM fuel_moisture_sensor_readings
            ORDER BY received_at DESC, id DESC
            LIMIT ?
            """,
            (recent_limit,),
        ).fetchall()
        connection.close()
    except sqlite3.Error as exc:
        raise HTTPException(status_code=500, detail=f"Unable to read fuel-sensor data: {exc}") from exc

    devices = []
    for row in latest_rows:
        item = dict(row)
        health, age_seconds = _health_for(item.get("last_upload_at"), stale_after_minutes=stale_after_minutes)
        devices.append({**item, "health": health, "age_seconds": age_seconds})

    health_counts = {state: sum(device["health"] == state for device in devices) for state in ("online", "delayed", "offline", "unknown")}
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "stale_after_minutes": stale_after_minutes,
        "summary": {"sensor_count": len(devices), **health_counts},
        "devices": devices,
        "recent_readings": [dict(row) for row in recent_rows],
    }
