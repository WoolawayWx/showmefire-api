from __future__ import annotations

import bisect
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.database import get_db_path
from .repository import ensure_schema


OBSERVATION_FIELDS = {
    "air_temp": ("temperature_c", lambda value: (value - 32.0) * 5.0 / 9.0),
    "dew_point_temperature": ("dewpoint_c", lambda value: (value - 32.0) * 5.0 / 9.0),
    "relative_humidity": ("relative_humidity", float),
    "wind_speed": ("wind_speed_ms", lambda value: value * 0.44704),
    "wind_gust": ("wind_gust_ms", lambda value: value * 0.44704),
    "precip_accum": ("precipitation_mm", lambda value: value * 25.4),
    "solar_radiation": ("solar_wm2", float),
    "fuel_moisture": ("fuel_moisture", float),
}

VERIFICATION_FIELDS = {
    "temperature_2m": ("temperature_c", "temperature_c"),
    "dewpoint_2m": ("dewpoint_c", "dewpoint_c"),
    "relative_humidity_2m": ("relative_humidity", "relative_humidity"),
    "wind_speed_10m": ("wind_speed_ms", "wind_speed_ms"),
    "wind_gust_10m": ("wind_gust_ms", "wind_gust_ms"),
    "shortwave_down": ("solar_wm2", "solar_wm2"),
}


def _parse_time(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def persist_station_observations(
    stations: list[dict], db_path: str | Path | None = None,
) -> int:
    """Normalize the English-unit Synoptic snapshot without losing sensor timestamps."""
    database = Path(db_path or get_db_path())
    ensure_schema(database)
    ingested = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    rows: dict[tuple[str, str], dict] = {}
    for station in stations:
        station_id = station.get("stid") or station.get("id")
        if not station_id:
            continue
        qc_state = "fail" if station.get("qc_flagged") else "pass"
        qc_reasons = ["synoptic_station_qc_flag"] if qc_state == "fail" else []
        for source_name, (column, converter) in OBSERVATION_FIELDS.items():
            observation = (station.get("observations") or {}).get(source_name) or {}
            if observation.get("value") is None or not observation.get("time"):
                continue
            try:
                value = converter(float(observation["value"]))
                # Validate parseability, but retain the provider's original timestamp text.
                _parse_time(observation["time"])
            except (TypeError, ValueError, OverflowError):
                continue
            key = (str(station_id), str(observation["time"]))
            row = rows.setdefault(key, {
                "station_id": key[0], "observed_at_utc": key[1],
                "network": str(station.get("network") or "unknown"), "source": "synoptic",
                "qc_state": qc_state, "qc_reasons_json": json.dumps(qc_reasons),
                "ingested_at_utc": ingested,
            })
            row[column] = value
    columns = (
        "station_id", "observed_at_utc", "temperature_c", "dewpoint_c", "relative_humidity",
        "wind_u_ms", "wind_v_ms", "wind_speed_ms", "wind_gust_ms", "precipitation_mm",
        "solar_wm2", "fuel_moisture", "network", "source", "qc_state", "qc_reasons_json",
        "ingested_at_utc",
    )
    if not rows:
        return 0
    assignments = ",".join(
        f"{column}=COALESCE(excluded.{column},station_observations_v2.{column})"
        for column in columns[2:12]
    )
    assignments += ",network=excluded.network,source=excluded.source,qc_state=excluded.qc_state,"
    assignments += "qc_reasons_json=excluded.qc_reasons_json,ingested_at_utc=excluded.ingested_at_utc"
    with sqlite3.connect(database) as connection:
        connection.executemany(
            f"INSERT INTO station_observations_v2 ({','.join(columns)}) "
            f"VALUES ({','.join('?' for _ in columns)}) "
            f"ON CONFLICT(station_id,observed_at_utc) DO UPDATE SET {assignments}",
            ([row.get(column) for column in columns] for row in rows.values()),
        )
    return len(rows)


def update_forecast_verification(
    db_path: str | Path | None = None, *, now: datetime | None = None,
) -> int:
    """Match recent completed hours to the nearest observation within 30 minutes.

    The Synoptic latest feed covers the preceding 70 minutes. A two-hour
    forecast window catches delayed refreshes without rescanning 30 days of
    station forecasts after every refresh; the matched error rows themselves
    are retained for the rolling 30-day confidence component.
    """
    database = Path(db_path or get_db_path())
    ensure_schema(database)
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    cutoff = current - timedelta(hours=2)
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        forecasts = connection.execute(
            """SELECT h.*,r.model_version,r.config_version FROM station_forecast_hours h
               JOIN forecast_runs r ON r.run_id=h.run_id
               WHERE r.status IN ('complete','superseded') AND h.valid_time_utc BETWEEN ? AND ?""",
            (cutoff.isoformat().replace("+00:00", "Z"), current.isoformat().replace("+00:00", "Z")),
        ).fetchall()
        observations = connection.execute(
            "SELECT * FROM station_observations_v2 WHERE observed_at_utc BETWEEN ? AND ?",
            ((cutoff - timedelta(minutes=30)).isoformat().replace("+00:00", "Z"),
             (current + timedelta(minutes=30)).isoformat().replace("+00:00", "Z")),
        ).fetchall()
        by_station_field: dict[tuple[str, str], list[tuple[datetime, sqlite3.Row]]] = {}
        for observation in observations:
            try:
                stamp = _parse_time(observation["observed_at_utc"])
            except ValueError:
                continue
            for _, observation_column in VERIFICATION_FIELDS.values():
                if observation[observation_column] is not None:
                    by_station_field.setdefault((observation["station_id"], observation_column), []).append(
                        (stamp, observation)
                    )
        for values in by_station_field.values():
            values.sort(key=lambda item: item[0])
        station_field_times = {
            key: [item[0] for item in values] for key, values in by_station_field.items()
        }
        output = []
        for forecast in forecasts:
            valid_time = _parse_time(forecast["valid_time_utc"])
            bucket = "0-24" if forecast["lead_hour"] <= 24 else "25-48" if forecast["lead_hour"] <= 48 else "49-72"
            for variable, (forecast_column, observation_column) in VERIFICATION_FIELDS.items():
                candidates = by_station_field.get((forecast["station_id"], observation_column), [])
                if not candidates:
                    continue
                index = bisect.bisect_left(
                    station_field_times[(forecast["station_id"], observation_column)], valid_time
                )
                nearby = candidates[max(0, index - 1):min(len(candidates), index + 1)]
                if not nearby:
                    continue
                observation_time, observation = min(
                    nearby, key=lambda item: abs((item[0] - valid_time).total_seconds())
                )
                if abs((observation_time - valid_time).total_seconds()) > 1800:
                    continue
                forecast_value = forecast[forecast_column]
                observation_value = observation[observation_column]
                if forecast_value is None or observation_value is None:
                    continue
                output.append((
                    forecast["run_id"], forecast["station_id"], forecast["valid_time_utc"], variable,
                    forecast_value, observation_value, float(forecast_value) - float(observation_value),
                    bucket, forecast["model_version"], forecast["config_version"],
                    int(observation["qc_state"] == "pass"),
                ))
        connection.executemany(
            """INSERT OR REPLACE INTO forecast_verification
               (run_id,station_id,valid_time_utc,variable,forecast_value,observation_value,error,
                lead_bucket,model_version,config_version,qc_eligible)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""", output,
        )
    return len(output)
