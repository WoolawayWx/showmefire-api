import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any, Iterable
from urllib.parse import quote

import httpx

from core.database import get_db_path

logger = logging.getLogger(__name__)

EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"
EXPO_RECEIPTS_URL = "https://exp.host/--/api/v2/push/getReceipts"
EXPO_ACCESS_TOKEN = os.getenv("EXPO_ACCESS_TOKEN", "").strip()


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(get_db_path())
    connection.row_factory = sqlite3.Row
    return connection


def _delete_subscription_rows(connection: sqlite3.Connection, installation_id: str) -> None:
    """Remove a subscription and all delivery data linked to its installation."""
    connection.execute(
        "DELETE FROM mobile_push_receipts WHERE ticket_id IN "
        "(SELECT ticket_id FROM mobile_push_tickets WHERE installation_id = ?)",
        (installation_id,),
    )
    connection.execute(
        "DELETE FROM mobile_push_tickets WHERE installation_id = ?",
        (installation_id,),
    )
    connection.execute(
        "DELETE FROM mobile_push_subscriptions WHERE installation_id = ?",
        (installation_id,),
    )


def upsert_subscription(
    *,
    installation_id: str,
    expo_push_token: str,
    platform: str,
    app_version: str,
    forecast: bool,
    sitrep: bool,
    fire_weather: bool,
    county_fips: list[str],
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as connection:
        if not (forecast or sitrep or fire_weather):
            _delete_subscription_rows(connection, installation_id)
            return {"registered": False, "updatedAt": now}

        duplicate_installations = connection.execute(
            "SELECT installation_id FROM mobile_push_subscriptions "
            "WHERE expo_push_token = ? AND installation_id != ?",
            (expo_push_token, installation_id),
        ).fetchall()
        for duplicate in duplicate_installations:
            _delete_subscription_rows(connection, duplicate["installation_id"])
        connection.execute(
            '''
            INSERT INTO mobile_push_subscriptions (
                installation_id, expo_push_token, platform, app_version,
                forecast_enabled, sitrep_enabled, fire_weather_enabled,
                county_fips_json, enabled, created_at, updated_at, last_seen_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
            ON CONFLICT(installation_id) DO UPDATE SET
                expo_push_token = excluded.expo_push_token,
                platform = excluded.platform,
                app_version = excluded.app_version,
                forecast_enabled = excluded.forecast_enabled,
                sitrep_enabled = excluded.sitrep_enabled,
                fire_weather_enabled = excluded.fire_weather_enabled,
                county_fips_json = excluded.county_fips_json,
                enabled = 1,
                updated_at = excluded.updated_at,
                last_seen_at = excluded.last_seen_at
            ''',
            (
                installation_id,
                expo_push_token,
                platform,
                app_version,
                int(forecast),
                int(sitrep),
                int(fire_weather),
                json.dumps(sorted(set(county_fips))),
                now,
                now,
                now,
            ),
        )
    return {"registered": True, "updatedAt": now}


def delete_subscription(installation_id: str) -> None:
    with _connect() as connection:
        _delete_subscription_rows(connection, installation_id)


def purge_delivery_records(retention_days: int = 7) -> dict[str, int]:
    """Delete push-delivery bookkeeping older than the retention window."""
    if retention_days < 1:
        raise ValueError("retention_days must be at least 1")

    modifier = f"-{retention_days} days"
    with _connect() as connection:
        receipt_cursor = connection.execute(
            "DELETE FROM mobile_push_receipts WHERE ticket_id IN "
            "(SELECT ticket_id FROM mobile_push_tickets WHERE created_at < datetime('now', ?))",
            (modifier,),
        )
        ticket_cursor = connection.execute(
            "DELETE FROM mobile_push_tickets WHERE created_at < datetime('now', ?)",
            (modifier,),
        )
    return {"receipts": receipt_cursor.rowcount, "tickets": ticket_cursor.rowcount}


def record_event(event_key: str, event_type: str, payload: dict[str, Any]) -> bool:
    with _connect() as connection:
        cursor = connection.execute(
            "INSERT OR IGNORE INTO mobile_push_events (event_key, event_type, payload_json) VALUES (?, ?, ?)",
            (event_key, event_type, json.dumps(payload, separators=(",", ":"))),
        )
        return cursor.rowcount == 1


def _eligible_subscriptions(event_type: str, county_fips: Iterable[str] | None = None) -> list[dict[str, str]]:
    county_set = set(county_fips or [])
    column = {
        "forecast": "forecast_enabled",
        "sitrep": "sitrep_enabled",
        "fire_weather": "fire_weather_enabled",
    }[event_type]
    with _connect() as connection:
        rows = connection.execute(
            f"SELECT installation_id, expo_push_token, county_fips_json FROM mobile_push_subscriptions WHERE enabled = 1 AND {column} = 1"
        ).fetchall()
    result: list[dict[str, str]] = []
    for row in rows:
        if event_type == "fire_weather":
            try:
                selected = set(json.loads(row["county_fips_json"] or "[]"))
            except json.JSONDecodeError:
                selected = set()
            if not county_set.intersection(selected):
                continue
        result.append({"installation_id": row["installation_id"], "expo_push_token": row["expo_push_token"]})
    return result


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if EXPO_ACCESS_TOKEN:
        headers["Authorization"] = f"Bearer {EXPO_ACCESS_TOKEN}"
    return headers


def _send_batch(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for attempt in range(3):
        try:
            with httpx.Client(timeout=10.0) as client:
                response = client.post(EXPO_PUSH_URL, headers=_headers(), json=messages)
            if response.status_code == 429 or response.status_code >= 500:
                raise httpx.HTTPStatusError("Transient Expo push error", request=response.request, response=response)
            response.raise_for_status()
            data = response.json().get("data", [])
            return data if isinstance(data, list) else [data]
        except (httpx.HTTPError, ValueError) as exc:
            if attempt == 2:
                logger.warning("Expo push batch failed after retries: %s", exc)
                return []
            time.sleep(2 ** attempt)
    return []


def send_mobile_event(
    *,
    event_type: str,
    event_key: str,
    title: str,
    body: str,
    url: str,
    county_fips: Iterable[str] | None = None,
    extra_data: dict[str, Any] | None = None,
    image_url: str | None = None,
) -> int:
    payload = {"title": title, "body": body, "url": url, **(extra_data or {})}
    if image_url:
        payload["imageUrl"] = image_url
    if not record_event(event_key, event_type, payload):
        return 0

    subscriptions = _eligible_subscriptions(event_type, county_fips)
    channel = {"forecast": "forecast", "sitrep": "sitrep", "fire_weather": "fire-weather"}[event_type]
    sent = 0
    for start in range(0, len(subscriptions), 100):
        batch_subscriptions = subscriptions[start:start + 100]
        messages = []
        for item in batch_subscriptions:
            message = {
                "to": item["expo_push_token"],
                "title": title,
                "body": body[:500],
                "sound": "default",
                "priority": "high" if event_type == "fire_weather" else "default",
                "channelId": channel,
                "data": {"url": url, "eventType": event_type, **(extra_data or {})},
            }
            if image_url:
                message["richContent"] = {"image": image_url}
                message["mutableContent"] = True
            messages.append(message)
        tickets = _send_batch(messages)
        with _connect() as connection:
            for subscription, ticket in zip(batch_subscriptions, tickets):
                if ticket.get("status") == "ok" and ticket.get("id"):
                    connection.execute(
                        "INSERT OR IGNORE INTO mobile_push_tickets (ticket_id, installation_id, event_key) VALUES (?, ?, ?)",
                        (ticket["id"], subscription["installation_id"], event_key),
                    )
                    sent += 1
                elif (ticket.get("details") or {}).get("error") == "DeviceNotRegistered":
                    _delete_subscription_rows(connection, subscription["installation_id"])
    return sent


def notify_forecast(forecast: dict[str, Any], image_url: str | None = None) -> int:
    identity = forecast.get("updated_at") or forecast.get("id") or forecast.get("valid_time")
    return send_mobile_event(
        event_type="forecast",
        event_key=f"forecast:{identity}",
        title=str(forecast.get("title") or "Show Me Fire Forecast"),
        body="Missouri's latest daily fire weather forecast is ready.",
        url="/forecasts",
        extra_data={"forecastRevision": str(identity)},
        image_url=image_url,
    )


def notify_sitrep(*, title: str, identity: str) -> int:
    return send_mobile_event(
        event_type="sitrep",
        event_key=f"sitrep:{identity}",
        title="New Show Me Fire SitRep",
        body=title or "A new Missouri situation report is available.",
        url="/sitrep",
    )


def process_fire_weather_alerts(alerts: list[dict[str, Any]]) -> int:
    baseline_key = "fire_weather:baseline:v1"
    with _connect() as connection:
        baseline_exists = connection.execute(
            "SELECT 1 FROM mobile_push_events WHERE event_key = ?", (baseline_key,)
        ).fetchone() is not None
    if not baseline_exists:
        for alert in alerts:
            record_event(f"fire_weather:{alert['id']}", "fire_weather", alert)
        record_event(baseline_key, "fire_weather", {"seeded": len(alerts)})
        return 0

    sent = 0
    for alert in alerts:
        alert_id = str(alert["id"])
        sent += send_mobile_event(
            event_type="fire_weather",
            event_key=f"fire_weather:{alert_id}",
            title=str(alert.get("event") or "Fire Weather Alert"),
            body=str(alert.get("headline") or alert.get("areaDescription") or "A new Missouri fire weather alert is active."),
            url=f"/alert/{quote(alert_id, safe='')}",
            county_fips=alert.get("countyFips") or [],
            extra_data={"alertId": alert_id},
        )
    return sent


def check_push_receipts() -> int:
    with _connect() as connection:
        rows = connection.execute(
            '''
            SELECT t.ticket_id, s.expo_push_token
            FROM mobile_push_tickets t
            JOIN mobile_push_subscriptions s ON s.installation_id = t.installation_id
            LEFT JOIN mobile_push_receipts r ON r.ticket_id = t.ticket_id
            WHERE r.ticket_id IS NULL AND t.created_at <= datetime('now', '-15 minutes')
            ORDER BY t.created_at
            LIMIT 1000
            '''
        ).fetchall()
    if not rows:
        return 0

    ticket_to_token = {row["ticket_id"]: row["expo_push_token"] for row in rows}
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(EXPO_RECEIPTS_URL, headers=_headers(), json={"ids": list(ticket_to_token)})
        response.raise_for_status()
        receipts = response.json().get("data", {})
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Expo receipt check failed: %s", exc)
        return 0

    checked = 0
    with _connect() as connection:
        for ticket_id, receipt in receipts.items():
            status = str(receipt.get("status") or "error")
            error = str((receipt.get("details") or {}).get("error") or receipt.get("message") or "") or None
            if error == "DeviceNotRegistered" and ticket_id in ticket_to_token:
                installation = connection.execute(
                    "SELECT installation_id FROM mobile_push_subscriptions WHERE expo_push_token = ?",
                    (ticket_to_token[ticket_id],),
                ).fetchone()
                if installation:
                    _delete_subscription_rows(connection, installation["installation_id"])
            else:
                ticket_exists = connection.execute(
                    "SELECT 1 FROM mobile_push_tickets WHERE ticket_id = ?",
                    (ticket_id,),
                ).fetchone()
                if ticket_exists:
                    connection.execute(
                        "INSERT OR REPLACE INTO mobile_push_receipts (ticket_id, status, error, checked_at) VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                        (ticket_id, status, error),
                    )
            checked += 1
    return checked
