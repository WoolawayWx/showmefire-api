import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterable
from urllib.parse import quote

import httpx

from core.database import get_db_path

logger = logging.getLogger(__name__)

EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"
EXPO_RECEIPTS_URL = "https://exp.host/--/api/v2/push/getReceipts"
EXPO_ACCESS_TOKEN = os.getenv("EXPO_ACCESS_TOKEN", "").strip()


@contextmanager
def _connect():
    connection = sqlite3.connect(get_db_path())
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


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
    fire_detection: bool,
    county_fips: list[str],
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as connection:
        if not (forecast or sitrep or fire_weather or fire_detection):
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
                forecast_enabled, sitrep_enabled, fire_weather_enabled, fire_detection_enabled,
                county_fips_json, enabled, created_at, updated_at, last_seen_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
            ON CONFLICT(installation_id) DO UPDATE SET
                expo_push_token = excluded.expo_push_token,
                platform = excluded.platform,
                app_version = excluded.app_version,
                forecast_enabled = excluded.forecast_enabled,
                sitrep_enabled = excluded.sitrep_enabled,
                fire_weather_enabled = excluded.fire_weather_enabled,
                fire_detection_enabled = excluded.fire_detection_enabled,
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
                int(fire_detection),
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


def _eligible_subscriptions(
    event_type: str, county_fips: Iterable[str] | None = None, *, all_counties: bool = False,
) -> list[dict[str, str]]:
    """Subscribers opted into `event_type`. County-scoped channels also need a
    county match unless `all_counties` (manual statewide broadcast) is set."""
    county_set = set(county_fips or [])
    column = {
        "forecast": "forecast_enabled",
        "sitrep": "sitrep_enabled",
        "fire_weather": "fire_weather_enabled",
        "fire_detection": "fire_detection_enabled",
    }[event_type]
    with _connect() as connection:
        rows = connection.execute(
            f"SELECT installation_id, expo_push_token, county_fips_json FROM mobile_push_subscriptions WHERE enabled = 1 AND {column} = 1"
        ).fetchall()
    result: list[dict[str, str]] = []
    for row in rows:
        if event_type in {"fire_weather", "fire_detection"} and not (all_counties and not county_set):
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
    all_counties: bool = False,
) -> int:
    payload = {"title": title, "body": body, "url": url, **(extra_data or {})}
    if image_url:
        payload["imageUrl"] = image_url
    if not record_event(event_key, event_type, payload):
        return 0

    subscriptions = _eligible_subscriptions(event_type, county_fips, all_counties=all_counties)
    unregistered = 0
    channel = {
        "forecast": "forecast",
        "sitrep": "sitrep",
        "fire_weather": "fire-weather",
        "fire_detection": "fire-detection",
    }[event_type]
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
                "priority": "high" if event_type in {"fire_weather", "fire_detection"} else "default",
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
                    unregistered += 1
    _record_send_result(event_key, recipients=len(subscriptions), sent=sent, unregistered=unregistered)
    return sent


def _record_send_result(event_key: str, *, recipients: int, sent: int, unregistered: int) -> None:
    """Keep the outcome of one alert on its event row. A tracking failure must never fail the alert itself."""
    try:
        with _connect() as connection:
            connection.execute(
                """UPDATE mobile_push_events SET recipients = ?, sent = ?, unregistered = ?, failed = ?,
                       delivered = 0, receipt_errors = 0 WHERE event_key = ?""",
                (recipients, sent, unregistered, max(0, recipients - sent - unregistered), event_key),
            )
    except sqlite3.Error:
        logger.warning("Could not record push result for %s", event_key, exc_info=True)


def _tally_receipts(connection: sqlite3.Connection, tallies: dict[str, dict[str, Any]]) -> None:
    """Add Expo's delivery receipts to their alerts: `tallies` is {event_key: {"delivered": n, "errors": {kind: n}}}."""
    for event_key, tally in tallies.items():
        row = connection.execute("SELECT error_counts_json FROM mobile_push_events WHERE event_key = ?", (event_key,)).fetchone()
        if row is None:  # the ticket outlived its event row (or never had one)
            continue
        try:
            counts = json.loads(row["error_counts_json"] or "{}")
        except json.JSONDecodeError:
            counts = {}
        for kind, number in tally["errors"].items():
            counts[kind] = counts.get(kind, 0) + number
        connection.execute(
            """UPDATE mobile_push_events SET delivered = COALESCE(delivered, 0) + ?,
                   receipt_errors = COALESCE(receipt_errors, 0) + ?, error_counts_json = ? WHERE event_key = ?""",
            (tally["delivered"], sum(tally["errors"].values()), json.dumps(counts, sort_keys=True), event_key),
        )


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


def notify_new_high_confidence_incidents(incidents: list[dict[str, Any]]) -> int:
    """Push one alert per newly high-confidence fire incident.

    Idempotent per incident_id via mobile_push_events (event_key), so this can be
    called every ingest cycle without re-notifying the same incident. Seeds a
    baseline on first run so a fresh deploy doesn't blast every currently-active
    high-confidence incident, mirroring process_fire_weather_alerts.
    """
    baseline_key = "fire_detection:baseline:v1"
    with _connect() as connection:
        baseline_exists = connection.execute(
            "SELECT 1 FROM mobile_push_events WHERE event_key = ?", (baseline_key,)
        ).fetchone() is not None
    if not baseline_exists:
        for incident in incidents:
            record_event(f"fire_detection:{incident['incident_id']}", "fire_detection", incident)
        record_event(baseline_key, "fire_detection", {"seeded": len(incidents)})
        return 0

    sent = 0
    for incident in incidents:
        incident_id = incident["incident_id"]
        slug = incident.get("incident_slug")
        county_fips = incident.get("county_fips")
        sent += send_mobile_event(
            event_type="fire_detection",
            event_key=f"fire_detection:{incident_id}",
            title="New high-confidence fire detected",
            body=f"A likely wildfire was detected near {incident.get('county_name') or 'your area'}.",
            url=f"/detections/{quote(slug, safe='')}/feedback" if slug else "/fires",
            county_fips=[county_fips] if county_fips else [],
            extra_data={"incidentSlug": slug} if slug else {},
        )
    return sent


def check_push_receipts() -> int:
    with _connect() as connection:
        rows = connection.execute(
            '''
            SELECT t.ticket_id, t.event_key, s.expo_push_token
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
    ticket_to_event = {row["ticket_id"]: row["event_key"] for row in rows}
    tallies: dict[str, dict[str, Any]] = {}
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
            event_key = ticket_to_event.get(ticket_id)
            if event_key:
                tally = tallies.setdefault(event_key, {"delivered": 0, "errors": {}})
                if status == "ok":
                    tally["delivered"] += 1
                else:
                    kind = (error or "unknown")[:80]
                    tally["errors"][kind] = tally["errors"].get(kind, 0) + 1
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
        _tally_receipts(connection, tallies)
    return checked


def push_stats(days: int = 30, recent: int = 25) -> dict[str, Any]:
    """How the app's alerts are doing: who can receive them, and what became of each one that was sent.

    `sent` is what Expo accepted; `delivered` is Expo's receipt that it reached Apple/Google (not that it was seen or
    opened, which the app does not report). Alerts with no counts were never sent through this pipeline (the baseline
    rows seeded on first run, or alerts from before this was tracked) and are left out."""
    days = max(1, min(int(days), 365))
    window = f"-{days} days"

    def pct(part: int, whole: int) -> float | None:
        return round(100 * part / whole, 1) if whole else None

    with _connect() as connection:
        audience_row = connection.execute(
            """SELECT COUNT(*) AS devices,
                      SUM(forecast_enabled) AS forecast, SUM(sitrep_enabled) AS sitrep,
                      SUM(fire_weather_enabled) AS fire_weather, SUM(fire_detection_enabled) AS fire_detection,
                      SUM(CASE WHEN created_at >= datetime('now', '-7 days') THEN 1 ELSE 0 END) AS new_7d,
                      SUM(CASE WHEN created_at >= datetime('now', ?) THEN 1 ELSE 0 END) AS new_in_period,
                      SUM(CASE WHEN last_seen_at >= datetime('now', '-30 days') THEN 1 ELSE 0 END) AS active_30d
               FROM mobile_push_subscriptions WHERE enabled = 1""", (window,)).fetchone()
        platforms = {row["platform"]: row["devices"] for row in connection.execute(
            "SELECT platform, COUNT(*) AS devices FROM mobile_push_subscriptions WHERE enabled = 1 GROUP BY platform")}
        awaiting = connection.execute(
            """SELECT COUNT(*) FROM mobile_push_tickets t LEFT JOIN mobile_push_receipts r ON r.ticket_id = t.ticket_id
               WHERE r.ticket_id IS NULL""").fetchone()[0]
        by_type = []
        for row in connection.execute(
            """SELECT event_type, CASE WHEN event_key LIKE 'manual:%' THEN 1 ELSE 0 END AS manual, COUNT(*) AS alerts,
                      SUM(recipients) AS recipients, SUM(sent) AS sent, SUM(failed) AS failed,
                      SUM(unregistered) AS unregistered, SUM(delivered) AS delivered, SUM(receipt_errors) AS receipt_errors,
                      SUM(CASE WHEN recipients = 0 THEN 1 ELSE 0 END) AS reached_nobody
               FROM mobile_push_events
               WHERE recipients IS NOT NULL AND created_at >= datetime('now', ?)
               GROUP BY event_type, manual ORDER BY event_type, manual""", (window,)):
            item = dict(row)
            item["manual"] = bool(item["manual"])
            item["accepted_pct"] = pct(item["sent"], item["recipients"])
            item["delivered_pct"] = pct(item["delivered"], item["sent"])
            item["awaiting_receipt"] = max(0, item["sent"] - item["delivered"] - item["receipt_errors"])
            by_type.append(item)
        daily = [dict(row) for row in connection.execute(
            """SELECT date(created_at) AS day, event_type, COUNT(*) AS alerts, SUM(recipients) AS recipients,
                      SUM(sent) AS sent, SUM(delivered) AS delivered
               FROM mobile_push_events WHERE recipients IS NOT NULL AND created_at >= datetime('now', ?)
               GROUP BY day, event_type ORDER BY day DESC, event_type""", (window,))]
        recent_alerts = []
        for row in connection.execute(
            """SELECT event_key, event_type, payload_json, created_at, recipients, sent, failed, unregistered, delivered,
                      receipt_errors, error_counts_json
               FROM mobile_push_events WHERE recipients IS NOT NULL
               ORDER BY created_at DESC, rowid DESC LIMIT ?""", (max(1, min(recent, 200)),)):
            item = dict(row)
            try:
                payload = json.loads(item.pop("payload_json") or "{}")
            except json.JSONDecodeError:
                payload = {}
            try:
                item["error_counts"] = json.loads(item.pop("error_counts_json") or "{}")
            except json.JSONDecodeError:
                item["error_counts"] = {}
            item["title"] = str(payload.get("title") or "")
            item["manual"] = item["event_key"].startswith("manual:")
            item["delivered_pct"] = pct(item["delivered"], item["sent"])
            item["awaiting_receipt"] = max(0, item["sent"] - item["delivered"] - item["receipt_errors"])
            recent_alerts.append(item)

    audience = {key: int(audience_row[key] or 0) for key in audience_row.keys()}
    totals = {key: sum(item[key] or 0 for item in by_type) for key in ("alerts", "recipients", "sent", "failed", "unregistered", "delivered", "receipt_errors", "reached_nobody")}
    totals["accepted_pct"] = pct(totals["sent"], totals["recipients"])
    totals["delivered_pct"] = pct(totals["delivered"], totals["sent"])
    return {"period_days": days, "audience": {**audience, "platforms": platforms, "awaiting_receipt": awaiting},
            "totals": totals, "by_type": by_type, "daily": daily, "recent": recent_alerts}
