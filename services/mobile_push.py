import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterable
from urllib.parse import quote
from zoneinfo import ZoneInfo

import httpx

from ai.briefing import DANGER_LABELS
from core.database import get_db_path
from core.sandbox import is_sandbox

logger = logging.getLogger(__name__)

EXPO_PUSH_URL = "https://exp.host/--/api/v2/push/send"
EXPO_RECEIPTS_URL = "https://exp.host/--/api/v2/push/getReceipts"
EXPO_ACCESS_TOKEN = os.getenv("EXPO_ACCESS_TOKEN", "").strip()

CHICAGO_TZ = ZoneInfo("America/Chicago")
# Fire-detection pushes are feedback requests, not emergency alerts: held
# overnight (Central) and capped per installation per rolling day.
FEEDBACK_PROMPT_QUIET_START_HOUR = 21
FEEDBACK_PROMPT_QUIET_END_HOUR = 7
FEEDBACK_PROMPT_DAILY_CAP = 3
# App versions from this one on create the quieter 'fire-detection-feedback'
# Android channel (mobile/src/notifications/push.ts); older installs only have
# the high-importance 'fire-detection' one, and a push to a channel the device
# doesn't have can be dropped or misrouted.
FEEDBACK_CHANNEL_MIN_APP_VERSION = (1, 2, 0)


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
    connection.execute("DELETE FROM mobile_forecast_queue WHERE installation_id = ?", (installation_id,))
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
    general: bool = False,
    local_forecast_titles: bool = False,
    forecast_not_before: str | None = None,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc).isoformat()
    with _connect() as connection:
        if not (forecast or sitrep or fire_weather or fire_detection or general):
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
                forecast_enabled, local_forecast_titles, forecast_not_before, sitrep_enabled, fire_weather_enabled, fire_detection_enabled, general_enabled,
                county_fips_json, enabled, created_at, updated_at, last_seen_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
            ON CONFLICT(installation_id) DO UPDATE SET
                expo_push_token = excluded.expo_push_token,
                platform = excluded.platform,
                app_version = excluded.app_version,
                forecast_enabled = excluded.forecast_enabled,
                local_forecast_titles = excluded.local_forecast_titles,
                forecast_not_before = excluded.forecast_not_before,
                sitrep_enabled = excluded.sitrep_enabled,
                fire_weather_enabled = excluded.fire_weather_enabled,
                fire_detection_enabled = excluded.fire_detection_enabled,
                general_enabled = excluded.general_enabled,
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
                int(local_forecast_titles),
                forecast_not_before,
                int(sitrep),
                int(fire_weather),
                int(fire_detection),
                int(general),
                json.dumps(sorted(set(county_fips))),
                now,
                now,
                now,
            ),
        )
        if not forecast:
            connection.execute("DELETE FROM mobile_forecast_queue WHERE installation_id = ?", (installation_id,))
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


HISTORY_DAYS = 7  # matches purge_delivery_records: tickets older than this are deleted anyway
HISTORY_MAX_ITEMS = 100


def notification_history(installation_id: str, days: int = HISTORY_DAYS) -> list[dict[str, Any]]:
    """What was sent to one installation in the last `days` days, newest first.

    Built from the delivery tickets (so only pushes Expo accepted for this
    device) joined to the event payload. Forecast copy personalised per device
    is not stored, so a forecast shows the statewide title/body. Rows vanish
    when the subscription is deleted, because its tickets are deleted with it.
    """
    days = max(1, min(days, HISTORY_DAYS))
    with _connect() as connection:
        rows = connection.execute(
            """SELECT e.event_key, e.event_type, e.payload_json, MIN(t.created_at) AS sent_at
               FROM mobile_push_tickets t JOIN mobile_push_events e ON e.event_key = t.event_key
               WHERE t.installation_id = ? AND t.created_at >= datetime('now', ?)
               GROUP BY e.event_key
               ORDER BY sent_at DESC, e.event_key DESC
               LIMIT ?""",
            (installation_id, f"-{days} days", HISTORY_MAX_ITEMS),
        ).fetchall()
    items: list[dict[str, Any]] = []
    for row in rows:
        try:
            payload = json.loads(row["payload_json"] or "{}")
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or not payload.get("title"):
            continue
        item = {
            "id": row["event_key"],
            "eventType": row["event_type"],
            "title": str(payload["title"]),
            "body": str(payload.get("body") or ""),
            "url": str(payload.get("url") or ""),
            "sentAt": row["sent_at"].replace(" ", "T") + "Z",
        }
        if payload.get("imageUrl"):
            item["imageUrl"] = str(payload["imageUrl"])
        items.append(item)
    return items


def record_event(event_key: str, event_type: str, payload: dict[str, Any]) -> bool:
    with _connect() as connection:
        cursor = connection.execute(
            "INSERT OR IGNORE INTO mobile_push_events (event_key, event_type, payload_json) VALUES (?, ?, ?)",
            (event_key, event_type, json.dumps(payload, separators=(",", ":"))),
        )
        return cursor.rowcount == 1


def _eligible_subscriptions(
    event_type: str, county_fips: Iterable[str] | None = None, *, all_counties: bool = False,
) -> list[dict[str, Any]]:
    """Subscribers opted into `event_type`. County-scoped channels also need a
    county match unless `all_counties` (manual statewide broadcast) is set.
    fire_detection also skips installations that already got
    FEEDBACK_PROMPT_DAILY_CAP of them in the last 24 h."""
    county_set = set(county_fips or [])
    column = {
        "forecast": "forecast_enabled",
        "sitrep": "sitrep_enabled",
        "fire_weather": "fire_weather_enabled",
        "fire_detection": "fire_detection_enabled",
        "general": "general_enabled",
    }[event_type]
    capped: set[str] = set()
    with _connect() as connection:
        rows = connection.execute(
            f"SELECT installation_id, expo_push_token, county_fips_json, app_version, local_forecast_titles, forecast_not_before FROM mobile_push_subscriptions WHERE enabled = 1 AND {column} = 1"
        ).fetchall()
        if event_type == "fire_detection":
            capped = {
                row["installation_id"]
                for row in connection.execute(
                    '''
                    SELECT installation_id FROM mobile_push_tickets
                    WHERE event_key LIKE 'fire_detection:%' AND created_at >= datetime('now', '-1 day')
                    GROUP BY installation_id HAVING COUNT(*) >= ?
                    ''',
                    (FEEDBACK_PROMPT_DAILY_CAP,),
                ).fetchall()
            }
    result: list[dict[str, Any]] = []
    for row in rows:
        if row["installation_id"] in capped:
            continue
        scoped = event_type in {"fire_weather", "fire_detection"} and not (all_counties and not county_set)
        personalized = event_type == "forecast" and row["local_forecast_titles"]
        selected = set()
        if scoped or personalized:
            try:
                selected = set(json.loads(row["county_fips_json"] or "[]"))
            except (TypeError, json.JSONDecodeError):
                pass
        if scoped and not county_set.intersection(selected):
            continue
        result.append({
            "installation_id": row["installation_id"],
            "expo_push_token": row["expo_push_token"],
            "app_version": row["app_version"] or "",
            "forecast_title_counties": selected if personalized else set(),
            "forecast_not_before": row["forecast_not_before"],
        })
    return result


def _parse_app_version(value: str) -> tuple[int, ...] | None:
    """'1.10.0' -> (1, 10, 0). None for empty/unparseable values, which callers
    treat as old. Compared as numbers: as strings '1.10.0' < '1.9.0'."""
    try:
        parts = tuple(int(part) for part in value.strip().split("."))
    except ValueError:
        return None
    return parts or None


def _channel_for(event_type: str, app_version: str) -> str:
    if event_type == "fire_detection":
        parsed = _parse_app_version(app_version)
        if parsed is not None and parsed >= FEEDBACK_CHANNEL_MIN_APP_VERSION:
            return "fire-detection-feedback"
    return {
        "forecast": "forecast",
        "sitrep": "sitrep",
        "fire_weather": "fire-weather",
        "fire_detection": "fire-detection",
        "general": "general",
    }[event_type]


# Notification action category the app registers at runtime (src/notifications/push.ts); it has one
# "Open in Maps" button. Apps that don't know the id simply show the notification without buttons.
FIRE_DETECTION_MAP_CATEGORY = "fire-detection-map"


def _valid_coordinate(latitude: Any, longitude: Any) -> tuple[float | None, float | None]:
    try:
        lat, lon = round(float(latitude), 5), round(float(longitude), 5)
    except (TypeError, ValueError):
        return None, None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None, None
    return lat, lon


def _headers() -> dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if EXPO_ACCESS_TOKEN:
        headers["Authorization"] = f"Bearer {EXPO_ACCESS_TOKEN}"
    return headers


def _send_batch(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if is_sandbox():  # reviewer sandbox: no push leaves the server
        return []
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


def _forecast_county_copy(
    counties: set[str], danger: dict[str, tuple[int, str]],
) -> tuple[str, str] | None:
    groups: dict[int, list[str]] = {}
    for fips in counties:
        if fips in danger:
            level, name = danger[fips]
            groups.setdefault(level, []).append(name)
    if not groups:
        return None

    low, high = DANGER_LABELS[min(groups)], DANGER_LABELS[max(groups)]
    category = high if low == high else f"{low} to {high}"
    missing = len(counties) - sum(len(names) for names in groups.values())
    if len(counties) == 1:
        return f"{category} fire danger • {next(iter(groups.values()))[0]}", "Your daily fire weather forecast is ready."

    title = f"{category} fire danger • Your counties"
    levels = sorted(groups, reverse=True)
    unavailable = f" • No forecast for {missing} selected" if missing else ""
    details = " • ".join(f"{DANGER_LABELS[level]}: {', '.join(sorted(groups[level]))}" for level in levels)
    if len(details) + len(unavailable) > 500:
        details = " • ".join(f"{DANGER_LABELS[level]}: {len(groups[level])}" for level in levels)
        details = f"Selected counties — {details}"
    return title, details + unavailable


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
    forecast_county_danger: dict[str, tuple[int, str]] | None = None,
) -> int:
    if is_sandbox():  # reviewer sandbox: nothing is recorded or sent
        return 0
    payload = {"title": title, "body": body, "url": url, **(extra_data or {})}
    if image_url:
        payload["imageUrl"] = image_url
    subscriptions = _eligible_subscriptions(event_type, county_fips, all_counties=all_counties)
    if event_type == "forecast":
        now = datetime.now(CHICAGO_TZ)
        clock = now.strftime("%H:%M")
        pending = [item for item in subscriptions if (item["forecast_not_before"] or "") > clock]
        subscriptions = [item for item in subscriptions if (item["forecast_not_before"] or "") <= clock]
        with _connect() as connection:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO mobile_push_events (event_key, event_type, payload_json) VALUES (?, ?, ?)",
                (event_key, event_type, json.dumps({
                    **payload, "_countyDanger": forecast_county_danger or {},
                } if pending else payload)),
            )
            if not cursor.rowcount:
                return 0
            connection.executemany(
                "INSERT INTO mobile_forecast_queue (event_key, installation_id, forecast_day) VALUES (?, ?, ?)",
                [(event_key, item["installation_id"], now.date().isoformat()) for item in pending],
            )
    elif not record_event(event_key, event_type, payload):
        return 0

    return _deliver_event(
        event_type, event_key, subscriptions, title, body, url,
        extra_data=extra_data, image_url=image_url, forecast_county_danger=forecast_county_danger,
    )


def _deliver_event(
    event_type: str, event_key: str, subscriptions: list[dict[str, Any]], title: str, body: str, url: str,
    *, extra_data: dict[str, Any] | None = None, image_url: str | None = None,
    forecast_county_danger: dict[str, tuple[int, str]] | None = None,
) -> int:
    unregistered = 0
    sent = 0
    for start in range(0, len(subscriptions), 100):
        batch_subscriptions = subscriptions[start:start + 100]
        messages = []
        for item in batch_subscriptions:
            local_copy = _forecast_county_copy(item["forecast_title_counties"], forecast_county_danger or {})
            message_title, message_body = local_copy or (title, body)
            message = {
                "to": item["expo_push_token"],
                "title": message_title,
                "body": message_body[:500],
                "sound": "default",
                "priority": "high" if event_type == "fire_weather" else "default",
                "channelId": _channel_for(event_type, item["app_version"]),
                "data": {"url": url, "eventType": event_type, **(extra_data or {})},
            }
            if event_type == "fire_detection" and (extra_data or {}).get("latitude") is not None:
                message["categoryId"] = FIRE_DETECTION_MAP_CATEGORY
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


def deliver_scheduled_forecasts(now: datetime | None = None) -> int:
    """Claim today's due forecast notifications once across scheduler processes."""
    if is_sandbox():
        return 0
    now = (now or datetime.now(timezone.utc)).astimezone(CHICAGO_TZ)
    today, clock = now.date().isoformat(), now.strftime("%H:%M")
    with _connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("DELETE FROM mobile_forecast_queue WHERE forecast_day < ?", (today,))
        due = connection.execute(
            """SELECT q.event_key, q.installation_id, s.expo_push_token, s.app_version,
                      s.local_forecast_titles, s.county_fips_json, e.payload_json
               FROM mobile_forecast_queue q
               JOIN mobile_push_subscriptions s USING (installation_id)
               JOIN mobile_push_events e USING (event_key)
               WHERE q.forecast_day = ? AND s.enabled = 1 AND s.forecast_enabled = 1
                 AND COALESCE(s.forecast_not_before, '') <= ?""",
            (today, clock),
        ).fetchall()
        groups = {}
        payloads = {}
        for row in due:
            item = dict(row)
            key = item["event_key"]
            payloads[key] = item.pop("payload_json")
            groups.setdefault(key, []).append(item)
        connection.executemany(
            "DELETE FROM mobile_forecast_queue WHERE event_key = ? AND installation_id = ?",
            [(row["event_key"], row["installation_id"]) for row in due],
        )

    sent = 0
    for key, subscriptions in groups.items():
        payload = json.loads(payloads[key])
        for item in subscriptions:
            item["forecast_title_counties"] = (
                set(json.loads(item["county_fips_json"])) if item["local_forecast_titles"] else set()
            )
        sent += _deliver_event(
            "forecast", key, subscriptions, payload["title"], payload["body"], payload["url"],
            extra_data={name: value for name, value in payload.items()
                        if name not in {"title", "body", "url", "imageUrl", "_countyDanger"}},
            image_url=payload.get("imageUrl"), forecast_county_danger=payload.get("_countyDanger"),
        )
    return sent


def _record_send_result(event_key: str, *, recipients: int, sent: int, unregistered: int) -> None:
    """Keep the outcome of one alert on its event row. A tracking failure must never fail the alert itself."""
    try:
        with _connect() as connection:
            connection.execute(
                """UPDATE mobile_push_events SET recipients = COALESCE(recipients, 0) + ?,
                       sent = COALESCE(sent, 0) + ?, unregistered = COALESCE(unregistered, 0) + ?,
                       failed = COALESCE(failed, 0) + ?, delivered = COALESCE(delivered, 0),
                       receipt_errors = COALESCE(receipt_errors, 0) WHERE event_key = ?""",
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


def notify_forecast(
    forecast: dict[str, Any], image_url: str | None = None,
    *, county_danger: dict[str, tuple[int, str]] | None = None,
) -> int:
    identity = forecast.get("updated_at") or forecast.get("id") or forecast.get("valid_time")
    return send_mobile_event(
        event_type="forecast",
        event_key=f"forecast:{identity}",
        title=str(forecast.get("title") or "Show Me Fire Forecast"),
        body="Missouri's latest daily fire weather forecast is ready.",
        url="/forecasts",
        extra_data={"forecastRevision": str(identity)},
        image_url=image_url,
        forecast_county_danger=county_danger,
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


def _in_feedback_quiet_hours(now: datetime | None = None) -> bool:
    hour = (now or datetime.now(timezone.utc)).astimezone(CHICAGO_TZ).hour
    return hour >= FEEDBACK_PROMPT_QUIET_START_HOUR or hour < FEEDBACK_PROMPT_QUIET_END_HOUR


def notify_feedback_candidates(incidents: list[dict[str, Any]], now: datetime | None = None) -> int:
    """Push one feedback request per new candidate incident
    (fire_confidence.list_feedback_candidate_incidents: moderate/high, recent).

    The goal is a public label (confirmed fire / controlled burn / not a fire)
    for the detection-confidence models; hearing about a nearby fire is a side
    effect. Idempotent per incident_id via mobile_push_events (event_key), so
    this can be called every ingest cycle. During quiet hours nothing is
    recorded, so the first cycle after they end sends whatever is still a
    candidate. Seeds a baseline on first run so a fresh deploy doesn't prompt
    for every current candidate - v2 because widening from high-only to
    moderate+high left the v1 baseline not covering active moderate incidents.
    """
    if _in_feedback_quiet_hours(now):
        return 0
    baseline_key = "fire_detection:baseline:v2"
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
        county_name = incident.get("county_name")
        where = f"near {county_name} County" if county_name else "near you"
        extra_data = {"confidenceLabel": incident.get("confidence_label")}
        if incident.get("smf_number"):
            extra_data["smfNumber"] = incident["smf_number"]
        if slug:
            extra_data["incidentSlug"] = slug
        # The app's "Open in Maps" notification action drops a pin here (see FIRE_DETECTION_MAP_CATEGORY).
        latitude, longitude = _valid_coordinate(incident.get("latitude"), incident.get("longitude"))
        if latitude is not None:
            extra_data["latitude"] = latitude
            extra_data["longitude"] = longitude
        delivered = send_mobile_event(
            event_type="fire_detection",
            event_key=f"fire_detection:{incident_id}",
            title="Help verify a satellite detection",
            body=f"Satellite picked up heat {where}. Wildfire, controlled burn, or nothing? Tap to tell us.",
            url=f"/detections/{quote(slug, safe='')}/feedback" if slug else "/map",
            county_fips=[county_fips] if county_fips else [],
            extra_data=extra_data,
        )
        sent += delivered
        _note_incident_push(incident_id, f"fire_detection:{incident_id}", delivered,
                            f"Asked {_people(delivered)} {where} to help verify it")
    return sent


def _people(count: int) -> str:
    return "1 person" if count == 1 else f"{count} people"


def _note_incident_push(incident_id: int | None, event_key: str, delivered: int, summary: str) -> None:
    """Put "a notification went out" on the incident's public timeline. Never lets a failure disturb the send."""
    if not incident_id or delivered <= 0:
        return
    try:
        from services.incident_history import record

        with _connect() as connection:
            record(connection, incident_id, "notification", summary, data={"recipients": delivered}, dedupe_key=f"push:{event_key}")
    except Exception:  # noqa: BLE001
        logger.exception("could not record the push on incident %s", incident_id)


def notify_official_listing(now: datetime | None = None) -> int:
    """Tell fire-detection subscribers when an agency or dispatch lists a fire: NIFC WFIGS (services/official_incidents.py)
    or WildCAD-E dispatch (services/wildcad.py).

    Same channel as the detection prompts: the "Help verify fire detections" opt-in, the incident's county, quiet hours
    (held, not dropped) and the per-device daily cap all apply. One push per INCIDENT, whichever source listed it first,
    so dispatch followed by NIFC never notifies twice; only for listings first seen in the last 12 hours, and never for an
    incident staff rejected.
    """
    from services.official_incidents import NOTIFY_WINDOW

    now = now or datetime.now(timezone.utc)
    if _in_feedback_quiet_hours(now):
        return 0
    cutoff = (now - NOTIFY_WINDOW).strftime("%Y-%m-%d %H:%M:%S")
    with _connect() as connection:
        base = """FROM {table} s JOIN fire_incidents i ON i.id = s.incident_id
                  WHERE s.notified_at IS NULL AND s.first_seen_at >= ? AND {extra}
                    AND i.status NOT IN ('deleted', 'merged') AND COALESCE(i.review_decision, '') != 'rejected'
                    AND i.public_slug IS NOT NULL"""
        nifc = connection.execute(
            "SELECT 'nifc' AS source, s.irwin_id AS ref, s.incident_id, s.name, s.acres, s.first_seen_at, i.public_slug, "
            "i.smf_number, i.county_fips, i.county_name " + base.format(table="fire_incident_official", extra="s.ended_at IS NULL"),
            (cutoff,),
        ).fetchall()
        dispatch = connection.execute(
            "SELECT 'wildcad' AS source, s.incident_key AS ref, s.incident_id, s.name, s.acres, s.first_seen_at, i.public_slug, "
            "i.smf_number, i.county_fips, i.county_name " + base.format(table="wildcad_incidents", extra="s.out_at IS NULL"),
            (cutoff,),
        ).fetchall()
    by_incident: dict[int, list[Any]] = {}
    for row in sorted([*nifc, *dispatch], key=lambda item: item["first_seen_at"]):
        by_incident.setdefault(row["incident_id"], []).append(row)

    sent = 0
    for incident_id, rows in by_incident.items():
        first = rows[0]
        where = f"near {first['county_name']} County" if first["county_name"] else "in Missouri"
        size = f", about {first['acres']:g} acres" if first["acres"] else ""
        name = first["name"] or "a fire"
        if first["source"] == "wildcad":
            title, body = f"Fire crews dispatched: {name}", f"Reported {where}{size}. Tap to see what we know about it."
        else:
            title, body = f"Forest Service is working the {name}", f"Listed {where}{size}. Tap to see what we know about it."
        event_key = f"fire_detection:official:incident:{incident_id}"
        delivered = send_mobile_event(
            event_type="fire_detection",
            event_key=event_key,
            title=title,
            body=body,
            url=f"/incidents/{quote(first['public_slug'], safe='')}",
            county_fips=[first["county_fips"]] if first["county_fips"] else [],
            extra_data={"incidentSlug": first["public_slug"], "smfNumber": first["smf_number"], "official": True},
        )
        sent += delivered
        what = "dispatch" if first["source"] == "wildcad" else "the Forest Service"
        _note_incident_push(incident_id, event_key, delivered, f"Told {_people(delivered)} {where} {what} listed it")
        with _connect() as connection:
            for row in rows:
                table, column = (("fire_incident_official", "irwin_id") if row["source"] == "nifc" else ("wildcad_incidents", "incident_key"))
                connection.execute(f"UPDATE {table} SET notified_at = CURRENT_TIMESTAMP WHERE {column} = ?", (row["ref"],))
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


def _detection_alert_details(connection: sqlite3.Connection, event_key: str, sent_at: str | None, sent: int) -> dict[str, Any] | None:
    """Staff-only context for one fire_detection alert: the county it went to (alerts match whole counties, never a
    distance), the incident behind it, how fast it went out, and what the public answered. None if the incident is gone."""
    try:
        incident_id = int(event_key.split(":", 1)[1])
    except (IndexError, ValueError):
        return None
    incident = connection.execute(
        """SELECT public_slug, county_fips, county_name, detection_count, first_detected_at, last_detected_at, status
           FROM fire_incidents WHERE id = ?""", (incident_id,)).fetchone()
    if incident is None:
        return None
    responses = {row["classification"]: row["n"] for row in connection.execute(
        "SELECT classification, COUNT(*) AS n FROM fire_incident_feedback WHERE incident_id = ? GROUP BY classification",
        (incident_id,))}
    lag_minutes = None
    try:
        first = datetime.fromisoformat(str(incident["first_detected_at"]).replace("Z", "+00:00"))
        alerted = datetime.fromisoformat(str(sent_at).replace("Z", "+00:00"))
        if first.tzinfo is None:
            first = first.replace(tzinfo=timezone.utc)
        if alerted.tzinfo is None:
            alerted = alerted.replace(tzinfo=timezone.utc)
        lag_minutes = max(0, round((alerted - first).total_seconds() / 60))
    except (TypeError, ValueError):
        pass
    total_responses = sum(responses.values())
    return {
        "incident_id": incident_id,
        "incident_slug": incident["public_slug"],
        "incident_status": incident["status"],
        "county_fips": incident["county_fips"],
        "county_name": incident["county_name"],
        "detection_count": incident["detection_count"],
        "first_detected_at": incident["first_detected_at"],
        "last_detected_at": incident["last_detected_at"],
        "minutes_to_alert": lag_minutes,
        "responses": responses,
        "response_total": total_responses,
        "response_pct": round(100 * total_responses / sent, 1) if sent else None,
    }


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
                      SUM(general_enabled) AS general,
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
            if item["event_type"] == "fire_detection":
                item["confidence_label"] = payload.get("confidenceLabel")
                item["detection"] = _detection_alert_details(connection, item["event_key"], item["created_at"], item["sent"] or 0)
            item["delivered_pct"] = pct(item["delivered"], item["sent"])
            item["awaiting_receipt"] = max(0, item["sent"] - item["delivered"] - item["receipt_errors"])
            recent_alerts.append(item)

    audience = {key: int(audience_row[key] or 0) for key in audience_row.keys()}
    totals = {key: sum(item[key] or 0 for item in by_type) for key in ("alerts", "recipients", "sent", "failed", "unregistered", "delivered", "receipt_errors", "reached_nobody")}
    totals["accepted_pct"] = pct(totals["sent"], totals["recipients"])
    totals["delivered_pct"] = pct(totals["delivered"], totals["sent"])
    return {"period_days": days, "audience": {**audience, "platforms": platforms, "awaiting_receipt": awaiting},
            "totals": totals, "by_type": by_type, "daily": daily, "recent": recent_alerts}
