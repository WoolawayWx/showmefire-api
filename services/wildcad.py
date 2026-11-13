"""WildCAD-E dispatch feed (Missouri-Iowa Interagency Dispatch Center, MOMOC) into the fire incident system.

Two JSON endpoints behind WildWeb are polled, under strict rules that are enforced here, not just by the schedule:

* at most ONE request per endpoint per 5 minutes. The guard lives in the database (`wildcad_feed_state`) and is claimed
  atomically before any request, so a restart, a second worker or a manual call cannot exceed it. Incidents and resources
  are polled by separate jobs ~150 s apart, so the combined rate is 2 requests / 5 min;
* a descriptive User-Agent with a contact address;
* failures are never retried inside a cycle: 5 -> 10 -> 20 -> 30 min back-off, and an HTTP 429/403 (or Retry-After) pauses
  that endpoint for an hour and alerts staff;
* every response is stored locally (`wildcad_snapshots`, only when its content changed) and parsed into local tables.
  Everything else (APIs, website, app, history, alerts) reads those tables and never fetches upstream;
* `WILDCAD_ENABLED=1` is required; without it nothing is ever requested.

Dedupe key: center + incident number + year (the dispatcher's `inc_num` restarts every year). A dispatcher editing acreage,
name or units updates the same record. `irwin_id` is kept when the feed starts providing it, to tie the record to NIFC.

Privacy: incident commanders and individual responders are people. They are stored for staff but never reach public
output (see `public_resources`).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Callable
from zoneinfo import ZoneInfo

import httpx

from core.database import upsert_detection_event
from services import incident_history

logger = logging.getLogger(__name__)


def get_db_path():
    """Resolved at call time so tests (and anything else) that patch core.database.get_db_path also reach this module."""
    from core import database

    return database.get_db_path()

CENTER = "MOMOC"
BASE_URL = "https://snknmqmon6.execute-api.us-west-2.amazonaws.com/centers"
USER_AGENT = "ShowMeFire (contact@showmefire.org)"
KINDS = ("incidents", "resources")
MIN_INTERVAL = timedelta(minutes=5)
BACKOFF_STEPS = (5, 10, 20, 30)          # minutes after the 1st, 2nd, 3rd... consecutive failure
BLOCKED_PAUSE = timedelta(hours=1)       # 403 / 429
MAX_RETRY_AFTER = timedelta(hours=6)
TIMEOUT_SECONDS = 20
MAX_RESPONSE_BYTES = 2_000_000
SNAPSHOT_RETENTION_DAYS = 7
CENTRAL = ZoneInfo("America/Chicago")
SOURCE = "wildcad"
SYSTEM_NAME = "WildCAD-E (Missouri-Iowa Interagency Dispatch)"
BASELINE_KEY = "wildcad:baseline:v1"
RECENT_DAYS = 7                           # an eligible incident older than this is stored but not turned into an incident
MO_BOUNDS = (35.9, 40.7, -95.9, -89.0)    # lat min/max, lon min/max: a cheap check before the county lookup
# Incident types that become fire incidents; everything else is kept for staff but is not a fire.
WILDFIRE_TYPES = {"wildfire"}
PRESCRIBED_WORDS = ("prescribed", " rx")


class ParseError(ValueError):
    """The upstream payload no longer looks like what this parser was written for."""


class FeedBlocked(RuntimeError):
    """403/429: stop and tell staff; do not keep asking."""


# --------------------------------------------------------------------------- time / small helpers

def _utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def _iso(moment: datetime | None) -> str | None:
    return _utc(moment).astimezone(timezone.utc).isoformat().replace("+00:00", "Z") if moment else None


def _parse_iso(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return _utc(datetime.fromisoformat(_normalize_fraction(str(value).replace("Z", "+00:00"))))
    except ValueError:
        return None


def _normalize_fraction(text: str) -> str:
    """Python 3.10 only accepts 3 or 6 fractional-second digits; WildCAD sends 2 ("10:24:45.11") as well as 3."""
    return re.sub(r"(\.\d+)", lambda match: (match.group(1) + "000000")[:7], text, count=1)


def _local_to_utc(value: Any) -> tuple[datetime | None, int | None]:
    """WildCAD stamps are Central local time with no zone. Returns (utc moment, local year)."""
    if not value:
        return None, None
    try:
        local = datetime.fromisoformat(_normalize_fraction(str(value)))
    except ValueError:
        return None, None
    if local.tzinfo is None:
        local = local.replace(tzinfo=CENTRAL)
    return local.astimezone(timezone.utc), local.astimezone(CENTRAL).year


def _number(value: Any) -> float | None:
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def _text(value: Any) -> str:
    return str(value).strip() if value not in (None, "") else ""


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(get_db_path(), timeout=15)
    connection.row_factory = sqlite3.Row
    return connection


def enabled() -> bool:
    return os.getenv("WILDCAD_ENABLED", "").strip().lower() in {"1", "true", "yes"}


# --------------------------------------------------------------------------- request guard

def _backoff(failures: int) -> timedelta:
    return timedelta(minutes=BACKOFF_STEPS[min(max(failures, 1), len(BACKOFF_STEPS)) - 1])


def claim_slot(kind: str, now: datetime) -> bool:
    """Atomically reserve the next request for `kind`. False means "too soon": make no request."""
    with _connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute("INSERT OR IGNORE INTO wildcad_feed_state (kind) VALUES (?)", (kind,))
        row = connection.execute("SELECT next_allowed_at FROM wildcad_feed_state WHERE kind = ?", (kind,)).fetchone()
        allowed_at = _parse_iso(row["next_allowed_at"])
        if allowed_at is not None and now < allowed_at:
            return False
        # Reserved before the request goes out, so nothing else can start one in the meantime.
        connection.execute(
            "UPDATE wildcad_feed_state SET last_attempt_at = ?, next_allowed_at = ?, updated_at = CURRENT_TIMESTAMP WHERE kind = ?",
            (_iso(now), _iso(now + MIN_INTERVAL), kind),
        )
    return True


def _record_success(kind: str, now: datetime, status: int, headers: dict[str, str]) -> None:
    with _connect() as connection:
        connection.execute(
            "UPDATE wildcad_feed_state SET last_success_at = ?, consecutive_failures = 0, last_status = ?, last_error = NULL, "
            "paused_reason = NULL, etag = ?, last_modified = ?, next_allowed_at = ?, updated_at = CURRENT_TIMESTAMP WHERE kind = ?",
            (_iso(now), status, headers.get("etag"), headers.get("last-modified"), _iso(now + MIN_INTERVAL), kind),
        )


def _record_failure(kind: str, now: datetime, status: int | None, error: str, *, blocked: bool = False,
                    retry_after: timedelta | None = None) -> int:
    with _connect() as connection:
        row = connection.execute("SELECT consecutive_failures FROM wildcad_feed_state WHERE kind = ?", (kind,)).fetchone()
        failures = (row["consecutive_failures"] if row else 0) + 1
        wait = _backoff(failures)
        reason = None
        if blocked:
            wait, reason = max(BLOCKED_PAUSE, retry_after or timedelta(0)), f"blocked (HTTP {status})"
        elif retry_after:
            wait = max(wait, retry_after)
        connection.execute(
            "UPDATE wildcad_feed_state SET consecutive_failures = ?, last_status = ?, last_error = ?, paused_reason = ?, "
            "next_allowed_at = ?, updated_at = CURRENT_TIMESTAMP WHERE kind = ?",
            (failures, status, error[:300], reason, _iso(now + min(wait, MAX_RETRY_AFTER)), kind),
        )
    return failures


def _retry_after(headers: dict[str, str]) -> timedelta | None:
    value = headers.get("retry-after")
    if value and value.strip().isdigit():
        return timedelta(seconds=int(value))
    return None


def fetch_json(url: str, extra_headers: dict[str, str], client: httpx.Client | None = None) -> tuple[int, dict[str, str], bytes]:
    """One GET with the Show Me Fire user agent, a size cap and no redirects. The only place that touches the network."""
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json", **extra_headers}
    owns = client is None
    client = client or httpx.Client(timeout=TIMEOUT_SECONDS, follow_redirects=False)
    try:
        with client.stream("GET", url, headers=headers) as response:
            body = bytearray()
            for chunk in response.iter_bytes():
                body.extend(chunk)
                if len(body) > MAX_RESPONSE_BYTES:
                    raise ParseError(f"response larger than {MAX_RESPONSE_BYTES} bytes")
            return response.status_code, {k.lower(): v for k, v in response.headers.items()}, bytes(body)
    finally:
        if owns:
            client.close()


# --------------------------------------------------------------------------- parsing

def _records(payload: Any) -> tuple[list[dict[str, Any]], str | None]:
    """The feed answers `[ {data: [...], retrieved, webComment} ]`; tolerate a bare dict or list as well."""
    if isinstance(payload, list):
        if len(payload) == 1 and isinstance(payload[0], dict) and "data" in payload[0]:
            payload = payload[0]
        elif all(isinstance(item, dict) for item in payload):
            return payload, None
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return [item for item in payload["data"] if isinstance(item, dict)], _text(payload.get("retrieved")) or None
    raise ParseError("unexpected WildCAD payload shape")


def _json_field(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(value) if isinstance(value, str) else {}
    except ValueError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def parse_incident(record: dict[str, Any], center: str = CENTER) -> dict[str, Any] | None:
    """One feed record -> normalized dict, or None when it has no incident number or date."""
    inc_num = _text(record.get("inc_num"))
    reported_at, year = _local_to_utc(record.get("date"))
    if not inc_num or reported_at is None or year is None:
        return None
    fiscal, status = _json_field(record.get("fiscal_data")), _json_field(record.get("fire_status"))
    latitude, longitude = _number(record.get("latitude")), _number(record.get("longitude"))
    if longitude is not None and longitude > 0:
        longitude = -longitude          # the feed drops the minus sign for the western hemisphere
    irwin = _text(record.get("irwin_id") or record.get("irwinId") or fiscal.get("irwin_id") or fiscal.get("irwinId"))
    resources = [str(code).strip() for code in (record.get("resources") or []) if code and not set(str(code)) <= {"*"}]
    return {
        "incident_key": f"{center}:{inc_num}:{year}",
        "center": center,
        "inc_num": inc_num,
        "year": year,
        "uuid": _text(record.get("uuid")) or None,
        "irwin_id": irwin.strip("{} ").lower() or None,
        "dispatch_unit": _text(fiscal.get("wfdssunit")) or None,
        "name": _text(record.get("name")),
        "incident_type": _text(record.get("type")),
        "acres": _number(record.get("acres")),
        "latitude": latitude,
        "longitude": longitude,
        "web_comment": _text(record.get("webComment")),
        "ic_name": _text(record.get("ic")),
        "reported_at": reported_at,
        "contained_at": _local_to_utc(status.get("contain"))[0],
        "controlled_at": _local_to_utc(status.get("control"))[0],
        "out_at": _local_to_utc(status.get("out"))[0],
        "resources": resources,
        "raw": record,
    }


def parse_incidents(payload: Any) -> list[dict[str, Any]]:
    records, _retrieved = _records(payload)
    parsed = [item for item in (parse_incident(record) for record in records) if item]
    if records and not parsed:
        raise ParseError("no incident in the payload could be read (fields changed?)")
    return parsed


# Resource classification. Anything that might be a person is treated as one (hidden from the public) unless a
# clear equipment/crew keyword says otherwise.
_KIND_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("aircraft", re.compile(r"\b(heli|helicopter|aircraft|air tanker|tanker|bell \d+|n\d{2,4}[a-z]{0,2})\b|^h-", re.I)),
    ("uas", re.compile(r"\buas\b|drone", re.I)),
    ("dozer", re.compile(r"dozer|dzr|^d-\d", re.I)),
    ("engine", re.compile(r"\beng(ine)?\b|^e-?\d|-e\d+\b|\beng\d", re.I)),
    ("crew", re.compile(r"\bcrew\b|job ?corps?|\bmod(ule)?\b|\bjcc\b|hotshot", re.I)),
    ("utility", re.compile(r"utility|\bchase\b|^ut-|\btruck\b|skid steer|water tender|\btender\b", re.I)),
]
_PERSON_CODE = re.compile(r"^[A-Za-z'.\- ]+,\s?[A-Za-z.\- ]+$")
_PERSON_DESCRIPTION = re.compile(r"^[A-Z][a-z'.\-]+(?: [A-Z][a-z'.\-]+){1,2}(?: [A-Z]{2,})?$")


def classify_resource(code: str, description: str) -> tuple[str, bool]:
    """(kind, is_person). Unknown things are people-until-proven-otherwise so they can never leak publicly."""
    haystack = f"{code} {description}"
    if _PERSON_CODE.match(code.strip()):
        return "person", True
    for kind, pattern in _KIND_PATTERNS:
        if pattern.search(haystack):
            return kind, False
    if _PERSON_DESCRIPTION.match(description.strip()):
        return "person", True
    return "unknown", True


def parse_resources(payload: Any, center: str = CENTER) -> list[dict[str, Any]]:
    records, _retrieved = _records(payload)
    parsed = []
    for record in records:
        code = _text(record.get("code"))
        if not code:
            continue
        description = _text(record.get("description"))
        identifiers = record.get("incident_identifiers") if isinstance(record.get("incident_identifiers"), dict) else {}
        kind, is_person = classify_resource(code, description)
        parsed.append({
            "center": center,
            "code": code,
            "upstream_id": record.get("id") if isinstance(record.get("id"), int) else None,
            "description": description,
            "status": _text(record.get("status")),
            "location": _text(record.get("location")),
            "is_public": 1 if record.get("is_public", True) else 0,
            "dispatch_unit": _text(identifiers.get("unit")) or None,
            "inc_num": _text(identifiers.get("incident_num")) or None,
            "incident_name": _text(record.get("incident_name")),
            "kind": kind,
            "is_person": 1 if is_person else 0,
        })
    if records and not parsed:
        raise ParseError("no resource in the payload could be read (fields changed?)")
    return parsed


# --------------------------------------------------------------------------- snapshots (the local cache)

def store_snapshot(kind: str, body: bytes, record_count: int, now: datetime) -> bool:
    """Keep the raw response, but only when its content changed. Returns True when a new snapshot was written."""
    digest = hashlib.sha256(body).hexdigest()
    with _connect() as connection:
        cursor = connection.execute(
            "INSERT OR IGNORE INTO wildcad_snapshots (kind, fetched_at, sha256, record_count, body_json) VALUES (?, ?, ?, ?, ?)",
            (kind, _iso(now), digest, record_count, body.decode("utf-8", errors="replace")),
        )
        connection.execute("DELETE FROM wildcad_snapshots WHERE fetched_at < ?",
                           (_iso(now - timedelta(days=SNAPSHOT_RETENTION_DAYS)),))
        return cursor.rowcount == 1


# --------------------------------------------------------------------------- polling

def poll(kind: str, now: datetime | None = None, client: httpx.Client | None = None,
         fetch: Callable[..., tuple[int, dict[str, str], bytes]] | None = None) -> dict[str, Any]:
    """One guarded poll of `kind` ("incidents" | "resources"). Never raises. Returns a small status dict."""
    if kind not in KINDS:
        raise ValueError(kind)
    now = now or datetime.now(timezone.utc)
    if not enabled():
        return {"kind": kind, "requested": False, "reason": "disabled"}
    if not claim_slot(kind, now):
        return {"kind": kind, "requested": False, "reason": "too soon"}

    with _connect() as connection:
        validators = connection.execute("SELECT etag, last_modified FROM wildcad_feed_state WHERE kind = ?", (kind,)).fetchone()
    extra: dict[str, str] = {}
    if validators and validators["etag"]:
        extra["If-None-Match"] = validators["etag"]
    elif validators and validators["last_modified"]:
        extra["If-Modified-Since"] = validators["last_modified"]
    url = f"{BASE_URL}/{CENTER}/{kind}"
    try:
        status, headers, body = (fetch or fetch_json)(url, extra, client)
    except ParseError as error:
        _record_failure(kind, now, None, str(error))
        _alert(kind, f"WildCAD {kind} response rejected: {error}")
        return {"kind": kind, "requested": True, "ok": False, "error": str(error)}
    except Exception as error:  # noqa: BLE001 - network problems are expected; back off and try later
        failures = _record_failure(kind, now, None, f"{type(error).__name__}: {error}")
        logger.warning("WildCAD %s request failed (%s)", kind, error)
        return {"kind": kind, "requested": True, "ok": False, "error": str(error), "failures": failures}

    if status in (403, 429):
        _record_failure(kind, now, status, f"HTTP {status}", blocked=True, retry_after=_retry_after(headers))
        _alert(kind, f"WildCAD {kind} answered HTTP {status}. Polling is paused for an hour; check with MOMOC before resuming.", blocked=True)
        return {"kind": kind, "requested": True, "ok": False, "status": status, "blocked": True}
    if status == 304:
        _record_success(kind, now, status, headers)
        return {"kind": kind, "requested": True, "ok": True, "changed": False}
    if status != 200:
        _record_failure(kind, now, status, f"HTTP {status}", retry_after=_retry_after(headers))
        return {"kind": kind, "requested": True, "ok": False, "status": status}

    try:
        payload = json.loads(body.decode("utf-8"))
        parsed = parse_incidents(payload) if kind == "incidents" else parse_resources(payload)
    except (ValueError, UnicodeDecodeError) as error:    # ParseError and JSONDecodeError are ValueErrors
        _record_failure(kind, now, status, f"unreadable payload: {error}")
        _alert(kind, f"WildCAD {kind} payload could not be read ({error}). The feed format may have changed.")
        return {"kind": kind, "requested": True, "ok": False, "error": str(error)}

    changed = store_snapshot(kind, body, len(parsed), now)
    _record_success(kind, now, status, headers)
    _write_heartbeat(kind, now)
    _clear_problems(kind)
    try:
        result = apply_incidents(parsed, now) if kind == "incidents" else apply_resources(parsed, now)
    except Exception:  # noqa: BLE001
        logger.exception("WildCAD %s: could not apply %d records", kind, len(parsed))
        return {"kind": kind, "requested": True, "ok": True, "changed": changed, "applied": False}
    return {"kind": kind, "requested": True, "ok": True, "changed": changed, "applied": True, **result}


def poll_incidents_job() -> dict[str, Any]:
    result = poll("incidents")
    if result.get("ok"):
        incident_history.sync_incident_history()
    return result


def poll_resources_job() -> dict[str, Any]:
    result = poll("resources")
    if result.get("ok"):
        incident_history.sync_incident_history()
    return result


# --------------------------------------------------------------------------- health hooks (filled in with the alerting)

def _write_heartbeat(kind: str, now: datetime) -> None:
    try:
        from services.wildcad_health import write_heartbeat

        write_heartbeat(kind, now)
    except Exception:  # noqa: BLE001
        logger.exception("could not write the WildCAD heartbeat")


def _clear_problems(kind: str) -> None:
    try:
        from services.wildcad_health import clear_problem

        clear_problem(kind)
    except Exception:  # noqa: BLE001
        logger.exception("could not clear the WildCAD problem state")


def _alert(kind: str, message: str, *, blocked: bool = False) -> None:
    try:
        from services.wildcad_health import alert_problem

        alert_problem(kind, message, blocked=blocked)
    except Exception:  # noqa: BLE001
        logger.exception("could not send the WildCAD alert")


# --------------------------------------------------------------------------- applying incidents

def _eligible(incident: dict[str, Any], now: datetime) -> str | None:
    """Return the cause category when this dispatch record should be a fire incident, else None."""
    kind = incident["incident_type"].lower()
    if kind in WILDFIRE_TYPES:
        cause = "wildfire"
    elif any(word in f" {kind}" for word in PRESCRIBED_WORDS):
        cause = "prescribed"
    else:
        return None
    lat, lon = incident["latitude"], incident["longitude"]
    if lat is None or lon is None or not (MO_BOUNDS[0] <= lat <= MO_BOUNDS[1] and MO_BOUNDS[2] <= lon <= MO_BOUNDS[3]):
        return None
    if now - incident["reported_at"] > timedelta(days=RECENT_DAYS):
        return None
    return cause


def _acres_text(acres: float | None) -> str:
    return "size not reported" if acres is None else f"{acres:g} acre{'' if acres == 1 else 's'}"


def _county(incident: dict[str, Any]) -> tuple[str | None, str | None]:
    from services.county_lookup import county_for_point

    return county_for_point(incident["latitude"], incident["longitude"])


def _ingest_event(incident: dict[str, Any], cause: str, county_fips: str, county_name: str) -> tuple[int, int | None]:
    result = upsert_detection_event(
        source=SOURCE,
        external_id=incident["incident_key"],
        latitude=incident["latitude"],
        longitude=incident["longitude"],
        occurred_at=_iso(incident["reported_at"]),
        county_fips=county_fips,
        county_name=county_name,
        verification_tier="official_source_confirmed",
        cause_category=cause,
        acres=incident["acres"],
        official_source_system=SYSTEM_NAME,
        official_source_ref=f"{incident['dispatch_unit'] or incident['center']}-{incident['inc_num']}",
        origin={"channel": f"{CENTER}/incidents", "provider_ref": incident["incident_key"],
                "raw": {"name": incident["name"], "type": incident["incident_type"], "acres": incident["acres"]}},
    )
    with _connect() as connection:
        row = connection.execute("SELECT incident_id FROM fire_events WHERE id = ?", (result["event_id"],)).fetchone()
    return result["event_id"], (row["incident_id"] if row else None)


def apply_incidents(incidents: list[dict[str, Any]], now: datetime) -> dict[str, int]:
    """Upsert the dispatch records, tie eligible ones to incidents, and write timeline entries for changes."""
    from services.mobile_push import record_event

    with _connect() as connection:
        first_run = connection.execute("SELECT 1 FROM mobile_push_events WHERE event_key = ?", (BASELINE_KEY,)).fetchone() is None
        known = {row["incident_key"]: dict(row) for row in connection.execute("SELECT * FROM wildcad_incidents")}

    created = updated = linked = 0
    for incident in incidents:
        key = incident["incident_key"]
        before = known.get(key)
        cause = _eligible(incident, now)
        county_fips = county_name = None
        if cause is not None:
            county_fips, county_name = _county(incident)
            if county_fips is None:
                cause = None   # inside the coarse box but not a Missouri county (Iowa, Arkansas...)
        event_id, incident_id = (before or {}).get("event_id"), (before or {}).get("incident_id")
        if cause is not None:
            event_id, incident_id = _ingest_event(incident, cause, county_fips, county_name)
            linked += 1
        with _connect() as connection:
            if before is None:
                created += 1
                connection.execute(
                    "INSERT INTO wildcad_incidents (incident_key, center, inc_num, year, uuid, irwin_id, dispatch_unit, name, "
                    "incident_type, acres, latitude, longitude, county_fips, county_name, reported_at, contained_at, controlled_at, "
                    "out_at, web_comment, ic_name, raw_json, event_id, incident_id, notified_at, first_seen_at, last_seen_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (key, incident["center"], incident["inc_num"], incident["year"], incident["uuid"], incident["irwin_id"],
                     incident["dispatch_unit"], incident["name"], incident["incident_type"], incident["acres"],
                     incident["latitude"], incident["longitude"], county_fips, county_name, _iso(incident["reported_at"]),
                     _iso(incident["contained_at"]), _iso(incident["controlled_at"]), _iso(incident["out_at"]),
                     incident["web_comment"], incident["ic_name"], json.dumps(incident["raw"], separators=(",", ":")),
                     event_id, incident_id, _iso(now) if (first_run or cause is None) else None,
                     now.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d %H:%M:%S")),
                )
                if incident_id is not None:
                    name_part = ", " + incident["name"] if incident["name"] else ""
                    incident_history.record(
                        connection, incident_id, "dispatch",
                        f"Reported to dispatch as incident {incident['inc_num']}{name_part} ({_acres_text(incident['acres'])})",
                        occurred_at=incident["reported_at"], data={"incident_number": incident["inc_num"], "acres": incident["acres"]},
                        dedupe_key=f"dispatch:{key}:reported",
                    )
            else:
                updated += 1
                _diff_and_record(connection, before, incident, incident_id, now)
                connection.execute(
                    "UPDATE wildcad_incidents SET uuid = ?, irwin_id = COALESCE(?, irwin_id), dispatch_unit = ?, name = ?, "
                    "incident_type = ?, acres = ?, latitude = ?, longitude = ?, county_fips = COALESCE(?, county_fips), "
                    "county_name = COALESCE(?, county_name), contained_at = ?, controlled_at = ?, out_at = ?, web_comment = ?, "
                    "ic_name = ?, raw_json = ?, event_id = COALESCE(?, event_id), incident_id = COALESCE(?, incident_id), "
                    "last_seen_at = ? WHERE incident_key = ?",
                    (incident["uuid"], incident["irwin_id"], incident["dispatch_unit"], incident["name"], incident["incident_type"],
                     incident["acres"], incident["latitude"], incident["longitude"], county_fips, county_name,
                     _iso(incident["contained_at"]), _iso(incident["controlled_at"]), _iso(incident["out_at"]),
                     incident["web_comment"], incident["ic_name"], json.dumps(incident["raw"], separators=(",", ":")),
                     event_id, incident_id, now.strftime("%Y-%m-%d %H:%M:%S"), key),
                )
                if incident_id is not None and incident["acres"] != before["acres"] and event_id:
                    connection.execute("UPDATE fire_events SET acres = ? WHERE id = ?", (incident["acres"], event_id))
    if first_run:
        record_event(BASELINE_KEY, "fire_detection", {"seeded": len(incidents)})
    return {"created": created, "updated": updated, "linked": linked}


def _diff_and_record(connection: sqlite3.Connection, before: dict[str, Any], incident: dict[str, Any],
                     incident_id: int | None, now: datetime) -> None:
    if incident_id is None:
        return
    key = incident["incident_key"]
    if incident["name"] and incident["name"] != before["name"]:
        incident_history.record(
            connection, incident_id, "dispatch",
            f"Dispatch changed the name from {before['name'] or 'unnamed'} to {incident['name']}", occurred_at=now,
            data={"name": incident["name"]}, dedupe_key=f"dispatch:{key}:name:{incident['name']}",
        )
    if incident["acres"] != before["acres"] and incident["acres"] is not None:
        incident_history.record(
            connection, incident_id, "dispatch",
            f"Dispatch updated the size from {_acres_text(before['acres'])} to {_acres_text(incident['acres'])}", occurred_at=now,
            data={"acres": incident["acres"]}, dedupe_key=f"dispatch:{key}:acres:{incident['acres']:g}",
        )
    for field, label in (("contained_at", "contained"), ("controlled_at", "controlled"), ("out_at", "out")):
        if incident[field] is not None and before.get(field) is None:
            incident_history.record(
                connection, incident_id, "dispatch", f"Dispatch reported the fire {label}", occurred_at=incident[field],
                data={"status": label}, dedupe_key=f"dispatch:{key}:{label}",
            )


# --------------------------------------------------------------------------- applying resources

ON_SCENE = "on scene"
IDLE_STATUSES = {"out of svc", "available", "out of service"}
_KIND_PLURALS = {"engine": "engines", "dozer": "dozers", "crew": "crews", "aircraft": "aircraft", "uas": "drones",
                 "utility": "utility vehicles", "person": "personnel", "unknown": "other resources"}


def _signature_text(counts: dict[str, int]) -> str:
    parts = []
    for kind in ("engine", "dozer", "crew", "aircraft", "uas", "utility", "person", "unknown"):
        n = counts.get(kind, 0)
        if n:
            noun = _KIND_PLURALS[kind] if n != 1 else {"engine": "engine", "dozer": "dozer", "crew": "crew", "aircraft": "aircraft",
                                                       "uas": "drone", "utility": "utility vehicle", "person": "person",
                                                       "unknown": "other resource"}[kind]
            parts.append(f"{n} {noun}")
    return ", ".join(parts)


def apply_resources(resources: list[dict[str, Any]], now: datetime) -> dict[str, int]:
    """Replace the resource list with the latest snapshot, tie assigned resources to their incidents, and note changes."""
    stamp = now.strftime("%Y-%m-%d %H:%M:%S")
    with _connect() as connection:
        incidents = {(row["dispatch_unit"], row["inc_num"]): dict(row) for row in connection.execute(
            "SELECT incident_key, dispatch_unit, inc_num, incident_id, resource_signature FROM wildcad_incidents")}
        existing = {row["code"]: dict(row) for row in connection.execute("SELECT code, status FROM wildcad_resources")}
        seen = set()
        for item in resources:
            seen.add(item["code"])
            incident = incidents.get((item["dispatch_unit"], item["inc_num"])) if item["inc_num"] else None
            incident_id = incident["incident_id"] if incident else None
            before = existing.get(item["code"])
            connection.execute(
                "INSERT INTO wildcad_resources (center, code, upstream_id, description, status, location, kind, is_person, is_public, "
                "dispatch_unit, inc_num, incident_name, incident_id, first_seen_at, last_seen_at, status_changed_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(center, code) DO UPDATE SET upstream_id = excluded.upstream_id, description = excluded.description, "
                "location = excluded.location, kind = excluded.kind, is_person = excluded.is_person, is_public = excluded.is_public, "
                "dispatch_unit = excluded.dispatch_unit, inc_num = excluded.inc_num, incident_name = excluded.incident_name, "
                "incident_id = excluded.incident_id, last_seen_at = excluded.last_seen_at, "
                "status_changed_at = CASE WHEN wildcad_resources.status != excluded.status THEN excluded.last_seen_at "
                "ELSE wildcad_resources.status_changed_at END, status = excluded.status",
                (item["center"], item["code"], item["upstream_id"], item["description"], item["status"], item["location"],
                 item["kind"], item["is_person"], item["is_public"], item["dispatch_unit"], item["inc_num"],
                 item["incident_name"], incident_id, stamp, stamp, stamp),
            )
        # The upstream list is the whole inventory: anything absent is gone.
        if seen:
            placeholders = ",".join("?" * len(seen))
            connection.execute(f"DELETE FROM wildcad_resources WHERE code NOT IN ({placeholders})", list(seen))

        changes = 0
        for (unit, number), incident in incidents.items():
            if incident["incident_id"] is None:
                continue
            counts: dict[str, int] = {}
            for item in resources:
                if item["dispatch_unit"] == unit and item["inc_num"] == number and item["status"].lower() == ON_SCENE:
                    counts[item["kind"]] = counts.get(item["kind"], 0) + 1
            signature = json.dumps(counts, sort_keys=True)
            if signature == (incident["resource_signature"] or "") or (not counts and not incident["resource_signature"]):
                continue
            summary = f"On scene: {_signature_text(counts)}" if counts else "No dispatched resources are on scene any more"
            incident_history.record(
                connection, incident["incident_id"], "dispatch", summary, occurred_at=now, data={"on_scene": counts},
                dedupe_key=f"dispatch:{incident['incident_key']}:resources:{stamp}",
            )
            connection.execute("UPDATE wildcad_incidents SET resource_signature = ? WHERE incident_key = ?",
                               (signature if counts else "", incident["incident_key"]))
            changes += 1
    return {"resources": len(resources), "resource_changes": changes}


# --------------------------------------------------------------------------- reading the local cache (public vs staff)

def public_resources(incident_id: int) -> dict[str, Any]:
    """Equipment that dispatch marks public, plus a head-count of people. Never a name or where a person is based."""
    with _connect() as connection:
        rows = [dict(row) for row in connection.execute(
            "SELECT description, code, status, kind, is_person FROM wildcad_resources WHERE incident_id = ? AND is_public = 1 "
            "AND LOWER(status) NOT IN ('out of svc', 'available', 'out of service') ORDER BY kind, code", (incident_id,))]
    equipment = [{"name": row["description"] or row["code"], "kind": row["kind"], "status": row["status"]}
                 for row in rows if not row["is_person"]]
    return {"equipment": equipment, "personnel": sum(1 for row in rows if row["is_person"])}


def staff_resources(incident_id: int) -> list[dict[str, Any]]:
    with _connect() as connection:
        return [dict(row) for row in connection.execute(
            "SELECT code, description, status, location, kind, is_person, is_public, inc_num, incident_name, status_changed_at "
            "FROM wildcad_resources WHERE incident_id = ? ORDER BY status, code", (incident_id,))]


def dispatch_for(incident_id: int) -> dict[str, Any] | None:
    """The dispatch record attached to an incident, in the shape the public API returns (no IC, no raw payload)."""
    with _connect() as connection:
        row = connection.execute(
            "SELECT center, inc_num, year, dispatch_unit, name, incident_type, acres, reported_at, out_at, contained_at, last_seen_at "
            "FROM wildcad_incidents WHERE incident_id = ? ORDER BY reported_at LIMIT 1", (incident_id,)).fetchone()
    if row is None:
        return None
    return {
        "center": row["center"],
        "incident_number": f"{row['dispatch_unit'] or row['center']}-{row['inc_num']}",
        "name": row["name"],
        "type": row["incident_type"],
        "acres": row["acres"],
        "reported_at": row["reported_at"],
        "contained_at": row["contained_at"],
        "out_at": row["out_at"],
        "last_updated": row["last_seen_at"],
    }


def official_sources_for(incident_id: int, nifc: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Every agency/dispatch record on an incident (newest design of `official_source`, which stays for older apps)."""
    sources: list[dict[str, Any]] = []
    dispatch = dispatch_for(incident_id)
    if dispatch:
        sources.append({"system": "WildCAD-E dispatch", "name": dispatch["name"], "agency_name": "Missouri-Iowa Interagency Dispatch",
                        "incident_number": dispatch["incident_number"], "acres": dispatch["acres"], "active": dispatch["out_at"] is None})
    if nifc:
        sources.append({"system": "NIFC WFIGS", "name": nifc["name"], "agency_name": nifc["agency_name"], "incident_number": None,
                        "acres": nifc["acres"], "active": nifc["active"]})
    return sources


def feed_status() -> dict[str, Any]:
    """Freshness of the local cache, for GET /api/dispatch/status. Reads the database only."""
    with _connect() as connection:
        state = {row["kind"]: dict(row) for row in connection.execute("SELECT * FROM wildcad_feed_state")}
        counts = {kind: connection.execute("SELECT record_count FROM wildcad_snapshots WHERE kind = ? ORDER BY fetched_at DESC LIMIT 1",
                                           (kind,)).fetchone() for kind in KINDS}
    feeds = {}
    for kind in KINDS:
        row = state.get(kind) or {}
        feeds[kind] = {"last_success_at": row.get("last_success_at"), "last_attempt_at": row.get("last_attempt_at"),
                       "consecutive_failures": row.get("consecutive_failures", 0), "paused": bool(row.get("paused_reason")),
                       "records": counts[kind]["record_count"] if counts[kind] else None}
    return {"enabled": enabled(), "center": CENTER, "feeds": feeds}
