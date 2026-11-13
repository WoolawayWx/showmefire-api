"""U.S. Forest Service wildfires in Missouri, folded into the fire incident system.

Source: NIFC's public WFIGS "Current Wildland Fire Incident Locations" ArcGIS
service (fed by IRWIN, which WildCAD dispatch writes to). No key. Records
usually appear 1-3 hours after discovery (median 1.4 h over 25 Missouri USFS
fires; a few took days), and fires dispatch never enters nationally are absent,
so this confirms and names fires; it is not a first alert.

Each active record becomes a `fire_events` row (source `nifc_wfigs`, already
approved, tier official_source_confirmed) that joins the satellite-detection
incident it belongs to, or starts one, exactly like an approved public report.
The agency's name/acres/containment live in `fire_incident_official`, changes are
written to the incident's timeline (services/incident_history.py), and a new
listing goes out through the existing `fire_detection` push channel (same
opt-in, county filter, quiet hours and daily cap), see
mobile_push.notify_official_listing.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from core.database import get_db_path, upsert_detection_event
from services import incident_history

logger = logging.getLogger(__name__)

WFIGS_CURRENT_URL = (
    "https://services3.arcgis.com/T4QMspbfLg3qTGWY/arcgis/rest/services/"
    "WFIGS_Incident_Locations_Current/FeatureServer/0/query"
)
WHERE = "POOState = 'US-MO' AND POOProtectingAgency = 'USFS' AND IncidentTypeCategory = 'WF'"
OUT_FIELDS = ",".join((
    "IrwinID", "IncidentName", "POOCounty", "POOFips", "IncidentSize", "PercentContained",
    "FireDiscoveryDateTime", "CreatedOnDateTime_dt",
))
SOURCE = "nifc_wfigs"
SYSTEM_NAME = "NIFC WFIGS (U.S. Forest Service)"
BASELINE_KEY = "official_incident:baseline:v1"
# A record that first appears long after discovery (late data entry) is history, not news.
MAX_DISCOVERY_AGE = timedelta(days=3)
NOTIFY_WINDOW = timedelta(hours=12)
ACRES_STEP = 1.0           # note an acreage change of at least this many acres...
ACRES_RELATIVE_STEP = 0.2  # ...or this fraction of the previous size
TIMEOUT_SECONDS = 20


def _millis(value: Any) -> datetime | None:
    try:
        return datetime.fromtimestamp(float(value) / 1000, tz=timezone.utc)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def _iso(moment: datetime | None) -> str | None:
    return moment.astimezone(timezone.utc).isoformat().replace("+00:00", "Z") if moment else None


def parse_incident(feature: dict[str, Any]) -> dict[str, Any] | None:
    attributes = feature.get("attributes") or {}
    irwin_id = str(attributes.get("IrwinID") or "").strip("{} ").lower()
    name = str(attributes.get("IncidentName") or "").strip()
    geometry = feature.get("geometry") or {}
    if not irwin_id or not name or geometry.get("x") is None or geometry.get("y") is None:
        return None
    size, contained = attributes.get("IncidentSize"), attributes.get("PercentContained")
    fips = str(attributes.get("POOFips") or "").strip()
    return {
        "irwin_id": irwin_id,
        "name": name,
        "county_name": str(attributes.get("POOCounty") or "").strip() or None,
        "county_fips": fips if len(fips) == 5 and fips.startswith("29") else None,
        "acres": float(size) if isinstance(size, (int, float)) else None,
        "percent_contained": float(contained) if isinstance(contained, (int, float)) else None,
        "discovered_at": _millis(attributes.get("FireDiscoveryDateTime")),
        "listed_at": _millis(attributes.get("CreatedOnDateTime_dt")),
        "latitude": float(geometry["y"]),
        "longitude": float(geometry["x"]),
    }


def fetch_usfs_missouri_incidents(client: httpx.Client | None = None) -> list[dict[str, Any]]:
    """Active USFS wildfires in Missouri. Raises on a network/HTTP/format problem so the caller can log it."""
    params = {"where": WHERE, "outFields": OUT_FIELDS, "outSR": "4326", "returnGeometry": "true", "f": "json"}
    owns_client = client is None
    client = client or httpx.Client(timeout=TIMEOUT_SECONDS)
    try:
        response = client.get(WFIGS_CURRENT_URL, params=params)
        response.raise_for_status()
        body = response.json()
    finally:
        if owns_client:
            client.close()
    if not isinstance(body, dict) or "features" not in body:
        raise ValueError(f"unexpected WFIGS response: {str(body)[:200]}")
    return [item for item in (parse_incident(feature) for feature in body["features"]) if item]


def official_source_for(incident_id: int) -> dict[str, Any] | None:
    """The agency record attached to an incident, in the shape the public API returns (None when there is none)."""
    with _connect() as connection:
        row = connection.execute(
            "SELECT name, agency, acres, percent_contained, discovered_at, first_seen_at, ended_at "
            "FROM fire_incident_official WHERE incident_id = ? ORDER BY ended_at IS NOT NULL, last_seen_at DESC LIMIT 1",
            (incident_id,),
        ).fetchone()
    if row is None:
        return None
    return {
        "name": row["name"],
        "agency": row["agency"],
        "agency_name": "U.S. Forest Service" if row["agency"] == "USFS" else row["agency"],
        "acres": row["acres"],
        "percent_contained": row["percent_contained"],
        "active": row["ended_at"] is None,
        "source": "NIFC WFIGS",
    }


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(get_db_path(), timeout=15)
    connection.row_factory = sqlite3.Row
    return connection


def _acres_text(acres: float | None) -> str:
    return "size not reported" if acres is None else f"{acres:g} acre{'' if acres == 1 else 's'}"


def _meaningfully_different(before: float | None, after: float | None) -> bool:
    if after is None or before == after:
        return False
    if before is None:
        return True
    return abs(after - before) >= max(ACRES_STEP, ACRES_RELATIVE_STEP * before)


def _ingest(incident: dict[str, Any]) -> tuple[int, int | None]:
    """Store/refresh the fire_events row and return (event_id, incident_id)."""
    from services.county_lookup import county_for_point

    county_fips, county_name = incident["county_fips"], incident["county_name"]
    if county_fips is None:
        county_fips, looked_up = county_for_point(incident["latitude"], incident["longitude"])
        county_name = county_name or looked_up
    discovered = incident["discovered_at"] or incident["listed_at"] or datetime.now(timezone.utc)
    result = upsert_detection_event(
        source=SOURCE,
        external_id=incident["irwin_id"],
        latitude=incident["latitude"],
        longitude=incident["longitude"],
        occurred_at=_iso(discovered),
        county_fips=county_fips,
        county_name=county_name,
        verification_tier="official_source_confirmed",
        cause_category="wildfire",
        acres=incident["acres"],
        official_source_system=SYSTEM_NAME,
        official_source_ref=incident["irwin_id"],
        origin={"channel": "WFIGS_Incident_Locations_Current", "provider_ref": incident["irwin_id"],
                "raw": {"name": incident["name"], "acres": incident["acres"], "county": county_name}},
    )
    with _connect() as connection:
        row = connection.execute("SELECT incident_id FROM fire_events WHERE id = ?", (result["event_id"],)).fetchone()
    return result["event_id"], (row["incident_id"] if row else None)


def sync_official_incidents(now: datetime | None = None) -> int:
    """Pull the feed, update incidents and their timelines, and push new listings. Never raises."""
    now = now or datetime.now(timezone.utc)
    try:
        incidents = fetch_usfs_missouri_incidents()
    except Exception as error:  # noqa: BLE001 - logged, retried next cycle
        logger.warning("USFS incident feed unavailable: %s", error)
        return 0
    try:
        return _apply(incidents, now)
    except Exception:  # noqa: BLE001
        logger.exception("USFS incident sync failed")
        return 0


def _apply(incidents: list[dict[str, Any]], now: datetime) -> int:
    from services.mobile_push import notify_official_listing, record_event

    with _connect() as connection:
        first_run = connection.execute(
            "SELECT 1 FROM mobile_push_events WHERE event_key = ?", (BASELINE_KEY,)
        ).fetchone() is None
        known = {row["irwin_id"]: dict(row) for row in connection.execute("SELECT * FROM fire_incident_official")}

    seen: set[str] = set()
    for incident in incidents:
        irwin_id = incident["irwin_id"]
        seen.add(irwin_id)
        event_id, incident_id = _ingest(incident)
        before = known.get(irwin_id)
        with _connect() as connection:
            if before is None:
                stale = (
                    incident["discovered_at"] is not None and now - incident["discovered_at"] > MAX_DISCOVERY_AGE
                )
                connection.execute(
                    "INSERT INTO fire_incident_official (irwin_id, incident_id, event_id, name, acres, percent_contained, "
                    "discovered_at, first_seen_at, last_seen_at, notified_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (irwin_id, incident_id, event_id, incident["name"], incident["acres"], incident["percent_contained"],
                     _iso(incident["discovered_at"]), now.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d %H:%M:%S"),
                     _iso(now) if (first_run or stale) else None),
                )
                if incident_id is not None:
                    incident_history.record(
                        connection, incident_id, "official",
                        f"Listed by the U.S. Forest Service as the {incident['name']} ({_acres_text(incident['acres'])})",
                        occurred_at=incident["listed_at"] or now,
                        data={"name": incident["name"], "acres": incident["acres"], "agency": "USFS"},
                        dedupe_key=f"official:{irwin_id}:listed",
                    )
            else:
                connection.execute(
                    "UPDATE fire_incident_official SET incident_id = COALESCE(?, incident_id), name = ?, "
                    "last_seen_at = CURRENT_TIMESTAMP, ended_at = NULL WHERE irwin_id = ?",
                    (incident_id, incident["name"], irwin_id),
                )
                target = incident_id or before["incident_id"]
                if target is not None:
                    if before["name"] != incident["name"]:
                        incident_history.record(
                            connection, target, "official",
                            f"Name updated from {before['name']} to {incident['name']}", occurred_at=now,
                            data={"name": incident["name"]}, dedupe_key=f"official:{irwin_id}:name:{incident['name']}",
                        )
                    if _meaningfully_different(before["acres"], incident["acres"]):
                        incident_history.record(
                            connection, target, "official",
                            f"Size updated from {_acres_text(before['acres'])} to {_acres_text(incident['acres'])}",
                            occurred_at=now, data={"acres": incident["acres"]},
                            dedupe_key=f"official:{irwin_id}:acres:{incident['acres']}",
                        )
                        connection.execute("UPDATE fire_events SET acres = ? WHERE id = ?", (incident["acres"], event_id))
                    pct = incident["percent_contained"]
                    if pct is not None and pct != before["percent_contained"]:
                        incident_history.record(
                            connection, target, "official", f"Containment updated to {pct:g}%", occurred_at=now,
                            data={"percent_contained": pct}, dedupe_key=f"official:{irwin_id}:contained:{pct:g}",
                        )
                if _meaningfully_different(before["acres"], incident["acres"]):
                    connection.execute("UPDATE fire_incident_official SET acres = ? WHERE irwin_id = ?",
                                       (incident["acres"], irwin_id))
                connection.execute("UPDATE fire_incident_official SET percent_contained = ? WHERE irwin_id = ?",
                                   (incident["percent_contained"], irwin_id))

    # Records that dropped off the active list: contained/out, or taken down.
    with _connect() as connection:
        for irwin_id, row in known.items():
            if irwin_id in seen or row["ended_at"] is not None:
                continue
            connection.execute("UPDATE fire_incident_official SET ended_at = ? WHERE irwin_id = ?", (_iso(now), irwin_id))
            if row["incident_id"] is not None:
                incident_history.record(
                    connection, row["incident_id"], "official",
                    "No longer on the Forest Service's active incident list (usually means contained or out)",
                    occurred_at=now, dedupe_key=f"official:{irwin_id}:ended:{_iso(now)[:13]}",
                )

    if first_run:
        record_event(BASELINE_KEY, "fire_detection", {"seeded": len(incidents)})
        return 0
    return notify_official_listing(now)
