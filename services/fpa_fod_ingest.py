"""
Ingest USFS Fire Program Analysis Fire-Occurrence Database (FPA-FOD)
records for Missouri into the fire_events store as
verification_tier='official_source_confirmed'.

FPA-FOD is a public-domain, federally curated dataset (USDA Forest
Service Rocky Mountain Research Station) of 2.3M US wildfire records,
1992-2020, compiled from federal/state/local reporting systems - see
https://www.fs.usda.gov/rds/archive/catalog/RDS-2013-0009.6. It is
queried live via the USFS ArcGIS REST endpoint rather than downloading
the ~214MB bulk file, since only the Missouri subset is needed.

Why this can land directly at official_source_confirmed rather than
starting at unverified like satellite/NGFS detections: FPA-FOD's scope
is wildfires specifically (unplanned, reportable fire responses), which
structurally excludes prescribed and agricultural burns at the source -
exactly the label contamination the project's cause_filter gate exists
to catch. Individual records still carry their own NWCG cause
classification, mapped to this store's cause_category below.

Two date-handling subtleties, both load-bearing:

1. `discovery_date` is an Esri date field storing ONLY a calendar date,
   encoded as milliseconds since epoch at UTC midnight (e.g.
   1447113600000 -> 2015-11-10T00:00:00Z, verified against the sibling
   `discovery_doy` field in a live sample). Converting that timestamp to
   Central time before extracting the date would shift many records back
   a calendar day (UTC midnight = 6-7pm the PREVIOUS day in Central) -
   the date must be read directly from the UTC value, never localized.
2. `discovery_time` (a separate "HHMM" string field, may be missing) is
   the actual local time of discovery. It is combined with the date above
   and localized to America/Chicago - not UTC - since FPA-FOD's
   discovery_time is documented as local standard/daylight time as
   reported by the fire organization, and Missouri is what's being
   ingested here. When discovery_time is missing, the record gets a noon
   placeholder and occurred_at_precision='day'.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, Iterator, Optional
from zoneinfo import ZoneInfo

import httpx

from core.database import upsert_detection_event
from services.county_lookup import county_for_point

logger = logging.getLogger(__name__)

QUERY_URL = "https://apps.fs.usda.gov/ArcX/rest/services/EDW/EDW_FireOccurrence6thEdition_01/MapServer/29/query"
PAGE_SIZE = 2000
CENTRAL = ZoneInfo("America/Chicago")

OUT_FIELDS = ",".join([
    "fod_id", "fpa_id", "nwcg_reporting_agency", "nwcg_reporting_unit_name",
    "fire_name", "fire_code", "discovery_date", "discovery_time",
    "nwcg_cause_classification", "nwcg_general_cause", "fire_size",
    "latitude", "longitude", "state", "fips_code", "fips_name",
])


def _map_cause(classification: Optional[str], general_cause: Optional[str]) -> str:
    """
    Maps NWCG's cause taxonomy onto this store's coarser cause_category
    enum. The fine-grained NWCG values are not lost - they're preserved
    verbatim in official_source_ref - this mapping only feeds the
    cause_filter gate, which cares about "lightning vs. everything else"
    and "clearly unwanted fire vs. ambiguous", not the full NWCG detail.
    """
    classification = (classification or "").strip().lower()
    general_cause = (general_cause or "").strip().lower()

    if classification == "natural":
        # Documented assumption, not verified per-record: Missouri's
        # natural wildfire ignition is overwhelmingly lightning. If this
        # ever needs to be precise, nwcg_general_cause for natural-class
        # records should be re-checked against the live NWCG taxonomy.
        return "lightning"
    if "debris" in general_cause:
        return "debris_burn"
    if "equipment" in general_cause:
        return "equipment"
    if "arson" in general_cause or "incendiary" in general_cause:
        return "incendiary"
    if "missing" in general_cause or "undetermined" in general_cause or not general_cause:
        return "unknown"
    if classification == "human":
        return "wildfire"
    return "unknown"


def _occurred_at(discovery_date_ms: Optional[int], discovery_time: Optional[str]) -> tuple:
    """Returns (occurred_at_iso_utc, precision). See module docstring for the UTC/Central subtlety."""
    if discovery_date_ms is None:
        raise ValueError("discovery_date is required")

    calendar_date = datetime.fromtimestamp(discovery_date_ms / 1000, tz=timezone.utc).date()

    digits = (discovery_time or "").strip()
    if digits.isdigit() and len(digits) == 4:
        hour, minute = int(digits[:2]), int(digits[2:])
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            local_dt = datetime(calendar_date.year, calendar_date.month, calendar_date.day,
                                hour, minute, tzinfo=CENTRAL)
            return local_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "minute"

    local_dt = datetime(calendar_date.year, calendar_date.month, calendar_date.day, 12, 0, tzinfo=CENTRAL)
    return local_dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "day"


def map_record(attributes: Dict) -> Optional[Dict]:
    """Maps one raw FPA-FOD feature's attributes to upsert_detection_event kwargs, or None if unmappable."""
    lat, lon = attributes.get("latitude"), attributes.get("longitude")
    fod_id = attributes.get("fod_id")
    if lat is None or lon is None or fod_id is None:
        return None

    try:
        occurred_at, precision = _occurred_at(attributes.get("discovery_date"), attributes.get("discovery_time"))
    except (ValueError, OverflowError, OSError):
        return None

    fips_code = attributes.get("fips_code")
    fips_name = (attributes.get("fips_name") or "").replace(" County", "").strip() or None
    if not fips_code:
        # FPA-FOD's own county attribution is genuinely missing for a
        # meaningful share of records - verified against the live source
        # (Forest Service-reported fires in particular often lack
        # fips_code/fips_name upstream, likely because they're tracked by
        # national forest unit rather than county). Every record still has
        # lat/lon, so fall back to the same point-in-polygon lookup Track 1
        # built for user submissions rather than leaving the county blank.
        fips_code, resolved_name = county_for_point(float(lat), float(lon))
        fips_name = resolved_name or fips_name
    ref_parts = [
        f"FPA-FOD {attributes.get('fpa_id') or fod_id}",
        attributes.get("nwcg_reporting_unit_name"),
        f"fire_name={attributes['fire_name']}" if attributes.get("fire_name") else None,
    ]
    official_source_ref = " / ".join(part for part in ref_parts if part)

    return {
        "source": "official",
        "external_id": f"fpa_fod:{fod_id}",
        "latitude": float(lat),
        "longitude": float(lon),
        "occurred_at": occurred_at,
        "occurred_at_precision": precision,
        "county_fips": fips_code,
        "county_name": fips_name,
        "verification_tier": "official_source_confirmed",
        "cause_category": _map_cause(attributes.get("nwcg_cause_classification"), attributes.get("nwcg_general_cause")),
        "acres": attributes.get("fire_size"),
        "official_source_system": "USFS FPA-FOD",
        "official_source_ref": official_source_ref,
    }


def fetch_missouri_records(since_year: Optional[int] = None, until_year: Optional[int] = None,
                           page_size: int = PAGE_SIZE, client: Optional[httpx.Client] = None) -> Iterator[Dict]:
    """Paginates through the ArcGIS REST endpoint for Missouri, yielding raw feature attribute dicts."""
    clauses = ["state='MO'"]
    if since_year is not None:
        clauses.append(f"fire_year>={int(since_year)}")
    if until_year is not None:
        clauses.append(f"fire_year<={int(until_year)}")
    where = " AND ".join(clauses)

    owns_client = client is None
    client = client or httpx.Client(timeout=30.0)
    try:
        offset = 0
        while True:
            response = client.get(QUERY_URL, params={
                "where": where,
                "outFields": OUT_FIELDS,
                "resultOffset": offset,
                "resultRecordCount": page_size,
                "orderByFields": "fod_id",
                "f": "json",
            })
            response.raise_for_status()
            body = response.json()
            if "error" in body:
                raise RuntimeError(f"FPA-FOD query error: {body['error']}")
            features = body.get("features", [])
            if not features:
                return
            for feature in features:
                yield feature.get("attributes", {})
            if len(features) < page_size:
                return
            offset += page_size
    finally:
        if owns_client:
            client.close()


def ingest_fpa_fod(since_year: Optional[int] = None, until_year: Optional[int] = None,
                   dry_run: bool = False, client: Optional[httpx.Client] = None) -> Dict:
    """
    Fetches and upserts Missouri FPA-FOD records. Returns
    {"inserted", "updated", "skipped", "would_process", "errors"}.

    dry_run never writes to the database, so it cannot distinguish insert
    from update (that requires checking existing rows) - would_process
    counts every record that mapped successfully and would have been
    upserted, so a dry run still reports a meaningful non-zero number
    instead of always showing 0/0.
    """
    inserted = updated = skipped = would_process = 0
    errors = []

    for attributes in fetch_missouri_records(since_year, until_year, client=client):
        try:
            mapped = map_record(attributes)
            if mapped is None:
                skipped += 1
                continue
            if dry_run:
                would_process += 1
                continue
            result = upsert_detection_event(**mapped)
            if result["inserted"]:
                inserted += 1
            else:
                updated += 1
        except Exception as exc:
            errors.append(f"fod_id={attributes.get('fod_id')}: {exc}")

    summary = {"inserted": inserted, "updated": updated, "skipped": skipped,
               "would_process": would_process, "errors": errors, "dry_run": dry_run}
    logger.info("fpa_fod_ingest: %s", {**summary, "errors": len(errors)})
    return summary
