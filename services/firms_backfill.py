"""Backfill VIIRS/MODIS detections from before the live feed (and this DB) started.

The live poller (tools/firedetections.py) only reads FIRMS' last day of near-real-time data, and fire_events only
exists from when ingest was switched on. FIRMS also serves the Standard Processing (SP) archive through the same
area API, up to 5 days per request: .../api/area/csv/{MAP_KEY}/{PRODUCT}/{west,south,east,north}/{days}/{YYYY-MM-DD}.

The backfill NEVER writes to the database. It fetches, normalises with the live poller's own parser, and writes a flat
CSV (data/firms_backfill/firms_history_<start>_<end>.csv). That CSV is compared with NERIS on the training side
(model-training: `python -m detection_dataset.overlap`), and only the overlapping rows come back to be added to
fire_events with `ingest_csv` (scripts/ingest_firms_history_csv.py), detections only, no incidents.

An SP row is given the id its NRT twin would have had (`VIIRS_SNPP_NRT:...`), so a detection that exists in both is one
row, and an ingested row lands on (or fills in) the same fire_events row the live poller would have written.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import requests

from core.config import DATA_DIR
from tools.firedetections import MISSOURI_BBOX_PARAM, _dedup_key, _is_missouri_feature, _normalize_csv_row, _parse_csv_rows

logger = logging.getLogger(__name__)

FIRMS_AREA_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{product}/{bbox}/{days}/{day}"
MAX_DAYS_PER_REQUEST = 5  # FIRMS' limit on DAY_RANGE

# Standard Processing archive product -> the near-real-time product whose ids it must share.
ARCHIVE_PRODUCTS = {
    "VIIRS_SNPP_SP": "VIIRS_SNPP_NRT",
    "VIIRS_NOAA20_SP": "VIIRS_NOAA20_NRT",
    "VIIRS_NOAA21_SP": "VIIRS_NOAA21_NRT",
    "MODIS_SP": "MODIS_NRT",
}
NRT_PRODUCTS = tuple(ARCHIVE_PRODUCTS.values())
# Archive first: when both carry a detection the better-processed SP record wins the dedup.
DEFAULT_PRODUCTS = tuple(ARCHIVE_PRODUCTS) + NRT_PRODUCTS
BACKFILL_DIR = DATA_DIR / "firms_backfill"


class FirmsError(RuntimeError):
    pass


def day_windows(start: date, end: date, size: int = MAX_DAYS_PER_REQUEST) -> List[Tuple[date, int]]:
    """Cover [start, end] inclusive with (first_day, n_days) windows of at most `size` days."""
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    windows, cursor = [], start
    while cursor <= end:
        n = min(size, (end - cursor).days + 1)
        windows.append((cursor, n))
        cursor += timedelta(days=n)
    return windows


def _map_key() -> str:
    key = os.getenv("MODIS_KEY", "").strip().strip("'").strip('"')  # the same FIRMS key the live poller uses
    if not key:
        raise FirmsError("MODIS_KEY (the FIRMS MAP_KEY) is not set")
    return key


def _rows_from_response(text: str, product: str) -> List[Dict[str, Any]]:
    """FIRMS answers a bad key, an unknown product or an exhausted quota with a plain-text body and HTTP 200."""
    body = (text or "").lstrip("﻿").strip()
    if not body:
        return []
    if "latitude" not in body.splitlines()[0].lower():
        raise FirmsError(f"{product}: {body[:160]}")
    return _parse_csv_rows(body)


def _get(url: str, key: str, fetch: Callable, attempts: int, sleep: Callable) -> str:
    for attempt in range(1, attempts + 1):
        try:
            response = fetch(url, timeout=60)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as exc:
            if attempt == attempts:
                raise FirmsError(f"FIRMS request failed: {str(exc).replace(key, '[redacted]')}") from None  # the key is in the URL
            sleep(2 ** attempt)
    return ""


def fetch_history(
    start: date,
    end: date,
    products: Iterable[str] = DEFAULT_PRODUCTS,
    *,
    key: Optional[str] = None,
    bbox: str = MISSOURI_BBOX_PARAM,
    fetch: Callable = requests.get,
    sleep: Callable = time.sleep,
    pause_s: float = 0.25,
    attempts: int = 3,
    progress: Optional[Callable[[str, int, int, int], None]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Fetch every product over [start, end]. Returns (deduplicated Missouri features, per-product stats)."""
    key = key or _map_key()
    features: List[Dict[str, Any]] = []
    seen: set = set()
    stats: Dict[str, Any] = {"requests": 0, "per_product": {}, "duplicates": 0}
    for product in products:
        nrt_name = ARCHIVE_PRODUCTS.get(product, product)
        kept = 0
        windows = day_windows(start, end)
        for index, (first_day, days) in enumerate(windows, 1):
            url = FIRMS_AREA_URL.format(key=key, product=product, bbox=bbox, days=days, day=first_day.isoformat())
            rows = _rows_from_response(_get(url, key, fetch, attempts, sleep), product)
            stats["requests"] += 1
            for row in rows:
                feature = _normalize_csv_row(row, nrt_name)  # nrt_name => the same SOURCE_ID the live poller would write
                if not feature or not _is_missouri_feature(feature):
                    continue
                dedup = _dedup_key(feature)
                if dedup in seen:
                    stats["duplicates"] += 1
                    continue
                seen.add(dedup)
                features.append(feature)
                kept += 1
            if progress:
                progress(product, index, len(windows), kept)
            sleep(pause_s)
        stats["per_product"][product] = kept
        logger.info("firms_backfill: %s %s..%s kept %d", product, start, end, kept)
    return features, stats


# Column names match fire_events, so the same CSV reads naturally in the training repo and rebuilds a feed feature here.
HISTORY_COLUMNS = ["external_id", "source", "product", "satellite", "occurred_at", "latitude", "longitude", "frp",
                   "confidence", "bright_t7", "bright_t13", "pixel_area", "daynight"]


def feature_to_row(feature: Dict[str, Any]) -> Dict[str, Any]:
    props = feature.get("properties") or {}
    lon, lat = feature["geometry"]["coordinates"][:2]
    typename = str(props.get("TYPENAME") or "")
    return {
        "external_id": props.get("SOURCE_ID"), "source": "modis" if "MODIS" in typename.upper() else "viirs", "product": typename,
        "satellite": props.get("SATELLITE"), "occurred_at": props.get("ACQ_DATE_TIME"), "latitude": lat, "longitude": lon,
        "frp": props.get("FRP"), "confidence": props.get("CONFIDENCE"),
        "bright_t7": props.get("BRIGHT_MID"), "bright_t13": props.get("BRIGHT_LONG"),  # the ~4 um / ~11 um channels
        "pixel_area": props.get("PIXEL_AREA"), "daynight": props.get("DAYNIGHT"),
    }


def _num(value: Any) -> Optional[float]:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def row_to_feature(row: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild the feed feature fire_ingest._ingest_satdet_feature expects from a history-CSV row (extra columns ignored)."""
    lat, lon = float(row["latitude"]), float(row["longitude"])
    return {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]}, "properties": {
        "SOURCE": "FIRMS", "SOURCE_ID": row["external_id"], "TYPENAME": row.get("product") or "", "SATELLITE": row.get("satellite") or None,
        "ACQ_DATE_TIME": row["occurred_at"], "LATITUDE": lat, "LONGITUDE": lon, "FRP": _num(row.get("frp")),
        "CONFIDENCE": row.get("confidence") or None, "BRIGHT_MID": _num(row.get("bright_t7")), "BRIGHT_LONG": _num(row.get("bright_t13")),
        "PIXEL_AREA": _num(row.get("pixel_area")), "DAYNIGHT": row.get("daynight") or None}}


def backfill(
    start: date,
    end: date,
    *,
    products: Iterable[str] = DEFAULT_PRODUCTS,
    dry_run: bool = False,
    key: Optional[str] = None,
    out_dir: Optional[Path] = None,
    fetch: Callable = requests.get,
    sleep: Callable = time.sleep,
    progress: Optional[Callable[[str, int, int, int], None]] = None,
) -> Dict[str, Any]:
    """Fetch [start, end] from FIRMS and write them to a history CSV. Never touches the database."""
    import csv
    today = datetime.now(timezone.utc).date()
    end = min(end, today)
    features, stats = fetch_history(start, end, products, key=key, fetch=fetch, sleep=sleep, progress=progress)
    features.sort(key=lambda f: (f.get("properties") or {}).get("ACQ_DATE_TIME") or "")
    summary: Dict[str, Any] = {"start": start.isoformat(), "end": end.isoformat(), "fetched": len(features), "dry_run": dry_run, **stats}
    if dry_run or not features:
        return summary
    out_dir = out_dir or BACKFILL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"firms_history_{start.isoformat()}_{end.isoformat()}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        writer.writerows(feature_to_row(feature) for feature in features)
    summary["csv"] = str(path)
    return summary


def ingest_csv(csv_path: Path, *, cluster_incidents: bool = False, dry_run: bool = False) -> Dict[str, Any]:
    """Add the detections in a history CSV (typically the NERIS-overlap subset) to fire_events.

    Detections only by default: no incidents are created, so old history never reaches the review queue or the map.
    Idempotent: a detection already stored is updated in place, not duplicated."""
    import csv
    import tempfile
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    features = [row_to_feature(row) for row in rows]
    features.sort(key=lambda f: f["properties"]["ACQ_DATE_TIME"])
    if dry_run:
        return {"rows": len(rows), "dry_run": True}
    from services.fire_ingest import ingest_detection_files  # lazy: pulls in the DB layer
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"{Path(csv_path).stem}.geojson"  # the file name is recorded as the detection's source channel
        path.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
        result = ingest_detection_files(paths={"satdet": path, "ngfs": Path(tmp) / "_none.geojson"}, cluster_incidents=cluster_incidents)
    return {"rows": len(rows), "dry_run": False, "clustered_into_incidents": cluster_incidents, "ingest": result}
