#!/usr/bin/env python3
"""Backfill the rolling one-year Synoptic FM archive in resumable UTC days."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from services.archive_bundler import ROOT_DIR
from services.synoptic import SYNOPTIC_API_TOKEN

API = "https://api.synopticdata.com/v2/stations"
STATES = "MO,OK,AR,TN,KY,IL,IA,NE,KS"
BBOX = (-96.8, -88.1, 34.8, 41.8)  # west, east, south, north
VARS = "fuel_moisture,relative_humidity,air_temp,wind_speed,wind_gust,solar_radiation,precip_accum"
MAX_STATION_HOURS = 100_000


def _within(station):
    try:
        lon, lat = float(station["LONGITUDE"]), float(station["LATITUDE"])
        west, east, south, north = BBOX
        return west <= lon <= east and south <= lat <= north
    except (KeyError, TypeError, ValueError):
        return False


async def _get(session, service, params):
    for attempt in range(4):
        try:
            async with session.get(f"{API}/{service}", params=params, timeout=180) as response:
                body = await response.json()
                if response.status == 429 or response.status >= 500:
                    raise RuntimeError(f"retryable HTTP {response.status}")
                response.raise_for_status()
                if int(body.get("SUMMARY", {}).get("RESPONSE_CODE", 1)) not in (1, 2):
                    raise RuntimeError(f"Synoptic error: {body.get('SUMMARY')}")
                return body
        except Exception:
            if attempt == 3:
                raise
            await asyncio.sleep(2 ** attempt)


async def discover_stations(session):
    body = await _get(session, "metadata", {
        "token": SYNOPTIC_API_TOKEN, "state": STATES, "network": "2",
        "vars": "fuel_moisture", "complete": "1", "sitinghistory": "1",
        "status": "active,inactive", "obtimezone": "UTC",
    })
    return [station for station in body.get("STATION", []) if _within(station)]


def _dedupe_station(station):
    obs = station.get("OBSERVATIONS", {})
    times = obs.get("date_time", [])
    keep = []
    seen = set()
    for index, stamp in enumerate(times):
        if stamp not in seen:
            seen.add(stamp)
            keep.append(index)
    for key, values in list(obs.items()):
        if isinstance(values, list) and len(values) == len(times):
            obs[key] = [values[i] for i in keep]
    return station


async def fetch_day(session, day, station_ids):
    start = day.strftime("%Y%m%d0000")
    end = (day + timedelta(days=1) - timedelta(minutes=1)).strftime("%Y%m%d%H%M")
    max_stations = max(1, MAX_STATION_HOURS // 24)
    chunks = [station_ids[i:i + min(250, max_stations)] for i in range(0, len(station_ids), min(250, max_stations))]
    combined = {}
    summaries = []
    for chunk in chunks:
        body = await _get(session, "timeseries", {
            "token": SYNOPTIC_API_TOKEN, "stid": ",".join(chunk), "start": start, "end": end,
            "vars": VARS, "units": "metric", "obtimezone": "UTC", "complete": "1", "qc": "on",
        })
        summaries.append(body.get("SUMMARY", {}))
        for station in body.get("STATION", []):
            combined[station.get("STID") or station.get("ID")] = _dedupe_station(station)
    return {
        "SUMMARY": {"NUMBER_OF_OBJECTS": len(combined), "chunk_summaries": summaries},
        "STATION": list(combined.values()),
        "_metadata": {"start_utc": start, "end_utc": end, "vars": VARS, "bbox": BBOX, "fetched_at": datetime.now(timezone.utc).isoformat()},
    }


def _write_json_atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


async def run(args):
    if not SYNOPTIC_API_TOKEN:
        raise RuntimeError("SYNOPTIC_API_TOKEN is not configured")
    archive = Path(args.archive_dir or ROOT_DIR / "archive" / "raw_data")
    manifest_path = archive / "synoptic_backfill_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"days": {}, "version": 1}
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc) if args.end else datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc) if args.start else end - timedelta(days=365)

    async with aiohttp.ClientSession() as session:
        stations = await discover_stations(session)
        station_ids = [station["STID"] for station in stations if station.get("STID")]
        print(f"Synoptic backfill {start.date()} through {end.date()}: {len(station_ids)} FM stations")
        if args.dry_run:
            return
        _write_json_atomic(archive / "synoptic_station_metadata.json", {"STATION": stations, "fetched_at": datetime.now(timezone.utc).isoformat()})
        day = start
        while day <= end:
            key = day.strftime("%Y%m%d")
            target = archive / f"raw_data_{key}.json"
            if manifest["days"].get(key, {}).get("status") == "complete" and target.exists() and not args.force:
                day += timedelta(days=1)
                continue
            try:
                body = await fetch_day(session, day, station_ids)
                _write_json_atomic(target, body)
                today = datetime.now(timezone.utc).date()
                status = "partial" if day.date() >= today else "complete"
                manifest["days"][key] = {"status": status, "stations": len(body["STATION"]), "updated_at": datetime.now(timezone.utc).isoformat()}
            except Exception as exc:
                logging.exception("backfill failed for %s", key)
                manifest["days"][key] = {"status": "failed", "error": str(exc), "updated_at": datetime.now(timezone.utc).isoformat()}
            _write_json_atomic(manifest_path, manifest)
            day += timedelta(days=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", help="YYYY-MM-DD; default one year ago")
    parser.add_argument("--end", help="YYYY-MM-DD; default today UTC")
    parser.add_argument("--archive-dir")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
