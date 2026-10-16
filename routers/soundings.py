"""Balloon soundings in and around Missouri (see maps/sounding_maps.py).

GET /api/soundings                       every site with its kept soundings (same as images/soundings.json)
GET /api/soundings/{site}                one site; `site` is the station id or short name (KSGF / sgf, USCOU / cou)
GET /api/soundings/{site}/{valid}        a full archived sounding (profile + metrics); valid = YYYYMMDDHHMM UTC
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException

from services import soundings as snd

router = APIRouter(prefix="/api/soundings", tags=["soundings"])

PROJECT_DIR = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_DIR / "images" / "soundings.json"
ARCHIVE_DIR = PROJECT_DIR / "data" / "soundings"


def _manifest() -> dict:
    try:
        return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise HTTPException(status_code=404, detail="No soundings yet")


def _station_id(site: str) -> str:
    site = site.strip()
    for station in snd.STATIONS:
        if site.upper() == station or site.lower() == snd.site_slug(station):
            return station
    raise HTTPException(status_code=404, detail=f"Unknown sounding site {site!r}")


@router.get("")
def all_sites():
    return _manifest()


@router.get("/{site}")
def one_site(site: str):
    station = _station_id(site)
    entry = _manifest().get("stations", {}).get(station)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No soundings for {station}")
    return {"station": station, **entry}


@router.get("/{site}/{valid}")
def one_sounding(site: str, valid: str):
    station = _station_id(site)
    try:
        when = datetime.strptime(valid, "%Y%m%d%H%M")
    except ValueError:
        raise HTTPException(status_code=422, detail="valid must be YYYYMMDDHHMM (UTC)")
    path = ARCHIVE_DIR / f"{when:%Y}" / f"{when:%m}" / f"{station}_{valid}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise HTTPException(status_code=404, detail=f"No archived sounding for {station} at {valid}")
