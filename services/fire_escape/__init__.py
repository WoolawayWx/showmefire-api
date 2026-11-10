"""Fire escape maps: three experimental views of the day's fire danger, from the HRRR forecast.

SHADOW ONLY - NOT FOR OPERATIONS. Comparison products shown on the Testbed behind its acknowledgement; nothing
here feeds the public forecast.

1. Physical Fire Danger Index (services/fire_weather/physical_danger.py): no training at all. The flame
   length a head fire would have in cured grass at the afternoon's peak wind, as a level 0 Low .. 4 Extreme.
2. If a fire starts here (model `fire`, trained on 23,000 Missouri wildfires 1992-2020): the chance it
   reaches 10 acres, and 100 acres. Per grid cell; it knows the fuels around the cell.
3. A fire that gets away, in this county, today (model `county`, trained on every county-day 1992-2020):
   the chance a fire of 10 acres or more, and of 100 acres or more, is reported. The chance a fire starts
   AND grows; weather only.

Inputs: the 12z HRRR file the daily forecast caches (forecast hours 4-15), and gridMET for the rain of the
last 30 days and the slow fuel moisture (fetched, about a day behind). The models were fitted on gridMET and
airport observations, so HRRR is mapped onto those features first (hrrr_day.py; the fit is in the bundle).

features.py, hrrr_day.py and predict.py are byte-identical mirrors of model-training/fire_escape/
(core/contract_mirrors.json). The trained release is published from model-training (`python -m fire_escape.publish`)
and installed here with `python scripts/install_fire_escape.py --latest` (install.py), into DATA_DIR/models/
fire_escape_release. No training data lives on the server. Without a release the job reports "no bundle" and stops.
"""
from __future__ import annotations

import os
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
MODEL = "fire_escape_county"            # the tag fire danger tracking scores (services/fire_danger_tracking.py)


def enabled() -> bool:
    return os.getenv("SMF_FIRE_ESCAPE_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")


def bundle_dir() -> Path:
    return Path(os.getenv("SMF_FIRE_ESCAPE_BUNDLE") or DATA_DIR / "models" / "fire_escape_release")


def evidence_dir() -> Path:
    """Run records, grids and every day's map images. The Testbed only gets a copy of the latest day's maps."""
    return Path(os.getenv("SMF_FIRE_ESCAPE_EVIDENCE_ROOT") or DATA_DIR / "model-shadow" / "fire-escape")


def gridmet_cache_dir() -> Path:
    return Path(os.getenv("SMF_FIRE_ESCAPE_GRIDMET_CACHE") or DATA_DIR / "cache" / "gridmet")


def hrrr_cache_dir() -> Path:
    return Path(os.getenv("SMF_HRRR_CACHE_DIR") or "/app/cache/hrrr")
