"""Correlates a fire_events row's county with the most recently published
county-level fire-weather-danger summary (services/ensemble_fire_danger/
runtime.py), so the detection-confidence model and the v2 API can report how
dangerous conditions were where a detection/report occurred - goal 4's
"correlate reported fires with the day's forecast/observed data".

Reads the latest run's evidence JSON directly rather than re-running any
model. Falls back to (None, None) if no run has published yet, or the
county isn't present in that run's grid - this is contextual enrichment,
never a hard dependency for scoring or ingest.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def _latest_evidence_path() -> Optional[Path]:
    from services.ensemble_fire_danger.runtime import EVIDENCE_ROOT

    if not EVIDENCE_ROOT.exists():
        return None
    candidates = sorted(EVIDENCE_ROOT.glob("*.ensemble_fire_danger.json"))
    return candidates[-1] if candidates else None


@lru_cache(maxsize=1)
def _load_evidence(path_str: str, mtime: float) -> Optional[dict]:
    try:
        return json.loads(Path(path_str).read_text())
    except Exception:
        logger.exception("weather_context: failed to read %s", path_str)
        return None


def _current_record() -> Optional[dict]:
    path = _latest_evidence_path()
    if path is None:
        return None
    return _load_evidence(str(path), path.stat().st_mtime)


def county_fire_danger_today(county_fips: Optional[str]) -> Tuple[Optional[str], Optional[float]]:
    """Returns (category_label, probability) for the latest published
    ensemble fire-danger run at this county, or (None, None) if unavailable.
    category_label is one of "below_moderate", "moderate", "elevated",
    "critical", "extreme"."""
    if not county_fips:
        return None, None
    record = _current_record()
    if not record:
        return None, None
    track = (record.get("tracks") or {}).get(record.get("primary_track") or "")
    if not track:
        return None, None
    categorical = (track.get("county_max_categorical") or {}).get(county_fips)
    if categorical is None:
        return None, None

    from services.ensemble_fire_danger.core import CATEGORY_KEYS

    categorical = int(round(categorical))
    category_key = CATEGORY_KEYS.get(categorical)
    probability = None
    if category_key:
        probability = ((track.get("county_max_neighborhood_probability") or {}).get(category_key) or {}).get(county_fips)
    return category_key or "below_moderate", probability
