"""NWS alert event -> Oklahoma-Mesonet-style advisory category mapping.

Pure functions, no I/O - operates on the same enriched alerts GeoJSON that
alerts/activemoalerts.py's run_active_mo_alerts() already refreshes every 5
minutes (core/scheduler.py's fetch_active_mo_alerts job) and writes to
gis/active.json. Reused by maps/advisory_category_maps.py to split that one
combined feed into 8 separate category maps, at zero extra external calls.
"""
from __future__ import annotations

from typing import Optional

# Keyword matching is deliberately case-insensitive substring matching
# against the NWS "event" string (e.g. "Red Flag Warning", "Heat Advisory"),
# not an exhaustive enum - NWS event names are a loosely-governed free-text
# list and new/rare event types should fail safely (uncategorized) rather
# than raise.
CATEGORY_EVENT_KEYWORDS: dict[str, list[str]] = {
    "severe_weather": ["severe thunderstorm", "tornado", "special weather statement"],
    "flood": ["flood"],
    "fire": ["red flag", "fire weather"],
    "wind": ["wind advisory", "high wind"],
    "heat": ["heat advisory", "excessive heat"],
    "winter_weather": ["winter storm", "winter weather", "ice storm", "snow"],
    "frost_freeze": ["frost advisory", "freeze warning", "freeze watch"],
    "visibility": ["dense fog", "smoke", "dust"],
}

CATEGORY_LABELS: dict[str, str] = {
    "severe_weather": "Severe Weather",
    "flood": "Flood",
    "fire": "Fire",
    "wind": "Wind",
    "heat": "Heat",
    "winter_weather": "Winter Weather",
    "frost_freeze": "Frost/Freeze",
    "visibility": "Visibility",
}


def categorize_event(event_name: Optional[str]) -> Optional[str]:
    """Map an NWS alert 'event' string to one of CATEGORY_EVENT_KEYWORDS'
    keys, or None if it doesn't match any known category."""
    if not event_name:
        return None
    lowered = event_name.lower()
    for category, keywords in CATEGORY_EVENT_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return category
    return None


def filter_alerts_by_category(alerts_geojson: dict, category: str) -> dict:
    """Return a new FeatureCollection containing only the features whose
    'event' property maps to `category`. Never mutates the input."""
    features = alerts_geojson.get("features") or []
    matched = [
        feature for feature in features
        if categorize_event((feature.get("properties") or {}).get("event")) == category
    ]
    return {**alerts_geojson, "features": matched}
