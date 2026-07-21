import json
import re
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable
from urllib.parse import urlencode

import shapefile

from core.database import get_briefing_config, get_latest_forecast

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ACTIVE_ALERTS_PATH = PROJECT_ROOT / "gis" / "active.json"
FIRE_ZONES_PATH = PROJECT_ROOT / "gis" / "MOFireWxZones.geojson"
COUNTIES_SHP_PATH = PROJECT_ROOT / "maps" / "shapefiles" / "MO_County_Boundaries" / "MO_County_Boundaries.shp"
OPSBRIEF_DIR = PROJECT_ROOT / "files" / "opsbrief"
FIRE_EVENTS = {"Red Flag Warning", "Fire Weather Watch"}
PUBLIC_API_BASE_URL = "https://api.showmefire.org"
PUBLIC_CDN_BASE_URL = "https://cdn.showmefire.org/latest"

FORECAST_MAPS = [
    ("fire-danger", "Peak Fire Danger", "mo-forecastfiredanger.png"),
    ("fuel-moisture", "Minimum Fuel Moisture", "mo-forecastfuelmoisture.png"),
    ("minimum-rh", "Minimum Relative Humidity", "mo-forecastminrh.png"),
    ("maximum-temperature", "Maximum Temperature", "mo-forecastmaxtemp.png"),
    ("maximum-wind", "Maximum Wind", "mo-forecastmaxwind.png"),
    ("rainfall", "Forecast Rainfall", "mo-forecastrainfall.png"),
    ("snow-water-equivalent", "Snow Water Equivalent", "mo-forecastswe.png"),
]


def _normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower().replace("county", ""))


@lru_cache(maxsize=1)
def county_catalog() -> list[dict[str, str]]:
    reader = shapefile.Reader(str(COUNTIES_SHP_PATH))
    fields = [field[0] for field in reader.fields[1:]]
    records: list[dict[str, str]] = []
    for record in reader.records():
        row = dict(zip(fields, record))
        fips = f"29{str(row.get('COUNTYFIPS') or '').zfill(3)}"
        records.append({"fips": fips, "name": str(row.get("COUNTYNAME") or "").strip()})
    return sorted(records, key=lambda item: item["name"])


@lru_cache(maxsize=1)
def _zone_to_county_fips() -> dict[str, str]:
    name_to_fips = {_normalized_name(item["name"]): item["fips"] for item in county_catalog()}
    try:
        zones = json.loads(FIRE_ZONES_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    result: dict[str, str] = {}
    for feature in zones.get("features", []):
        props = feature.get("properties") or {}
        zone = str(props.get("ZONE") or "").zfill(3)
        fips = name_to_fips.get(_normalized_name(props.get("NAME") or ""))
        if zone and fips:
            result[zone] = fips
    return result


def county_fips_for_alert(properties: Dict[str, Any]) -> list[str]:
    result: set[str] = set()
    geocode = properties.get("geocode") or {}
    for same in geocode.get("SAME") or []:
        value = str(same)
        if len(value) == 6 and value.startswith("029"):
            result.add(f"29{value[-3:]}")

    zone_map = _zone_to_county_fips()
    ugc_values: Iterable[str] = geocode.get("UGC") or []
    affected = [str(url).rsplit("/", 1)[-1] for url in properties.get("affectedZones") or []]
    for raw in [*ugc_values, *affected]:
        match = re.fullmatch(r"MOZ(\d{3})", str(raw).upper())
        if match and match.group(1) in zone_map:
            result.add(zone_map[match.group(1)])
    return sorted(result)


def _parse_time(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def active_fire_weather_alerts(
    path: Path = ACTIVE_ALERTS_PATH,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    now = now or datetime.now(timezone.utc)
    try:
        document = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return []

    alerts: list[dict[str, Any]] = []
    for feature in document.get("features", []):
        properties = feature.get("properties") or {}
        if properties.get("event") not in FIRE_EVENTS:
            continue
        expires = _parse_time(properties.get("ends") or properties.get("expires"))
        if expires and expires <= now:
            continue
        alert_id = str(properties.get("id") or properties.get("@id") or "").strip()
        if not alert_id:
            continue
        alerts.append({
            "id": alert_id,
            "event": str(properties.get("event") or "Fire Weather Alert"),
            "headline": str(properties.get("headline") or properties.get("event") or "Fire Weather Alert"),
            "severity": str(properties.get("severity") or "Unknown"),
            "onset": properties.get("onset") or properties.get("effective"),
            "expires": properties.get("ends") or properties.get("expires"),
            "areaDescription": str(properties.get("areaDesc") or "Missouri"),
            "description": str(properties.get("description") or ""),
            "instruction": properties.get("instruction"),
            "countyFips": county_fips_for_alert(properties),
            "sent": properties.get("sent"),
        })
    return sorted(alerts, key=lambda alert: str(alert.get("onset") or ""))


def _active_opsbrief(api_base_url: str) -> dict[str, Any]:
    config = get_briefing_config() or {}
    file_name = Path(str(config.get("file_path") or "")).name
    active = bool(config.get("is_active")) and bool(file_name) and (OPSBRIEF_DIR / file_name).exists()
    expires = _parse_time(config.get("expires_at"))
    if expires and expires <= datetime.now(timezone.utc):
        active = False
    return {
        "active": active,
        "title": config.get("title") if active else None,
        "url": f"{api_base_url}/api/opsbrief/view" if active else None,
    }


def build_mobile_content(api_base_url: str = PUBLIC_API_BASE_URL, cdn_base_url: str = PUBLIC_CDN_BASE_URL) -> dict[str, Any]:
    api_base_url = api_base_url.rstrip("/")
    cdn_base_url = cdn_base_url.rstrip("/")
    forecast = get_latest_forecast()
    forecast_payload = None
    if forecast:
        revision = str(forecast.get("updated_at") or forecast.get("id") or forecast.get("valid_time") or "")
        version_query = f"?{urlencode({'v': revision})}" if revision else ""
        forecast_payload = {
            "id": int(forecast["id"]),
            "title": str(forecast.get("title") or "Daily Fire Weather Forecast"),
            "discussion": str(forecast.get("discussion") or ""),
            "validTime": str(forecast.get("valid_time") or ""),
            "updatedAt": str(forecast.get("updated_at") or "") or None,
            "maps": [
                {"key": key, "title": title, "url": f"{cdn_base_url}/{filename}{version_query}"}
                for key, title, filename in FORECAST_MAPS
            ],
        }
    return {
        "updatedAt": datetime.now(timezone.utc).isoformat(),
        "forecast": forecast_payload,
        "outlooks": [
            {"day": day, "title": f"Day {day} Missouri Fire Weather Outlook", "imageUrl": f"{api_base_url}/images/mo-outlook-day{day}.webp"}
            for day in (2, 3)
        ],
        "opsBrief": _active_opsbrief(api_base_url),
        "alerts": [{key: value for key, value in alert.items() if key != "sent"} for alert in active_fire_weather_alerts()],
        "counties": county_catalog(),
    }
