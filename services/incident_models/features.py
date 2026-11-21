"""Inputs shared by the incident models (persistence.py, incident_type.py).

Everything here is computed the same way when a model is TRAINED on history and when it SERVES a live incident, which is
the only way a model's measured accuracy means anything. Two rules follow from that:

  * Use only what was knowable at the time. The persistence model looks at the first WINDOW_MINUTES of an incident and
    nothing later; the type model uses everything the incident has so far (its labels are applied after the fact).
  * Never read a stored value that is not reliably there. Detections do not store fuel, canopy or weather danger (null on
    nearly every row), so ground cover is sampled from the static product for the place, and weather comes from the
    hourly RTMA analysis for the hour of the first detection. Either can be missing; a missing value is NaN and XGBoost
    learns to cope (the logistic model fills it with the training median).

None of this touches the confidence score that drives alerts (services/incident_confidence.py); these models only add
information beside it.
"""
from __future__ import annotations

import json
import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

SATELLITE_SOURCES = ("viirs", "modis", "ngfs")
WINDOW_MINUTES = 30
SCAN_BUCKET_MINUTES = 10
LOCAL_TZ = ZoneInfo("America/Chicago")
NAN = float("nan")

WEATHER_VARIABLES = ("temp_f", "rh_pct", "wind_speed_mph", "vpd_hpa", "fuel_moisture_pct", "precip_mm_1h")

# Order matters: a trained model remembers it.
WINDOW_FEATURES = (
    "n_w", "scans_w", "frp_max_w", "frp_mean_w", "frp_sum_w", "n_ngfs_w", "n_viirs_w", "n_modis_w",
    "det_conf_max_w", "det_conf_mean_w", "span_km_w",
    "hour_sin", "hour_cos", "month", "weekend",
)
COVER_FEATURES = ("crop", "pasture_grass", "forest", "canopy")
NEIGHBOR_FEATURES = ("n_near_24h", "n_near_7d")
WEATHER_FEATURES = tuple(f"wx_{name}" for name in WEATHER_VARIABLES)
PERSISTENCE_FEATURES = WINDOW_FEATURES + COVER_FEATURES + NEIGHBOR_FEATURES + WEATHER_FEATURES


def parse_time(value: Any) -> Optional[datetime]:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def satellite_members(members: Iterable[dict]) -> list[dict]:
    """Satellite detections with a usable time, oldest first. Reports, leads and agency records are left out: they arrive
    on their own schedule and would make "how the detections behave" mean different things in training and serving."""
    stamped = []
    for member in members:
        if member.get("source") not in SATELLITE_SOURCES:
            continue
        when = parse_time(member.get("occurred_at"))
        if when is not None:
            stamped.append((when, member))
    stamped.sort(key=lambda pair: pair[0])
    return [dict(member, _t=when) for when, member in stamped]


def first_window(sat: list[dict], minutes: int = WINDOW_MINUTES) -> tuple[Optional[datetime], list[dict]]:
    """(time of the first detection, the detections within `minutes` of it)."""
    if not sat:
        return None, []
    t0 = sat[0]["_t"]
    return t0, [m for m in sat if m["_t"] <= t0 + timedelta(minutes=minutes)]


def _km_span(rows: list[dict]) -> float:
    lats = [float(m["latitude"]) for m in rows if m.get("latitude") is not None]
    lons = [float(m["longitude"]) for m in rows if m.get("longitude") is not None]
    if len(lats) < 2:
        return 0.0
    mid = (max(lats) + min(lats)) / 2
    return math.hypot((max(lons) - min(lons)) * 111.32 * math.cos(math.radians(mid)), (max(lats) - min(lats)) * 110.574)


def _num(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def window_features(t0: Optional[datetime], rows: list[dict], *, suffix: str = "_w") -> dict[str, float]:
    """Counts and strength of a set of satellite detections, plus when the first one was seen (local clock and season).
    `suffix` is "_w" for the first-window features and "" for whole-incident features."""
    feats: dict[str, float] = {}
    frps = [v for v in (_num(m.get("frp")) for m in rows) if v is not None]
    confs = [v for v in (_num(m.get("detection_confidence_pct")) for m in rows) if v is not None]
    bucket = SCAN_BUCKET_MINUTES * 60
    feats[f"n{suffix}"] = float(len(rows))
    feats[f"scans{suffix}"] = float(len({int(m["_t"].timestamp() // bucket) for m in rows}))
    feats[f"frp_max{suffix}"] = max(frps) if frps else 0.0
    feats[f"frp_mean{suffix}"] = sum(frps) / len(frps) if frps else 0.0
    feats[f"frp_sum{suffix}"] = sum(frps)
    for source in SATELLITE_SOURCES:
        feats[f"n_{source}{suffix}"] = float(sum(1 for m in rows if m.get("source") == source))
    feats[f"det_conf_max{suffix}"] = max(confs) if confs else NAN
    feats[f"det_conf_mean{suffix}"] = sum(confs) / len(confs) if confs else NAN
    feats[f"span_km{suffix}"] = _km_span(rows)
    if t0 is not None:
        local = t0.astimezone(LOCAL_TZ)
        angle = 2 * math.pi * (local.hour + local.minute / 60) / 24
        feats["hour_sin"], feats["hour_cos"] = math.sin(angle), math.cos(angle)
        feats["month"] = float(local.month)
        feats["weekend"] = 1.0 if local.weekday() >= 5 else 0.0
    else:
        feats["hour_sin"] = feats["hour_cos"] = feats["month"] = feats["weekend"] = NAN
    return feats


def _share(groups: Any, names: Iterable[str]) -> float:
    wanted = set(names)
    return sum(float(g.get("pct") or 0) for g in groups or [] if g.get("group") in wanted) / 100.0


def cover_shares(cover: Optional[dict]) -> dict[str, float]:
    """Cultivated crops, pasture/grass, forest (each 0-1) and canopy cover % from a ground-cover summary; NaN when the
    ground-cover product is not available."""
    if not cover:
        return {name: NAN for name in COVER_FEATURES}
    nlcd = cover.get("nlcd_groups")
    if nlcd:
        crop = _share(nlcd, ["Cultivated crops"])
        pasture = _share(nlcd, ["Pasture/hay", "Grassland"]) + 0.5 * _share(nlcd, ["Shrub"])
        forest = _share(nlcd, ["Forest", "Wetlands"])
    elif cover.get("ground_cover"):
        crop = _share(cover["ground_cover"], ["Cropland"])
        pasture = _share(cover["ground_cover"], ["Grass/Herbs", "Shrubs"])
        forest = _share(cover["ground_cover"], ["Trees"])
    else:
        groups = cover.get("groups")
        crop = _share(groups, ["Cropland"])
        pasture = _share(groups, ["Grass", "Grass and shrub", "Shrub"])
        forest = _share(groups, ["Timber with understory", "Timber litter"])
    canopy = _num(cover.get("canopy_cover_pct"))
    return {"crop": min(1.0, crop), "pasture_grass": min(1.0, pasture), "forest": min(1.0, forest),
            "canopy": canopy if canopy is not None else NAN}


def cover_for(incident: dict, rows: list[dict]) -> dict[str, float]:
    """Ground cover around the incident, sampled over the given detections' footprints (static data, so it is the same
    answer today and when the incident happened). Never raises."""
    try:
        from services.incident_fuel import fuel_context

        cover = fuel_context(
            incident.get("centroid_latitude"), incident.get("centroid_longitude"),
            footprints=[m.get("footprint_geojson") for m in rows], outline=None,
        )
    except Exception:  # noqa: BLE001 - enrichment only
        logger.debug("ground cover unavailable", exc_info=True)
        cover = None
    return cover_shares(cover)


def neighbor_counts(conn, incident: dict, t0: Optional[datetime], km: float = 5.0) -> dict[str, float]:
    """Other incidents that started within `km` in the 24 h and 7 days BEFORE this one began: a recurring field-burn area
    or heat source looks very different from a first fire in a quiet place. Counts only incidents that existed then."""
    out = {"n_near_24h": NAN, "n_near_7d": NAN}
    lat, lon = _num(incident.get("centroid_latitude")), _num(incident.get("centroid_longitude"))
    if conn is None or t0 is None or lat is None or lon is None:
        return out
    dlat = km / 110.574
    dlon = km / max(0.1, 111.32 * math.cos(math.radians(lat)))
    base = ("SELECT COUNT(*) FROM fire_incidents WHERE id != ? AND status NOT IN ('deleted','merged') "
            "AND centroid_latitude BETWEEN ? AND ? AND centroid_longitude BETWEEN ? AND ? "
            "AND datetime(first_detected_at) >= datetime(?) AND datetime(first_detected_at) < datetime(?)")
    end = t0.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    try:
        for key, days in (("n_near_24h", 1), ("n_near_7d", 7)):
            start = (t0 - timedelta(days=days)).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            out[key] = float(conn.execute(base, (incident.get("id") or -1, lat - dlat, lat + dlat, lon - dlon, lon + dlon, start, end)).fetchone()[0])
    except Exception:  # noqa: BLE001
        logger.debug("neighbour count failed", exc_info=True)
    return out


def weather_at(lat: Any, lon: Any, when: Optional[datetime]) -> dict[str, float]:
    """The hourly RTMA analysis at the grid cell nearest the point, for the hour containing `when`; all NaN when that day
    is not in the archive (it keeps about 120 days) or the point is off the grid. Never raises."""
    out = {f"wx_{name}": NAN for name in WEATHER_VARIABLES}
    la, lo = _num(lat), _num(lon)
    if when is None or la is None or lo is None:
        return out
    try:
        from services import rtma_analysis
        from services.hrrr_hourly import live_store, store

        folder = rtma_analysis.root_dir() / rtma_analysis.run_id(when.astimezone(rtma_analysis.LOCAL_TZ).date())
        manifest_path = folder / "manifest.json"
        if not manifest_path.is_file() or not (folder / "forecast.zarr").is_dir():
            return out
        manifest = json.loads(manifest_path.read_text())
        result = live_store.point_series(folder / "forecast.zarr", manifest, la, lo, store.MAX_POINT_DISTANCE_KM, names=WEATHER_VARIABLES)
        if not result or result.get("error"):
            return out
        key = when.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0).strftime("%Y-%m-%dT%H:00:00Z")
        times = list(result.get("valid_times_utc") or [])
        if key not in times:
            return out
        index = times.index(key)
        for name in WEATHER_VARIABLES:
            value = _num((result["series"].get(name) or [None] * len(times))[index])
            if value is not None:
                out[f"wx_{name}"] = value
    except Exception:  # noqa: BLE001 - weather is an optional input
        logger.debug("weather lookup failed", exc_info=True)
    return out


def persistence_features(incident: dict, members: list[dict], conn=None, *, with_weather: bool = True,
                         cover: Optional[dict[str, float]] = None) -> tuple[Optional[datetime], dict[str, float]]:
    """(t0, features) for the persistence model, from the first WINDOW_MINUTES of the incident only.
    Returns (None, {}) when the incident has no timed satellite detection."""
    sat = satellite_members(members)
    t0, window = first_window(sat)
    if t0 is None:
        return None, {}
    feats = window_features(t0, window)
    feats.update(cover if cover is not None else cover_for(incident, window))
    feats.update(neighbor_counts(conn, incident, t0))
    centre = (window[0].get("latitude"), window[0].get("longitude"))
    feats.update(weather_at(incident.get("centroid_latitude") or centre[0], incident.get("centroid_longitude") or centre[1], t0)
                 if with_weather else {f"wx_{name}": NAN for name in WEATHER_VARIABLES})
    return t0, {name: feats.get(name, NAN) for name in PERSISTENCE_FEATURES}
