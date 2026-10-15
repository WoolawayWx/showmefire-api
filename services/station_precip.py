"""Station precipitation across Missouri and its neighbor states.

Synoptic's ``precip_accum_one_hour`` exists only on ASOS/AWOS (network 1), and the
account has no access to Synoptic's precipitation-totals service. This module
fetches the raw timeseries instead and derives 1-hour and 24-hour totals for
every station that reports any precipitation variable:

- ASOS/AWOS (1):  ``precip_accum_one_hour`` and ``precip_accum_24_hour`` used directly.
- RAWS (2):       cumulative ``precip_accum``; totals are the sum of its positive steps,
                  so a gauge reset or bucket empty never produces negative rain.

CWOP (network 65) is deliberately excluded: citizen stations with no hourly value and
unreliable gauges.

A station that has a precipitation sensor but reported no rain gets an explicit
``0.0``; a station without a sensor, or whose reading is stale, is ``None``
(missing) rather than zero. Values are inches. Buffer states exist only so the
Missouri interpolation has points past the border; ``in_state`` marks Missouri.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

import aiohttp
from dotenv import load_dotenv

from core.ignored_stations import get_ignored_stations

logger = logging.getLogger(__name__)
load_dotenv()
SYNOPTIC_API_TOKEN = os.getenv("SYNOPTIC_API_TOKEN")

STATES = ["MO", "IL", "IA", "NE", "KS", "OK", "AR", "TN", "KY"]
NETWORKS = [1, 2, 156]
LOOKBACK_MINUTES = 25 * 60
# A reading older than this is "unknown", not "dry".
MAX_READING_AGE = timedelta(minutes=int(os.getenv("SMF_PRECIP_MAX_AGE_MINUTES", "120")))
# Below the gauge resolution (ASOS reports 0.001 for trace); treat as no rain.
MEASURABLE_IN = 0.005
LIMITS_IN = {"1h": 6.0, "24h": 20.0}
# Two reports this far apart can't be in the same accumulation (routine reports are ~60 min
# apart), which matters for stations that report only hourly: their consecutive hourly values
# can rise with no drop between them.
SEGMENT_GAP = timedelta(minutes=57)
RESET_DROP_IN = 0.0005
# Hours with no observation at all that are tolerated before a 24 h total is withheld
# instead of under-reported.
MAX_MISSING_HOURS = 2
# A missing hour next to an hour with at least this much rain withholds the total.
WET_HOUR_IN = 0.01
SENSOR_CACHE_TTL = timedelta(hours=24)
_sensor_cache = {"stids": set(), "fetched": None}

precip_data = {"stations": None, "last_updated": None, "error": None}


def get_precip_data():
    return precip_data


def _series(observations: dict, name: str):
    """Values for ``name`` (any ``_set_N``), first set that has data."""
    for key, values in observations.items():
        if key.startswith(name + "_set_") and any(v is not None for v in values):
            return values
    return None


def _parse_times(raw_times):
    times = []
    for value in raw_times:
        try:
            times.append(datetime.fromisoformat(str(value).replace("Z", "+00:00")))
        except ValueError:
            times.append(None)
    return times


def _latest(values, times):
    for value, when in zip(reversed(values), reversed(times)):
        if value is not None and when is not None:
            return value, when
    return None, None


def _clean(value, window):
    if value is None or value < 0 or value > LIMITS_IN[window]:
        return None
    return 0.0 if value < MEASURABLE_IN else round(value, 3)


def _sum_positive_steps(values, times, since):
    """Rain over a cumulative gauge: sum of positive steps after ``since``.

    A negative step is a reset/empty and is skipped. Returns None when the gauge has
    fewer than two readings in the window, so a short record is not reported as dry.
    """
    points = [(t, v) for v, t in zip(values, times) if v is not None and t is not None and t >= since]
    if len(points) < 2:
        return None
    total = 0.0
    for (_, previous), (_, current) in zip(points, points[1:]):
        step = current - previous
        if step > 0:
            total += step
    return total


def _accum_in_window(reports, start, end):
    """Rain at an ASOS/AWOS station over ``(start, end]`` from its 1-hour reports.

    Synoptic's ``precip_accum_one_hour`` is not a rolling hour: it accumulates from the
    station's routine report and drops back to ~0 right after the next one (e.g. 0.72 in at
    11:49, 0.001 at 11:51). Reports are split into segments at each drop (or a gap over
    ``SEGMENT_GAP``: a dry hour, or a station that reports only hourly) and a segment's rain in the window is the rise of its
    value across it, which handles partial hours at either edge and never double counts the
    5-minute ramp within an hour.
    """
    segments, current, previous = [], [], None
    for when, value in sorted(reports):
        if previous is not None and (value < previous[1] - RESET_DROP_IN or when - previous[0] >= SEGMENT_GAP):
            segments.append(current)
            current = []
        current.append((when, value))
        previous = (when, value)
    if current:
        segments.append(current)

    def level(segment, moment):  # value at ``moment``: last report at or before it, else 0
        known = [v for t, v in segment if t <= moment]
        return known[-1] if known else 0.0

    total = 0.0
    for segment in segments:
        if segment[-1][0] <= start or segment[0][0] > end:
            continue
        total += max(0.0, level(segment, end) - level(segment, start))
    return total


def _missing_hours(obs_times, end):
    """Indexes (0 = the hour ending at ``end``) of the 24 hours with no observation at all."""
    return [k for k in range(24)
            if not any(end - timedelta(hours=k + 1) < t <= end - timedelta(hours=k) for t in obs_times)]


def _gap_may_hide_rain(missing, reports, end):
    """True when a missing hour sits next to an hour that had rain (the gap likely lost some)."""
    def wet(k):
        window_end = end - timedelta(hours=k)
        return k >= 0 and _accum_in_window(reports, window_end - timedelta(hours=1), window_end) >= WET_HOUR_IN
    return any(wet(k - 1) or wet(k + 1) for k in missing)


def derive_station_precip(station: dict, now: datetime | None = None, one_hour_sensors=frozenset()) -> dict | None:
    """Build one precip record from a Synoptic timeseries station, or None if it has no
    precipitation sensor.

    ``one_hour_sensors`` is the set of STIDs whose metadata lists a 1-hour precip sensor.
    METAR omits the precip group when no rain fell, so a dry ASOS has no 1-hour value at
    all; it is 0.0 only when the sensor is known to exist and the station is reporting
    (fresh ``air_temp``).
    """
    now = now or datetime.now(timezone.utc)
    observations = station.get("OBSERVATIONS") or {}
    times = _parse_times(observations.get("date_time") or [])
    one_hour = _series(observations, "precip_accum_one_hour")
    day = _series(observations, "precip_accum_24_hour")
    gauge = _series(observations, "precip_accum")
    one_hour_sensor = one_hour is not None or station.get("STID") in one_hour_sensors
    if not one_hour_sensor and day is None and gauge is None:
        return None

    precip_1h = precip_24h = None
    hours_missing = None
    sources = []

    air_temp = _series(observations, "air_temp")
    _, last_obs = _latest(air_temp, times) if air_temp is not None else (None, None)
    if one_hour_sensor and last_obs is not None:
        if now - last_obs <= MAX_READING_AGE:
            obs_times = [t for v, t in zip(air_temp, times) if v is not None and t is not None]
            reports = [(t, v) for v, t in zip(one_hour or [], times) if v is not None and t is not None]
            precip_1h = _clean(_accum_in_window(reports, last_obs - timedelta(hours=1), last_obs), "1h")
            missing = _missing_hours(obs_times, last_obs)
            hours_missing = len(missing)
            if hours_missing <= MAX_MISSING_HOURS and not _gap_may_hide_rain(missing, reports, last_obs):
                precip_24h = _clean(_accum_in_window(reports, last_obs - timedelta(hours=24), last_obs), "24h")
            sources.append("asos_accum_one_hour")
    elif one_hour is not None:
        value, when = _latest(one_hour, times)
        if value is not None and now - when <= MAX_READING_AGE:
            precip_1h = _clean(value, "1h")
            sources.append("precip_accum_one_hour")
    if day is not None and precip_24h is None and not (one_hour_sensor and last_obs is not None):
        # The NWS 24 h group (6-hourly, different window) is only a fallback for stations
        # we cannot build an hourly sum for.
        value, when = _latest(day, times)
        if value is not None and now - when <= MAX_READING_AGE:
            precip_24h = _clean(value, "24h")
            sources.append("precip_accum_24_hour")
    if gauge is not None:
        _, when = _latest(gauge, times)
        if when is not None and now - when <= MAX_READING_AGE:
            # Anchor to the gauge's own latest reading: RAWS report hourly, so a window
            # anchored to "now" can hold a single reading and wrongly yield nothing.
            if precip_1h is None:
                step_total = _sum_positive_steps(gauge, times, when - timedelta(hours=1, minutes=5))
                precip_1h = _clean(step_total, "1h") if step_total is not None else None
            if precip_24h is None:
                step_total = _sum_positive_steps(gauge, times, when - timedelta(hours=24, minutes=5))
                precip_24h = _clean(step_total, "24h") if step_total is not None else None
            sources.append("precip_accum")

    network_id = int(station["MNET_ID"]) if str(station.get("MNET_ID") or "").isdigit() else None
    return {
        "stid": station.get("STID"),
        "name": station.get("NAME"),
        "state": station.get("STATE"),
        "in_state": station.get("STATE") == "MO",
        "latitude": float(station["LATITUDE"]),
        "longitude": float(station["LONGITUDE"]),
        "network_id": network_id,
        "precip_1h_in": precip_1h,
        "precip_24h_in": precip_24h,
        "precip_24h_hours_missing": hours_missing,
        "sources": sources,
    }


def derive_all(stations: list[dict], ignored=frozenset(), now: datetime | None = None, one_hour_sensors=frozenset()) -> list[dict]:
    records = []
    for station in stations:
        if station.get("STID") in ignored:
            continue
        try:
            record = derive_station_precip(station, now, one_hour_sensors)
        except (KeyError, TypeError, ValueError):
            continue  # malformed station: skip it rather than lose the whole refresh
        if record is not None and (record["precip_1h_in"] is not None or record["precip_24h_in"] is not None):
            records.append(record)
    return records


async def _one_hour_sensor_stids(session) -> set:
    """STIDs of ASOS/AWOS stations with a 1-hour precip sensor (cached; metadata rarely changes)."""
    now = datetime.now(timezone.utc)
    cached = _sensor_cache["fetched"]
    if cached and now - cached < SENSOR_CACHE_TTL:
        return _sensor_cache["stids"]
    try:
        response = await session.get("https://api.synopticdata.com/v2/stations/metadata", params={
            "token": SYNOPTIC_API_TOKEN, "state": ",".join(STATES), "network": "1",
            "status": "active", "sensorvars": "1", "complete": "1"}, timeout=60)
        response.raise_for_status()
        payload = await response.json()
        _sensor_cache["stids"] = {s["STID"] for s in payload.get("STATION") or []
                                  if "precip_accum_one_hour" in (s.get("SENSOR_VARIABLES") or {})}
        _sensor_cache["fetched"] = now
    except Exception as error:  # fall back to the last known set (possibly empty)
        logger.warning("Could not refresh ASOS precip sensor list: %s", error)
    return _sensor_cache["stids"]


async def fetch_station_precip():
    """Refresh the cached station precipitation (MO + neighbor states)."""
    params = {
        "token": SYNOPTIC_API_TOKEN,
        "state": ",".join(STATES),
        "recent": LOOKBACK_MINUTES,
        # air_temp only establishes whether a station is currently reporting at all.
        "vars": "precip_accum,precip_accum_one_hour,precip_accum_24_hour,air_temp",
        "units": "english",
        "status": "active",
        "network": ",".join(map(str, NETWORKS)),
        "obtimezone": "UTC",
    }
    try:
        async with aiohttp.ClientSession() as session:
            response = await session.get("https://api.synopticdata.com/v2/stations/timeseries", params=params, timeout=120)
            response.raise_for_status()
            payload = await response.json()
            sensors = await _one_hour_sensor_stids(session)
        summary = payload.get("SUMMARY") or {}
        if summary.get("RESPONSE_CODE") not in (None, 1):
            raise RuntimeError(summary.get("RESPONSE_MESSAGE") or "Synoptic returned an error")
        records = derive_all(payload.get("STATION") or [], get_ignored_stations(), one_hour_sensors=sensors)
        precip_data.update(stations=records, count=len(records), error=None,
                           last_updated=datetime.now(timezone.utc).isoformat())
        logger.info("Station precip updated (%d stations, %d in Missouri)",
                    len(records), sum(r["in_state"] for r in records))
    except Exception as error:  # keep serving the last good data
        precip_data["error"] = str(error)
        logger.error("Error fetching station precipitation: %s", error)
