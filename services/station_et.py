"""Station reference evapotranspiration (ETo) from RAWS observations.

Only RAWS (network 2) reports solar radiation, so ET is computed there: FAO-56 Penman-Monteith
on the hourly time step (FAO Irrigation and Drainage Paper 56, eq. 53), for the grass
reference surface. Per station, over the rolling last 24 h (inches):

- ``et_current_in_hr``  the latest hour's rate
- ``et_24h_in``         sum of the 24 hourly values (up to ``24 - MIN_HOURS_24H`` missing hours
                        are filled by linear interpolation; otherwise None, never a short sum)
- ``et_max_in_hr`` / ``et_min_in_hr``  highest and lowest hourly rate in the window (the low is
                        ~0 overnight everywhere, as expected)

Assumptions worth knowing: the observation time is treated as the middle of its hour; wind is
measured at the standard RAWS height of 6.1 m (20 ft) and converted to 2 m; soil heat flux is
0.1 Rn by day and 0.5 Rn at night (FAO-56); at night the cloud ratio Rs/Rso is carried over
from the last daytime hour, since Rs/Rso is unstable near zero sun (FAO-56 sec. 3, hourly Rnl).
A station missing any input or a plausible value for an hour simply has no value for that hour.
"""
from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timedelta, timezone

import aiohttp
import numpy as np
from dotenv import load_dotenv

from core.ignored_stations import get_ignored_stations
from services.station_precip import MAX_READING_AGE, STATES, _parse_times, _series

logger = logging.getLogger(__name__)
load_dotenv()
SYNOPTIC_API_TOKEN = os.getenv("SYNOPTIC_API_TOKEN")

NETWORKS = [2]  # RAWS: the only network reporting solar radiation
LOOKBACK_MINUTES = 26 * 60
WIND_HEIGHT_M = 6.1
ALBEDO = 0.23
GSC = 0.0820  # solar constant, MJ m-2 min-1
SIGMA_HOURLY = 2.043e-10  # Stefan-Boltzmann, MJ K-4 m-2 h-1
DAY_SIN_ELEVATION = 0.2  # sun at least ~11.5 deg up: Rs/Rso is meaningful
DEFAULT_CLOUD_RATIO = 0.75  # night before any daytime hour in the window
MAX_HOURLY_ETO_MM = 1.5
MIN_HOURS_24H = 22
MM_PER_IN = 25.4
FEET_PER_METER = 3.28084
INPUT_LIMITS = {"air_temp": (-50.0, 60.0), "relative_humidity": (0.0, 105.0),
                "wind_speed": (0.0, 80.0), "solar_radiation": (0.0, 1500.0)}

et_data = {"stations": None, "last_updated": None, "error": None}


def get_et_data():
    return et_data


def saturation_vapor_pressure(temp_c: float) -> float:
    return 0.6108 * math.exp(17.27 * temp_c / (temp_c + 237.3))


def solar_geometry(lat_deg: float, lon_deg: float, when_utc: datetime, hours: float = 1.0):
    """(Ra in MJ m-2 per period, sin of solar elevation) for the period centred on ``when_utc``."""
    doy = when_utc.timetuple().tm_yday
    phi = math.radians(lat_deg)
    dr = 1 + 0.033 * math.cos(2 * math.pi * doy / 365)
    delta = 0.409 * math.sin(2 * math.pi * doy / 365 - 1.39)
    b = 2 * math.pi * (doy - 81) / 364
    seasonal_correction = 0.1645 * math.sin(2 * b) - 0.1255 * math.cos(b) - 0.025 * math.sin(b)
    utc_hours = when_utc.hour + when_utc.minute / 60 + when_utc.second / 3600
    solar_time = (utc_hours + lon_deg / 15 + seasonal_correction) % 24
    omega = math.pi / 12 * (solar_time - 12)
    sin_elevation = math.sin(phi) * math.sin(delta) + math.cos(phi) * math.cos(delta) * math.cos(omega)

    sunset = math.acos(max(-1.0, min(1.0, -math.tan(phi) * math.tan(delta))))
    w1 = max(omega - math.pi * hours / 24, -sunset)
    w2 = min(omega + math.pi * hours / 24, sunset)
    if w2 <= w1:
        return 0.0, sin_elevation
    ra = (12 * 60 / math.pi) * GSC * dr * (
        (w2 - w1) * math.sin(phi) * math.sin(delta)
        + math.cos(phi) * math.cos(delta) * (math.sin(w2) - math.sin(w1)))
    return max(0.0, ra), sin_elevation


def hourly_eto_mm(temp_c, rh_pct, wind_ms, rs_wm2, lat, lon, elevation_m, when_utc,
                  cloud_ratio=None, wind_height_m=WIND_HEIGHT_M):
    """FAO-56 hourly grass-reference ET (mm/h) for the hour centred on ``when_utc``.

    ``cloud_ratio`` is the Rs/Rso to use when the sun is too low to measure it (night).
    Returns ``(eto_mm, daytime_cloud_ratio_or_None)``.
    """
    es = saturation_vapor_pressure(temp_c)
    ea = es * max(0.0, min(100.0, rh_pct)) / 100
    slope = 4098 * es / (temp_c + 237.3) ** 2
    pressure = 101.3 * ((293 - 0.0065 * elevation_m) / 293) ** 5.26
    gamma = 0.000665 * pressure
    u2 = wind_ms * 4.87 / math.log(67.8 * wind_height_m - 5.42)

    ra, sin_elevation = solar_geometry(lat, lon, when_utc)
    rso = (0.75 + 2e-5 * elevation_m) * ra
    rs = rs_wm2 * 0.0036  # W m-2 -> MJ m-2 h-1
    daytime_ratio = None
    if sin_elevation > DAY_SIN_ELEVATION and rso > 0:
        daytime_ratio = max(0.3, min(1.0, rs / rso))
        ratio = daytime_ratio
    else:
        ratio = cloud_ratio if cloud_ratio is not None else DEFAULT_CLOUD_RATIO

    rns = (1 - ALBEDO) * rs
    rnl = SIGMA_HOURLY * (temp_c + 273.16) ** 4 * (0.34 - 0.14 * math.sqrt(ea)) * (1.35 * ratio - 0.35)
    rn = rns - rnl
    soil_flux = 0.1 * rn if rn > 0 else 0.5 * rn
    numerator = 0.408 * slope * (rn - soil_flux) + gamma * (37 / (temp_c + 273)) * u2 * (es - ea)
    return max(0.0, numerator / (slope + gamma * (1 + 0.34 * u2))), daytime_ratio


def _usable(name, value):
    low, high = INPUT_LIMITS[name]
    return value is not None and low <= value <= high


def derive_station_et(station: dict, now: datetime | None = None) -> dict | None:
    """One ET record from a Synoptic timeseries station (metric units), or None when it lacks
    the inputs, an elevation, or fresh data."""
    now = now or datetime.now(timezone.utc)
    observations = station.get("OBSERVATIONS") or {}
    times = _parse_times(observations.get("date_time") or [])
    series = {name: _series(observations, name) for name in INPUT_LIMITS}
    if any(values is None for values in series.values()):
        return None
    feet = station.get("ELEVATION") or station.get("ELEV_DEM")
    if feet is None:
        return None
    elevation_m = float(feet) / FEET_PER_METER
    lat, lon = float(station["LATITUDE"]), float(station["LONGITUDE"])

    rows = {}
    for index, when in enumerate(times):
        if when is None:
            continue
        values = {name: series[name][index] for name in INPUT_LIMITS}
        if all(_usable(name, value) for name, value in values.items()):
            rows[when.replace(minute=0, second=0, microsecond=0) + timedelta(hours=when.minute >= 30)] = (when, values)
    if not rows:
        return None
    ordered = [rows[key] for key in sorted(rows)]
    latest = ordered[-1][0]
    if now - latest > MAX_READING_AGE:
        return None

    slots = [None] * 24  # slot k = the hour ending k hours before the latest observation
    cloud_ratio = None
    for when, values in ordered:
        eto, daytime_ratio = hourly_eto_mm(values["air_temp"], values["relative_humidity"], values["wind_speed"],
                                           values["solar_radiation"], lat, lon, elevation_m, when, cloud_ratio)
        if daytime_ratio is not None:
            cloud_ratio = daytime_ratio
        k = int(round((latest - when).total_seconds() / 3600))
        if 0 <= k < 24 and eto <= MAX_HOURLY_ETO_MM:
            slots[k] = eto

    valid = [k for k, value in enumerate(slots) if value is not None]
    if not valid:
        return None
    total_mm = None
    if len(valid) >= MIN_HOURS_24H:
        filled = np.interp(range(24), valid, [slots[k] for k in valid])
        total_mm = float(np.sum(filled))
    hourly = [slots[k] for k in valid]

    def inches(mm, digits):
        return None if mm is None else round(mm / MM_PER_IN, digits)

    return {
        "stid": station.get("STID"),
        "name": station.get("NAME"),
        "state": station.get("STATE"),
        "in_state": station.get("STATE") == "MO",
        "latitude": lat,
        "longitude": lon,
        "elevation_m": round(elevation_m, 1),
        "observed_at": latest.isoformat(),
        "et_current_in_hr": inches(slots[0], 4),
        "et_24h_in": inches(total_mm, 3),
        "et_max_in_hr": inches(max(hourly), 4),
        "et_min_in_hr": inches(min(hourly), 4),
        "hours_used": len(valid),
    }


def derive_all(stations: list[dict], ignored=frozenset(), now: datetime | None = None) -> list[dict]:
    records = []
    for station in stations:
        if station.get("STID") in ignored:
            continue
        try:
            record = derive_station_et(station, now)
        except (KeyError, TypeError, ValueError):
            continue  # malformed station: skip it rather than lose the whole refresh
        if record is not None:
            records.append(record)
    return records


async def fetch_station_et():
    """Refresh the cached station ET (RAWS in MO + neighbor states)."""
    params = {
        "token": SYNOPTIC_API_TOKEN,
        "state": ",".join(STATES),
        "recent": LOOKBACK_MINUTES,
        "vars": ",".join(INPUT_LIMITS),
        "units": "metric",
        "status": "active",
        "network": ",".join(map(str, NETWORKS)),
        "obtimezone": "UTC",
    }
    try:
        async with aiohttp.ClientSession() as session:
            response = await session.get("https://api.synopticdata.com/v2/stations/timeseries", params=params, timeout=120)
            response.raise_for_status()
            payload = await response.json()
        summary = payload.get("SUMMARY") or {}
        if summary.get("RESPONSE_CODE") not in (None, 1):
            raise RuntimeError(summary.get("RESPONSE_MESSAGE") or "Synoptic returned an error")
        records = derive_all(payload.get("STATION") or [], get_ignored_stations())
        et_data.update(stations=records, count=len(records), error=None,
                       last_updated=datetime.now(timezone.utc).isoformat())
        logger.info("Station ET updated (%d stations, %d in Missouri)", len(records), sum(r["in_state"] for r in records))
    except Exception as error:  # keep serving the last good data
        et_data["error"] = str(error)
        logger.error("Error fetching station ET: %s", error)
