"""Anchor a gridded fuel-moisture estimate to RAWS stations.

The residual at a station is measured minus the grid's estimate there. The correction is the mean
residual everywhere (the estimate's statewide offset), plus each station's own departure from that
mean spread with inverse-distance weights w = 1 / (d^2 + s^2) and a background weight 1 / L^2 that
fades it out: with one station, d^2 + s^2 = L^2 gives half its departure, so a local reading reaches
about L (SMF_RAWS_FADE_KM, 75 km by default) and not across the state.

Why the mean is kept everywhere: the 3 + 0.25 x RH estimate's error is mostly statewide (RAWS 10-hr
sticks read ~6.5 points drier on average, ~9 at night). Leave-one-out over 864 station-hours of real
data (2026-10-20/21, 18 Missouri RAWS), RMSE in fuel-moisture points: uncalibrated 7.60, mean only
3.02, mean + faded departures 2.97, plain inverse distance 2.96, fading the whole correction to zero
3.10 (75 km) / 3.54 (50 km). So the fade keeps one station's local reading from spreading statewide
without costing accuracy; fading the whole correction did cost accuracy.

Used by the current-conditions map (services/spread_rate_moisture.py), the RTMA peak product
(services/rtma_peak.py) and the data viewer's "what happened" days (services/rtma_analysis.py).
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import numpy as np

KM_PER_DEG_LAT = 110.57
KM_PER_DEG_LON_EQUATOR = 111.32
SMOOTH_KM = 5.0
MIN_STATIONS = 3
RESIDUAL_LIMIT = 15.0
FM_RANGE = (1.0, 40.0)
MAX_OBSERVATION_AGE = timedelta(minutes=75)


def fade_km() -> float:
    try:
        return max(1.0, float(os.getenv("SMF_RAWS_FADE_KM", "75")))
    except ValueError:
        return 75.0


def _km_scale(lat: np.ndarray) -> tuple[float, float]:
    return KM_PER_DEG_LON_EQUATOR * float(np.cos(np.deg2rad(np.nanmean(lat)))), KM_PER_DEG_LAT


def correction_field(lon: np.ndarray, lat: np.ndarray, station_lon, station_lat, residuals,
                     fade: float | None = None) -> np.ndarray:
    """Station values spread across the grid, fading to zero away from the stations."""
    fade = fade or fade_km()
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    kx, ky = _km_scale(lat)
    weighted = np.zeros(lon.shape, dtype=float)
    total = np.full(lon.shape, 1.0 / fade ** 2)
    for s_lon, s_lat, residual in zip(station_lon, station_lat, residuals):
        weight = 1.0 / (np.square((lon - s_lon) * kx) + np.square((lat - s_lat) * ky) + SMOOTH_KM ** 2)
        weighted += weight * residual
        total += weight
    return weighted / total


def station_correction(lon: np.ndarray, lat: np.ndarray, station_lon, station_lat, residuals,
                       fade: float | None = None) -> np.ndarray:
    """The correction to add to a grid: the mean residual, plus each station's faded departure from it."""
    residuals = np.asarray(residuals, dtype=float)
    mean = float(np.mean(residuals))
    return mean + correction_field(lon, lat, station_lon, station_lat, residuals - mean, fade)


def nearest_cell(lon: np.ndarray, lat: np.ndarray, station_lon: float, station_lat: float) -> int:
    kx, ky = _km_scale(lat)
    return int(np.nanargmin(np.square((lon - station_lon) * kx) + np.square((lat - station_lat) * ky)))


def calibrate(estimate: np.ndarray, lon: np.ndarray, lat: np.ndarray, observations: list[dict],
              fade: float | None = None) -> tuple[np.ndarray, dict]:
    """Calibrate ``estimate`` with ``observations`` ({longitude, latitude, fuel_moisture}).

    Returns the calibrated grid (the estimate unchanged with fewer than MIN_STATIONS) and a summary.
    """
    fade = fade or fade_km()
    flat = np.asarray(estimate, dtype=float).ravel()
    used, residuals = [], []
    for observation in observations:
        at_station = flat[nearest_cell(lon, lat, observation["longitude"], observation["latitude"])]
        if np.isfinite(at_station):
            used.append(observation)
            residuals.append(float(np.clip(observation["fuel_moisture"] - at_station, -RESIDUAL_LIMIT, RESIDUAL_LIMIT)))
    if len(used) < MIN_STATIONS:
        return estimate, {"station_count": len(used), "applied": False, "fade_km": fade}
    correction = station_correction(lon, lat, [o["longitude"] for o in used], [o["latitude"] for o in used],
                                    residuals, fade)
    corrected = np.clip(estimate + correction, *FM_RANGE)
    return corrected, {"station_count": len(used), "applied": True, "fade_km": fade,
                       "mean_residual": round(float(np.mean(residuals)), 2)}


# --- RAWS observations from a Synoptic timeseries payload (archive/raw_data*, raw_data_full*) ---------------

def _observation_series(observations: dict, *names: str) -> list:
    for name in names:
        values = observations.get(name)
        if isinstance(values, dict):
            values = values.get("value")
        if values is not None:
            return values if isinstance(values, list) else [values]
    return []


def synoptic_fuel_observations(payload: dict | None, valid_time: datetime,
                               max_age: timedelta = MAX_OBSERVATION_AGE) -> list[dict]:
    """Each station's fuel-moisture reading closest to ``valid_time`` (within ``max_age``)."""
    if not payload:
        return []
    selected = []
    target = valid_time.astimezone(timezone.utc)
    for station in payload.get("STATION", []):
        observations = station.get("OBSERVATIONS", {})
        times = _observation_series(observations, "date_time")
        values = _observation_series(observations, "fuel_moisture_value_1", "fuel_moisture_set_1", "fuel_moisture")
        candidates = []
        for index, raw_time in enumerate(times):
            if index >= len(values) or values[index] is None:
                continue
            try:
                observed_time = datetime.fromisoformat(str(raw_time).replace("Z", "+00:00"))
                if observed_time.tzinfo is None:
                    observed_time = observed_time.replace(tzinfo=timezone.utc)
                age = abs(target - observed_time.astimezone(timezone.utc))
                value = float(values[index])
            except (TypeError, ValueError):
                continue
            if age <= max_age and np.isfinite(value) and 0 < value <= 60:
                candidates.append((age, value, observed_time))
        if not candidates:
            continue
        try:
            longitude = float(station["LONGITUDE"])
            latitude = float(station["LATITUDE"])
        except (KeyError, TypeError, ValueError):
            continue
        _, value, observed_time = min(candidates, key=lambda candidate: candidate[0])
        selected.append({"station": station.get("STID") or station.get("ID"), "longitude": longitude,
                         "latitude": latitude, "fuel_moisture": value, "observation_time": observed_time.isoformat()})
    return selected


def fresh_station_fuel_moisture(station: dict, now: datetime | None = None,
                                max_age: timedelta = MAX_OBSERVATION_AGE) -> float | None:
    """A live-feed station's (services/synoptic.get_station_data) fuel moisture, if it is recent."""
    fuel = (station.get("observations") or {}).get("fuel_moisture") or {}
    value, when = fuel.get("value"), fuel.get("time")
    if value is None or not when:
        return None
    try:
        observed = datetime.fromisoformat(str(when).replace("Z", "+00:00"))
    except ValueError:
        return None
    if observed.tzinfo is None:
        observed = observed.replace(tzinfo=timezone.utc)
    return float(value) if (now or datetime.now(timezone.utc)) - observed <= max_age else None
