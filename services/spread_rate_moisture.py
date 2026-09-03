"""Gridded dead-fuel moisture conditioning with RAWS 10-hour anchoring."""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Iterable

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

from core.fire_danger import _green_factor
from services.rtma_capture import _root, count_cached_hours, latest_complete_hour
from services.seasonal_fuel_state import current_gdd_accum

logger = logging.getLogger(__name__)

MIN_CONDITIONING_HOURS = 5 * 24
TARGET_CONDITIONING_HOURS = 7 * 24
MIN_RAW_STATIONS = 3
MAX_RAW_AGE_MINUTES = 180
FM_MIN_PERCENT = 1.0
FM_MAX_PERCENT = 40.0

TAU_1_HR = 1.0
TAU_10_HR = 10.0
TAU_100_HR = 100.0


def nelson_emc(temp_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    rh = np.asarray(rh, dtype=float)
    temp_c = np.asarray(temp_c, dtype=float)
    emc = np.where(
        rh <= 10,
        0.03 + 0.2626 * rh - 0.00104 * rh * temp_c,
        np.where(
            rh <= 50,
            2.22 - 0.160 * rh + 0.01660 * temp_c,
            21.06 - 0.4944 * rh + 0.005565 * rh ** 2 - 0.00063 * rh * temp_c,
        ),
    )
    return np.clip(emc, FM_MIN_PERCENT, FM_MAX_PERCENT)


def live_moisture_percent(gdd_accum: float | None) -> tuple[float, float]:
    """Map statewide GDD to Scott–Burgan L2 (cured) vs L4 (green) live moistures."""
    green = _green_factor(gdd_accum)
    herbaceous = 60.0 + green * 60.0
    woody = 90.0 + green * 60.0
    return herbaceous, woody


def _interpolate_field(source_lat, source_lon, values, target_lat, target_lon) -> np.ndarray:
    source_lon = np.where(np.asarray(source_lon) > 180, np.asarray(source_lon) - 360, source_lon)
    tree = cKDTree(np.column_stack((np.asarray(source_lat).ravel(), np.asarray(source_lon).ravel())))
    index = tree.query(np.column_stack((target_lat.ravel(), target_lon.ravel())))[1]
    return np.asarray(values).ravel()[index].reshape(target_lat.shape)


def _squeeze2d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    while values.ndim > 2:
        values = np.take(values, 0, axis=0)
    return values


def _load_rtma_hour(stamp: pd.Timestamp, cache_dir) -> xr.Dataset:
    path = cache_dir / f"rtma_{stamp:%Y%m%d_%H}z.nc"
    if not path.is_file():
        raise FileNotFoundError(path)
    return xr.open_dataset(path)


def _temp_c_and_fields(ds: xr.Dataset, target_lat: np.ndarray, target_lon: np.ndarray):
    temp = _squeeze2d(np.asarray(ds["t2m"].values, dtype=float))
    temp = np.where(temp > 150, temp - 273.15, temp)
    rh = _squeeze2d(np.asarray(ds["r2"].values, dtype=float))
    precip = _squeeze2d(np.asarray(ds.get("apcp", xr.zeros_like(ds["t2m"])).values, dtype=float))
    u = _squeeze2d(np.asarray(ds["u10"].values, dtype=float))
    v = _squeeze2d(np.asarray(ds["v10"].values, dtype=float))
    src_lat = _squeeze2d(np.asarray(ds["latitude"].values, dtype=float))
    src_lon = _squeeze2d(np.asarray(ds["longitude"].values, dtype=float))
    return {
        "temp_c": _interpolate_field(src_lat, src_lon, temp, target_lat, target_lon),
        "rh": _interpolate_field(src_lat, src_lon, rh, target_lat, target_lon),
        "precip_mm": _interpolate_field(src_lat, src_lon, precip, target_lat, target_lon),
        "wind_ms": _interpolate_field(src_lat, src_lon, np.hypot(u, v), target_lat, target_lon),
        "u_ms": _interpolate_field(src_lat, src_lon, u, target_lat, target_lon),
        "v_ms": _interpolate_field(src_lat, src_lon, v, target_lat, target_lon),
    }


def _advance_moisture(state: dict, temp_c: np.ndarray, rh: np.ndarray, precip_mm: np.ndarray) -> dict:
    emc = nelson_emc(temp_c, rh)
    fm1 = state["fm1"] + (emc - state["fm1"]) * (1.0 - np.exp(-1.0 / TAU_1_HR))
    fm10 = state["fm10"] + (emc - state["fm10"]) * (1.0 - np.exp(-1.0 / TAU_10_HR))
    fm100 = state["fm100"] + (emc - state["fm100"]) * (1.0 - np.exp(-1.0 / TAU_100_HR))
    wet = precip_mm > 0.5
    if np.any(wet):
        fm1 = np.where(wet, np.minimum(fm1 + 0.5 * precip_mm, FM_MAX_PERCENT), fm1)
        fm10 = np.where(wet, np.minimum(fm10 + 0.15 * precip_mm, FM_MAX_PERCENT), fm10)
        fm100 = np.where(wet, np.minimum(fm100 + 0.05 * precip_mm, FM_MAX_PERCENT), fm100)
    return {"fm1": fm1, "fm10": fm10, "fm100": fm100}


def extract_raws_fuel_points(raws_payload: dict, states: Iterable[str] = ("MO",)) -> list[dict]:
    now = datetime.now(timezone.utc)
    points = []
    for station in raws_payload.get("stations") or []:
        if station.get("state") not in states:
            continue
        obs = station.get("observations") or {}
        fuel = obs.get("fuel_moisture")
        if not isinstance(fuel, dict):
            continue
        value = fuel.get("value")
        obs_time = fuel.get("time")
        if value is None:
            continue
        try:
            fm = float(value)
        except (TypeError, ValueError):
            continue
        if not (FM_MIN_PERCENT <= fm <= FM_MAX_PERCENT):
            continue
        age_minutes = None
        if obs_time:
            try:
                stamp = datetime.fromisoformat(str(obs_time).replace("Z", "+00:00"))
                if stamp.tzinfo is None:
                    stamp = stamp.replace(tzinfo=timezone.utc)
                age_minutes = (now - stamp.astimezone(timezone.utc)).total_seconds() / 60.0
            except Exception:
                age_minutes = None
        if age_minutes is not None and age_minutes > MAX_RAW_AGE_MINUTES:
            continue
        lon = station.get("longitude")
        lat = station.get("latitude")
        if lon is None or lat is None:
            continue
        points.append(
            {
                "stid": station.get("stid"),
                "lon": float(lon),
                "lat": float(lat),
                "fm10": fm,
                "age_minutes": age_minutes,
            }
        )
    return points


def apply_raws_10hr_correction(
    fm10: np.ndarray,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    stations: list[dict],
) -> tuple[np.ndarray, dict]:
    if len(stations) < MIN_RAW_STATIONS:
        raise ValueError(f"fewer than {MIN_RAW_STATIONS} fresh RAWS fuel-moisture observations")
    coords = np.asarray([[point["lat"], point["lon"]] for point in stations], dtype=float)
    tree = cKDTree(coords)
    model_at_stations = np.asarray(
        [_interpolate_field(target_lat, target_lon, fm10, np.array([[point["lat"]]]), np.array([[point["lon"]]]))[0, 0]
         for point in stations],
        dtype=float,
    )
    residuals = np.asarray([point["fm10"] for point in stations], dtype=float) - model_at_stations
    distance, index = tree.query(np.column_stack((target_lat.ravel(), target_lon.ravel())), k=min(8, len(stations)))
    if distance.ndim == 1:
        distance, index = distance[:, None], index[:, None]
    weights = 1.0 / np.maximum(distance, 1e-3) ** 2
    weights /= weights.sum(axis=1, keepdims=True)
    correction = (weights * residuals[index]).sum(axis=1).reshape(target_lat.shape)
    corrected = np.clip(fm10 + correction, FM_MIN_PERCENT, FM_MAX_PERCENT)
    nearest_distance, _ = tree.query(np.column_stack((target_lat.ravel(), target_lon.ravel())), k=1)
    meta = {
        "station_count": len(stations),
        "mean_residual": float(np.mean(residuals)),
        "nearest_station_distance_deg": nearest_distance.reshape(target_lat.shape),
    }
    return corrected, meta


def condition_moisture(
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    analysis_hour: datetime | None = None,
    raws_payload: dict | None = None,
) -> dict:
    root = _root()
    cache_dir = root / "cache" / "rtma"
    analysis_hour = analysis_hour or latest_complete_hour()
    if analysis_hour.tzinfo is not None:
        analysis_hour = analysis_hour.astimezone(timezone.utc).replace(tzinfo=None)
    end_stamp = pd.Timestamp(analysis_hour)
    available_hours = count_cached_hours(analysis_hour, TARGET_CONDITIONING_HOURS, cache_dir)
    if available_hours < MIN_CONDITIONING_HOURS:
        raise RuntimeError(
            f"insufficient RTMA history for spread-rate conditioning: {available_hours}/{MIN_CONDITIONING_HOURS} hours"
        )
    stamps = [end_stamp - pd.Timedelta(hours=offset) for offset in range(available_hours - 1, -1, -1)]
    state = None
    missing = 0
    for stamp in stamps:
        try:
            with _load_rtma_hour(stamp, cache_dir) as ds:
                fields = _temp_c_and_fields(ds, target_lat, target_lon)
        except Exception:
            missing += 1
            if state is None:
                continue
            fields = None
        if fields is None:
            continue
        if state is None:
            emc = nelson_emc(fields["temp_c"], fields["rh"])
            state = {"fm1": emc.copy(), "fm10": emc.copy(), "fm100": emc.copy()}
        else:
            state = _advance_moisture(state, fields["temp_c"], fields["rh"], fields["precip_mm"])
    if state is None:
        raise RuntimeError("no usable RTMA hours for moisture conditioning")
    with _load_rtma_hour(end_stamp, cache_dir) as ds:
        weather = _temp_c_and_fields(ds, target_lat, target_lon)
    herbaceous, woody = live_moisture_percent(current_gdd_accum())
    correction_meta = {"station_count": 0, "mean_residual": None}
    fm10 = state["fm10"]
    if raws_payload:
        stations = extract_raws_fuel_points(raws_payload)
        if len(stations) >= MIN_RAW_STATIONS:
            fm10, correction_meta = apply_raws_10hr_correction(fm10, target_lat, target_lon, stations)
    return {
        "analysis_hour": end_stamp.isoformat(),
        "fm1_pct": state["fm1"],
        "fm10_pct": fm10,
        "fm100_pct": state["fm100"],
        "live_herbaceous_pct": np.full_like(fm10, herbaceous, dtype=float),
        "live_woody_pct": np.full_like(fm10, woody, dtype=float),
        "temp_c": weather["temp_c"],
        "rh": weather["rh"],
        "wind_ms": weather["wind_ms"],
        "precip_mm": weather["precip_mm"],
        "u_ms": weather["u_ms"],
        "v_ms": weather["v_ms"],
        "conditioning_hours_available": available_hours,
        "conditioning_hours_missing": missing,
        "raws_correction": correction_meta,
    }
