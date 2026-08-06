"""Canonical forecast-precipitation contract used by map and model paths."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr

CONTRACT_PATH = Path(__file__).with_name("precipitation_contract.json")
VARIABLE_NAMES = ("tp", "apcp", "APCP", "precipitation")


def _contract_bytes() -> bytes:
    payload = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


PRECIPITATION_CONTRACT_SHA256 = hashlib.sha256(_contract_bytes()).hexdigest()
PRECIPITATION_CONTRACT_VERSION = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))["version"]


@dataclass(frozen=True)
class DecodedPrecipitation:
    cumulative_mm: xr.DataArray
    interval_mm: xr.DataArray
    interval_hours: xr.DataArray
    reset_flag: xr.DataArray
    partial_window_flag: xr.DataArray
    variable_name: str
    source_units: str
    accumulation_kind: str


def get_precip_dataarray(dataset: xr.Dataset) -> tuple[str, xr.DataArray]:
    for name in VARIABLE_NAMES:
        if name in dataset.data_vars:
            return name, dataset[name]
    raise ValueError(f"precipitation variable unavailable; expected one of {VARIABLE_NAMES}")


def normalize_to_mm(values, units: str | None):
    if not units or not str(units).strip():
        raise ValueError("precipitation units are missing")
    normalized = str(units).lower().replace(" ", "").replace("**", "^")
    if normalized in {"mm", "millimeter", "millimeters", "kgm^-2", "kgm-2", "kg/m^2", "kg/m2"}:
        return values
    if normalized in {"m", "meter", "meters"}:
        return values * 1000.0
    if normalized in {"in", "inch", "inches", "[in_i]"}:
        return values * 25.4
    raise ValueError(f"unsupported precipitation units: {units!r}")


def _time_dimension(array: xr.DataArray) -> str | None:
    for name in ("step", "valid_time", "time"):
        if name in array.dims:
            return name
    return None


def _lead_hours(array: xr.DataArray, dimension: str) -> np.ndarray:
    coordinate = array.coords.get(dimension)
    if coordinate is None:
        return np.arange(array.sizes[dimension], dtype=float)
    values = np.asarray(coordinate.values)
    if np.issubdtype(values.dtype, np.timedelta64):
        return values / np.timedelta64(1, "h")
    if np.issubdtype(values.dtype, np.datetime64):
        return (values - values[0]) / np.timedelta64(1, "h")
    return values.astype(float)


def decode_forecast_precipitation(dataset: xr.Dataset) -> DecodedPrecipitation:
    name, raw = get_precip_dataarray(dataset)
    units = raw.attrs.get("units") or raw.attrs.get("GRIB_parameterUnits")
    millimeters = normalize_to_mm(raw.astype(float), units).clip(min=0)
    dimension = _time_dimension(millimeters)
    step_type = str(raw.attrs.get("GRIB_stepType", raw.attrs.get("stepType", ""))).lower()
    accumulated = step_type == "accum" or name in {"tp", "apcp", "APCP"}
    if dimension is None:
        cumulative, interval = millimeters, millimeters
        duration = xr.zeros_like(millimeters, dtype=float) + np.nan
        reset = xr.zeros_like(millimeters, dtype=bool)
        partial = xr.ones_like(millimeters, dtype=bool)
    else:
        leads = _lead_hours(millimeters, dimension)
        durations = np.diff(np.concatenate(([0.0], leads))).astype(float)
        if np.any(durations < 0):
            raise ValueError("precipitation leads are not chronological")
        duration = xr.DataArray(durations, dims=(dimension,), coords={dimension: millimeters[dimension]}).broadcast_like(millimeters)
        partial = duration != 1.0
        if accumulated:
            previous = millimeters.shift({dimension: 1}, fill_value=0.0)
            raw_interval = millimeters - previous
            reset = raw_interval < -1e-6
            interval = raw_interval.where(~reset, 0.0).clip(min=0)
            cumulative = millimeters
        else:
            interval = millimeters
            cumulative = interval.cumsum(dimension)
            reset = xr.zeros_like(interval, dtype=bool)
    return DecodedPrecipitation(cumulative, interval, duration, reset, partial, name, str(units),
                                "cumulative_since_init" if accumulated else "interval")


def final_accumulation_mm(dataset: xr.Dataset) -> xr.DataArray:
    decoded = decode_forecast_precipitation(dataset)
    dimension = _time_dimension(decoded.cumulative_mm)
    return decoded.cumulative_mm.isel({dimension: -1}) if dimension else decoded.cumulative_mm
