from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import xarray as xr

from .contracts import OPTIONAL_VARIABLES, REQUIRED_VARIABLES, VARIABLE_UNITS


ALIASES: dict[str, tuple[str, ...]] = {
    "temperature_2m": ("temperature_2m", "t2m", "tmp2m"),
    "dewpoint_2m": ("dewpoint_2m", "d2m", "dpt2m", "dpt"),
    "relative_humidity_2m": ("relative_humidity_2m", "rh2m", "r2"),
    "wind_u_10m": ("wind_u_10m", "u10", "ugrd10m"),
    "wind_v_10m": ("wind_v_10m", "v10", "vgrd10m"),
    "wind_gust_10m": ("wind_gust_10m", "gust", "gust10m", "fg10"),
    "precipitation_increment": ("precipitation_increment", "tp", "apcp", "prate"),
    "shortwave_down": ("shortwave_down", "dswrf", "sdswrf", "ssrd"),
    "cloud_cover": ("cloud_cover", "tcc", "tcdc"),
    "mixing_height": ("mixing_height", "hpbl", "blh", "pblh", "mixht"),
    "soil_moisture": ("soil_moisture", "soilw", "mstav", "swvl1"),
    "snow_water_equivalent": ("snow_water_equivalent", "weasd", "sde", "sdwe", "sd"),
    "temperature_850hpa": ("temperature_850hpa", "t850"),
    "temperature_700hpa": ("temperature_700hpa", "t700"),
    "wind_u_850hpa": ("wind_u_850hpa", "u850"),
    "wind_v_850hpa": ("wind_v_850hpa", "v850"),
    "wind_u_700hpa": ("wind_u_700hpa", "u700"),
    "wind_v_700hpa": ("wind_v_700hpa", "v700"),
}


@dataclass
class SourceCube:
    model: str
    cycle_time: datetime
    dataset: xr.Dataset
    member_ids: tuple[str, ...]
    quality_flags: tuple[str, ...] = ()
    checksum: str | None = None
    object_key: str | None = None

    def ensemble_mean(self, variable: str) -> xr.DataArray:
        data = self.dataset[variable]
        return data.mean("member", skipna=True) if "member" in data.dims else data


def _first_present(dataset: xr.Dataset, aliases: tuple[str, ...]) -> str | None:
    return next((name for name in aliases if name in dataset.data_vars), None)


def _to_celsius(data: xr.DataArray) -> xr.DataArray:
    units = str(data.attrs.get("units", "")).lower()
    if units in {"k", "kelvin"} or (not units and float(data.mean(skipna=True)) > 150):
        data = data - 273.15
    return data.astype("float32")


def _normalize_units(variable: str, data: xr.DataArray) -> xr.DataArray:
    units = str(data.attrs.get("units", "")).lower()
    if variable.startswith("temperature") or variable == "dewpoint_2m":
        data = _to_celsius(data)
    elif variable in {"relative_humidity_2m", "cloud_cover"}:
        if ("fraction" in units or units in {"1", "0-1"}) or float(data.max(skipna=True)) <= 1.5:
            data = data * 100.0
        data = data.clip(0, 100).astype("float32")
    elif variable in {"precipitation_increment", "snow_water_equivalent"}:
        if units in {"m", "meter", "metre"}:
            data = data * 1000.0
        data = data.clip(min=0).astype("float32")
    elif variable == "shortwave_down":
        data = data.clip(min=0).astype("float32")
    elif variable == "soil_moisture":
        if units in {"%", "percent", "percentage"} or float(data.max(skipna=True)) > 1.5:
            data = data / 100.0
        data = data.clip(0, 1).astype("float32")
    else:
        data = data.astype("float32")
    data.attrs["units"] = VARIABLE_UNITS.get(variable, units)
    return data


def _precip_to_increments(data: xr.DataArray) -> xr.DataArray:
    semantics = str(data.attrs.get("accumulation_semantics", "incremental")).lower()
    if semantics not in {"cumulative", "accumulated", "since_initialization"}:
        return data
    diffs = data.diff("time", label="upper")
    first = data.isel(time=0)
    increments = xr.concat([first, xr.where(diffs < 0, data.isel(time=slice(1, None)), diffs)], dim="time")
    increments = increments.assign_coords(time=data.time)
    increments.attrs.update(data.attrs)
    increments.attrs["accumulation_semantics"] = "incremental"
    return increments


class DatasetAdapter:
    """Normalize a clipped native-grid model dataset to the v1 contract."""

    model: str

    def __init__(self, model: str, *, strict_required: bool = True):
        self.model = model.lower()
        self.strict_required = strict_required

    def open(self, path: str | Path, cycle_time: datetime | None = None) -> SourceCube:
        with xr.open_dataset(path) as source:
            dataset = source.load()
        return self.normalize(dataset, cycle_time=cycle_time)

    def normalize(self, source: xr.Dataset, cycle_time: datetime | None = None) -> SourceCube:
        renamed: dict[str, xr.DataArray] = {}
        missing: list[str] = []
        for variable in (*REQUIRED_VARIABLES, *OPTIONAL_VARIABLES):
            found = _first_present(source, ALIASES[variable])
            if found is None:
                if variable in REQUIRED_VARIABLES:
                    missing.append(variable)
                continue
            data = _normalize_units(variable, source[found])
            if variable == "precipitation_increment":
                data = _precip_to_increments(data)
            renamed[variable] = data
        if missing and self.strict_required:
            raise ValueError(f"{self.model} missing required fields: {', '.join(missing)}")
        if missing:
            if not renamed:
                raise ValueError(f"{self.model} contains none of the forecast-v1 variables")
            template = next(iter(renamed.values()))
            for variable in missing:
                renamed[variable] = xr.full_like(template, np.nan, dtype="float32")
        normalized = xr.Dataset(renamed, attrs=dict(source.attrs))
        if "time" not in normalized.dims:
            raise ValueError(f"{self.model} dataset has no time dimension")
        if "member" not in normalized.dims:
            normalized = normalized.expand_dims(member=["deterministic"])
        normalized = normalized.transpose("member", "time", ...)
        cycle = cycle_time or _cycle_from_attrs(source.attrs)
        valid_times = normalized.time.values.astype("datetime64[s]")
        if len(np.unique(valid_times)) != len(valid_times) or (len(valid_times) > 1 and not bool(np.all(np.diff(valid_times).astype("timedelta64[s]") > np.timedelta64(0, "s")))):
            raise ValueError(f"{self.model} valid times must be unique and increasing")
        expected_start = np.datetime64(cycle.astimezone(timezone.utc).replace(tzinfo=None), "s")
        if valid_times[0] != expected_start:
            raise ValueError(f"{self.model} first valid time does not match its initialization time")
        members = tuple(str(v) for v in normalized.member.values)
        normalized.attrs.update(
            model=self.model,
            cycle_time=cycle.isoformat().replace("+00:00", "Z"),
            variable_contract="forecast-v1",
        )
        flags = tuple(
            [f"required_field_unavailable:{name}" for name in missing]
            + [f"optional_field_missing:{name}" for name in OPTIONAL_VARIABLES if name not in normalized]
        )
        return SourceCube(self.model, cycle, normalized, members, flags)


def _cycle_from_attrs(attrs: Mapping) -> datetime:
    raw = attrs.get("cycle_time") or attrs.get("initialization_time")
    if raw is None:
        raise ValueError("cycle_time must be supplied or present in dataset attributes")
    parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    return parsed.replace(tzinfo=parsed.tzinfo or timezone.utc).astimezone(timezone.utc)


class HRRRAdapter(DatasetAdapter):
    def __init__(self): super().__init__("hrrr")


class RRFSAdapter(DatasetAdapter):
    def __init__(self): super().__init__("rrfs")


class REFSAdapter(DatasetAdapter):
    def __init__(self): super().__init__("refs")


class GEFSAdapter(DatasetAdapter):
    def __init__(self): super().__init__("gefs", strict_required=False)


ADAPTERS = {"hrrr": HRRRAdapter, "rrfs": RRFSAdapter, "refs": REFSAdapter, "gefs": GEFSAdapter}
