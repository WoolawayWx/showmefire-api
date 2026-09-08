"""Acquire the daily forecast-v1 source cube directly from NOAA via Herbie.

The acquisition layer is intentionally separate from the product pipeline.  It
downloads only indexed GRIB messages needed by forecast-v1, clips them to the
Missouri research buffer, converts the grid to explicit projected x/y axes,
and returns the same normalized ``SourceCube`` contract used by staged runs.
"""
from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pyproj
import xarray as xr

from .adapters import ADAPTERS, SourceCube
from .contracts import HORIZON_HOURS

logger = logging.getLogger(__name__)

MO_BUFFERED_BBOX = (-96.8, -88.1, 34.8, 41.8)  # west, east, south, north
SURFACE_SEARCHES = (
    # Keep critical fields separate so a cfgrib/read failure for one message
    # cannot discard temperature, humidity, and dewpoint together.
    r":TMP:2 m above ground:",
    r":DPT:2 m above ground:",
    r":RH:2 m above ground:",
    r":UGRD:10 m above ground:",
    r":VGRD:10 m above ground:",
    r":TCDC:entire atmosphere(?: \(considered as a single layer\))?:",
    r":(?:SOILW|MSTAV):",
    r":(?:HGT:planetary boundary layer|HPBL|MIXHT):",
    r":GUST:(?:surface|10 m above ground):",
    r":APCP:surface:",
    r":DSWRF:surface:",
    # HRRR exposes forecast-state and interval-accumulation WEASD messages;
    # only the state field belongs in the SWE forecast cube.
    r":WEASD:surface:(?:anl|[0-9]+ hour fcst)",
)
# Kept as a public union for inventory diagnostics and compatibility.
SURFACE_SEARCH = "|".join(SURFACE_SEARCHES)
UPPER_AIR_SEARCH = r":(?:TMP|UGRD|VGRD):(?:700|850) mb:"


@dataclass(frozen=True)
class AcquisitionSpec:
    public_name: str
    herbie_model: str
    product: str
    leads: tuple[int, ...]
    members: tuple[str | int | None, ...]
    domain: str | None = None
    required: bool = False


@dataclass(frozen=True)
class AcquisitionResult:
    cycle_time: datetime
    cubes: dict[str, SourceCube]
    warnings: tuple[str, ...]


def latest_publishable_12z(now: datetime | None = None, *, minimum_age_hours: int = 6) -> datetime:
    """Return the newest 12Z cycle old enough for its extended fields to exist."""
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    candidate = current.replace(hour=12, minute=0, second=0, microsecond=0)
    if current < candidate + timedelta(hours=minimum_age_hours):
        candidate -= timedelta(days=1)
    return candidate


def _parse_members(raw: str, *, integer: bool) -> tuple[str | int, ...]:
    values: list[str | int] = []
    for value in raw.split(","):
        value = value.strip()
        if value:
            values.append(int(value) if integer else value)
    if not values:
        raise ValueError("ensemble member list cannot be empty")
    return tuple(values)


def default_specs() -> tuple[AcquisitionSpec, ...]:
    """Build source requests, allowing member counts/products to be tuned by env."""
    refs = _parse_members(os.getenv("SMF_REFS_MEMBERS", "1,2,3,4,5,6,7"), integer=True)
    gefs = _parse_members(
        os.getenv("SMF_GEFS_MEMBERS", "c00," + ",".join(f"p{i:02d}" for i in range(1, 31))),
        integer=False,
    )
    rrfs_product = os.getenv("SMF_RRFS_PRODUCT", "natlev")
    rrfs_domain = os.getenv("SMF_RRFS_DOMAIN", "conus") or None
    return (
        AcquisitionSpec("hrrr", "hrrr", "sfc", tuple(range(49)), (None,), required=True),
        AcquisitionSpec("rrfs", "rrfs", rrfs_product, tuple(range(HORIZON_HOURS + 1)), ("control",), domain=rrfs_domain),
        # REFS members are delivered in the RRFS ensemble feed; keeping the
        # public source name separate preserves blend and provenance semantics.
        AcquisitionSpec("refs", "rrfs", rrfs_product, tuple(range(HORIZON_HOURS + 1)), refs, domain=rrfs_domain),
        AcquisitionSpec("gefs", "gefs", "atmos.25", tuple(range(0, HORIZON_HOURS + 1, 3)), gefs),
    )


def _sanitize(dataset: xr.Dataset) -> xr.Dataset:
    dataset = dataset.copy(deep=False)
    for name in dataset.variables:
        dataset[name].attrs.pop("dtype", None)
        dataset[name].attrs.pop("source", None)
        dataset[name].encoding.pop("dtype", None)
    return dataset


def _canonicalize_variables(dataset: xr.Dataset) -> xr.Dataset:
    """Use cfgrib names and GRIB metadata to produce stable adapter aliases."""
    aliases = {
        "2t": "t2m", "t2m": "t2m", "2d": "d2m", "d2m": "d2m",
        "2r": "r2", "r2": "r2", "10u": "u10", "u10": "u10",
        "10v": "v10", "v10": "v10", "gust": "gust", "10fg": "gust",
        "tp": "tp", "apcp": "apcp", "dswrf": "dswrf", "sdswrf": "dswrf", "tcc": "tcc",
        "tcdc": "tcc", "hpbl": "hpbl", "blh": "hpbl", "mixht": "mixht", "soilw": "soilw",
        "mstav": "soilw",
        "swvl1": "soilw", "weasd": "weasd", "sdwe": "weasd", "sde": "sde",
        "sd": "sde", "snod": "sde",
    }
    output: dict[str, xr.DataArray] = {}
    for name, data in dataset.data_vars.items():
        if name == "gribfile_projection":
            continue
        short_name = str(data.attrs.get("GRIB_shortName") or name).lower()
        canonical = aliases.get(short_name) or aliases.get(name.lower())
        if canonical:
            output.setdefault(canonical, data)
            continue
        if short_name in {"t", "u", "v"} and "isobaricInhPa" in data.dims:
            for level in (700, 850):
                if level in data.isobaricInhPa:
                    prefix = {"t": "t", "u": "u", "v": "v"}[short_name]
                    output[f"{prefix}{level}"] = data.sel(isobaricInhPa=level, drop=True)
    normalized = xr.Dataset(output, attrs=dataset.attrs)
    for coord in ("latitude", "longitude"):
        if coord in dataset.coords:
            normalized = normalized.assign_coords({coord: dataset.coords[coord]})
    projection = dataset.get("gribfile_projection")
    if projection is not None:
        normalized["gribfile_projection"] = projection
    if "r2" not in normalized and "t2m" in normalized and "d2m" in normalized:
        temperature = normalized.t2m - 273.15 if float(normalized.t2m.mean(skipna=True)) > 150 else normalized.t2m
        dewpoint = normalized.d2m - 273.15 if float(normalized.d2m.mean(skipna=True)) > 150 else normalized.d2m
        normalized["r2"] = (
            100 * np.exp(17.625 * dewpoint / (243.04 + dewpoint) - 17.625 * temperature / (243.04 + temperature))
        ).clip(0, 100).astype("float32")
        normalized.r2.attrs["units"] = "%"
    return normalized


def _time_axis(dataset: xr.Dataset, cycle: datetime) -> xr.Dataset:
    """Turn Herbie/cfgrib scalar time + step coordinates into valid-time axis."""
    dataset = _sanitize(dataset)
    initialization = np.datetime64(cycle.replace(tzinfo=None), "ns")
    if "valid_time" in dataset.dims:
        if "time" in dataset.coords and "time" not in dataset.dims:
            dataset = dataset.drop_vars("time")
        return dataset.rename(valid_time="time")
    if "step" in dataset.dims:
        valid = initialization + dataset.step.values.astype("timedelta64[ns]")
        if "time" in dataset.coords:
            dataset = dataset.drop_vars("time")
        return dataset.rename(step="time").assign_coords(time=valid)
    if "time" in dataset.dims:
        time_values = np.asarray(dataset.time.values)
        if len(np.unique(time_values)) != len(time_values):
            # Some GRIB indexes expose duplicate messages (notably HRRR SWE).
            # FastHerbie concatenates those records on initialization time;
            # coalesce complementary/non-null values before using valid time.
            dataset = dataset.groupby("time").first(skipna=True)
        if "valid_time" in dataset.coords and dataset.valid_time.dims == dataset.time.dims:
            valid = dataset.valid_time.values
            dataset = dataset.drop_vars("valid_time").assign_coords(time=valid)
        elif "valid_time" in dataset.coords and dataset.valid_time.ndim == 0 and dataset.sizes["time"] == 1:
            valid = np.asarray(dataset.valid_time.values).reshape(-1)[0]
            dataset = dataset.drop_vars("valid_time").assign_coords(time=[valid])
        return dataset
    valid = dataset.coords.get("valid_time")
    if valid is not None:
        value = np.asarray(valid.values).reshape(-1)[0]
        dataset = dataset.drop_vars([name for name in ("time", "valid_time", "step") if name in dataset.coords])
        return dataset.expand_dims(time=[value])
    step = dataset.coords.get("step")
    value = initialization if step is None else initialization + np.asarray(step.values).reshape(-1)[0]
    dataset = dataset.drop_vars([name for name in ("time", "step") if name in dataset.coords])
    return dataset.expand_dims(time=[value])


def _merge_herbie_result(value, cycle: datetime) -> xr.Dataset:
    items = value if isinstance(value, list) else [value]
    if not items:
        raise RuntimeError("Herbie returned no datasets")
    timed = [_time_axis(item, cycle) for item in items]
    merged = xr.merge(timed, compat="override", join="outer")
    return merged.sortby("time")


def _detect_crs(dataset: xr.Dataset) -> pyproj.CRS:
    projection = dataset.get("gribfile_projection")
    if projection is not None and projection.attrs.get("grid_mapping_name"):
        return pyproj.CRS.from_cf(projection.attrs)
    for variable in dataset.data_vars.values():
        attrs = variable.attrs
        if attrs.get("GRIB_gridType") == "lambert" and attrs.get("GRIB_LoVInDegrees") is not None:
            lon0 = float(attrs["GRIB_LoVInDegrees"])
            if lon0 > 180:
                lon0 -= 360
            lat0 = float(attrs["GRIB_LaDInDegrees"])
            lat1 = float(attrs["GRIB_Latin1InDegrees"])
            lat2 = float(attrs.get("GRIB_Latin2InDegrees", lat1))
            return pyproj.CRS.from_proj4(
                f"+proj=lcc +lat_0={lat0} +lon_0={lon0} +lat_1={lat1} "
                f"+lat_2={lat2} +R=6371229 +units=m +no_defs"
            )
    # GEFS is a regular geographic grid.
    return pyproj.CRS.from_epsg(4326)


def _clip_and_project_axes(dataset: xr.Dataset) -> xr.Dataset:
    if "longitude" not in dataset.coords or "latitude" not in dataset.coords:
        raise ValueError("Herbie dataset is missing latitude/longitude coordinates")
    longitude = xr.where(dataset.longitude > 180, dataset.longitude - 360, dataset.longitude)
    latitude = dataset.latitude
    west, east, south, north = MO_BUFFERED_BBOX
    mask = (longitude >= west) & (longitude <= east) & (latitude >= south) & (latitude <= north)
    if longitude.ndim == 1 and latitude.ndim == 1:
        clipped = dataset.where((longitude >= west) & (longitude <= east), drop=True)
        clipped = clipped.where((latitude >= south) & (latitude <= north), drop=True)
    elif mask.ndim == 1:
        clipped = dataset.where(mask, drop=True)
    else:
        rows, columns = np.where(mask.values)
        if not len(rows):
            raise ValueError("forecast grid does not intersect the Missouri buffer")
        ydim, xdim = mask.dims[-2:]
        clipped = dataset.isel({ydim: slice(rows.min(), rows.max() + 1), xdim: slice(columns.min(), columns.max() + 1)})
    longitude = xr.where(clipped.longitude > 180, clipped.longitude - 360, clipped.longitude)
    latitude = clipped.latitude
    crs = _detect_crs(clipped)
    if longitude.ndim == 1 and latitude.ndim == 1 and crs.is_geographic:
        x_values, y_values = longitude.values, latitude.values
        xdim, ydim = longitude.dims[0], latitude.dims[0]
    else:
        xgrid, ygrid = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(
            longitude.values, latitude.values
        )
        ydim, xdim = longitude.dims[-2:]
        x_values, y_values = np.asarray(xgrid)[0, :], np.asarray(ygrid)[:, 0]
    rename = {name: target for name, target in ((xdim, "x"), (ydim, "y")) if name != target}
    clipped = clipped.rename(rename).assign_coords(x=("x", x_values), y=("y", y_values))
    if clipped.x.values[0] > clipped.x.values[-1]:
        clipped = clipped.sortby("x")
    if clipped.y.values[0] < clipped.y.values[-1]:
        clipped = clipped.sortby("y", ascending=False)
    clipped.attrs["crs"] = crs.to_string()
    return clipped


def _fetch_member(
    spec: AcquisitionSpec,
    cycle: datetime,
    member: str | int | None,
    save_dir: Path,
    factory: Callable,
) -> xr.Dataset:
    kwargs = dict(
        DATES=[cycle.replace(tzinfo=None)], fxx=list(spec.leads), model=spec.herbie_model,
        product=spec.product, save_dir=save_dir, max_threads=int(os.getenv("SMF_HERBIE_THREADS", "4")),
        verbose=False,
    )
    if member is not None:
        kwargs["member"] = member
    if spec.domain is not None or spec.herbie_model == "rrfs":
        kwargs["domain"] = spec.domain
    client = factory(**kwargs)
    groups = []
    errors = []
    for search in SURFACE_SEARCHES:
        attempts = max(1, int(os.getenv("SMF_HERBIE_QUERY_ATTEMPTS", "3")))
        for attempt in range(1, attempts + 1):
            try:
                groups.append(_merge_herbie_result(client.xarray(search, remove_grib=True), cycle))
                break
            except Exception as error:
                if attempt == attempts:
                    errors.append(f"{search}:{type(error).__name__}")
                else:
                    logger.warning(
                        "%s %s query failed on attempt %d/%d: %s",
                        spec.public_name, search, attempt, attempts, error,
                    )
                    time.sleep(min(attempt, 2))
    if not groups:
        raise RuntimeError("Herbie returned no forecast-v1 surface fields")
    surface = xr.merge(groups, compat="override", join="outer")
    if errors:
        surface.attrs["acquisition_warnings"] = ";".join(errors)
    try:
        upper = _merge_herbie_result(client.xarray(UPPER_AIR_SEARCH, remove_grib=True), cycle)
        surface = xr.merge([surface, upper], compat="override", join="outer")
    except Exception as error:
        logger.info("%s optional upper-air fields unavailable for %s: %s", spec.public_name, member, error)
    canonical = _canonicalize_variables(surface)
    flags = []
    if "weasd" not in canonical and cycle.month in {5, 6, 7, 8, 9} and "t2m" in canonical:
        canonical["weasd"] = xr.zeros_like(canonical.t2m, dtype="float32")
        canonical.weasd.attrs.update(units="mm", long_name="seasonal zero-snow fallback")
        flags.append("seasonal_swe_assumed_zero")
    if flags:
        canonical.attrs["acquisition_quality_flags"] = ";".join(flags)
    return _clip_and_project_axes(canonical)


def _hourly_gefs(dataset: xr.Dataset, cycle: datetime) -> xr.Dataset:
    target = np.array([np.datetime64(cycle.replace(tzinfo=None)) + np.timedelta64(hour, "h") for hour in range(HORIZON_HOURS + 1)])
    precipitation = next((name for name in ("tp", "apcp", "precipitation_increment") if name in dataset), None)
    continuous = dataset.drop_vars(precipitation) if precipitation else dataset
    hourly = continuous.interp(time=target)
    if precipitation:
        source = dataset[precipitation]
        output = xr.full_like(source.reindex(time=target), np.nan, dtype="float32")
        source_times = source.time.values.astype("datetime64[h]")
        for index, valid in enumerate(source_times):
            end = int((valid - np.datetime64(cycle.replace(tzinfo=None), "h")) / np.timedelta64(1, "h"))
            previous = 0 if index == 0 else int((source_times[index - 1] - np.datetime64(cycle.replace(tzinfo=None), "h")) / np.timedelta64(1, "h"))
            width = max(end - previous, 1)
            for lead in range(previous + (0 if index == 0 else 1), end + 1):
                output.loc[{"time": target[lead]}] = source.isel(time=index) / width
        output.attrs.update(source.attrs, accumulation_semantics="incremental")
        hourly[precipitation] = output
    return hourly


def acquire_source(
    spec: AcquisitionSpec,
    cycle: datetime,
    cache_dir: str | Path,
    *,
    fast_herbie_factory: Callable | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> SourceCube:
    if fast_herbie_factory is None:
        from herbie import FastHerbie
        fast_herbie_factory = FastHerbie
    def fetch(member):
        return member, _fetch_member(spec, cycle, member, Path(cache_dir), fast_herbie_factory)

    workers = min(len(spec.members), max(1, int(os.getenv("SMF_ENSEMBLE_MEMBER_THREADS", "2"))))
    fetched = []
    if workers == 1:
        iterator = map(fetch, spec.members)
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"forecast-{spec.public_name}") as executor:
            iterator = executor.map(fetch, spec.members)
            for completed, item in enumerate(iterator, 1):
                fetched.append(item)
                if progress_callback:
                    progress_callback({"event": "member_completed", "member": "deterministic" if item[0] is None else str(item[0]), "completed": completed, "total": len(spec.members)})
    if workers == 1:
        for completed, item in enumerate(iterator, 1):
            fetched.append(item)
            if progress_callback:
                progress_callback({"event": "member_completed", "member": "deterministic" if item[0] is None else str(item[0]), "completed": completed, "total": len(spec.members)})
    member_datasets = []
    member_names = []
    for member, dataset in fetched:
        member_name = "deterministic" if member is None else str(member)
        member_datasets.append(dataset.expand_dims(member=[member_name]))
        member_names.append(member_name)
    combined = xr.concat(member_datasets, dim="member", join="outer", compat="override", coords="minimal")
    if spec.public_name == "gefs":
        combined = _hourly_gefs(combined, cycle)
    combined.attrs.update(cycle_time=cycle.isoformat().replace("+00:00", "Z"), acquisition="herbie-indexed-grib")
    try:
        cube = ADAPTERS[spec.public_name]().normalize(combined, cycle)
    except ValueError as error:
        query_errors = combined.attrs.get("acquisition_warnings")
        detail = f"; Herbie query failures: {query_errors}" if query_errors else ""
        raise ValueError(f"{error}{detail}") from error
    acquisition_flags = tuple(filter(None, str(combined.attrs.get("acquisition_quality_flags", "")).split(";")))
    if acquisition_flags:
        cube = SourceCube(
            cube.model, cube.cycle_time, cube.dataset, cube.member_ids,
            tuple(sorted(set(cube.quality_flags).union(acquisition_flags))), cube.checksum, cube.object_key,
        )
    expected = len(spec.leads) if spec.public_name != "gefs" else HORIZON_HOURS + 1
    if cube.dataset.sizes["time"] != expected:
        raise RuntimeError(f"{spec.public_name} incomplete: expected {expected} valid hours, received {cube.dataset.sizes['time']}")
    return cube


def acquire_cycle(
    cycle: datetime,
    cache_dir: str | Path,
    *,
    specs: Iterable[AcquisitionSpec] | None = None,
    fast_herbie_factory: Callable | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> AcquisitionResult:
    cycle = cycle.astimezone(timezone.utc)
    cubes: dict[str, SourceCube] = {}
    warnings: list[str] = []
    requested_specs = tuple(specs or default_specs())
    for source_index, spec in enumerate(requested_specs, 1):
        def source_progress(update: dict) -> None:
            if progress_callback:
                progress_callback({
                    **update, "model": spec.public_name, "source_index": source_index,
                    "source_total": len(requested_specs),
                })

        if progress_callback:
            progress_callback({
                "event": "source_started", "model": spec.public_name,
                "source_index": source_index, "source_total": len(requested_specs),
                "completed": 0, "total": len(spec.members),
            })
        if spec.public_name == "refs" and "rrfs" not in cubes:
            warnings.append("source_unavailable:refs:rrfs_feed_unavailable")
            if progress_callback:
                progress_callback({
                    "event": "source_skipped", "model": spec.public_name,
                    "source_index": source_index, "source_total": len(requested_specs),
                    "reason": "rrfs_feed_unavailable",
                })
            continue
        try:
            cubes[spec.public_name] = acquire_source(
                spec, cycle, cache_dir, fast_herbie_factory=fast_herbie_factory,
                progress_callback=source_progress,
            )
            if progress_callback:
                progress_callback({
                    "event": "source_completed", "model": spec.public_name,
                    "source_index": source_index, "source_total": len(requested_specs),
                    "completed": len(spec.members), "total": len(spec.members),
                })
        except Exception as error:
            if progress_callback:
                progress_callback({
                    "event": "source_failed", "model": spec.public_name,
                    "source_index": source_index, "source_total": len(requested_specs),
                    "reason": type(error).__name__,
                })
            if spec.required:
                raise RuntimeError(f"required {spec.public_name} acquisition failed: {error}") from error
            warnings.append(f"source_unavailable:{spec.public_name}:{type(error).__name__}")
            logger.warning("Optional %s acquisition failed: %s", spec.public_name, error, exc_info=True)
    if "rrfs" not in cubes and "gefs" not in cubes:
        raise RuntimeError("lead hours 49-72 require RRFS or the GEFS coarse fallback")
    if "rrfs" not in cubes:
        warnings.append("coarse_synoptic_fallback:gefs:49-72")
    return AcquisitionResult(cycle, cubes, tuple(warnings))
