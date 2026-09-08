from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from zoneinfo import ZoneInfo

import numpy as np
import xarray as xr
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject

from core.database import get_db_path
from core.config import FORECAST_V1_DIR, GIS_DIR
from .adapters import ADAPTERS, SourceCube
from .artifacts import artifact_record, promote_directory, sha256_file, write_cog, write_manifest, write_netcdf, write_points_parquet, write_static_graphic
from .contracts import HORIZON_HOURS, PUBLIC_GRID, PUBLIC_LAYER_STYLES, QUALITY_BITS, TIME_COUNT, VARIABLE_UNITS, run_id_for_cycle, utc_rfc3339
from .engine import build_forecast_cube, local_day_slices
from .repository import ensure_schema, json_text, replace_assets, transaction, upsert_run
from .r2_store import ForecastR2Store


SCHEMA_VERSION = "forecast-v1"
PUBLIC_RASTER_VARIABLES = (
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "wind_gust_10m",
    "precipitation_increment", "precipitation_accumulated", "fuel_moisture_p10",
    "fuel_moisture_p50", "fuel_moisture_p90", "fire_danger", "meteorological_confidence",
    "category_confidence", "probability_rh_le_25", "probability_gust_ge_30mph", "probability_concurrent",
)
DAILY_RASTER_MAP = {
    "peak_fire_danger": "fire_danger",
    "minimum_rh": "relative_humidity_2m",
    "minimum_fuel_moisture_p50": "fuel_moisture_p50",
    "maximum_temperature": "temperature_2m",
    "maximum_wind": "wind_speed_10m",
    "maximum_gust": "wind_gust_10m",
    "total_precipitation": "precipitation_increment",
    "category_confidence": "category_confidence",
}
STATIC_TITLES = {
    "peak_fire_danger": "Peak Fire Danger", "minimum_rh": "Minimum Relative Humidity",
    "minimum_fuel_moisture_p50": "Minimum 10-Hour Fuel Moisture (P50)", "maximum_temperature": "Maximum Temperature",
    "maximum_wind": "Maximum Sustained Wind", "maximum_gust": "Maximum Wind Gust",
    "total_precipitation": "Total Precipitation", "category_confidence": "Fire-Danger Category Confidence",
    "maximum_snow_water_equivalent": "Maximum Snow-Water Equivalent",
}


@dataclass(frozen=True)
class FuelInitialization:
    values: xr.DataArray
    quality_mask: xr.DataArray
    source_mask: xr.DataArray


def initialize_fuel_moisture(
    atmosphere: xr.Dataset,
    cycle_time: datetime,
    observations: list[dict],
    spatial_analysis: xr.DataArray | None = None,
    *,
    station_radius_m: float = 45_000,
) -> FuelInitialization:
    """Build the public-grid initial state without using future observations."""
    from .contracts import QUALITY_BITS, SOURCE_BITS
    from .engine import nelson_emc

    template = atmosphere.temperature_2m.isel(time=0)
    if spatial_analysis is not None:
        values = spatial_analysis.broadcast_like(template).clip(1, 40).astype("float32")
        quality = xr.full_like(template, QUALITY_BITS["fuel_spatial_init"], dtype="uint16")
        sources = xr.zeros_like(template, dtype="uint16")
    else:
        values = nelson_emc(template, atmosphere.relative_humidity_2m.isel(time=0))
        quality = xr.full_like(template, QUALITY_BITS["fuel_physics_init"], dtype="uint16")
        sources = xr.zeros_like(template, dtype="uint16")
    cutoff = cycle_time.astimezone(timezone.utc).timestamp() - 3 * 3600
    eligible = []
    transformer = Transformer.from_crs("EPSG:4326", PUBLIC_GRID.crs, always_xy=True)
    for observation in observations:
        try:
            observed_at = datetime.fromisoformat(str(observation["observed_at_utc"]).replace("Z", "+00:00")).astimezone(timezone.utc)
            moisture = float(observation["fuel_moisture"])
        except (KeyError, TypeError, ValueError):
            continue
        if observation.get("network") != "RAWS" or observation.get("qc_state") != "pass":
            continue
        if not cutoff <= observed_at.timestamp() <= cycle_time.astimezone(timezone.utc).timestamp() or not 1 <= moisture <= 40:
            continue
        sx, sy = transformer.transform(float(observation["longitude"]), float(observation["latitude"]))
        eligible.append((sx, sy, moisture))
    if eligible:
        xx, yy = np.meshgrid(template.x.values, template.y.values)
        best_distance = np.full(template.shape, np.inf)
        best_value = np.full(template.shape, np.nan, dtype=np.float32)
        for sx, sy, moisture in eligible:
            distance = np.hypot(xx - sx, yy - sy)
            replace = distance < best_distance
            best_distance[replace] = distance[replace]
            best_value[replace] = moisture
        station_cells = best_distance <= station_radius_m
        values = xr.where(station_cells, best_value, values).astype("float32")
        quality = xr.where(station_cells, 0, quality).astype("uint16")
        sources = xr.where(station_cells, SOURCE_BITS["observations"], sources).astype("uint16")
    return FuelInitialization(values, quality, sources)


def public_coordinates() -> tuple[np.ndarray, np.ndarray]:
    x = PUBLIC_GRID.west + (np.arange(PUBLIC_GRID.width) + 0.5) * PUBLIC_GRID.resolution_m
    y = PUBLIC_GRID.north - (np.arange(PUBLIC_GRID.height) + 0.5) * PUBLIC_GRID.resolution_m
    return x, y


def archive_source_cube(cube: SourceCube, archive_root: str | Path, r2: ForecastR2Store | None = None) -> SourceCube:
    """Pack one already-clipped native-grid source cycle and deduplicate by checksum."""
    root = Path(archive_root)
    cube.dataset.attrs.setdefault("source_member_count", len(cube.member_ids))
    cycle_path = cube.cycle_time.astimezone(timezone.utc).strftime("%Y%m%d/%H")
    temporary = root / ".staging" / f"{cube.model}-{cube.cycle_time:%Y%m%d%H}.nc"
    write_netcdf(cube.dataset, temporary)
    checksum = sha256_file(temporary)
    object_key = f"forecast-v1/sources/{cube.model}/{cycle_path}/{checksum}.nc"
    final = root / "sources" / cube.model / cycle_path / f"{checksum}.nc"
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        if sha256_file(final) != checksum:
            raise RuntimeError(f"source checksum collision at {final}")
        temporary.unlink()
    else:
        os.replace(temporary, final)
    if r2 and r2.configured:
        r2.upload_immutable(final, object_key, checksum)
    return SourceCube(cube.model, cube.cycle_time, cube.dataset, cube.member_ids, cube.quality_flags, checksum, object_key)


def source_member_count(cube: SourceCube) -> int:
    return int(cube.dataset.attrs.get("source_member_count", len(cube.member_ids)))


def summarize_ensemble_for_public(cube: SourceCube) -> SourceCube:
    """Preserve ensemble mean/spread while avoiding full-member reprojection."""
    count = source_member_count(cube)
    if count <= 2 or "member" not in cube.dataset.dims:
        return cube
    variables = {}
    summary_members = ["ensemble_mean_minus_spread", "ensemble_mean_plus_spread"]
    for name, data in cube.dataset.data_vars.items():
        if "member" not in data.dims:
            variables[name] = data
            continue
        mean = data.mean("member", skipna=True).astype("float32")
        spread = data.std("member", skipna=True).astype("float32")
        variables[name] = xr.concat([mean - spread, mean + spread], dim="member").assign_coords(member=summary_members).astype("float32")
    dataset = xr.Dataset(
        variables,
        attrs={**cube.dataset.attrs, "source_member_count": count, "public_member_reduction": "mean_plus_minus_population_spread"},
    )
    return SourceCube(cube.model, cube.cycle_time, dataset, tuple(summary_members), cube.quality_flags, cube.checksum, cube.object_key)


def regrid_to_public(cube: SourceCube) -> SourceCube:
    """Reproject a clipped native-grid cube onto the authoritative public grid."""
    if "x" not in cube.dataset.coords or "y" not in cube.dataset.coords:
        raise ValueError(f"{cube.model} must expose projected x/y coordinates before public regridding")
    source_crs = str(cube.dataset.attrs.get("crs", PUBLIC_GRID.crs))
    x, y = public_coordinates()
    variables: dict[str, xr.DataArray] = {}
    nearest = {"snow_water_equivalent"}
    if source_crs == PUBLIC_GRID.crs:
        for name, data in cube.dataset.data_vars.items():
            variables[name] = data.interp(x=x, y=y, method="nearest" if name in nearest else "linear")
        dataset = xr.Dataset(variables, attrs={**cube.dataset.attrs, "crs": PUBLIC_GRID.crs, "grid_id": PUBLIC_GRID.id})
        return SourceCube(cube.model, cube.cycle_time, dataset, cube.member_ids, cube.quality_flags, cube.checksum, cube.object_key)

    source_x = np.asarray(cube.dataset.x.values, dtype=float)
    source_y = np.asarray(cube.dataset.y.values, dtype=float)
    if source_x.size < 2 or source_y.size < 2:
        raise ValueError(f"{cube.model} source grid is too small to reproject")
    dx = abs(float(np.median(np.diff(source_x))))
    dy = abs(float(np.median(np.diff(source_y))))
    source_transform = from_origin(float(source_x.min() - dx / 2), float(source_y.max() + dy / 2), dx, dy)
    destination_transform = from_origin(PUBLIC_GRID.west, PUBLIC_GRID.north, PUBLIC_GRID.resolution_m, PUBLIC_GRID.resolution_m)
    for name, data in cube.dataset.data_vars.items():
        if "x" not in data.dims or "y" not in data.dims:
            continue
        source = data.transpose(..., "y", "x")
        values = np.asarray(source.values, dtype=np.float32)
        if source_y[0] < source_y[-1]:
            values = np.flip(values, axis=-2)
        if source_x[0] > source_x[-1]:
            values = np.flip(values, axis=-1)
        leading_shape = values.shape[:-2]
        destination = np.full((*leading_shape, PUBLIC_GRID.height, PUBLIC_GRID.width), np.nan, dtype=np.float32)
        indexes = np.ndindex(leading_shape) if leading_shape else [()]
        for index in indexes:
            reproject(
                values[index], destination[index], src_transform=source_transform, src_crs=source_crs,
                dst_transform=destination_transform, dst_crs=PUBLIC_GRID.crs,
                src_nodata=np.nan, dst_nodata=np.nan,
                resampling=Resampling.nearest if name in nearest else Resampling.bilinear,
            )
        dims = (*source.dims[:-2], "y", "x")
        coords = {dim: source.coords[dim] for dim in source.dims[:-2] if dim in source.coords}
        coords.update(x=x, y=y)
        variables[name] = xr.DataArray(destination, dims=dims, coords=coords, attrs=data.attrs)
    dataset = xr.Dataset(variables, attrs={**cube.dataset.attrs, "crs": PUBLIC_GRID.crs, "grid_id": PUBLIC_GRID.id})
    return SourceCube(cube.model, cube.cycle_time, dataset, cube.member_ids, cube.quality_flags, cube.checksum, cube.object_key)


def reproject_fuel_quantiles_to_public(spatial_output: xr.Dataset) -> dict[str, xr.DataArray]:
    """Reproject immutable-grid P10/P50/P90 residual output onto the public grid."""
    x_name = "x" if "x" in spatial_output.coords else "longitude"
    y_name = "y" if "y" in spatial_output.coords else "latitude"
    if x_name not in spatial_output.coords or y_name not in spatial_output.coords:
        raise ValueError("spatial fuel output must provide x/y or longitude/latitude coordinates")
    source_x = np.asarray(spatial_output[x_name].values, dtype=float)
    source_y = np.asarray(spatial_output[y_name].values, dtype=float)
    if source_x.size < 2 or source_y.size < 2:
        raise ValueError("spatial fuel grid requires at least two cells per axis")
    dx = abs(float(np.median(np.diff(source_x))))
    dy = abs(float(np.median(np.diff(source_y))))
    left = float(source_x.min() - dx / 2)
    top = float(source_y.max() + dy / 2)
    source_transform = from_origin(left, top, dx, dy)
    destination_transform = from_origin(PUBLIC_GRID.west, PUBLIC_GRID.north, PUBLIC_GRID.resolution_m, PUBLIC_GRID.resolution_m)
    destination_x, destination_y = public_coordinates()
    result = {}
    aliases = {"p10": ("p10", "fuel_moisture_p10"), "p50": ("p50", "fuel_moisture_p50"), "p90": ("p90", "fuel_moisture_p90")}
    for output_name, candidates in aliases.items():
        variable = next((name for name in candidates if name in spatial_output), None)
        if variable is None:
            raise ValueError(f"spatial fuel output is missing {output_name}")
        source = spatial_output[variable].transpose(..., y_name, x_name)
        source_values = np.asarray(source.values, dtype=np.float32)
        if source_y[0] < source_y[-1]:
            source_values = np.flip(source_values, axis=-2)
        if source_x[0] > source_x[-1]:
            source_values = np.flip(source_values, axis=-1)
        leading_shape = source_values.shape[:-2]
        destination = np.full((*leading_shape, PUBLIC_GRID.height, PUBLIC_GRID.width), np.nan, dtype=np.float32)
        for index in np.ndindex(leading_shape or (1,)):
            source_slice = source_values[index] if leading_shape else source_values
            destination_slice = destination[index] if leading_shape else destination
            reproject(
                source_slice, destination_slice, src_transform=source_transform,
                src_crs=str(spatial_output.attrs.get("crs", "EPSG:4326")),
                dst_transform=destination_transform, dst_crs=PUBLIC_GRID.crs,
                src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.bilinear,
            )
        dims = (*source.dims[:-2], "y", "x")
        coords = {name: source.coords[name] for name in source.dims[:-2] if name in source.coords}
        coords.update(y=destination_y, x=destination_x)
        result[output_name] = xr.DataArray(destination, dims=dims, coords=coords, attrs={"units": "%", "grid_id": PUBLIC_GRID.id})
    return result


def _value(value: Any) -> Any:
    if value is None:
        return None
    scalar = value.item() if hasattr(value, "item") else value
    if isinstance(scalar, (float, np.floating)) and not np.isfinite(scalar):
        return None
    if isinstance(scalar, (np.integer,)):
        return int(scalar)
    return scalar


def extract_station_rows(dataset: xr.Dataset, stations: list[dict], run_id: str) -> tuple[list[dict], list[dict]]:
    transformer = Transformer.from_crs("EPSG:4326", PUBLIC_GRID.crs, always_xy=True)
    hourly_rows: list[dict] = []
    daily_rows: list[dict] = []
    day_groups = local_day_slices(dataset.time)
    for station in stations:
        px, py = transformer.transform(float(station["longitude"]), float(station["latitude"]))
        point = dataset.sel(x=px, y=py, method="nearest")
        for lead in range(point.sizes["time"]):
            valid = datetime.fromisoformat(np.datetime_as_string(point.time.values[lead], unit="s")).replace(tzinfo=timezone.utc)
            danger = _value(point.fire_danger.values[lead])
            hourly_rows.append({
                "run_id": run_id, "station_id": station["station_id"], "valid_time_utc": utc_rfc3339(valid), "lead_hour": lead,
                "temperature_c": _value(point.temperature_2m.values[lead]), "dewpoint_c": _value(point.dewpoint_2m.values[lead]) if "dewpoint_2m" in point else None,
                "relative_humidity": _value(point.relative_humidity_2m.values[lead]), "wind_u_ms": _value(point.wind_u_10m.values[lead]),
                "wind_v_ms": _value(point.wind_v_10m.values[lead]), "wind_speed_ms": _value(point.wind_speed_10m.values[lead]),
                "wind_gust_ms": _value(point.wind_gust_10m.values[lead]), "precipitation_mm": _value(point.precipitation_increment.values[lead]),
                "solar_wm2": _value(point.shortwave_down.values[lead]), "cloud_cover_pct": _value(point.cloud_cover.values[lead]),
                "mixing_height_m": _value(point.mixing_height.values[lead]), "fuel_moisture_p10": _value(point.fuel_moisture_p10.values[lead]),
                "fuel_moisture_p50": _value(point.fuel_moisture_p50.values[lead]), "fuel_moisture_p90": _value(point.fuel_moisture_p90.values[lead]),
                "danger_class": None if danger == 255 else danger, "danger_score": None,
                "meteorological_confidence": None if _value(point.meteorological_confidence.values[lead]) == 255 else _value(point.meteorological_confidence.values[lead]),
                "category_confidence": None if _value(point.category_confidence.values[lead]) == 255 else _value(point.category_confidence.values[lead]),
                "probability_rh_le_25": None if _value(point.probability_rh_le_25.values[lead]) == 255 else _value(point.probability_rh_le_25.values[lead]),
                "probability_gust_ge_30mph": None if _value(point.probability_gust_ge_30mph.values[lead]) == 255 else _value(point.probability_gust_ge_30mph.values[lead]),
                "probability_concurrent": None if _value(point.probability_concurrent.values[lead]) == 255 else _value(point.probability_concurrent.values[lead]),
                "source_mask": _value(point.source_mask.values[lead]), "quality_mask": _value(point.quality_mask.values[lead]),
            })
        for day_index, (local_date, indexes) in enumerate(day_groups, 1):
            rows = [hourly_rows[-point.sizes["time"] + i] for i in indexes]
            available = [row for row in rows if row["danger_class"] is not None]
            peak = max((row["danger_class"] for row in available), default=None)
            critical = [row for row in available if row["danger_class"] >= 2]
            at_peak = [row for row in available if row["danger_class"] == peak]
            confidence_window = at_peak if peak is not None and peak >= 2 else [
                row for row in rows
                if 12 <= datetime.fromisoformat(row["valid_time_utc"].replace("Z", "+00:00")).astimezone(ZoneInfo("America/Chicago")).hour <= 18
            ]
            def numbers(name: str) -> list[float]: return [float(row[name]) for row in rows if row[name] is not None]
            daily_rows.append({
                "run_id": run_id, "station_id": station["station_id"], "local_date": local_date, "day_index": day_index,
                "temperature_low_c": min(numbers("temperature_c"), default=None), "temperature_high_c": max(numbers("temperature_c"), default=None),
                "minimum_rh": min(numbers("relative_humidity"), default=None), "maximum_wind_ms": max(numbers("wind_speed_ms"), default=None),
                "maximum_gust_ms": max(numbers("wind_gust_ms"), default=None), "precipitation_total_mm": sum(numbers("precipitation_mm")),
                "minimum_fuel_moisture_p50": min(numbers("fuel_moisture_p50"), default=None),
                "fuel_moisture_p10": min(numbers("fuel_moisture_p10"), default=None), "fuel_moisture_p90": min(numbers("fuel_moisture_p90"), default=None),
                "peak_category": peak, "critical_window_start_utc": critical[0]["valid_time_utc"] if critical else None,
                "critical_window_end_utc": critical[-1]["valid_time_utc"] if critical else None,
                "concurrent_threshold_hours": sum(1 for row in rows if (row["probability_concurrent"] or 0) >= 50),
                "category_confidence": int(np.median([row["category_confidence"] for row in confidence_window if row["category_confidence"] is not None])) if confidence_window and any(row["category_confidence"] is not None for row in confidence_window) else None,
                "meteorological_confidence": int(np.median(numbers("meteorological_confidence"))) if numbers("meteorological_confidence") else None,
            })
    return hourly_rows, daily_rows


def extract_source_member_rows(cubes: dict[str, SourceCube], stations: list[dict], run_id: str) -> list[dict]:
    rows: list[dict] = []
    selected = ("temperature_2m", "relative_humidity_2m", "wind_u_10m", "wind_v_10m", "wind_gust_10m", "precipitation_increment")
    for station in stations:
        for model, cube in cubes.items():
            source_crs = str(cube.dataset.attrs.get("crs", PUBLIC_GRID.crs))
            transformer = Transformer.from_crs("EPSG:4326", source_crs, always_xy=True)
            px, py = transformer.transform(float(station["longitude"]), float(station["latitude"]))
            point = cube.dataset.sel(x=px, y=py, method="nearest")
            for member_index, member in enumerate(point.member.values):
                for lead in range(point.sizes["time"]):
                    valid = datetime.fromisoformat(np.datetime_as_string(point.time.values[lead], unit="s")).replace(tzinfo=timezone.utc)
                    row = {"run_id": run_id, "station_id": station["station_id"], "model": model, "member": str(member), "valid_time_utc": utc_rfc3339(valid), "lead_hour": lead}
                    for variable in selected:
                        if variable in point:
                            row[variable] = _value(point[variable].isel(member=member_index, time=lead).values)
                    rows.append(row)
    return rows


def _insert_rows(connection, table: str, rows: list[dict]) -> None:
    if not rows:
        return
    columns = tuple(rows[0])
    connection.executemany(
        f"INSERT OR REPLACE INTO {table} ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
        ([row[name] for name in columns] for row in rows),
    )


def register_source_cycles(cubes: list[SourceCube], archive_root: str | Path, db_path: str | Path | None = None) -> None:
    database = db_path or get_db_path()
    ensure_schema(database)
    root = Path(archive_root)
    with transaction(database) as connection:
        for cube in cubes:
            if not cube.object_key or not cube.checksum:
                raise ValueError(f"{cube.model} must be archived before registration")
            source_path = root / cube.object_key.removeprefix("forecast-v1/")
            connection.execute(
                "INSERT OR IGNORE INTO source_cycles(model,initialization_time_utc,member_count,variables_json,source_status,object_key,checksum,byte_size,acquisition_json) VALUES(?,?,?,?,?,?,?,?,?)",
                (cube.model, utc_rfc3339(cube.cycle_time), source_member_count(cube), json_text(sorted(cube.dataset.data_vars)), "verified", cube.object_key, cube.checksum, source_path.stat().st_size, json_text({"qualityFlags": list(cube.quality_flags)})),
            )


def _update_legacy_aliases(final: Path, root: Path, aliases: dict[str, str], r2: ForecastR2Store) -> None:
    for alias_key, relative_source in aliases.items():
        source = final / relative_source
        if not source.is_file():
            raise RuntimeError(f"legacy alias source is missing: {source}")
        local_alias = root / alias_key
        local_alias.parent.mkdir(parents=True, exist_ok=True)
        temporary = local_alias.with_suffix(local_alias.suffix + ".tmp")
        shutil.copy2(source, temporary)
        os.replace(temporary, local_alias)
        if r2.configured:
            r2.put_alias(source, alias_key)


def _publish_run_impl(
    cubes: dict[str, SourceCube], stations: list[dict], *, initial_fuel_moisture: float | xr.DataArray | FuelInitialization,
    publish_root: str | Path, config_version: str = "beta-1", model_version: str = "physics-only-beta-1",
    make_public: bool = False, db_path: str | Path | None = None, r2_store: ForecastR2Store | None = None,
    confidence_components: dict[str, xr.DataArray | float] | None = None,
    predicted_residuals: dict[str, xr.DataArray] | None = None,
    fuel_quantile_residuals: dict[str, xr.DataArray] | None = None,
    source_member_rows: list[dict] | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    cycle = next(iter(cubes.values())).cycle_time.astimezone(timezone.utc)
    if cycle.hour != 12:
        raise ValueError("forecast-v1 public pipeline accepts only the daily 12Z cycle")
    run_id = run_id_for_cycle(cycle)
    root = Path(publish_root)
    staging = root / ".staging" / run_id
    final = root / "runs" / cycle.strftime("%Y/%m/%d") / run_id
    if staging.exists():
        # The path is fully resolved from the fixed publish root and validated run id.
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    now = utc_rfc3339(datetime.now(timezone.utc))
    if progress_callback:
        progress_callback({"phase": "forecast_cube", "message": "Calculating the hourly forecast and fire danger", "fraction": 0.03})
    initialization = initial_fuel_moisture
    initial_values = initialization.values if isinstance(initialization, FuelInitialization) else initialization
    forecast, daily = build_forecast_cube(
        cubes, initial_values, predicted_residuals=predicted_residuals,
        fuel_quantile_residuals=fuel_quantile_residuals, confidence_components=confidence_components,
    )
    if isinstance(initialization, FuelInitialization):
        forecast["quality_mask"] = forecast.quality_mask | initialization.quality_mask.broadcast_like(forecast.quality_mask)
        forecast["source_mask"] = forecast.source_mask | initialization.source_mask.broadcast_like(forecast.source_mask)
    forecast.attrs.update(schema_version=SCHEMA_VERSION, run_id=run_id, cycle_time=utc_rfc3339(cycle), grid_id=PUBLIC_GRID.id, crs=PUBLIC_GRID.crs)
    assets: list[dict] = []
    warnings = sorted({flag for cube in cubes.values() for flag in cube.quality_flags})
    if bool(((forecast.quality_mask & QUALITY_BITS["coarse_synoptic_fallback"]) != 0).any()):
        warnings.append("coarse_synoptic_fallback:gefs:49-72")
    if confidence_components is None:
        warnings.append("meteorological_confidence_inputs_unavailable")
    warnings = sorted(set(warnings))
    valid_times = [utc_rfc3339(datetime.fromisoformat(np.datetime_as_string(value, unit="s")).replace(tzinfo=timezone.utc)) for value in forecast.time.values]
    cube_path = write_netcdf(forecast, staging / "derived" / "forecast_cube.nc")
    assets.append(artifact_record(cube_path, run_id=run_id, kind="cube", object_key=f"forecast-v1/runs/{cycle:%Y/%m/%d}/{run_id}/derived/forecast_cube.nc", dtype="NetCDF4", valid_start=valid_times[0], valid_end=valid_times[-1]))
    for variable_index, variable in enumerate(PUBLIC_RASTER_VARIABLES, 1):
        if progress_callback:
            progress_callback({
                "phase": "hourly_rasters", "message": f"Writing hourly raster: {variable}",
                "fraction": 0.08 + 0.27 * variable_index / len(PUBLIC_RASTER_VARIABLES),
                "current": variable_index, "total": len(PUBLIC_RASTER_VARIABLES), "item": variable,
            })
        categorical = variable == "fire_danger"
        byte_data = variable in {"meteorological_confidence", "category_confidence", "probability_rh_le_25", "probability_gust_ge_30mph", "probability_concurrent"}
        path = write_cog(forecast[variable], staging / "rasters" / "hourly" / f"{variable}.tif", variable=variable, band_times=valid_times, categorical=categorical, byte_data=byte_data)
        assets.append(artifact_record(path, run_id=run_id, kind="raster", variable=variable, aggregation="hourly", object_key=f"forecast-v1/runs/{cycle:%Y/%m/%d}/{run_id}/rasters/hourly/{variable}.tif", dtype="uint8" if categorical or byte_data else "float32", unit=VARIABLE_UNITS.get(variable), valid_start=valid_times[0], valid_end=valid_times[-1]))
    daily_dates = [str(value) for value in daily.local_date.values]
    daily_raster_map = dict(DAILY_RASTER_MAP)
    if cycle.month in {11, 12, 1, 2, 3}:
        daily_raster_map["maximum_snow_water_equivalent"] = "snow_water_equivalent"
    for daily_index, (daily_name, public_name) in enumerate(daily_raster_map.items(), 1):
        if progress_callback:
            progress_callback({
                "phase": "daily_products", "message": f"Rendering Day 1-3 products: {public_name}",
                "fraction": 0.35 + 0.35 * daily_index / len(daily_raster_map),
                "current": daily_index, "total": len(daily_raster_map), "item": public_name,
            })
        categorical = daily_name == "peak_fire_danger"
        byte_data = daily_name == "category_confidence"
        path = write_cog(daily[daily_name], staging / "rasters" / "daily" / f"{public_name}.tif", variable=public_name, band_times=daily_dates, categorical=categorical, byte_data=byte_data)
        assets.append(artifact_record(path, run_id=run_id, kind="raster", variable=public_name, aggregation="daily", object_key=f"forecast-v1/runs/{cycle:%Y/%m/%d}/{run_id}/rasters/daily/{public_name}.tif", dtype="uint8" if categorical or byte_data else "float32", unit=VARIABLE_UNITS.get(public_name)))
        for day_index, local_date in enumerate(daily_dates, 1):
            graphic_root = staging / "graphics" / f"day-{day_index}"
            day_confidence = daily.category_confidence.isel(day=day_index - 1).where(daily.category_confidence.isel(day=day_index - 1) != 255)
            median_confidence = _value(day_confidence.median(skipna=True).values)
            confidence_label = "Unavailable" if median_confidence is None else "High" if median_confidence >= 80 else "Moderate" if median_confidence >= 50 else "Low"
            graphic_data = daily[daily_name].isel(day=day_index - 1)
            if categorical or byte_data:
                graphic_data = graphic_data.where(graphic_data != 255)
            png, webp = write_static_graphic(
                graphic_data, graphic_root / f"{public_name}.png", graphic_root / f"{public_name}.webp",
                title=f"Day {day_index} {STATIC_TITLES[daily_name]}", valid_period=f"{local_date} America/Chicago",
                run_time=utc_rfc3339(cycle), status=f"{confidence_label} category confidence • " + ("Degraded inputs" if warnings else "All configured sources available"),
                categorical=daily_name == "peak_fire_danger",
                boundary_geojson=GIS_DIR / "state_missouri.geojson",
            )
            for graphic in (png, webp):
                assets.append(artifact_record(graphic, run_id=run_id, kind="graphic", variable=public_name, aggregation=f"day-{day_index}-{graphic.suffix[1:]}", object_key=f"forecast-v1/runs/{cycle:%Y/%m/%d}/{run_id}/graphics/day-{day_index}/{graphic.name}", dtype=graphic.suffix[1:].upper(), unit=VARIABLE_UNITS.get(public_name)))
    if progress_callback:
        progress_callback({"phase": "station_products", "message": "Writing station forecasts and member extracts", "fraction": 0.73})
    hourly_rows, daily_rows = extract_station_rows(forecast, stations, run_id)
    points_path = write_points_parquet(hourly_rows, staging / "points" / "public.parquet")
    assets.append(artifact_record(points_path, run_id=run_id, kind="points", variable="public", object_key=f"forecast-v1/runs/{cycle:%Y/%m/%d}/{run_id}/points/public.parquet", dtype="Parquet/Zstd"))
    source_member_rows = source_member_rows if source_member_rows is not None else extract_source_member_rows(cubes, stations, run_id)
    member_path = write_points_parquet(source_member_rows, staging / "points" / "source-members.parquet")
    assets.append(artifact_record(member_path, run_id=run_id, kind="points", variable="source-members", object_key=f"forecast-v1/runs/{cycle:%Y/%m/%d}/{run_id}/points/source-members.parquet", dtype="Parquet/Zstd"))
    layers = [{"variable": a["variable"], "aggregation": a["aggregation"], "unit": a["unit"], "dataType": a["storage_data_type"], "style": PUBLIC_LAYER_STYLES.get(a["variable"]), "tileUrl": f"/tiles/forecast/{run_id}/{a['variable']}/{{lead_hour}}/{{z}}/{{x}}/{{y}}.png" if a["aggregation"] == "hourly" else None} for a in assets if a["kind"] == "raster"]
    legacy_aliases = {
        "latest/mo-forecastfiredanger.png": "graphics/day-1/fire_danger.png",
        "latest/mo-forecastfuelmoisture.png": "graphics/day-1/fuel_moisture_p50.png",
        "latest/mo-forecastminrh.png": "graphics/day-1/relative_humidity_2m.png",
        "latest/mo-forecastmaxtemp.png": "graphics/day-1/temperature_2m.png",
        "latest/mo-forecastmaxwind.png": "graphics/day-1/wind_speed_10m.png",
        "latest/mo-forecastrainfall.png": "graphics/day-1/precipitation_increment.png",
    }
    if "maximum_snow_water_equivalent" in daily_raster_map:
        legacy_aliases["latest/mo-forecastswe.png"] = "graphics/day-1/snow_water_equivalent.png"
    manifest = {
        "schemaVersion": SCHEMA_VERSION, "runId": run_id, "cycleTime": utc_rfc3339(cycle), "issuedAt": now,
        "horizonHours": HORIZON_HOURS, "timeCount": TIME_COUNT,
        "grid": {"id": PUBLIC_GRID.id, "crs": PUBLIC_GRID.crs, "width": PUBLIC_GRID.width, "height": PUBLIC_GRID.height, "resolutionMeters": PUBLIC_GRID.resolution_m, "bounds": PUBLIC_GRID.bounds},
        "sources": [{"model": name, "cycleTime": utc_rfc3339(cube.cycle_time), "members": source_member_count(cube), "publicSummaryMembers": len(cube.member_ids), "status": "available", "qualityFlags": list(cube.quality_flags)} for name, cube in cubes.items()],
        "modelVersion": model_version, "configVersion": config_version, "warnings": warnings, "layers": layers,
        "confidenceMethod": {
            "meteorological": {"modelAgreement": 0.30, "ensembleSpread": 0.25, "cycleConsistency": 0.20, "rollingVerification": 0.15, "leadTime": 0.10},
            "category": "percentage of calibrated forecast scenarios matching the deterministic category",
            "labels": {"High": [80, 100], "Moderate": [50, 79], "Low": [0, 49]},
        },
        "stationFeedUrl": f"/api/forecast/v1/stations?run_id={run_id}",
        "staticAssets": [{"day": int(a["aggregation"].split("-")[1]), "format": a["aggregation"].split("-")[2], "variable": a["variable"], "url": "/forecast-v1-assets/" + a["object_key"].removeprefix("forecast-v1/")} for a in assets if a["kind"] == "graphic"],
        "legacyAliases": legacy_aliases,
        "checksums": {},
    }
    manifest["checksums"] = {str(Path(a["local_path"]).relative_to(staging)): a["checksum"] for a in assets}
    manifest_path = write_manifest(manifest, staging / "manifest.json")
    manifest_checksum = sha256_file(manifest_path)
    assets.append(artifact_record(manifest_path, run_id=run_id, kind="manifest", object_key=f"forecast-v1/runs/{cycle:%Y/%m/%d}/{run_id}/manifest.json", dtype="JSON"))
    r2 = r2_store or ForecastR2Store()
    if r2.configured:
        for asset_index, asset in enumerate(assets, 1):
            if progress_callback:
                progress_callback({
                    "phase": "uploading", "message": f"Uploading {asset['kind']} assets",
                    "fraction": 0.80 + 0.15 * asset_index / len(assets),
                    "current": asset_index, "total": len(assets), "item": asset["kind"],
                })
            r2.upload_immutable(asset["local_path"], asset["object_key"], asset["checksum"])
    if progress_callback:
        progress_callback({"phase": "finalizing", "message": "Committing the run index and latest pointers", "fraction": 0.97})
    promote_directory(staging, final)
    for asset in assets:
        relative = Path(asset["local_path"]).relative_to(staging)
        asset["local_path"] = str((final / relative).resolve())
    if make_public:
        _update_legacy_aliases(final, root, manifest["legacyAliases"], r2)
    database = db_path or get_db_path()
    ensure_schema(database)
    with transaction(database) as connection:
        previous = connection.execute("SELECT run_id FROM forecast_runs WHERE is_public=1 ORDER BY cycle_time_utc DESC LIMIT 1").fetchone()
        if make_public:
            connection.execute("UPDATE forecast_runs SET is_public=0, status=CASE WHEN status='complete' THEN 'superseded' ELSE status END WHERE is_public=1")
        upsert_run(connection, {
            "run_id": run_id, "cycle_time_utc": utc_rfc3339(cycle), "issued_at_utc": now, "completed_at_utc": utc_rfc3339(datetime.now(timezone.utc)),
            "status": "complete", "is_public": int(make_public), "grid_id": PUBLIC_GRID.id, "horizon_hours": HORIZON_HOURS,
            "schema_version": SCHEMA_VERSION, "config_version": config_version, "model_version": model_version,
            "source_cycles_json": json_text({name: utc_rfc3339(cube.cycle_time) for name, cube in cubes.items()}),
            "manifest_key": assets[-1]["object_key"], "manifest_checksum": manifest_checksum, "warnings_json": json_text(warnings),
            "superseded_run_id": previous[0] if make_public and previous else None, "created_at_utc": now,
        })
        replace_assets(connection, run_id, assets)
        _insert_rows(connection, "station_forecast_hours", hourly_rows)
        _insert_rows(connection, "station_forecast_days", daily_rows)
        for cube in cubes.values():
            if cube.object_key and cube.checksum:
                source_path = root / cube.object_key.removeprefix("forecast-v1/")
                connection.execute(
                    "INSERT OR IGNORE INTO source_cycles(model,initialization_time_utc,member_count,variables_json,source_status,object_key,checksum,byte_size,acquisition_json) VALUES(?,?,?,?,?,?,?,?,?)",
                    (cube.model, utc_rfc3339(cube.cycle_time), source_member_count(cube), json_text(sorted(cube.dataset.data_vars)), "verified", cube.object_key, cube.checksum, source_path.stat().st_size if source_path.exists() else 0, json_text({"qualityFlags": list(cube.quality_flags)})),
                )
        for station in stations:
            connection.execute(
                "INSERT OR REPLACE INTO forecast_station_metadata(station_id,name,latitude,longitude,network_type,elevation_m,timezone,is_active,sensor_capabilities_json,updated_at_utc) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (station["station_id"], station.get("name", station["station_id"]), station["latitude"], station["longitude"], station.get("network_type", "other"), station.get("elevation_m"), station.get("timezone", "America/Chicago"), int(station.get("is_active", True)), json_text(station.get("sensor_capabilities", {})), now),
            )
    if make_public:
        latest = {"schemaVersion": SCHEMA_VERSION, "runId": run_id, "manifest": f"/api/forecast/v1/runs/{run_id}", "updatedAt": now}
        write_manifest(latest, root / "latest.json")
        if r2.configured:
            r2.put_latest(latest)
    return manifest


def publish_run(
    cubes: dict[str, SourceCube], stations: list[dict], *, initial_fuel_moisture: float | xr.DataArray | FuelInitialization,
    publish_root: str | Path, config_version: str = "beta-1", model_version: str = "physics-only-beta-1",
    make_public: bool = False, db_path: str | Path | None = None, r2_store: ForecastR2Store | None = None,
    confidence_components: dict[str, xr.DataArray | float] | None = None,
    predicted_residuals: dict[str, xr.DataArray] | None = None,
    fuel_quantile_residuals: dict[str, xr.DataArray] | None = None,
    source_member_rows: list[dict] | None = None,
    progress_callback: Callable[[dict], None] | None = None,
) -> dict:
    """Track staging/failure state around the all-or-nothing publisher."""
    if not cubes:
        raise ValueError("at least one source cube is required")
    cycle = next(iter(cubes.values())).cycle_time.astimezone(timezone.utc)
    run_id = run_id_for_cycle(cycle)
    database = db_path or get_db_path()
    ensure_schema(database)
    started = utc_rfc3339(datetime.now(timezone.utc))
    with transaction(database) as connection:
        existing = connection.execute("SELECT status FROM forecast_runs WHERE run_id=?", (run_id,)).fetchone()
        if existing and existing["status"] in {"complete", "superseded"}:
            raise FileExistsError(f"immutable forecast run already exists: {run_id}")
        upsert_run(connection, {
            "run_id": run_id, "cycle_time_utc": utc_rfc3339(cycle), "issued_at_utc": None, "completed_at_utc": None,
            "status": "staging", "is_public": 0, "grid_id": PUBLIC_GRID.id, "horizon_hours": HORIZON_HOURS,
            "schema_version": SCHEMA_VERSION, "config_version": config_version, "model_version": model_version,
            "source_cycles_json": json_text({name: utc_rfc3339(cube.cycle_time) for name, cube in cubes.items()}),
            "manifest_key": None, "manifest_checksum": None, "warnings_json": "[]", "superseded_run_id": None,
            "created_at_utc": started,
        })
    try:
        return _publish_run_impl(
            cubes, stations, initial_fuel_moisture=initial_fuel_moisture, publish_root=publish_root,
            config_version=config_version, model_version=model_version, make_public=make_public, db_path=database,
            r2_store=r2_store, confidence_components=confidence_components, predicted_residuals=predicted_residuals,
            fuel_quantile_residuals=fuel_quantile_residuals, source_member_rows=source_member_rows,
            progress_callback=progress_callback,
        )
    except Exception as exc:
        with transaction(database) as connection:
            failed = connection.execute("SELECT is_public,superseded_run_id FROM forecast_runs WHERE run_id=?", (run_id,)).fetchone()
            connection.execute(
                "UPDATE forecast_runs SET status='failed',is_public=0,completed_at_utc=?,warnings_json=? WHERE run_id=?",
                (utc_rfc3339(datetime.now(timezone.utc)), json_text([f"publication_failed:{type(exc).__name__}"]), run_id),
            )
            if failed and failed["is_public"] and failed["superseded_run_id"]:
                connection.execute("UPDATE forecast_runs SET status='complete',is_public=1 WHERE run_id=?", (failed["superseded_run_id"],))
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish a staged 12Z forecast-v1 run")
    parser.add_argument("--cycle", required=True, help="UTC cycle such as 2026-09-06T12:00:00Z")
    parser.add_argument("--source", action="append", required=True, help="model=/path/to/normalized.nc")
    parser.add_argument("--stations", help="Station metadata JSON list (required unless --archive-only)")
    parser.add_argument("--publish-root", default=str(FORECAST_V1_DIR))
    parser.add_argument("--public", action="store_true")
    parser.add_argument("--archive-only", action="store_true", help="Archive a 00Z/06Z/12Z/18Z source cycle without deriving public products")
    args = parser.parse_args()
    cycle = datetime.fromisoformat(args.cycle.replace("Z", "+00:00"))
    archived = []
    r2 = ForecastR2Store()
    for definition in args.source:
        model, path = definition.split("=", 1)
        native = archive_source_cube(ADAPTERS[model]().open(path, cycle), args.publish_root, r2)
        archived.append(native)
    if args.archive_only:
        register_source_cycles(archived, args.publish_root)
        print(json.dumps({"cycleTime": utc_rfc3339(cycle), "status": "archived", "sources": sorted(cube.model for cube in archived)}))
        return
    if not args.stations:
        parser.error("--stations is required unless --archive-only is used")
    stations = json.loads(Path(args.stations).read_text(encoding="utf-8"))
    run_id = run_id_for_cycle(cycle)
    native_cubes = {cube.model: cube for cube in archived}
    source_member_rows = extract_source_member_rows(native_cubes, stations, run_id)
    archived.clear()
    cubes = {}
    for model in list(native_cubes):
        native = native_cubes.pop(model)
        cubes[model] = regrid_to_public(summarize_ensemble_for_public(native))
    manifest = publish_run(
        cubes, stations, initial_fuel_moisture=12.0, publish_root=args.publish_root,
        make_public=args.public, r2_store=r2, source_member_rows=source_member_rows,
    )
    print(json.dumps({"runId": manifest["runId"], "status": "complete"}))


if __name__ == "__main__":
    main()
