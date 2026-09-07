from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import xarray as xr

from core.fire_danger import MPS_TO_KNOTS, RULE_SPEC
from .adapters import SourceCube
from .contracts import (
    CONVECTION_ALLOWING_MODELS,
    CORRECTION_LIMITS,
    HORIZON_HOURS,
    LOCAL_TIMEZONE,
    QUALITY_BITS,
    REQUIRED_VARIABLES,
    SOURCE_BITS,
    VARIABLE_UNITS,
    correction_multiplier,
    weights_for_lead,
)


@dataclass(frozen=True)
class CorrectionResult:
    dataset: xr.Dataset
    capped: xr.DataArray


def _available_for_hour(
    cubes: dict[str, SourceCube], variable: str, lead: int, *, include_gefs: bool = False,
) -> dict[str, xr.DataArray]:
    values: dict[str, xr.DataArray] = {}
    for model, cube in cubes.items():
        if (model == "gefs" and not include_gefs) or variable not in cube.dataset or lead >= cube.dataset.sizes["time"]:
            continue
        value = cube.ensemble_mean(variable).isel(time=lead)
        if bool(value.notnull().any()):
            values[model] = value
    return values


def blend_sources(cubes: dict[str, SourceCube], horizon: int = HORIZON_HOURS) -> xr.Dataset:
    """Blend normalized sources. Missing model weights are renormalized per hour/cell."""
    if not CONVECTION_ALLOWING_MODELS.intersection(cubes):
        raise ValueError("at least one convection-allowing deterministic source is required")
    output: dict[str, xr.DataArray] = {}
    quality: list[xr.DataArray] = []
    source_masks: list[xr.DataArray] = []
    times = None
    for variable in REQUIRED_VARIABLES:
        hourly: list[xr.DataArray] = []
        for lead in range(horizon + 1):
            values = _available_for_hour(cubes, variable, lead, include_gefs=lead >= 49)
            expected = weights_for_lead(lead)
            convection_sources = CONVECTION_ALLOWING_MODELS.intersection(values).intersection(expected)
            coarse_fallback = lead >= 49 and not convection_sources and "gefs" in values
            if not convection_sources and not coarse_fallback:
                if values:
                    template = next(iter(values.values()))
                else:
                    template_cube = next(iter(cubes.values()))
                    template = template_cube.ensemble_mean(REQUIRED_VARIABLES[0]).isel(time=min(lead, template_cube.dataset.sizes["time"] - 1))
                hourly.append(xr.full_like(template, np.nan, dtype="float32"))
                if variable == REQUIRED_VARIABLES[0]:
                    quality.append(xr.full_like(template, QUALITY_BITS["source_degraded"], dtype="uint16"))
                    source_masks.append(xr.zeros_like(template, dtype="uint16"))
                continue
            active = {"gefs": 1.0} if coarse_fallback else {name: weight for name, weight in expected.items() if name in values}
            total = sum(active.values())
            weights = {name: weight / total for name, weight in active.items()}
            numerator = sum(values[name] * weight for name, weight in weights.items())
            denominator = sum(values[name].notnull() * weight for name, weight in weights.items())
            hourly.append((numerator / denominator.where(denominator > 0)).astype("float32"))
            if variable == REQUIRED_VARIABLES[0]:
                template = next(iter(values.values()))
                degraded = set(active) != set(expected)
                quality_value = QUALITY_BITS["source_degraded"] if degraded else 0
                if coarse_fallback:
                    quality_value |= QUALITY_BITS["coarse_synoptic_fallback"]
                quality.append(xr.full_like(template, quality_value, dtype="uint16"))
                mask_value = sum(SOURCE_BITS[name] for name in active)
                source_masks.append(xr.full_like(template, mask_value, dtype="uint16"))
        output[variable] = xr.concat(hourly, dim="time")
        if times is None:
            cycle = next(iter(cubes.values())).cycle_time.astimezone(timezone.utc)
            times = np.array([np.datetime64(cycle.replace(tzinfo=None) + timedelta(hours=i), "ns") for i in range(horizon + 1)])
        output[variable] = output[variable].assign_coords(time=times)
        output[variable].attrs["units"] = VARIABLE_UNITS[variable]
    dataset = xr.Dataset(output)
    dataset["source_mask"] = xr.concat(source_masks, dim="time").assign_coords(time=times)
    dataset["quality_mask"] = xr.concat(quality, dim="time").assign_coords(time=times)
    dataset["relative_humidity_2m"] = dataset.relative_humidity_2m.clip(0, 100)
    dataset["cloud_cover"] = dataset.cloud_cover.clip(0, 100)
    dataset["precipitation_increment"] = dataset.precipitation_increment.clip(min=0)
    dataset["precipitation_accumulated"] = dataset.precipitation_increment.fillna(0).cumsum("time").where(dataset.precipitation_increment.notnull())
    dataset["wind_speed_10m"] = np.hypot(dataset.wind_u_10m, dataset.wind_v_10m).astype("float32")
    dataset["wind_direction_10m"] = ((270 - np.degrees(np.arctan2(dataset.wind_v_10m, dataset.wind_u_10m))) % 360).astype("float32")
    return dataset


def apply_corrections(dataset: xr.Dataset, predicted_residuals: dict[str, xr.DataArray] | None) -> CorrectionResult:
    corrected = dataset.copy()
    capped_any = xr.zeros_like(dataset.quality_mask, dtype=bool)
    for variable, limit in CORRECTION_LIMITS.items():
        if not predicted_residuals or variable not in predicted_residuals:
            continue
        residual = predicted_residuals[variable].broadcast_like(dataset[variable])
        lead_caps = xr.DataArray(
            [limit * correction_multiplier(i) for i in range(dataset.sizes["time"])],
            dims=("time",), coords={"time": dataset.time},
        ).broadcast_like(dataset[variable])
        clipped = residual.clip(-lead_caps, lead_caps)
        capped_any = capped_any | (abs(residual) > lead_caps)
        corrected[variable] = (dataset[variable] + clipped).astype("float32")
    corrected["relative_humidity_2m"] = corrected.relative_humidity_2m.clip(0, 100)
    corrected["wind_speed_10m"] = np.hypot(corrected.wind_u_10m, corrected.wind_v_10m).astype("float32")
    corrected["wind_direction_10m"] = ((270 - np.degrees(np.arctan2(corrected.wind_v_10m, corrected.wind_u_10m))) % 360).astype("float32")
    corrected["quality_mask"] = corrected.quality_mask | xr.where(capped_any, QUALITY_BITS["correction_capped"], 0).astype("uint16")
    return CorrectionResult(corrected, capped_any)


def nelson_emc(temperature_c: xr.DataArray, rh: xr.DataArray) -> xr.DataArray:
    """Operational Nelson-style equilibrium moisture approximation."""
    r = rh.clip(0, 100)
    dry = 0.942 * r**0.679 + 11.0 * np.exp((r - 100.0) / 10.0) + 0.18 * (21.1 - temperature_c) * (1 - np.exp(-0.115 * r))
    wet = 0.618 * r**0.753 + 10.0 * np.exp((r - 100.0) / 10.0) + 0.18 * (21.1 - temperature_c) * (1 - np.exp(-0.115 * r))
    return ((dry + wet) / 2.0).clip(1, 40).astype("float32")


def evolve_fuel_moisture(
    dataset: xr.Dataset,
    initial_fm: xr.DataArray | float,
    quantile_residuals: dict[str, xr.DataArray] | None = None,
) -> xr.Dataset:
    state = xr.full_like(dataset.temperature_2m.isel(time=0), float(initial_fm), dtype="float32") if np.isscalar(initial_fm) else initial_fm.astype("float32")
    trajectory: list[xr.DataArray] = []
    for lead in range(dataset.sizes["time"]):
        temp = dataset.temperature_2m.isel(time=lead, drop=True)
        rh = dataset.relative_humidity_2m.isel(time=lead, drop=True)
        target = nelson_emc(temp, rh)
        tau = xr.where(target > state, 6.0, 10.0)
        candidate = state + (target - state) * (1.0 - np.exp(-1.0 / tau))
        rain = dataset.precipitation_increment.isel(time=lead, drop=True).fillna(0)
        solar = dataset.shortwave_down.isel(time=lead, drop=True).fillna(0)
        candidate = candidate + (40.0 - candidate) * (1.0 - np.exp(-rain / 2.5))
        candidate = candidate - np.clip(solar - 250.0, 0, None) / 8000.0
        if trajectory:
            prior_output = trajectory[-1].fillna(state)
            candidate = xr.where(abs(candidate - prior_output) > 8, prior_output + np.sign(candidate - prior_output) * 8, candidate)
        valid = np.isfinite(temp) & np.isfinite(rh)
        output = candidate.clip(1, 40).where(valid).astype("float32")
        trajectory.append(output)
        state = xr.where(valid, candidate.clip(1, 40), state).astype("float32")
    physics = xr.concat(trajectory, dim="time").assign_coords(time=dataset.time)
    result = dataset.copy()
    residuals = quantile_residuals or {}
    p10 = (physics + residuals.get("p10", 0)).clip(1, 40)
    p50 = (physics + residuals.get("p50", 0)).clip(1, 40)
    p90 = (physics + residuals.get("p90", 0)).clip(1, 40)
    stacked = xr.concat([p10, p50, p90], dim="quantile")
    ordered = np.sort(stacked.values, axis=0)
    for index, name in enumerate(("fuel_moisture_p10", "fuel_moisture_p50", "fuel_moisture_p90")):
        result[name] = xr.DataArray(ordered[index], dims=physics.dims, coords=physics.coords, attrs={"units": "%"}).astype("float32")
    return result


def _classify_arrays(fm: xr.DataArray, rh: xr.DataArray, wind_ms: xr.DataArray) -> xr.DataArray:
    fuel = fm.values
    humidity = rh.values
    wind = wind_ms.values * MPS_TO_KNOTS
    values = np.full(fm.shape, 255, dtype=np.uint8)
    valid = np.isfinite(fuel) & np.isfinite(humidity) & np.isfinite(wind)
    thresholds = RULE_SPEC["thresholds"]
    values[valid] = 0
    dry_fuel = fuel < thresholds["low_fm"]
    moderate = dry_fuel & ((humidity < thresholds["moderate_rh"]) | (wind >= thresholds["moderate_wind"]))
    elevated = (fuel < thresholds["elevated_fm"]) & (
        ((humidity < thresholds["elevated_rh"]) & (wind >= thresholds["elevated_wind"]))
        | ((humidity < thresholds["elevated_very_dry_rh"]) & (wind >= thresholds["elevated_very_dry_wind"]))
    )
    critical = (fuel < thresholds["elevated_fm"]) & (humidity < thresholds["critical_rh"]) & (wind >= thresholds["critical_wind"])
    extreme = (fuel < thresholds["extreme_fm"]) & (humidity < thresholds["extreme_rh"]) & (wind >= thresholds["extreme_wind"])
    values[valid & moderate] = 1
    values[valid & elevated] = 2
    values[valid & critical] = 3
    values[valid & extreme] = 4
    return xr.DataArray(values, dims=fm.dims, coords=fm.coords, attrs={"units": "category", "nodata": 255})


def add_risk_and_confidence(
    dataset: xr.Dataset,
    confidence_components: dict[str, xr.DataArray | float] | None = None,
    scenario_scales: dict[str, xr.DataArray] | None = None,
    scenario_count: int = 51,
    seed: int = 42,
) -> xr.Dataset:
    result = dataset.copy()
    result["fire_danger"] = _classify_arrays(result.fuel_moisture_p50, result.relative_humidity_2m, result.wind_speed_10m)
    lead_scores = np.array([100 if i <= 18 else 90 if i <= 36 else 75 if i <= 48 else 60 for i in range(result.sizes["time"])], dtype=np.float32)
    lead_score = xr.DataArray(lead_scores, dims="time", coords={"time": result.time}).broadcast_like(result.temperature_2m)
    valid = result.fire_danger != 255
    required_components = {"model_agreement", "ensemble_spread", "cycle_consistency", "rolling_verification"}
    if confidence_components and required_components <= set(confidence_components):
        def component(name: str) -> xr.DataArray:
            value = confidence_components[name]
            return xr.full_like(result.temperature_2m, float(value)) if np.isscalar(value) else value.broadcast_like(result.temperature_2m)
        met = (
            0.30 * component("model_agreement") + 0.25 * component("ensemble_spread")
            + 0.20 * component("cycle_consistency") + 0.15 * component("rolling_verification")
            + 0.10 * lead_score
        )
        result["meteorological_confidence"] = xr.where(valid & np.isfinite(met), np.rint(met).clip(0, 100), 255).astype("uint8")
    else:
        result["meteorological_confidence"] = xr.full_like(result.fire_danger, 255, dtype="uint8")
        result["quality_mask"] = result.quality_mask | QUALITY_BITS["insufficient_confidence"]

    rng = np.random.default_rng(seed)
    published = result.fire_danger.values
    matches = np.zeros(published.shape, dtype=np.uint16)
    rh_hits = np.zeros(published.shape, dtype=np.uint16)
    gust_hits = np.zeros(published.shape, dtype=np.uint16)
    both_hits = np.zeros(published.shape, dtype=np.uint16)
    scales = scenario_scales or {}
    rh_scale = scales.get("relative_humidity_2m", xr.full_like(result.relative_humidity_2m, 4.0))
    gust_scale = scales.get("wind_gust_10m", xr.full_like(result.wind_gust_10m, 1.5))
    wind_scale = scales.get("wind_speed_10m", xr.full_like(result.wind_speed_10m, 1.2))
    fuel_scale = scales.get("fuel_moisture", ((result.fuel_moisture_p90 - result.fuel_moisture_p10) / 2.563).clip(min=.75))
    for _ in range(scenario_count):
        noise = xr.DataArray(rng.normal(size=published.shape), dims=result.relative_humidity_2m.dims, coords=result.relative_humidity_2m.coords)
        rh = (result.relative_humidity_2m + noise * rh_scale).clip(0, 100)
        gust = (result.wind_gust_10m + xr.DataArray(rng.normal(size=published.shape), dims=noise.dims, coords=noise.coords) * gust_scale).clip(min=0)
        wind = (result.wind_speed_10m + xr.DataArray(rng.normal(size=published.shape), dims=noise.dims, coords=noise.coords) * wind_scale).clip(min=0)
        fm = (result.fuel_moisture_p50 + xr.DataArray(rng.normal(size=published.shape), dims=noise.dims, coords=noise.coords) * fuel_scale).clip(1, 40)
        scenario = _classify_arrays(fm, rh, wind).values
        matches += scenario == published
        low_rh = rh.values <= 25
        high_gust = gust.values >= 13.4112
        rh_hits += low_rh
        gust_hits += high_gust
        both_hits += low_rh & high_gust
    def probability(hits: np.ndarray) -> xr.DataArray:
        values = np.rint(hits * 100 / scenario_count).astype(np.uint8)
        values[~valid.values] = 255
        return xr.DataArray(values, dims=result.fire_danger.dims, coords=result.fire_danger.coords)
    result["category_confidence"] = probability(matches)
    coarse = (result.quality_mask & QUALITY_BITS["coarse_synoptic_fallback"]) != 0
    result["category_confidence"] = xr.where(
        coarse & (result.category_confidence != 255), result.category_confidence.clip(max=49), result.category_confidence
    ).astype("uint8")
    result["meteorological_confidence"] = xr.where(
        coarse & (result.meteorological_confidence != 255), result.meteorological_confidence.clip(max=49), result.meteorological_confidence
    ).astype("uint8")
    result["probability_rh_le_25"] = probability(rh_hits)
    result["probability_gust_ge_30mph"] = probability(gust_hits)
    result["probability_concurrent"] = probability(both_hits)
    return result


def scenario_scales_from_sources(cubes: dict[str, SourceCube], dataset: xr.Dataset) -> dict[str, xr.DataArray]:
    """Combine member spread, deterministic disagreement, and residual floors."""
    def source_spreads(variable: str, derived_speed: bool = False) -> list[xr.DataArray]:
        candidates = []
        for model in ("refs", "gefs"):
            cube = cubes.get(model)
            if not cube or cube.dataset.sizes.get("member", 1) < 2:
                continue
            if derived_speed:
                values = np.hypot(cube.dataset.wind_u_10m, cube.dataset.wind_v_10m)
            elif variable in cube.dataset:
                values = cube.dataset[variable]
            else:
                continue
            spread = values.std("member", skipna=True)
            if "time" in spread.coords and not np.array_equal(spread.time.values, dataset.time.values):
                spread = spread.interp(time=dataset.time)
            candidates.append(spread.broadcast_like(dataset.temperature_2m))
        return candidates

    result = {}
    definitions = {
        "relative_humidity_2m": (4.0, False),
        "wind_gust_10m": (1.5, False),
        "wind_speed_10m": (1.2, True),
    }
    for variable, (residual_floor, derived_speed) in definitions.items():
        candidates = source_spreads(variable, derived_speed)
        hrrr, rrfs = cubes.get("hrrr"), cubes.get("rrfs")
        if hrrr and rrfs:
            if derived_speed:
                first = np.hypot(hrrr.ensemble_mean("wind_u_10m"), hrrr.ensemble_mean("wind_v_10m"))
                second = np.hypot(rrfs.ensemble_mean("wind_u_10m"), rrfs.ensemble_mean("wind_v_10m"))
            else:
                first, second = hrrr.ensemble_mean(variable), rrfs.ensemble_mean(variable)
            disagreement = abs(first - second) / 2.0
            candidates.append(disagreement.reindex(time=dataset.time).broadcast_like(dataset.temperature_2m))
        combined = xr.full_like(dataset.temperature_2m, residual_floor)
        for candidate in candidates:
            combined = np.hypot(combined, candidate.fillna(0))
        result[variable] = combined.astype("float32")
    return result


def local_day_slices(times: xr.DataArray) -> list[tuple[str, np.ndarray]]:
    local = [datetime.fromisoformat(np.datetime_as_string(value, unit="s")).replace(tzinfo=timezone.utc).astimezone(ZoneInfo(LOCAL_TIMEZONE)) for value in times.values]
    dates: list[str] = []
    for value in local:
        key = value.date().isoformat()
        if key not in dates:
            dates.append(key)
    return [(key, np.array([i for i, value in enumerate(local) if value.date().isoformat() == key], dtype=int)) for key in dates[:3]]


def daily_aggregates(dataset: xr.Dataset) -> xr.Dataset:
    days = local_day_slices(dataset.time)
    outputs: dict[str, list[xr.DataArray]] = {name: [] for name in (
        "peak_fire_danger", "minimum_rh", "minimum_fuel_moisture_p50", "maximum_temperature",
        "maximum_wind", "maximum_gust", "total_precipitation", "category_confidence",
        "maximum_snow_water_equivalent",
    )}
    for _, indexes in days:
        subset = dataset.isel(time=indexes)
        peak = subset.fire_danger.where(subset.fire_danger != 255).max("time", skipna=True).fillna(255).astype("uint8")
        outputs["peak_fire_danger"].append(peak)
        outputs["minimum_rh"].append(subset.relative_humidity_2m.min("time", skipna=True))
        outputs["minimum_fuel_moisture_p50"].append(subset.fuel_moisture_p50.min("time", skipna=True))
        outputs["maximum_temperature"].append(subset.temperature_2m.max("time", skipna=True))
        outputs["maximum_wind"].append(subset.wind_speed_10m.max("time", skipna=True))
        outputs["maximum_gust"].append(subset.wind_gust_10m.max("time", skipna=True))
        outputs["total_precipitation"].append(subset.precipitation_increment.sum("time", skipna=True))
        outputs["maximum_snow_water_equivalent"].append(subset.snow_water_equivalent.max("time", skipna=True))
        equal_peak = subset.fire_danger == peak
        peak_confidence = subset.category_confidence.where(equal_peak & (subset.category_confidence != 255)).median("time", skipna=True)
        local_hours = []
        for value in subset.time.values:
            utc = datetime.fromisoformat(np.datetime_as_string(value, unit="s")).replace(tzinfo=timezone.utc)
            local_hours.append(utc.astimezone(ZoneInfo(LOCAL_TIMEZONE)).hour)
        afternoon = xr.DataArray(np.array([12 <= hour <= 18 for hour in local_hours]), dims="time", coords={"time": subset.time})
        afternoon_confidence = subset.category_confidence.where(afternoon & (subset.category_confidence != 255)).median("time", skipna=True)
        confidence = xr.where(peak <= 1, afternoon_confidence, peak_confidence)
        outputs["category_confidence"].append(confidence.fillna(255).astype("uint8"))
    result = xr.Dataset({name: xr.concat(values, dim="day") for name, values in outputs.items()})
    result = result.assign_coords(day=np.arange(1, len(days) + 1), local_date=("day", [key for key, _ in days]))
    return result


def build_forecast_cube(
    cubes: dict[str, SourceCube],
    initial_fuel_moisture: xr.DataArray | float,
    predicted_residuals: dict[str, xr.DataArray] | None = None,
    fuel_quantile_residuals: dict[str, xr.DataArray] | None = None,
    confidence_components: dict[str, xr.DataArray | float] | None = None,
) -> tuple[xr.Dataset, xr.Dataset]:
    blended = blend_sources(cubes)
    corrected = apply_corrections(blended, predicted_residuals).dataset
    fueled = evolve_fuel_moisture(corrected, initial_fuel_moisture, fuel_quantile_residuals)
    final = add_risk_and_confidence(
        fueled, confidence_components=confidence_components,
        scenario_scales=scenario_scales_from_sources(cubes, fueled),
    )
    return final, daily_aggregates(final)
