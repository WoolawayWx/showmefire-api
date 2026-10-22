from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import xarray as xr

from core.fire_danger import MPS_TO_KNOTS, RULE_SPEC
from services.rule_uncertainty import CATEGORY_CRITICAL, CATEGORY_ELEVATED, category_vectorized
from .adapters import SourceCube
from .contracts import (
    CONVECTION_ALLOWING_MODELS,
    CORRECTION_LIMITS,
    HORIZON_HOURS,
    LOCAL_TIMEZONE,
    QUALITY_BITS,
    REQUIRED_VARIABLES,
    SCENARIO_SIGMA_FLOORS,
    SOURCE_BITS,
    VARIABLE_UNITS,
    correction_multiplier,
    weights_for_lead,
)

logger = logging.getLogger(__name__)

CALM_WIND_MS = 0.5


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
    non_core_models = [model for model in cubes if model not in ("hrrr", "rrfs", "refs", "gefs")]
    metric_rows: list[tuple[str, datetime, str, int, bool, float | None]] = []
    cycle_for_metrics = next(iter(cubes.values())).cycle_time.astimezone(timezone.utc) if non_core_models else None
    output: dict[str, xr.DataArray] = {}
    quality: list[xr.DataArray] = []
    source_masks: list[xr.DataArray] = []
    times = None
    for variable in REQUIRED_VARIABLES:
        hourly: list[xr.DataArray] = []
        for lead in range(horizon + 1):
            values = _available_for_hour(cubes, variable, lead, include_gefs=lead >= 49)
            expected = weights_for_lead(lead)
            if non_core_models:
                # Local import: registry.py depends on core.database, which this
                # module has no other reason to import.
                from .registry import weight_overlay
                expected = weight_overlay(expected, lead)
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
                for model in non_core_models:
                    metric_rows.append((model, cycle_for_metrics, variable, lead, model in values, None))
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
            for model in non_core_models:
                present = model in values
                diff = float(abs(values[model] - hourly[-1]).mean(skipna=True)) if present else None
                metric_rows.append((model, cycle_for_metrics, variable, lead, present, diff))
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
    if metric_rows:
        # Best-effort: a metrics-logging failure should never fail a forecast run.
        try:
            from .registry import record_metrics_bulk
            record_metrics_bulk(metric_rows)
        except Exception:
            logger.warning("Failed to log forecast source model metrics", exc_info=True)
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
        if variable == "wind_speed_10m":
            # Speed is derived from u/v below, so correct it by scaling both
            # components; calm cells keep their (undefined) direction as is.
            speed = np.hypot(corrected.wind_u_10m, corrected.wind_v_10m)
            scale = ((speed + clipped).clip(min=0) / speed.where(speed > CALM_WIND_MS)).fillna(1.0)
            corrected["wind_u_10m"] = (corrected.wind_u_10m * scale).astype("float32")
            corrected["wind_v_10m"] = (corrected.wind_v_10m * scale).astype("float32")
            continue
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


def _classify_values(fm: np.ndarray, rh: np.ndarray, wind_ms: np.ndarray) -> np.ndarray:
    """The canonical rule (services/rule_uncertainty.category_vectorized, parity-checked
    against core.fire_danger.calculate_fire_danger), 255 where an input is missing."""
    category = category_vectorized(fm, rh, wind_ms * MPS_TO_KNOTS, RULE_SPEC["thresholds"])
    return np.where(category < 0, 255, category).astype(np.uint8)


def _classify_arrays(fm: xr.DataArray, rh: xr.DataArray, wind_ms: xr.DataArray) -> xr.DataArray:
    values = _classify_values(fm.values, rh.values, wind_ms.values)
    return xr.DataArray(values, dims=fm.dims, coords=fm.coords, attrs={"units": "category", "nodata": 255})


def _scenario_sigma(
    variable: str, template: xr.DataArray, spread: xr.DataArray | None,
    error_sigma: dict[str, xr.DataArray] | None,
) -> np.ndarray:
    """Per-hour error spread: the measured verification spread when there is one,
    otherwise the lead-growing floor, raised wherever the models disagree more."""
    hours = template.sizes["time"]
    start, end = SCENARIO_SIGMA_FLOORS[variable]
    base = start + (end - start) * np.minimum(np.arange(hours), HORIZON_HOURS) / HORIZON_HOURS
    measured = (error_sigma or {}).get(variable)
    if measured is not None:
        values = np.asarray(measured.values, dtype=np.float64)
        base = np.where(np.isfinite(values) & (values > 0), values, base)
    sigma = np.broadcast_to(base[:, None, None], template.shape)
    if spread is not None:
        sigma = np.maximum(sigma, spread.broadcast_like(template).fillna(0).values)
    return np.asarray(sigma, dtype=np.float32)


def _emc_rh_sensitivity(temperature_c: xr.DataArray, rh: xr.DataArray) -> np.ndarray:
    """Equilibrium fuel moisture change per 1% RH, so fuel errors follow RH errors."""
    up = nelson_emc(temperature_c, (rh + 1).clip(0, 100))
    down = nelson_emc(temperature_c, (rh - 1).clip(0, 100))
    return ((up - down) / 2.0).values.astype(np.float32)


def _local_dates(times: xr.DataArray) -> list[str]:
    return [
        datetime.fromisoformat(np.datetime_as_string(value, unit="s")).replace(tzinfo=timezone.utc)
        .astimezone(ZoneInfo(LOCAL_TIMEZONE)).date().isoformat()
        for value in times.values
    ]


def add_risk_and_confidence(
    dataset: xr.Dataset,
    confidence_components: dict[str, xr.DataArray | float] | None = None,
    scenario_scales: dict[str, xr.DataArray] | None = None,
    scenario_count: int = 51,
    seed: int = 42,
    error_sigma: dict[str, xr.DataArray] | None = None,
) -> xr.Dataset:
    result = dataset.copy()
    result["fire_danger"] = _classify_arrays(result.fuel_moisture_p50, result.relative_humidity_2m, result.wind_speed_10m)
    lead_scores = np.array([100 if i <= 18 else 90 if i <= 36 else 75 if i <= 48 else 60 for i in range(result.sizes["time"])], dtype=np.float32)
    lead_score = xr.DataArray(lead_scores, dims="time", coords={"time": result.time}).broadcast_like(result.temperature_2m)
    valid = result.fire_danger != 255
    component_weights = {
        "model_agreement": 0.30, "ensemble_spread": 0.25,
        "cycle_consistency": 0.20, "rolling_verification": 0.15,
    }
    component_values: dict[str, xr.DataArray] = {}
    weighted = xr.zeros_like(result.temperature_2m, dtype="float32")
    available_weight = xr.zeros_like(result.temperature_2m, dtype="float32")
    for name, weight in component_weights.items():
        raw = (confidence_components or {}).get(name)
        value = (
            xr.full_like(result.temperature_2m, np.nan, dtype="float32") if raw is None
            else xr.full_like(result.temperature_2m, float(raw), dtype="float32") if np.isscalar(raw)
            else raw.broadcast_like(result.temperature_2m).astype("float32")
        )
        component_values[name] = value
        finite = np.isfinite(value)
        weighted = weighted + xr.where(finite, value * weight, 0.0)
        available_weight = available_weight + xr.where(finite, weight, 0.0)
    realtime = np.isfinite(component_values["model_agreement"]) | np.isfinite(component_values["ensemble_spread"])
    total_weight = available_weight + 0.10
    supported = realtime & (total_weight >= 0.35)
    met = (weighted + 0.10 * lead_score) / total_weight.where(total_weight > 0)
    confidence_valid = valid & supported & np.isfinite(met)
    result["meteorological_confidence"] = xr.where(
        confidence_valid, np.rint(met).clip(0, 100), 255
    ).astype("uint8")
    partial = confidence_valid & (available_weight < sum(component_weights.values()))
    insufficient = valid & ~confidence_valid
    result["quality_mask"] = xr.where(
        partial, result.quality_mask | QUALITY_BITS["partial_confidence_inputs"], result.quality_mask
    ).astype("uint16")
    result["quality_mask"] = xr.where(
        insufficient, result.quality_mask | QUALITY_BITS["insufficient_confidence"], result.quality_mask
    ).astype("uint16")

    # Scenarios. Each draw holds its error for a whole local day (forecast
    # errors persist for hours, and independent hourly noise would inflate the
    # chance that *some* hour crosses a threshold), and fuel moisture moves with
    # the RH error through the equilibrium-moisture slope instead of on its own.
    template = result.temperature_2m.transpose("time", "y", "x")
    scales = scenario_scales or {}
    rh_sigma = _scenario_sigma("relative_humidity_2m", template, scales.get("relative_humidity_2m"), error_sigma)
    wind_sigma = _scenario_sigma("wind_speed_10m", template, scales.get("wind_speed_10m"), error_sigma)
    gust_sigma = _scenario_sigma("wind_gust_10m", template, scales.get("wind_gust_10m"), error_sigma)
    fuel_spread = scales.get("fuel_moisture", (result.fuel_moisture_p90 - result.fuel_moisture_p10) / 2.563)
    fuel_sigma = _scenario_sigma("fuel_moisture_p50", template, fuel_spread, error_sigma)
    sensitivity = _emc_rh_sensitivity(result.temperature_2m, result.relative_humidity_2m)
    fuel_independent = np.sqrt(np.clip(fuel_sigma**2 - (sensitivity * rh_sigma) ** 2, 0.25, None))
    dates = _local_dates(result.time)
    unique_dates = list(dict.fromkeys(dates))
    day_of_hour = np.array([unique_dates.index(value) for value in dates])
    published_days = local_day_slices(result.time)

    rng = np.random.default_rng(seed)
    published = result.fire_danger.transpose("time", "y", "x").values
    rh0 = result.relative_humidity_2m.transpose("time", "y", "x").values
    wind0 = result.wind_speed_10m.transpose("time", "y", "x").values
    gust0 = result.wind_gust_10m.transpose("time", "y", "x").values
    fm0 = result.fuel_moisture_p50.transpose("time", "y", "x").values
    counters = {name: np.zeros(published.shape, dtype=np.uint16) for name in (
        "matches", "rh", "gust", "both", "elevated", "critical",
    )}
    daily_counters = {name: np.zeros((len(published_days),) + published.shape[1:], dtype=np.uint16) for name in ("elevated", "critical")}
    for _ in range(scenario_count):
        z = rng.standard_normal((3, len(unique_dates)) + published.shape[1:]).astype(np.float32)
        z_rh, z_wind, z_fuel = (z[index][day_of_hour] for index in range(3))
        rh = np.clip(rh0 + rh_sigma * z_rh, 0, 100)
        wind = np.clip(wind0 + wind_sigma * z_wind, 0, None)
        gust = np.clip(gust0 + gust_sigma * z_wind, 0, None)
        fm = np.clip(fm0 + sensitivity * (rh - rh0) + fuel_independent * z_fuel, 1, 40)
        scenario = _classify_values(fm, rh, wind)
        found = scenario != 255
        elevated = found & (scenario >= CATEGORY_ELEVATED)
        critical = found & (scenario >= CATEGORY_CRITICAL)
        low_rh = rh <= 25
        high_gust = gust >= 13.4112
        counters["matches"] += scenario == published
        counters["rh"] += low_rh
        counters["gust"] += high_gust
        counters["both"] += low_rh & high_gust
        counters["elevated"] += elevated
        counters["critical"] += critical
        for day, (_, indexes) in enumerate(published_days):
            daily_counters["elevated"][day] += elevated[indexes].any(axis=0)
            daily_counters["critical"][day] += critical[indexes].any(axis=0)

    valid_values = published != 255

    def probability(hits: np.ndarray) -> xr.DataArray:
        values = np.rint(hits * 100 / scenario_count).astype(np.uint8)
        values[~valid_values] = 255
        return xr.DataArray(values, dims=template.dims, coords=template.coords)

    result["category_confidence"] = probability(counters["matches"])
    coarse = (result.quality_mask & QUALITY_BITS["coarse_synoptic_fallback"]) != 0
    result["category_confidence"] = xr.where(
        coarse & (result.category_confidence != 255), result.category_confidence.clip(max=49), result.category_confidence
    ).astype("uint8")
    result["meteorological_confidence"] = xr.where(
        coarse & (result.meteorological_confidence != 255), result.meteorological_confidence.clip(max=49), result.meteorological_confidence
    ).astype("uint8")
    result["probability_rh_le_25"] = probability(counters["rh"])
    result["probability_gust_ge_30mph"] = probability(counters["gust"])
    result["probability_concurrent"] = probability(counters["both"])
    result["probability_elevated_plus"] = probability(counters["elevated"])
    result["probability_critical_plus"] = probability(counters["critical"])
    # Chance that any hour of each local day reaches the category; picked up
    # by daily_aggregates.
    day_valid = np.stack([valid_values[indexes].any(axis=0) for _, indexes in published_days])
    spatial = {name: template.coords[name] for name in ("y", "x") if name in template.coords}
    for name in ("elevated", "critical"):
        values = np.rint(daily_counters[name] * 100 / scenario_count).astype(np.uint8)
        values[~day_valid] = 255
        result[f"daily_probability_{name}_plus"] = xr.DataArray(values, dims=("local_day", "y", "x"), coords=spatial)
    return result


def scenario_scales_from_sources(cubes: dict[str, SourceCube], dataset: xr.Dataset) -> dict[str, xr.DataArray]:
    """Ensemble member spread combined with HRRR/RRFS disagreement.

    Error floors are applied in add_risk_and_confidence (_scenario_sigma), from
    station verification when there is enough of it."""
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
        "relative_humidity_2m": False,
        "wind_gust_10m": False,
        "wind_speed_10m": True,
    }
    for variable, derived_speed in definitions.items():
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
        combined = xr.zeros_like(dataset.temperature_2m)
        for candidate in candidates:
            combined = np.hypot(combined, candidate.fillna(0))
        result[variable] = combined.astype("float32")
    return result


def local_day_slices(times: xr.DataArray) -> list[tuple[str, np.ndarray]]:
    dates = _local_dates(times)
    return [
        (key, np.array([i for i, value in enumerate(dates) if value == key], dtype=int))
        for key in list(dict.fromkeys(dates))[:3]
    ]


def daily_aggregates(dataset: xr.Dataset) -> xr.Dataset:
    days = local_day_slices(dataset.time)
    outputs: dict[str, list[xr.DataArray]] = {name: [] for name in (
        "peak_fire_danger", "minimum_rh", "minimum_fuel_moisture_p50", "maximum_temperature",
        "maximum_wind", "maximum_gust", "total_precipitation", "category_confidence",
        "maximum_snow_water_equivalent", "probability_elevated_plus", "probability_critical_plus",
    )}
    for day, (_, indexes) in enumerate(days):
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
        for name in ("elevated", "critical"):
            source = dataset.get(f"daily_probability_{name}_plus")
            outputs[f"probability_{name}_plus"].append(
                source.isel(local_day=day, drop=True) if source is not None
                else xr.full_like(peak, 255, dtype="uint8")
            )
    result = xr.Dataset({name: xr.concat(values, dim="day") for name, values in outputs.items()})
    result = result.assign_coords(day=np.arange(1, len(days) + 1), local_date=("day", [key for key, _ in days]))
    return result


def build_forecast_cube(
    cubes: dict[str, SourceCube],
    initial_fuel_moisture: xr.DataArray | float,
    predicted_residuals: dict[str, xr.DataArray] | None = None,
    fuel_quantile_residuals: dict[str, xr.DataArray] | None = None,
    confidence_components: dict[str, xr.DataArray | float] | None = None,
    error_sigma: dict[str, xr.DataArray] | None = None,
) -> tuple[xr.Dataset, xr.Dataset]:
    blended = blend_sources(cubes)
    corrected = apply_corrections(blended, predicted_residuals).dataset
    fueled = evolve_fuel_moisture(corrected, initial_fuel_moisture, fuel_quantile_residuals)
    final = add_risk_and_confidence(
        fueled, confidence_components=confidence_components,
        scenario_scales=scenario_scales_from_sources(cubes, fueled), error_sigma=error_sigma,
    )
    return final, daily_aggregates(final)
