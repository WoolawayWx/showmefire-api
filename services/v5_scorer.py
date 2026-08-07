"""Frozen, self-contained API scorer for station V5 shadow bundles."""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import xgboost as xgb
import xarray as xr
from scipy.spatial import cKDTree

from services.v5_shadow import validate_bundle

RUNTIME_SCHEMA = "api-v5-station-runtime-1"


def initialization_rows(run_time, station_records):
    """Add nearest cached RTMA initialization weather to causal station observations."""
    stamp = pd.Timestamp(run_time)
    stamp = stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    root = Path("/app") if Path("/app").exists() else Path(__file__).resolve().parent.parent
    path = root / "cache" / "rtma" / f"rtma_{stamp:%Y%m%d_%H}z.nc"
    with xr.open_dataset(path) as dataset:
        lat = np.asarray(dataset.latitude).ravel(); lon = np.asarray(dataset.longitude).ravel()
        lon = np.where(lon > 180, lon - 360, lon); tree = cKDTree(np.column_stack((lat, lon)))
        result = []
        for record in station_records:
            index = tree.query([record["latitude"], record["longitude"]])[1]
            temp = float(np.asarray(dataset.t2m).ravel()[index]); temp = temp - 273.15 if temp > 150 else temp
            rh = float(np.asarray(dataset.r2).ravel()[index])
            wind = float(np.hypot(np.asarray(dataset.u10).ravel()[index], np.asarray(dataset.v10).ravel()[index]))
            observed = pd.to_datetime(record.get("observation_time"), utc=True, errors="coerce")
            age = (stamp - observed).total_seconds() / 3600 if pd.notna(observed) else np.nan
            if not np.isfinite(age) or age < 0 or age > 3:
                continue
            result.append({**record, "initial_age_hours": age, "rtma_temp_c": temp, "rtma_rh": rh, "rtma_wind_ms": wind})
    return result


def _emc(temp, rh):
    temp, rh = np.asarray(temp, float), np.asarray(rh, float)
    value = np.where(rh <= 10, .03 + .2626 * rh - .00104 * rh * temp,
                     np.where(rh <= 50, 2.22 - .160 * rh + .01660 * temp,
                              21.06 - .4944 * rh + .005565 * rh ** 2 - .00063 * rh * temp))
    return np.clip(value, 1, 40)


def _evolve(initial, temp, rh, rain=None, saturation=25, scale=2):
    output, state = [], float(initial)
    rain = np.zeros(len(temp)) if rain is None else np.asarray(rain, float)
    for t, humidity, amount in zip(temp, rh, rain):
        if np.isfinite(amount) and amount > 0: state += (saturation - state) * (1 - np.exp(-amount / scale))
        equilibrium = float(_emc([t], [humidity])[0]); tau = 10 if equilibrium < state else 6
        state += (equilibrium - state) * (1 - np.exp(-1 / tau)); output.append(state)
    return output


def build_features(frame, base_prediction=None, physics_variant="rain25_scale2"):
    required = {"run_id", "station_id", "valid_time", "initial_fm", "initial_age_hours", "lead_hour",
                "rtma_temp_c", "rtma_rh", "rtma_wind_ms", "hrrr_temp_c", "hrrr_rh", "hrrr_wind_ms",
                "hrrr_precip_mm", "hrrr_precip_accum_mm", "hrrr_precip_increment_mm", "lat", "lon"}
    if missing := required.difference(frame.columns): raise ValueError(f"V5 shadow inputs missing: {sorted(missing)}")
    result = frame.copy(); valid = pd.to_datetime(result.valid_time, utc=True, errors="coerce")
    hour = 2 * np.pi * (valid.dt.hour + valid.dt.minute / 60) / 24
    doy = 2 * np.pi * (valid.dt.dayofyear - 1) / 365.2425
    result["valid_hour_sin"], result["valid_hour_cos"] = np.sin(hour), np.cos(hour)
    result["valid_doy_sin"], result["valid_doy_cos"] = np.sin(doy), np.cos(doy)
    result["temp_change_c"] = result.hrrr_temp_c - result.rtma_temp_c
    result["rh_change"] = result.hrrr_rh - result.rtma_rh
    result["wind_change_ms"] = result.hrrr_wind_ms - result.rtma_wind_ms
    result["vpd_kpa"] = .6108 * np.exp(17.27 * result.hrrr_temp_c / (result.hrrr_temp_c + 237.3)) * (1 - result.hrrr_rh.clip(0, 100) / 100)
    result["forecast_emc"] = _emc(result.hrrr_temp_c, result.hrrr_rh)
    result["precip_occurrence"] = (result.hrrr_precip_mm.fillna(0) > 0).astype(float)
    result["log1p_precip"] = np.log1p(result.hrrr_precip_mm.fillna(0).clip(lower=0))
    ordered = result.sort_values(["run_id", "station_id", "lead_hour"])
    grouped = ordered.groupby(["run_id", "station_id"], sort=False)
    for name, source in (("temp_lead_change_c", "hrrr_temp_c"), ("rh_lead_change", "hrrr_rh"),
                         ("wind_lead_change_ms", "hrrr_wind_ms"), ("precip_lead_change_mm", "hrrr_precip_mm")):
        result.loc[ordered.index, name] = grouped[source].diff().fillna(0).clip(lower=0 if "precip" in name else None).to_numpy()
    defaults = {"precip_interval_hours": 1., "precip_reset_flag": 0., "precip_partial_window_flag": 0., "precip_available": 1.}
    for name, value in defaults.items():
        if name not in result: result[name] = value
    for name in ("precip_duration_hours", "precip_3h_mm", "hours_since_forecast_rain", "wet_dry_transition",
                 "precip_6h_mm", "precip_24h_mm", "precip_6h_partial", "precip_24h_partial",
                 "precip_intensity_mmph", "rain_occurrence_interval", "active_rain_indicator",
                 "post_rain_3h_indicator", "precip_missing_indicator", "precip_quality_issue",
                 "forecast_drying_rate", "post_rain_drying_interaction"):
        result[name] = 0.0
    physics, rain_physics = pd.Series(index=result.index, dtype=float), pd.Series(index=result.index, dtype=float)
    for _, indices in ordered.groupby(["run_id", "station_id"], sort=False).groups.items():
        indices = list(indices); group = result.loc[indices].sort_values("lead_hour"); idx = group.index
        increments = group.hrrr_precip_increment_mm.fillna(0).clip(lower=0)
        physics.loc[idx] = _evolve(group.initial_fm.iloc[0], group.hrrr_temp_c, group.hrrr_rh)
        rain_physics.loc[idx] = _evolve(group.initial_fm.iloc[0], group.hrrr_temp_c, group.hrrr_rh, increments)
        rain_age, rain_duration = 999., 0.
        times = pd.to_datetime(group.valid_time, utc=True)
        for position, row_index in enumerate(idx):
            amount = float(increments.loc[row_index]); duration = max(0., float(group.precip_interval_hours.loc[row_index]))
            rain_duration, rain_age = ((rain_duration + duration, 0.) if amount > 0 else (0., min(999., rain_age + duration)))
            result.at[row_index, "precip_duration_hours"] = rain_duration
            result.at[row_index, "hours_since_forecast_rain"] = rain_age
            for hours, column, partial in ((3, "precip_3h_mm", None), (6, "precip_6h_mm", "precip_6h_partial"), (24, "precip_24h_mm", "precip_24h_partial")):
                selected = (times > times.iloc[position] - pd.Timedelta(hours=hours)) & (times <= times.iloc[position])
                result.at[row_index, column] = float(increments[selected].sum())
                if partial: result.at[row_index, partial] = float(selected.sum() < hours)
            intensity = amount / duration if duration else 0
            active, post = amount > .1, amount <= .1 and rain_age <= 3
            rh_change = 0 if position == 0 else float(group.hrrr_rh.iloc[position] - group.hrrr_rh.iloc[position - 1])
            drying = max(0., -rh_change) / max(duration, 1.)
            updates = {"precip_intensity_mmph": intensity, "rain_occurrence_interval": amount > 0,
                       "active_rain_indicator": active, "post_rain_3h_indicator": post,
                       "precip_missing_indicator": float(group.precip_available.loc[row_index] <= 0),
                       "precip_quality_issue": float(group.precip_reset_flag.loc[row_index] > 0 or group.precip_available.loc[row_index] <= 0 or group.precip_partial_window_flag.loc[row_index] > 0),
                       "forecast_drying_rate": drying, "post_rain_drying_interaction": drying * post}
            for name, value in updates.items(): result.at[row_index, name] = float(value)
    result["physics_fm"], result["rain_physics_fm"] = physics, rain_physics
    result["summer_indicator"] = valid.dt.month.isin([6, 7, 8]).astype(float)
    result["hot_dry_interaction"] = (result.hrrr_temp_c - 25).clip(lower=0) * result.vpd_kpa
    result["initial_emc_gap"] = result.initial_fm - result.forecast_emc
    if base_prediction is not None:
        result["incumbent_base_fm"] = np.asarray(base_prediction, float)
        result["physics_minus_incumbent"] = result.physics_fm - result.incumbent_base_fm
        result["rain_physics_minus_base"] = result.rain_physics_fm - result.incumbent_base_fm
    return result


def score(bundle_dir, rows):
    started = perf_counter(); contract = validate_bundle(bundle_dir); bundle_dir = Path(bundle_dir)
    base = xgb.XGBRegressor(); base.load_model(bundle_dir / "base_xgboost.json")
    specialist = xgb.XGBRegressor(); specialist.load_model(bundle_dir / "specialist_xgboost.json")
    raw = pd.DataFrame(rows); first = build_features(raw)
    base_prediction = base.predict(first[contract["base_features"]])
    prepared = build_features(raw, base_prediction, contract["physics_variant"])
    correction = specialist.predict(prepared[contract["specialist_features"]])
    guard = json.loads((bundle_dir / "guard.json").read_text()); uncertainty = json.loads((bundle_dir / "uncertainty.json").read_text())
    valid = pd.to_datetime(prepared.valid_time, utc=True); regimes = np.where(prepared.active_rain_indicator > 0, "active_rain",
        np.where(prepared.post_rain_3h_indicator > 0, "post_rain", np.where(valid.dt.month.isin([6, 7, 8]), "summer_dry", "other")))
    prediction, weights, reasons = base_prediction.copy(), [], []
    for index, (lead, regime) in enumerate(zip(prepared.lead_hour, regimes)):
        item = guard.get(f"{regime}|{float(lead)}", {}); weight = float(item.get("weight", 0)); cap = float(item.get("cap", 0))
        if prepared.precip_available.iloc[index] <= 0: weight, cap, reason = 0., 0., "unavailable_features"
        else: reason = item.get("reason", "missing_guard")
        prediction[index] += weight * np.clip(correction[index], -cap, cap); weights.append(weight); reasons.append(reason)
    widths = np.asarray([uncertainty.get("regimes", {}).get(str(r), {}).get("half_width", uncertainty["global"]) for r in regimes])
    return {"base": base_prediction, "prediction": prediction, "intervals": np.column_stack([prediction-widths, prediction, prediction+widths]),
            "raw_correction": correction, "guard_weights": np.asarray(weights), "guard_reasons": reasons,
            "regimes": regimes, "prepared": prepared, "latency_ms": (perf_counter()-started)*1000,
            "runtime_schema": RUNTIME_SCHEMA}
