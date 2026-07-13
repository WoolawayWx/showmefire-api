"""Guarded spatial fuel-moisture inference with an observable safe fallback."""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

from models.versioning import load_active_assets

logger = logging.getLogger(__name__)
_runtime = None
_diagnostics = {"available": False, "fallback": True, "fallback_reason": "not initialized", "last_success": None, "inference_ms": None}


def diagnostics(): return dict(_diagnostics)


def _load_runtime():
    global _runtime
    assets = load_active_assets("fuel_moisture_spatial", "stable")
    signature = tuple((role, str(value["path"]), value["sha256"]) for role, value in sorted(assets.items()))
    if _runtime and _runtime["signature"] == signature: return _runtime
    import onnxruntime as ort
    session = ort.InferenceSession(str(assets["model"]["path"]), providers=["CPUExecutionProvider"])
    contract = json.loads(session.get_modelmeta().custom_metadata_map["static_contract"])
    with xr.open_dataset(assets["static_bundle"]["path"]) as ds:
        if ds.attrs["grid_fingerprint"] != contract["grid_fingerprint"]: raise ValueError("active static grid mismatch")
        continuous = np.stack([np.zeros((256, 256)) if name == "__none__" else ds[name].values for name in contract["continuous_channels"]]).astype("float32")
        categorical = np.stack([np.zeros((256, 256)) if name == "__none__" else ds[name].values for name in contract["categorical_channels"]]).astype("int64")
        means = np.asarray(contract["normalization_mean"], "float32")[:, None, None]; stds = np.asarray(contract["normalization_std"], "float32")[:, None, None]
        if len(continuous): continuous = np.nan_to_num((continuous - means) / stds)
        static = {"continuous": continuous, "categorical": categorical, "x": ds.x.values, "y": ds.y.values,
                  "lat": ds.latitude.values, "lon": ds.longitude.values}
    _runtime = {"signature": signature, "assets": assets, "session": session, "contract": contract, "static": static}
    return _runtime


def _map(source_lat, source_lon, target_lat, target_lon, values):
    source_lon = np.where(np.asarray(source_lon) > 180, np.asarray(source_lon) - 360, source_lon)
    tree = cKDTree(np.column_stack((np.asarray(source_lat).ravel(), np.asarray(source_lon).ravel())))
    index = tree.query(np.column_stack((target_lat.ravel(), target_lon.ravel())))[1]
    return np.asarray(values).ravel()[index].reshape(target_lat.shape)


def _idw(points, lat, lon):
    coordinates = np.asarray([[p[1], p[0]] for p in points]); values = np.asarray([p[2] for p in points], "float32")
    distance, index = cKDTree(coordinates).query(np.column_stack((lat.ravel(), lon.ravel())), k=min(8, len(points)))
    if distance.ndim == 1: distance, index = distance[:, None], index[:, None]
    weights = 1 / np.maximum(distance, 1e-4) ** 2; weights /= weights.sum(axis=1, keepdims=True)
    shape = lat.shape
    return (weights * values[index]).sum(axis=1).reshape(shape), distance[:, 0].reshape(shape), (1 / np.square(weights).sum(axis=1)).reshape(shape)


def _physics(initial, temp, rh):
    state = initial.copy(); result = np.empty_like(temp)
    for step in range(len(temp)):
        equilibrium = np.clip(np.where(rh[step] <= 10, .03 + .2626 * rh[step] - .00104 * rh[step] * temp[step],
                                      np.where(rh[step] <= 50, 2.22 - .160 * rh[step] + .01660 * temp[step],
                                               21.06 - .4944 * rh[step] + .005565 * rh[step] ** 2 - .00063 * rh[step] * temp[step])), 1, 40)
        tau = np.where(equilibrium < state, 10.0, 6.0); state += (equilibrium - state) * (1 - np.exp(-1 / tau)); result[step] = state
    return result[:, None].astype("float32")


def _field(ds, name, x, y, steps, default=0):
    if name not in ds: return np.full((steps, 256, 256), default, "float32")
    ydim, xdim = ds.latitude.dims; values = np.asarray(ds[name].interp({xdim: x, ydim: y}).values).squeeze()
    if values.ndim == 2: values = np.repeat(values[None], steps, axis=0)
    return values.astype("float32")


def _normalizer(contract, name):
    item = contract["normalizers"][name]
    return (np.asarray(item["mean"], "float32").reshape(1, -1, 1, 1),
            np.asarray(item["std"], "float32").reshape(1, -1, 1, 1))


def _antecedent_rtma(root, init_stamp, lat, lon, allowed_missing=2):
    """Build 13 causal frames, carrying only earlier analyzed weather forward."""
    frames, missing, times = [], [], []
    previous = None
    for offset in range(-12, 1):
        stamp = init_stamp + pd.Timedelta(hours=offset); times.append(stamp.isoformat())
        path = root / "cache" / "rtma" / f"rtma_{stamp:%Y%m%d_%H}z.nc"
        try:
            with xr.open_dataset(path) as rtma:
                temp = _map(rtma.latitude.values, rtma.longitude.values, lat, lon, rtma.t2m.values)
                temp = np.where(temp > 150, temp - 273.15, temp)
                rh = _map(rtma.latitude.values, rtma.longitude.values, lat, lon, rtma.r2.values)
                wind = np.hypot(_map(rtma.latitude.values, rtma.longitude.values, lat, lon, rtma.u10.values),
                                _map(rtma.latitude.values, rtma.longitude.values, lat, lon, rtma.v10.values))
                previous = np.stack((temp, rh, wind)).astype("float32")
            frame = np.concatenate((previous, np.ones((1, *lat.shape), "float32")))
        except Exception:
            missing.append(stamp.isoformat())
            if previous is None:
                raise ValueError(f"earliest antecedent RTMA unavailable: {stamp.isoformat()}")
            frame = np.concatenate((previous, np.zeros((1, *lat.shape), "float32")))
        frames.append(frame)
    if len(missing) > allowed_missing:
        raise ValueError(f"antecedent RTMA missing {len(missing)} hours; maximum is {allowed_missing}")
    return np.stack(frames), missing, times


def try_predict(hrrr: xr.Dataset, fuel_points, run_date=None):
    """Return native-grid quantiles or None; never prevents legacy serving."""
    started = time.perf_counter()
    try:
        if not fuel_points or len(fuel_points) < 3: raise ValueError("fewer than three causal FM observations")
        runtime = _load_runtime(); static = runtime["static"]; contract = runtime["contract"]
        init_value = run_date if run_date is not None else np.asarray(hrrr.time.values).squeeze()
        init_stamp = pd.Timestamp(init_value)
        init_stamp = init_stamp.tz_localize("UTC") if init_stamp.tzinfo is None else init_stamp.tz_convert("UTC")
        root = Path("/app") if Path("/app").exists() else Path(__file__).resolve().parent.parent
        if contract.get("teacher_exported") is not False: raise ValueError("model contract does not prohibit teacher export")
        leads = [int(value) for value in contract.get("hrrr_leads", range(4, 16))]
        if leads != list(range(4, 16)): raise ValueError(f"unsupported HRRR lead contract: {leads}")
        all_steps = np.asarray(hrrr.step.values).reshape(-1) if "step" in hrrr.coords else np.arange(hrrr.sizes.get("step", 1))
        all_leads = [int(pd.to_timedelta(value).total_seconds() // 3600) for value in all_steps]
        if any(lead not in all_leads for lead in leads): raise ValueError("HRRR does not contain exact f04-f15 sequence")
        indices = [all_leads.index(lead) for lead in leads]; steps = len(leads)
        x, y, lat, lon = static["x"], static["y"], static["lat"], static["lon"]
        antecedent, missing_antecedent, antecedent_times = _antecedent_rtma(
            root, init_stamp, lat, lon, int(contract.get("allowed_missing_antecedent", 2)))
        initial, distance, effective = _idw(fuel_points, lat, lon); obs_mask = np.zeros_like(initial, "float32")
        station_tree = cKDTree(np.column_stack((lat.ravel(), lon.ravel())))
        for point in fuel_points: obs_mask.ravel()[station_tree.query([point[1], point[0]])[1]] = 1
        age = np.zeros_like(initial, "float32")
        temp = _field(hrrr, "t2m", x, y, len(all_leads))[indices]; temp = np.where(temp > 150, temp - 273.15, temp)
        rh = _field(hrrr, "r2", x, y, len(all_leads))[indices]
        wind = np.hypot(_field(hrrr, "u10", x, y, len(all_leads))[indices], _field(hrrr, "v10", x, y, len(all_leads))[indices])
        precip = _field(hrrr, "apcp", x, y, len(all_leads))[indices]
        hrrr_forecast = np.stack((temp, rh, wind, precip), axis=1).astype("float32")
        current = np.stack((initial, obs_mask, age, distance, effective)).astype("float32")
        antecedent_mean, antecedent_std = _normalizer(contract, "antecedent")
        hrrr_mean, hrrr_std = _normalizer(contract, "hrrr")
        feed = {"antecedent_rtma": ((antecedent - antecedent_mean) / antecedent_std)[None],
                "hrrr_forecast": ((hrrr_forecast - hrrr_mean) / hrrr_std)[None], "current_fm_state": current[None],
                "static_continuous": static["continuous"][None], "static_categorical": static["categorical"][None],
                "physics_trajectory": _physics(initial, temp, rh)[None]}
        output = runtime["session"].run(None, feed)[0][0]
        if not np.isfinite(output).all() or not (np.all(output[:, 0] <= output[:, 1]) and np.all(output[:, 1] <= output[:, 2])): raise ValueError("invalid spatial quantiles")
        native_lat, native_lon = hrrr.latitude.values, np.where(hrrr.longitude.values > 180, hrrr.longitude.values - 360, hrrr.longitude.values)
        result = {name: np.stack([_map(lat, lon, native_lat, native_lon, output[step, index]) for step in range(steps)])
                  for name, index in (("p10", 0), ("p50", 1), ("p90", 2))}
        interval_width = np.maximum(0, output[:, 2] - output[:, 0])
        confidence = np.exp(-distance[None] / 2) / (1 + interval_width / 10)
        result["confidence"] = np.stack([_map(lat, lon, native_lat, native_lon, confidence[step]) for step in range(steps)])
        result["nearest_station_distance_deg"] = _map(lat, lon, native_lat, native_lon, distance)
        result["effective_station_count"] = _map(lat, lon, native_lat, native_lon, effective)
        _diagnostics.update({"available": True, "fallback": False, "fallback_reason": None, "last_success": datetime.now(timezone.utc).isoformat(),
                             "inference_ms": round((time.perf_counter() - started) * 1000, 1), "bundle": contract["bundle_file"], "feature_set": contract["feature_set"],
                             "antecedent_expected": len(antecedent_times), "antecedent_missing": len(missing_antecedent),
                             "antecedent_missing_times": missing_antecedent, "teacher_exported": False})
        return result
    except Exception as exc:
        logger.warning("Spatial FM unavailable; retaining XGBoost forecast: %s", exc)
        _diagnostics.update({"available": False, "fallback": True, "fallback_reason": str(exc), "inference_ms": round((time.perf_counter() - started) * 1000, 1)})
        return None
