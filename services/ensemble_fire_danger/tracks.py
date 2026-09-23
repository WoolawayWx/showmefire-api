"""
Track M (real members) and Track E (synthetic draws from ensprod
mean/spread) -> per-member peak fire danger category on the target grid.

Contract-mirror discipline (api/core/contract_mirrors.json, pair
"ensemble_fire_danger_tracks"): byte-identical to
model-training/ensemble_fire_danger/tracks.py. The live product
(api/services/ensemble_fire_danger/runtime.py) and the training panel
(model-training/ensemble_fire_danger/build_panel.py) both call exactly
these functions, so calibration is fitted on the same numbers the live
graphics are drawn from.

Per member/draw, per valid hour:
  wind for the rule  = |V10| * wind_factor (DailyForecast's
                       HRRR_FIRE_WIND_REDUCTION_FACTOR, 0.8) in knots
  wind for FM model  = raw |V10| m/s (DailyForecast keeps ML input raw)
  FM                 = core.anchored_fm(member XGB FM, control XGB FM,
                       production FM)
  category           = categorize -> dampen -> snow forces Low
then the peak over the window, same as the public map.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

from . import core
from .fm_features import predict_fm_hourly
from .grib_idx import magnus_rh


def regridded_member_fields(ds: xr.Dataset, regridder) -> Dict[str, np.ndarray]:
    """Member Dataset(step,y,x) -> dict of (hours, ty, tx) arrays on the target grid."""
    t2m = regridder(ds["t2m"].values)
    rh = np.clip(regridder(ds["r2"].values), 0.0, 100.0)
    wind = regridder(np.hypot(ds["u10"].values, ds["v10"].values))
    precip = regridder(np.nan_to_num(ds["tp1h"].values)) if "tp1h" in ds else np.zeros_like(t2m)
    return {"temp_c": t2m - 273.15, "rh": rh, "wind_ms": wind, "precip_mm": np.nan_to_num(precip)}


def regridded_ensprod_fields(ds: xr.Dataset, regridder) -> Dict[str, np.ndarray]:
    out = {name: regridder(ds[name].values) for name in
           ("t2m_mean", "t2m_sprd", "d2m_mean", "d2m_sprd", "wind10_mean", "wind10_sprd")}
    out["precip_mm"] = np.nan_to_num(regridder(np.nan_to_num(ds["tp1h_mean"].values))) \
        if "tp1h_mean" in ds else np.zeros_like(out["t2m_mean"])
    for optional in ("jfwprb", "pwind_10p3"):
        if optional in ds:
            out[optional] = regridder(ds[optional].values)
    return out


def _peak_for(fields: Dict[str, np.ndarray], fm: np.ndarray, wind_factor: float, categorize, dampen,
              snow_mask) -> np.ndarray:
    wind_kts = fields["wind_ms"] * wind_factor * core.MPS_TO_KNOTS
    return core.peak_category(fm, fields["rh"], wind_kts, categorize, dampen=dampen, snow_mask=snow_mask)


def track_members(member_fields: Dict[str, Dict[str, np.ndarray]], weights: Dict[str, float],
                  wind_factors: Dict[str, float], control_id: Optional[str], booster, feature_names: Sequence[str],
                  valid_times: Sequence, categorize: Callable, dampen: Optional[Callable],
                  production_fm: Optional[np.ndarray], snow_mask: Optional[np.ndarray],
                  swe_grid: Optional[np.ndarray], fm_clip: Tuple[float, float]) -> Dict[str, object]:
    """Track M: returns {"member_ids", "weights", "peaks" (M,y,x), "min_rh" (M,y,x),
    "max_wind_kts" (M,y,x), "min_fm" (M,y,x)}."""
    ids = list(member_fields)
    xgb_fm = {mid: predict_fm_hourly(booster, feature_names, f["temp_c"], f["rh"], f["wind_ms"], f["precip_mm"],
                                     valid_times, swe_grid=swe_grid)
              for mid, f in member_fields.items()}
    if control_id in xgb_fm:
        control = xgb_fm[control_id]
    else:
        control = np.nanmean(np.stack(list(xgb_fm.values())), axis=0)
    peaks, min_rh, max_wind, min_fm = [], [], [], []
    for mid in ids:
        fields = member_fields[mid]
        fm = core.anchored_fm(xgb_fm[mid], control, production_fm, fm_clip)
        peaks.append(_peak_for(fields, fm, wind_factors.get(mid, 0.8), categorize, dampen, snow_mask))
        min_rh.append(np.nanmin(fields["rh"], axis=0))
        max_wind.append(np.nanmax(fields["wind_ms"] * wind_factors.get(mid, 0.8) * core.MPS_TO_KNOTS, axis=0))
        min_fm.append(np.nanmin(fm, axis=0))
    return {
        "member_ids": ids,
        "weights": [float(weights.get(mid, 1.0)) for mid in ids],
        "peaks": np.stack(peaks),
        "min_rh": np.stack(min_rh),
        "max_wind_kts": np.stack(max_wind),
        "min_fm": np.stack(min_fm),
    }


def track_synthetic(ensprod_fields: Dict[str, Dict[str, np.ndarray]], weights: Dict[str, float], synthetic: dict,
                    booster, feature_names: Sequence[str], valid_times: Sequence, categorize: Callable,
                    dampen: Optional[Callable], production_fm: Optional[np.ndarray], snow_mask: Optional[np.ndarray],
                    swe_grid: Optional[np.ndarray], fm_clip: Tuple[float, float]) -> Dict[str, object]:
    """Track E: pools every available ensprod source's mean/spread into
    one mixture, draws n_draws synthetic members, and scores each exactly
    like a Track M member (FM anchored on the XGBoost FM of the mixture
    MEAN weather, the Track-E analogue of the HRRR control)."""
    ids = list(ensprod_fields)
    w = [float(weights.get(i, 1.0)) for i in ids]

    def pooled(name: str):
        return core.pooled_mean_spread([ensprod_fields[i][f"{name}_mean"] for i in ids],
                                       [ensprod_fields[i][f"{name}_sprd"] for i in ids], w)

    t_mean, t_sprd = pooled("t2m")
    td_mean, td_sprd = pooled("d2m")
    w_mean, w_sprd = pooled("wind10")
    precip = sum(wi * ensprod_fields[i]["precip_mm"] for wi, i in zip(np.asarray(w) / sum(w), ids))
    wind_factor = float(synthetic.get("wind_factor", 0.8))

    mean_rh = magnus_rh(t_mean, np.minimum(td_mean, t_mean))
    control = predict_fm_hourly(booster, feature_names, t_mean - 273.15, mean_rh, w_mean, precip, valid_times,
                                swe_grid=swe_grid)
    peaks, min_rh, max_wind, min_fm = [], [], [], []
    for t, td, wind in core.synthetic_draws(t_mean, t_sprd, td_mean, td_sprd, w_mean, w_sprd,
                                            n_draws=int(synthetic.get("n_draws", 40)),
                                            seed=int(synthetic.get("seed", 0)),
                                            rho_t_td=float(synthetic.get("rho_t_td", -0.3)),
                                            spatial_sigma=float(synthetic.get("spatial_sigma_cells", 3.0))):
        rh = magnus_rh(t, td)
        fm_xgb = predict_fm_hourly(booster, feature_names, t - 273.15, rh, wind, precip, valid_times,
                                   swe_grid=swe_grid)
        fm = core.anchored_fm(fm_xgb, control, production_fm, fm_clip)
        fields = {"rh": rh, "wind_ms": wind}
        peaks.append(_peak_for(fields, fm, wind_factor, categorize, dampen, snow_mask))
        min_rh.append(np.nanmin(rh, axis=0))
        max_wind.append(np.nanmax(wind * wind_factor * core.MPS_TO_KNOTS, axis=0))
        min_fm.append(np.nanmin(fm, axis=0))
    n = len(peaks)
    extras = {}
    for optional in ("jfwprb", "pwind_10p3"):
        stacks = [ensprod_fields[i][optional] for i in ids if optional in ensprod_fields[i]]
        if stacks:
            extras[optional] = np.nanmax(np.nanmean(np.stack(stacks), axis=0), axis=0)
    return {
        "member_ids": [f"draw{i:02d}" for i in range(n)],
        "weights": [1.0] * n,
        "peaks": np.stack(peaks),
        "min_rh": np.stack(min_rh),
        "max_wind_kts": np.stack(max_wind),
        "min_fm": np.stack(min_fm),
        "sources": ids,
        "spread_rh_proxy": np.nanmax(np.nanmean(np.stack([t_sprd, td_sprd]), axis=0), axis=0),
        **extras,
    }
