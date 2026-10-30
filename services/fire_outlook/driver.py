"""
Per-member and ensemble drivers for the seven-day outlook.

Input is one member's normalized hourly dataset (forecast_v1 adapter contract, times
cycle+0..168 h): temperature_2m degC, relative_humidity_2m %, wind_u_10m / wind_v_10m m/s,
precipitation_increment mm (the amount ending at that time), snow_water_equivalent mm.

Time convention: step k = 1..168 uses fields valid at cycle+k h (instantaneous fields at the end
of the hour, precipitation ending then) and is the interval [k-1, k), so it belongs to window
(k-1)//6. Fuel moisture starts from `initial` (causally conditioned + RAWS-anchored by the caller;
see `seed_state` for the quality-labelled fallback) and runs continuously through all 168 steps.

Open decisions, exposed as parameters, not hidden constants:
  wind_factor     multiply |V10| before the rule. forecast_v1 uses 1.0; the day-1 ensemble product
                  uses 0.8 (HRRR_FIRE_WIND_REDUCTION_FACTOR). Default 1.0 until decided.
  snow_swe_mm     snow-water-equivalent at/above which a cell is forced Low.
  precip_interval_hours  source accumulation interval (GEFS 3 h) for the wet test, see moisture.py.
"""
from __future__ import annotations

import warnings
from typing import Callable, Dict, Iterable, Optional

import numpy as np

from services.fire_outlook import moisture, windows
from services.rule_uncertainty import MISSING_CATEGORY

MPS_TO_KNOTS = 1.9438444924406
REQUIRED = ("temperature_2m", "relative_humidity_2m", "wind_u_10m", "wind_v_10m",
            "precipitation_increment", "snow_water_equivalent")
STEPS = windows.HORIZON_HOURS


class IncompleteMember(ValueError):
    """The member does not cover all 168 steps / required variables; it must not be published as complete."""


def _fields(ds, cycle_time) -> Dict[str, np.ndarray]:
    missing = [v for v in REQUIRED if v not in ds.data_vars]
    if missing:
        raise IncompleteMember(f"missing variables: {missing}")
    expected = np.datetime64(cycle_time.replace(tzinfo=None), "h") + np.arange(STEPS + 1).astype("timedelta64[h]")
    got = ds["time"].values.astype("datetime64[h]")
    if got.shape != expected.shape or not np.array_equal(got, expected):
        raise IncompleteMember(f"time axis is not cycle+0..{STEPS}h hourly")
    out = {}
    for name in REQUIRED:
        arr = ds[name]
        if "member" in arr.dims:
            if arr.sizes["member"] != 1:
                raise IncompleteMember("run_member takes exactly one member")
            arr = arr.squeeze("member", drop=True)
        arr = arr.transpose("time", ...)
        out[name] = np.asarray(arr.values, dtype="float64")
        if out[name].ndim != 3:
            raise IncompleteMember(f"{name} must be (time, y, x), got {out[name].shape}")
    return out


def seed_state(fields: Dict[str, np.ndarray], fm10_anchor: Optional[np.ndarray] = None) -> tuple[moisture.State, str]:
    """Fallback seed from t=0 weather (EMC). Returns (state, quality label); 'emc_seed' means NOT
    causally conditioned and must be reported as such in the manifest."""
    state = moisture.initial_state(fields["temperature_2m"][0], fields["relative_humidity_2m"][0], fm10_anchor)
    return state, ("anchored_emc_seed" if fm10_anchor is not None else "emc_seed")


def run_member(ds, cycle_time, initial: moisture.State, categorize: Callable,
               dampen: Optional[Callable] = None, *, precip_interval_hours: int = 3,
               wind_factor: float = 1.0, snow_swe_mm: float = 1.0) -> Dict[str, np.ndarray]:
    """One member -> per-window outputs, all (28, y, x)."""
    f = _fields(ds, cycle_time)
    sl = slice(1, STEPS + 1)
    temp, rh = f["temperature_2m"][sl], f["relative_humidity_2m"][sl]
    wind_kts = np.hypot(f["wind_u_10m"][sl], f["wind_v_10m"][sl]) * wind_factor * MPS_TO_KNOTS
    precip = f["precipitation_increment"][sl]
    swe = f["snow_water_equivalent"][sl]

    # Wet test on the source interval total (hours 1..p, p+1..2p ...), depth spread evenly as given.
    if STEPS % precip_interval_hours:
        raise ValueError("precip_interval_hours must divide 168")
    blocks = precip.reshape((STEPS // precip_interval_hours, precip_interval_hours) + precip.shape[1:])
    wet = np.repeat(blocks.sum(axis=1) > moisture.WET_THRESHOLD_MM, precip_interval_hours, axis=0)
    wet &= ~np.isnan(precip)

    fuel = moisture.run_member(initial, temp, rh, precip, wet)
    fm10 = fuel["fm10"]
    snow = swe >= snow_swe_mm
    snow = np.where(np.isnan(swe), False, snow)  # unknown snow never forces Low

    hourly = windows.hourly_categories(fm10, rh, wind_kts, categorize, dampen, snow)
    out = windows.window_summary(hourly)

    def by_window(a):
        return a.reshape((windows.N_WINDOWS, windows.WINDOW_HOURS) + a.shape[1:])
    out["precip_6h_mm"] = np.where(np.isnan(by_window(precip)).any(axis=1), np.nan, by_window(precip).sum(axis=1)).astype("float32")
    out["fm10_min"] = by_window(fm10).min(axis=1).astype("float32")  # NaN propagates
    out["temp_max_c"] = by_window(temp).max(axis=1).astype("float32")
    out["rh_min_pct"] = by_window(rh).min(axis=1).astype("float32")
    out["wind_max_kts"] = by_window(wind_kts).max(axis=1).astype("float32")
    return out


class EnsembleRunner:
    """Fold members in one at a time. Keeps only (members, 28, y, x) window-min fm10 beyond the
    streaming agreement counters; everything else is reduced per member."""

    SAMPLE_KEYS = ("fm10_min", "temp_max_c", "rh_min_pct", "wind_max_kts", "precip_6h_mm")

    def __init__(self, n_members: int, grid_shape: tuple[int, int],
                 station_cells: Optional[list[tuple[int, int]]] = None):
        self.agreement = windows.AgreementAccumulator(n_members, grid_shape)
        # Full member samples at station cells only (n_members x 28 x stations): tiny, and what station
        # verification needs (member mean/p10/p90 of each window statistic) without gridding those distributions.
        self._cells = np.array(station_cells, dtype=int).reshape(-1, 2) if station_cells else None
        self._samples = (None if self._cells is None else
                         {k: np.full((n_members, windows.N_WINDOWS, len(self._cells)), np.nan, dtype="float32")
                          for k in self.SAMPLE_KEYS})
        self._fm10_min = np.full((n_members, windows.N_WINDOWS) + tuple(grid_shape), np.nan, dtype="float32")
        self._hours_e = np.zeros((windows.N_WINDOWS,) + tuple(grid_shape), dtype="float64")
        self._hours_c = np.zeros_like(self._hours_e)
        self._precip = np.zeros_like(self._hours_e)
        self._tmax = np.full(self._hours_e.shape, -np.inf)
        self._rhmin = np.full(self._hours_e.shape, np.inf)
        self._wmax = np.full(self._hours_e.shape, -np.inf)
        # Scenario MEANS of the window statistics (what the map shows); a NaN in any member keeps the mean NaN (NoData).
        self._tmax_sum = np.zeros(self._hours_e.shape)
        self._rhmin_sum = np.zeros(self._hours_e.shape)
        self._wmax_sum = np.zeros(self._hours_e.shape)
        self._n = 0

    def add(self, member: Dict[str, np.ndarray]) -> None:
        self.agreement.add(member["max_category"])
        self._fm10_min[self._n] = member["fm10_min"]
        if self._samples is not None:
            rows, cols = self._cells[:, 0], self._cells[:, 1]
            for key in self.SAMPLE_KEYS:
                self._samples[key][self._n] = member[key][:, rows, cols]
        ok_e, ok_c = member["hours_elevated"] != 255, member["hours_critical"] != 255
        self._hours_e += np.where(ok_e, member["hours_elevated"], 0)
        self._hours_c += np.where(ok_c, member["hours_critical"], 0)
        self._precip += np.nan_to_num(member["precip_6h_mm"])
        self._tmax = np.fmax(self._tmax, member["temp_max_c"])
        self._rhmin = np.fmin(self._rhmin, member["rh_min_pct"])
        self._wmax = np.fmax(self._wmax, member["wind_max_kts"])
        self._tmax_sum += member["temp_max_c"]
        self._rhmin_sum += member["rh_min_pct"]
        self._wmax_sum += member["wind_max_kts"]
        self._n += 1

    def finalize(self) -> Dict[str, np.ndarray]:
        agree = self.agreement.finalize()
        n = float(self._n)
        extra = {} if self._samples is None else {"station_samples": self._samples, "station_cells": self._cells}
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN cells are NoData by design
            q = np.nanpercentile(self._fm10_min, [10, 50, 90], axis=0).astype("float32")
        return {
            "agreement_elevated": agree["elevated"], "agreement_critical": agree["critical"],
            "nodata": agree["missing"], "n_members": agree["n_members"],
            "fm10_min_p10": q[0], "fm10_min_p50": q[1], "fm10_min_p90": q[2],
            "mean_hours_elevated": (self._hours_e / n).astype("float32"),
            "mean_hours_critical": (self._hours_c / n).astype("float32"),
            "mean_precip_6h_mm": (self._precip / n).astype("float32"),
            "temp_max_c_envelope": self._tmax.astype("float32"),
            "rh_min_pct_envelope": self._rhmin.astype("float32"),
            "wind_max_kts_envelope": self._wmax.astype("float32"),
            "temp_max_c_mean": (self._tmax_sum / n).astype("float32"),
            "rh_min_pct_mean": (self._rhmin_sum / n).astype("float32"),
            "wind_max_kts_mean": (self._wmax_sum / n).astype("float32"),
            **extra,
        }


def run_ensemble(members: Iterable, cycle_time, n_members: int, grid_shape: tuple[int, int],
                 initial_for: Callable[[object], moisture.State], categorize: Callable,
                 dampen: Optional[Callable] = None, station_cells: Optional[list[tuple[int, int]]] = None,
                 **kwargs) -> Dict[str, np.ndarray]:
    """members yields one normalized dataset at a time (sequential, disk-backed upstream). Any member that
    raises IncompleteMember aborts the run: a main run is only published when ALL members cover ALL windows."""
    runner = EnsembleRunner(n_members, grid_shape, station_cells)
    for ds in members:
        runner.add(run_member(ds, cycle_time, initial_for(ds), categorize, dampen, **kwargs))
    return runner.finalize()
