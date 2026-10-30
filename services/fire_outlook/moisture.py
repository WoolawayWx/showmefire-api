"""
Time-aware dead-fuel moisture for the seven-day outlook.

`advance` is the one-hour step of services/spread_rate_moisture.py::_advance_moisture
(parity-tested against it), extracted so it needs no xarray/scipy/RTMA cache. State is
carried continuously through a scenario's rain, drying and recovery; callers must not
re-seed it at six-hour window boundaries.

Coarse source intervals (GEFS gives 3 h / 6 h accumulations): the original wetting rule
fires when HOURLY precip > 0.5 mm, a threshold tuned for hourly RTMA. Spreading 1.2 mm
evenly over six hours gives 0.2 mm/h, which would silently suppress wetting. So `advance`
takes an optional `wet` mask, and the outlook passes `interval_total > 0.5 mm` (via
`allocate_precip`) while the hourly depth added is the even allocation. Interpolation here
does not imply hourly forecast skill; the source interval is kept in run metadata.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

FM_MIN_PERCENT = 1.0
FM_MAX_PERCENT = 40.0
TAU_HOURS = {"fm1": 1.0, "fm10": 10.0, "fm100": 100.0}
RAIN_GAIN = {"fm1": 0.5, "fm10": 0.15, "fm100": 0.05}  # percent per mm
WET_THRESHOLD_MM = 0.5

State = Dict[str, np.ndarray]


def nelson_emc(temp_c, rh) -> np.ndarray:
    rh = np.asarray(rh, dtype=float)
    temp_c = np.asarray(temp_c, dtype=float)
    emc = np.where(
        rh <= 10,
        0.03 + 0.2626 * rh - 0.00104 * rh * temp_c,
        np.where(
            rh <= 50,
            2.22 - 0.160 * rh + 0.01660 * temp_c,
            21.06 - 0.4944 * rh + 0.005565 * rh ** 2 - 0.00063 * rh * temp_c,
        ),
    )
    return np.clip(emc, FM_MIN_PERCENT, FM_MAX_PERCENT)


def initial_state(temp_c, rh, fm10_anchor=None) -> State:
    """Seed from EMC; where a (RAWS-anchored, causally conditioned) fm10 is given, use it for fm10."""
    emc = nelson_emc(temp_c, rh)
    state = {"fm1": emc.copy(), "fm10": emc.copy(), "fm100": emc.copy()}
    if fm10_anchor is not None:
        anchor = np.asarray(fm10_anchor, dtype=float)
        state["fm10"] = np.where(np.isfinite(anchor), np.clip(anchor, FM_MIN_PERCENT, FM_MAX_PERCENT), state["fm10"])
    return state


def advance(state: State, temp_c, rh, precip_mm, wet: Optional[np.ndarray] = None) -> State:
    """Advance one hour. NaN weather propagates NaN (NoData), never a zero-rain default."""
    emc = nelson_emc(temp_c, rh)
    precip_mm = np.asarray(precip_mm, dtype=float)
    out = {}
    for key, tau in TAU_HOURS.items():
        value = state[key] + (emc - state[key]) * (1.0 - np.exp(-1.0 / tau))
        mask = (precip_mm > WET_THRESHOLD_MM) if wet is None else np.asarray(wet, dtype=bool)
        value = np.where(mask, np.minimum(value + RAIN_GAIN[key] * np.nan_to_num(precip_mm), FM_MAX_PERCENT), value)
        out[key] = np.where(np.isnan(precip_mm), np.nan, value)  # missing rain is NoData, not "no rain"
    return out


def allocate_precip(interval_totals: np.ndarray, interval_hours: int) -> Tuple[np.ndarray, np.ndarray]:
    """Even allocation of per-interval totals (n, ...) -> hourly depth (n*h, ...) and wet mask
    (interval total > WET_THRESHOLD_MM, repeated across the interval's hours). NaN stays NaN."""
    totals = np.asarray(interval_totals, dtype=float)
    hourly = np.repeat(totals / interval_hours, interval_hours, axis=0)
    wet = np.repeat(totals > WET_THRESHOLD_MM, interval_hours, axis=0)
    return hourly, wet


def interpolate_hourly(values: np.ndarray, step_hours: int) -> np.ndarray:
    """Linear interpolation of instantaneous fields at 0, s, 2s, ... hours to hourly (len = (n-1)*s + 1).
    NaN at either end of a segment gives NaN across it."""
    values = np.asarray(values, dtype=float)
    n = values.shape[0]
    out = np.empty(((n - 1) * step_hours + 1,) + values.shape[1:], dtype=float)
    for i in range(n - 1):
        for k in range(step_hours):
            w = k / step_hours
            out[i * step_hours + k] = (1 - w) * values[i] + w * values[i + 1]
    out[-1] = values[-1]
    return out


def run_member(initial: State, temp_c, rh, precip_mm, wet: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """Advance continuously over (hours, ...) hourly inputs. Returns per-hour fm1/fm10/fm100, each (hours, ...).
    Hour h's fuel state is the state after applying hour h's weather."""
    state = initial
    series = {k: [] for k in state}
    for h in range(np.asarray(temp_c).shape[0]):
        state = advance(state, temp_c[h], rh[h], precip_mm[h], None if wet is None else wet[h])
        for k in series:
            series[k].append(state[k])
    return {k: np.stack(v) for k, v in series.items()}
