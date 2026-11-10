"""Versioned moisture state and explicit physical-backend boundary.

The vectorized lag model is a diagnostic baseline, NOT the Nelson stick model.
NFDRS-derived states can be supplied as a checked external trajectory. A
candidate cannot be certified NFDRS-compatible without reference parity evidence.
"""
from __future__ import annotations
import numpy as np
from .contracts import FUELS

PHYSICS_VERSION = "simard-lag-diagnostic-v1"


def equilibrium(temp_c, rh_pct):
    t, h = np.broadcast_arrays(np.asarray(temp_c, float), np.asarray(rh_pct, float))
    h = np.clip(h, 0, 100)
    out = np.where(h <= 10, .03 + .2626*h - .00104*h*t,
                   np.where(h <= 50, 1.7544 + .160107*h - .026611*t,
                            21.06 - .4944*h + .005565*h*h - .00063*h*t))
    return np.clip(out, 0, 60)


def step(state, temp_c, rh_pct, precip_mm, solar_wm2, *, elapsed_hours=1.):
    if elapsed_hours <= 0:
        raise ValueError("elapsed_hours must be positive")
    target = equilibrium(temp_c, rh_pct)
    rain = np.asarray(precip_mm, float)
    valid = np.isfinite(target) & np.isfinite(rain) & (rain >= 0) & np.isfinite(solar_wm2)
    target = np.clip(target - np.maximum(np.asarray(solar_wm2)-250, 0)/2000, 0, 60)
    out = {}
    for fuel in FUELS:
        prior = np.asarray(state[f"fm{fuel}_pct"], float)
        dry = prior + (target-prior)*(-np.expm1(-elapsed_hours/fuel))
        # Continuous rainfall wetting; sparse source intervals cannot suppress it
        # by being split below an arbitrary per-hour wetting threshold.
        wet = dry + (60-dry)*(-np.expm1(-rain/(2.5*np.sqrt(fuel))))
        out[f"fm{fuel}_pct"] = np.where(valid & np.isfinite(prior), np.clip(wet, 0, 60), np.nan).astype("float32")
    return out


def validate_state(state, shape):
    for fuel in FUELS:
        name = f"fm{fuel}_pct"
        value = np.asarray(state[name])
        if value.shape != shape or np.any((value[np.isfinite(value)] < 0) | (value[np.isfinite(value)] > 150)):
            raise ValueError(f"invalid {name} state")

