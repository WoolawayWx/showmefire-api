"""Frozen statewide physical-reference ranks, independent of retraining."""
from __future__ import annotations
import numpy as np


def _weighted_reference(values, weights):
    valid = np.isfinite(values) & (values >= 0) & np.isfinite(weights) & (weights > 0)
    x, w = np.asarray(values)[valid], np.asarray(weights)[valid]
    if not len(x):
        raise ValueError("risk reference has no valid burnable-area/time samples")
    order = np.argsort(x)
    x, w = x[order], w[order]
    unique, indices = np.unique(x, return_index=True)
    mass = np.add.reduceat(w, indices)
    # Equal physical values always have equal rank; no random tie breaking.
    rank = (np.cumsum(mass)-mass/2) / mass.sum()
    return unique.tolist(), rank.tolist()


def severity(ros, intensity, scale):
    def rank(values, name):
        ref = scale["reference"][name]
        x = np.asarray(values, float)
        result = np.interp(x, ref["values"], ref["ranks"], left=0., right=1.)
        return np.where(np.isfinite(x) & (x >= 0), result, np.nan)
    return np.maximum(rank(ros, "ros_m_min"), rank(intensity, "intensity_kw_m"))


def fit_scale(ros, intensity, weights=None, version="mo-relative-experimental-v1"):
    ros, intensity = np.broadcast_arrays(ros, intensity)
    weights = np.ones(ros.shape) if weights is None else np.broadcast_to(weights, ros.shape)
    valid = np.isfinite(ros) & np.isfinite(intensity) & (ros >= 0) & (intensity >= 0)
    scale = {"version": version, "reference": {}, "percentiles": [40, 70, 85, 95]}
    for name, values in (("ros_m_min", ros), ("intensity_kw_m", intensity)):
        x, ranks = _weighted_reference(values[valid], weights[valid])
        scale["reference"][name] = {"values": x, "ranks": ranks}
    s = severity(ros[valid], intensity[valid], scale)
    x, ranks = _weighted_reference(s, np.asarray(weights)[valid])
    cutpoints = np.interp([.4, .7, .85, .95], ranks, x)
    if np.any(np.diff(cutpoints) <= 0):
        raise ValueError("insufficient physical diversity for five distinct tiers")
    scale["cutpoints"] = cutpoints.tolist()
    return scale


def categories(ros, intensity, scale, burnable_fraction=1.):
    s = severity(ros, intensity, scale)
    out = np.searchsorted(scale["cutpoints"], s, side="right").astype("float32")
    return np.where(np.isfinite(s) & (np.asarray(burnable_fraction) > 0), out, np.nan)



# Absolute danger levels from flame length, after the fire-line handling chart
# (Andrews & Rothermel 1982): under 1 ft, 1-4 ft hand crews, 4-8 ft equipment,
# 8-11 ft serious control problems, over 11 ft crowning and major runs.
FLAME_LENGTH_SCALE = {"version": "flame-length-handling-chart-v1", "units": "m",
                      "levels": ["Low", "Moderate", "Elevated", "Critical", "Extreme"],
                      "upper_bounds_ft": [1, 4, 8, 11],
                      "upper_bounds_m": [0.3048, 1.2192, 2.4384, 3.3528]}


def flame_length_categories(flame_length_m, burnable_fraction=1., scale=FLAME_LENGTH_SCALE):
    """0-4 level per cell from absolute flame length; NaN where nothing can burn."""
    flame = np.asarray(flame_length_m, float)
    out = np.searchsorted(np.asarray(scale["upper_bounds_m"]), flame, side="right").astype("float32")
    return np.where(np.isfinite(flame) & (np.asarray(burnable_fraction) > 0), out, np.nan)
