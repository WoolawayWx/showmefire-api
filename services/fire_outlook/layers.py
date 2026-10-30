"""
Supporting map layers for the seven-day outlook: 10-hour fuel moisture and weather, as filled value bands per window.

The primary layers (Elevated-or-higher / Critical-or-higher scenario agreement) are in contours.py. These are the plan's
"supporting layers": what the weather scenarios say about the conditions behind the agreement. Each is a per-window field in
display units (the site shows US units), cut into bands at fixed edges and polygonized exactly like the agreement bands (disjoint,
holes preserved, NoData never covered). Band edges live here, are written to the manifest (`layers`), and the website colours them.

  fm10     modeled 10-hour fuel moisture: the driest value reached in each window, MEDIAN over the 31 scenarios (%)
  temp     window high temperature, scenario MEAN (deg F)
  rh       window low relative humidity, scenario MEAN (%)
  wind     window high 10 m wind, scenario MEAN (mph, before any fire-rule wind factor)
  precip   6-hour precipitation total, scenario MEAN (in)

Means and medians describe the typical scenario; the worst-case envelopes stay in the arrays for point queries. Fuel moisture is
MODELED (RAWS only anchors its starting state); the layer label says so.
"""
from __future__ import annotations

import json
from typing import Dict, List

import numpy as np

from services.fire_outlook import contours

KTS_TO_MPH = 1.15077945
MM_PER_INCH = 25.4

LAYER_DEFS: List[dict] = [
    {"id": "fm10", "label": "10-hour fuel moisture", "detail": "modeled; driest value in each window, median of the scenarios",
     "group": "Fuel moisture", "unit": "%", "edges": [4, 6, 8, 10, 12, 15, 20, 25], "decimals": 0},
    {"id": "temp", "label": "Temperature", "detail": "highest in each window, mean of the scenarios",
     "group": "Weather", "unit": "°F", "edges": [32, 40, 50, 60, 70, 80, 90, 100], "decimals": 0},
    {"id": "rh", "label": "Relative humidity", "detail": "lowest in each window, mean of the scenarios",
     "group": "Weather", "unit": "%", "edges": [10, 15, 20, 25, 30, 40, 50, 60, 80], "decimals": 0},
    {"id": "wind", "label": "Wind", "detail": "highest 10 m wind in each window, mean of the scenarios",
     "group": "Weather", "unit": "mph", "edges": [5, 10, 15, 20, 25, 30, 40], "decimals": 0},
    {"id": "precip", "label": "Precipitation", "detail": "6-hour total, mean of the scenarios",
     "group": "Weather", "unit": "in", "edges": [0.01, 0.05, 0.1, 0.25, 0.5, 1.0], "decimals": 2},
]
LAYER_IDS = [d["id"] for d in LAYER_DEFS]


def display_values(result: Dict[str, np.ndarray], wind_factor: float = 1.0) -> Dict[str, np.ndarray]:
    """(28, y, x) fields in display units, NaN where any input is NaN. Wind is undone from the rule's wind factor."""
    need = ("fm10_min_p50", "temp_max_c_mean", "rh_min_pct_mean", "wind_max_kts_mean", "mean_precip_6h_mm")
    missing = [k for k in need if k not in result]
    if missing:
        raise KeyError(f"result lacks {missing}; cannot build the supporting layers")
    factor = wind_factor or 1.0
    return {
        "fm10": result["fm10_min_p50"],
        "temp": result["temp_max_c_mean"] * 9.0 / 5.0 + 32.0,
        "rh": result["rh_min_pct_mean"],
        "wind": result["wind_max_kts_mean"] / factor * KTS_TO_MPH,
        "precip": result["mean_precip_6h_mm"] / MM_PER_INCH,
    }


def value_band_raster(values: np.ndarray, edges: List[float]) -> np.ndarray:
    """(y, x) -> class 1..len(edges)+1 (value < edges[0] is class 1; >= edges[-1] is the last); NaN -> 0 (not drawn)."""
    classes = (np.searchsorted(np.asarray(edges, dtype="float64"), values, side="right") + 1).astype("uint8")
    return np.where(np.isfinite(values), classes, 0).astype("uint8")


def band_labels(edges: List[float]) -> Dict[int, dict]:
    """class -> properties; the open ends have a None bound."""
    bounds = [None] + list(edges) + [None]
    return {i + 1: {"lower": bounds[i], "upper": bounds[i + 1]} for i in range(len(edges) + 1)}


def manifest_layers() -> List[dict]:
    """Layer metadata for the manifest: the two agreement layers first, then the supporting value layers."""
    agreement = [
        {"id": "elevated", "label": "Elevated or higher", "detail": "scenario agreement", "group": "Scenario agreement",
         "unit": "%", "kind": "agreement", "edges": list(contours.BAND_EDGES_PCT)},
        {"id": "critical", "label": "Critical or higher", "detail": "scenario agreement", "group": "Scenario agreement",
         "unit": "%", "kind": "agreement", "edges": list(contours.BAND_EDGES_PCT)},
    ]
    return agreement + [{**d, "kind": "value"} for d in LAYER_DEFS]


def layer_files(result: Dict[str, np.ndarray], grid: dict, *, wind_factor: float = 1.0, simplify_cells: float = 0.35,
                precision: int = 4) -> Dict[str, bytes]:
    """{'contours/<layer>/wNN.geojson': bytes} for every supporting layer and window. Coarser rounding/simplification than the
    agreement bands (value fields are noisier and nothing is stacked); vertices ~11 m."""
    values = display_values(result, wind_factor)
    files: Dict[str, bytes] = {}
    for d in LAYER_DEFS:
        labels = band_labels(d["edges"])
        for w in range(values[d["id"]].shape[0]):
            bands = value_band_raster(values[d["id"]][w], d["edges"])
            feats = contours.band_polygons(bands, west=grid["west"], north=grid["north"], resolution_m=grid["resolution_m"],
                                           crs=grid.get("crs", "EPSG:32615"), simplify_cells=simplify_cells,
                                           precision=precision, labels=labels)
            payload = {"type": "FeatureCollection", "features": feats,
                       "properties": {"layer": d["id"], "window_index": w, "unit": d["unit"], "edges": d["edges"]}}
            files[f"contours/{d['id']}/w{w:02d}.geojson"] = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return files
