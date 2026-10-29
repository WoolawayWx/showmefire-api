"""Beta/preview graphic for the Fire Weather Severity Index scored on BLENDED weather (HRRR + RRFS).

The live HRRR-only shadow (fire_weather_index_hook.py) is untouched. This runs inside the ensemble fire-danger run, which
has already fetched RRFS and regridded it onto the forecast grid for the same valid hours, so nothing new
is downloaded. It reduces each source's hourly fields with exactly the HRRR-only hook's four reductions, averages the
reduced grids (equal weight over the sources that have a value in a cell - the offline blend's rule,
model-training/fire_weather_index/grid_score.py), and scores the result with the same bundle and scoring as the shadow, so
the two products differ only in their weather input. Output: its own map (house style), evidence file and manifest entry.

Best-effort and gated by the shadow's enabled switch: never raises, never touches the HRRR-only shadow's state.
"""
from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

from core.risk_fusion_county_reference import county_cells
from core.risk_fusion_features import AFTERNOON_LEAD_HOURS, FULL_LEAD_HOURS, vapor_pressure_deficit_kpa
from services import fire_weather_index_shadow as fwis

logger = logging.getLogger(__name__)

MPS_TO_KNOTS = 1.9438444924406
# One member per source family, first available wins (RRFS falls back to the previous cycle, as the ensemble does).
FAMILIES = (
    ("HRRR", "HRRR", ("hrrr_0",)),
    ("RRFS", "RRFS", ("rrfs_0", "rrfs_m6")),
)
GRID_KEYS = ("rh_min_afternoon", "wind_kts_max", "vpd_kpa_max", "precip_24h_mm")


def lead_offsets(n_hours: int, lead_start: int = 4) -> tuple[list[int], list[int]]:
    """(afternoon offsets, full-window offsets) into an hourly array whose index 0 is lead `lead_start` - the same
    indexing as the HRRR-only hook (index 0..11 = leads 4..15)."""
    afternoon = [h - lead_start for h in AFTERNOON_LEAD_HOURS if 0 <= h - lead_start < n_hours]
    full = [h - lead_start for h in FULL_LEAD_HOURS if 0 <= h - lead_start < n_hours]
    return afternoon, full


def reduce_member(fields: Dict[str, np.ndarray], wind_factor: float = 0.8, lead_start: int = 4) -> Dict[str, np.ndarray]:
    """One source's hourly fields (temp_c, rh, wind_ms, precip_mm; hours x y x x) -> the four grids the index scores.
    Same reductions as fire_weather_index_hook.py."""
    n = len(fields["rh"])
    afternoon, full = lead_offsets(n, lead_start)
    if not full:
        raise ValueError("no leads available in the day-1 aggregation window")
    rh = np.asarray(fields["rh"], dtype="float64")
    wind_kts = np.asarray(fields["wind_ms"], dtype="float64") * wind_factor * MPS_TO_KNOTS
    temp = np.asarray(fields["temp_c"], dtype="float64")
    precip = np.asarray(fields["precip_mm"], dtype="float64")
    vpd = np.stack([vapor_pressure_deficit_kpa(temp[i], rh[i]) for i in full])
    rh_idx = afternoon or full
    return {
        "rh_min_afternoon": np.nanmin(rh[rh_idx], axis=0),
        "wind_kts_max": np.nanmax(wind_kts[full], axis=0),
        "vpd_kpa_max": np.nanmax(vpd, axis=0),
        "precip_24h_mm": np.nansum(precip[full], axis=0),
    }


def blend_grids(per_source: Dict[str, Dict[str, np.ndarray]]) -> tuple[Dict[str, np.ndarray], Dict[str, float]]:
    """Equal-weight mean over the sources with a finite value in each cell, per grid. A cell no source covers stays NaN.
    Also returns each source's coverage: the fraction of the grid it has a value for."""
    blended: Dict[str, np.ndarray] = {}
    for key in GRID_KEYS:
        stack = np.stack([np.asarray(g[key], dtype="float64") for g in per_source.values()])
        finite = np.isfinite(stack)
        total = np.where(finite, stack, 0.0).sum(axis=0)
        count = finite.sum(axis=0)
        blended[key] = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    coverage = {name: float(np.isfinite(np.asarray(g["rh_min_afternoon"])).mean()) for name, g in per_source.items()}
    return blended, coverage


def county_rows(grids: Dict[str, np.ndarray], cell_to_fips: Dict[str, str]) -> Dict[str, Dict[str, float]]:
    """Per-county weather row from the blended grids, equal to the HRRR-only hook's per-county reduction: worst cell for
    RH, wind and VPD, county-mean precipitation."""
    by_county: Dict[str, list] = {}
    for key, fips in cell_to_fips.items():
        r, c = (int(v) for v in key.split(","))
        by_county.setdefault(fips, []).append((r, c))
    rows: Dict[str, Dict[str, float]] = {}
    for fips, cells in by_county.items():
        rr, cc = zip(*cells)
        sel = lambda name: np.asarray(grids[name])[list(rr), list(cc)]
        rows[fips] = {
            "rh_min_afternoon": float(np.nanmin(sel("rh_min_afternoon"))),
            "wind_kts_max": float(np.nanmax(sel("wind_kts_max"))),
            "vpd_kpa_max": float(np.nanmax(sel("vpd_kpa_max"))),
            "precip_24h_mm": float(np.nanmean(sel("precip_24h_mm"))),
        }
    return rows


def pick_members(member_fields: Dict[str, dict], plan: Optional[Iterable] = None) -> Dict[str, dict]:
    """family -> {"member_id", "short", "cycle"} for the sources actually available this run."""
    cycles = {getattr(i, "member_id", None): getattr(i, "cycle", None) for i in (plan or [])}
    chosen: Dict[str, dict] = {}
    for family, short, ids in FAMILIES:
        for member_id in ids:
            if member_id in member_fields:
                cycle = cycles.get(member_id)
                chosen[family] = {"member_id": member_id, "short": short,
                                  "cycle": cycle.strftime("%Y-%m-%dT%H:%MZ") if hasattr(cycle, "strftime") else None}
                break
    return chosen


def run_for_ensemble(plan: Sequence, member_fields: Dict[str, dict], wind_factors: Dict[str, float], lat: np.ndarray,
                     lon: np.ndarray, anchor, lead_start: int = 4, **score_kwargs) -> Dict:
    """Called from the ensemble run with the fields it already regridded. Returns {"ok": bool, ...}; never raises."""
    try:
        if not fwis.diagnostics()["enabled"]:
            return {"ok": False, "reason": "fire_weather_index shadow is not enabled"}
        chosen = pick_members(member_fields, plan)
        if len(chosen) < 2:
            return {"ok": False, "reason": f"only {len(chosen)} source available ({', '.join(chosen) or 'none'}); a blend needs at least two"}
        cells = county_cells()
        if list(cells["grid_shape"]) != list(np.asarray(lat).shape):
            return {"ok": False, "reason": f"grid shape mismatch: ensemble grid {list(np.asarray(lat).shape)} != county_cells grid {cells['grid_shape']}"}

        per_source = {family: reduce_member(member_fields[info["member_id"]], wind_factors.get(info["member_id"], 0.8), lead_start)
                      for family, info in chosen.items()}
        blended, coverage = blend_grids(per_source)
        rows = county_rows(blended, cells["cell_to_fips"])
        county_list = sorted(rows)
        sources = {
            "rule": "equal-weight mean of the sources with a value in each cell, reduced exactly like the HRRR-only shadow",
            "used": [f"{family} ({info['member_id']})" for family, info in chosen.items()],
            "used_short": [info["short"] for info in chosen.values()],
            "members": {family: info for family, info in chosen.items()},
            "coverage_fraction": coverage,
            "anchor_cycle_utc": anchor.strftime("%Y-%m-%dT%H:%MZ"),
        }
        return fwis.score_blend_for_forecast(
            run_id=anchor.strftime("%Y%m%d_%H"), valid_local_date=anchor.strftime("%Y-%m-%d"), county_fips=county_list,
            weather_rows=rows, weather_grids=blended, lat=lat, lon=lon, sources=sources, **score_kwargs)
    except Exception as exc:
        logger.warning("fire_weather_index blend failed (non-fatal): %s", exc)
        return {"ok": False, "reason": str(exc)}
