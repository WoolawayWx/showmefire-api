"""
Wires a live forecast run's grids into the fire_risk_fusion Phase-A
shadow (services/risk_fusion_shadow.py::record_rule_mc). Called from
DailyForecast.py right after the hourly loop finishes - read-only with
respect to everything it's given, never raises, never affects the public
forecast path.

Deliberate simplifications for this first wiring, both documented rather
than hidden:

1. One county-day row per forecast run, using "day 1" of the run (the
   AFTERNOON_LEAD_HOURS/FULL_LEAD_HOURS lead windows from
   core/risk_fusion_features.py). The offline backfill
   (model-training/risk_fusion/build_county_days.py) is what builds the
   full multi-day panel; this hook's job is only to put TODAY's forecast
   through the same rule-MC path in shadow, once per run.

2. FM/RH/wind uncertainty use the documented fallback values from the
   project plan (V5's global regime half-width 3.41, RH sigma 7.0 pct,
   wind lognormal sigma 0.30) rather than the per-run
   spatial_fm_uncertainty_cache or forecast_verification.py outputs.
   Wiring those in is a follow-up once enough runs have accumulated in
   spatial_fm_uncertainty_cache to make per-run reads meaningful - using
   them from day one, before there is any cache to read, would just be
   the same fallback with extra code in between.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

import numpy as np

from core.fire_danger import RULE_SPEC
from core.risk_fusion_county_reference import county_cells
from core.risk_fusion_features import AFTERNOON_LEAD_HOURS, FULL_LEAD_HOURS, vapor_pressure_deficit_kpa
from services import risk_fusion_glm_shadow as rfgs
from services import risk_fusion_shadow as rfs

logger = logging.getLogger(__name__)

# See module docstring point 2. Global (not per-regime) fallback, since
# there is no per-run regime classification wired into this hook yet.
FALLBACK_FM_HALF_WIDTH = 3.41
FALLBACK_RH_SIGMA = 7.0
FALLBACK_WIND_SIGMA_LOG = 0.30
_TRUNCATED_NORMAL_80PCT_Z = 1.2816


def run_shadow_for_forecast(
    hourly_fm: list,
    hourly_rh: list,
    hourly_ws_kts: list,
    run_id: str,
    valid_local_date: str,
) -> bool:
    """
    hourly_fm/hourly_rh/hourly_ws_kts: lists of 2D grids, one per forecast
    hour, indexed the same way DailyForecast.py's own hourly_fm/hourly_rh/
    hourly_ws already are (hours_ahead 4..15 -> index 0..11).
    Returns True if shadow evidence was recorded, False otherwise
    (disabled, grid mismatch, or any internal failure) - callers should
    log the return value but never branch forecast behavior on it.
    """
    try:
        if not rfs.diagnostics()["enabled"]:
            return False

        cells = county_cells()
        if not hourly_fm:
            rfs.record_skipped_run("no hourly forecast grids available")
            return False

        grid_shape = list(np.asarray(hourly_fm[0]).shape)
        if grid_shape != cells["grid_shape"]:
            rfs.record_skipped_run(
                f"grid shape mismatch: forecast grid {grid_shape} != "
                f"vendored county_cells grid {cells['grid_shape']} - county_cells.json "
                "needs rebuilding against this repo's own HRRR crop before this hook can score"
            )
            return False

        afternoon_indices = [h for h in AFTERNOON_LEAD_HOURS if h < len(hourly_fm) + 4]
        full_indices = [h for h in FULL_LEAD_HOURS if h < len(hourly_fm) + 4]
        # hourly_fm[i] corresponds to hours_ahead = i + 4 (leads start at f04).
        afternoon_offsets = [h - 4 for h in afternoon_indices if 0 <= h - 4 < len(hourly_fm)]
        full_offsets = [h - 4 for h in full_indices if 0 <= h - 4 < len(hourly_fm)]
        if not full_offsets:
            rfs.record_skipped_run("no leads available in the day-1 aggregation window")
            return False

        cell_to_fips = cells["cell_to_fips"]
        county_list = sorted({fips for fips in cell_to_fips.values()})

        fm_by_county, rh_by_county, wind_by_county = [], [], []
        for fips in county_list:
            county_cell_map = {k: v for k, v in cell_to_fips.items() if v == fips}
            fm_vals = [_reduce(hourly_fm[i], county_cell_map, np.nanmean) for i in full_offsets]
            rh_vals = [_reduce(hourly_rh[i], county_cell_map, np.nanmin) for i in afternoon_offsets] or \
                      [_reduce(hourly_rh[i], county_cell_map, np.nanmin) for i in full_offsets]
            wind_vals = [_reduce(hourly_ws_kts[i], county_cell_map, np.nanmax) for i in full_offsets]
            fm_by_county.append(float(np.nanmean(fm_vals)))
            rh_by_county.append(float(np.nanmin(rh_vals)))
            wind_by_county.append(float(np.nanmax(wind_vals)))

        n = len(county_list)
        fm = np.asarray(fm_by_county)
        rh = np.asarray(rh_by_county)
        wind_kts = np.asarray(wind_by_county)
        fm_sigma = np.full(n, FALLBACK_FM_HALF_WIDTH / _TRUNCATED_NORMAL_80PCT_Z)
        rh_sigma = np.full(n, FALLBACK_RH_SIGMA)
        wind_sigma_log = np.full(n, FALLBACK_WIND_SIGMA_LOG)

        return rfs.record_rule_mc(
            run_id=run_id,
            county_fips=county_list,
            valid_local_date=valid_local_date,
            fm=fm, rh=rh, wind_kts=wind_kts,
            fm_sigma=fm_sigma, rh_sigma=rh_sigma, wind_sigma_log=wind_sigma_log,
            thresholds=RULE_SPEC["thresholds"],
        )
    except Exception as exc:
        logger.warning("risk_fusion shadow hook failed (non-fatal): %s", exc)
        try:
            rfs.record_skipped_run(str(exc))
        except Exception:
            pass
        return False


def _reduce(grid: np.ndarray, cell_to_fips: dict, reducer) -> float:
    values = [grid[int(k.split(",")[0]), int(k.split(",")[1])] for k in cell_to_fips]
    return float(reducer(np.asarray(values, dtype="float64")))


def run_glm_shadow_for_forecast(
    hourly_fm: list,
    hourly_rh: list,
    hourly_ws_kts: list,
    hourly_temp_c: list,
    hourly_precip_mm: list,
    run_id: str,
    valid_local_date: str,
) -> bool:
    """
    Wires a live forecast run's grids into the fire_risk_fusion Phase-B
    (learned GLM) shadow (services/risk_fusion_glm_shadow.py::
    score_glm_for_forecast). Mirrors run_shadow_for_forecast's shape
    exactly (same grid-mismatch guard, same read-only/never-raises
    contract) - kept as a separate function rather than folded into
    run_shadow_for_forecast because Phase B needs additional per-hour
    inputs (temperature, precipitation) that Phase A's rule-MC never did.

    hourly_temp_c: per-hour 2D grids in Celsius, same indexing as
    hourly_fm/hourly_rh/hourly_ws_kts (hours_ahead 4..15 -> index 0..11).
    hourly_precip_mm: per-hour 2D grids of that hour's precipitation
    INTERVAL (not cumulative), same indexing.
    """
    try:
        if not rfgs.diagnostics()["enabled"]:
            return False

        cells = county_cells()
        if not hourly_fm:
            rfgs.record_skipped_run("no hourly forecast grids available")
            return False

        grid_shape = list(np.asarray(hourly_fm[0]).shape)
        if grid_shape != cells["grid_shape"]:
            rfgs.record_skipped_run(
                f"grid shape mismatch: forecast grid {grid_shape} != "
                f"vendored county_cells grid {cells['grid_shape']} - county_cells.json "
                "needs rebuilding against this repo's own HRRR crop before this hook can score"
            )
            return False

        afternoon_indices = [h for h in AFTERNOON_LEAD_HOURS if h < len(hourly_fm) + 4]
        full_indices = [h for h in FULL_LEAD_HOURS if h < len(hourly_fm) + 4]
        afternoon_offsets = [h - 4 for h in afternoon_indices if 0 <= h - 4 < len(hourly_fm)]
        full_offsets = [h - 4 for h in full_indices if 0 <= h - 4 < len(hourly_fm)]
        if not full_offsets:
            rfgs.record_skipped_run("no leads available in the day-1 aggregation window")
            return False

        vpd_by_hour = [vapor_pressure_deficit_kpa(np.asarray(hourly_temp_c[i]), np.asarray(hourly_rh[i]))
                       for i in full_offsets]

        valid_date = datetime.strptime(valid_local_date, "%Y-%m-%d").date()
        calendar_row = rfgs.month_dummies(valid_date.month)
        calendar_row["is_weekend"] = valid_date.weekday() >= 5

        cell_to_fips = cells["cell_to_fips"]
        county_list = sorted({fips for fips in cell_to_fips.values()})
        calendar_rows, weather_rows = {}, {}
        for fips in county_list:
            county_cell_map = {k: v for k, v in cell_to_fips.items() if v == fips}
            def _cell_values(grid):
                return np.asarray([np.asarray(grid)[int(k.split(",")[0]), int(k.split(",")[1])]
                                   for k in county_cell_map], dtype="float64")

            rh_full = [_reduce(hourly_rh[i], county_cell_map, np.nanmean) for i in full_offsets]
            rh_afternoon = [_reduce(hourly_rh[i], county_cell_map, np.nanmin) for i in afternoon_offsets] or rh_full
            wind_all_cells = np.concatenate([_cell_values(hourly_ws_kts[i]) for i in full_offsets])
            vpd_full = [float(np.nanmax(_cell_values(grid))) for grid in vpd_by_hour]
            precip_full = [_reduce(hourly_precip_mm[i], county_cell_map, np.nanmean) for i in full_offsets]

            calendar_rows[fips] = calendar_row
            weather_rows[fips] = {
                "rh_mean": float(np.nanmean(rh_full)),
                "rh_min_afternoon": float(np.nanmin(rh_afternoon)),
                "wind_kts_max": float(np.nanmax(wind_all_cells)),
                "wind_kts_p90": float(np.nanpercentile(wind_all_cells, 90)),
                "vpd_kpa_max": float(np.nanmax(vpd_full)),
                "precip_24h_mm": float(np.nansum(precip_full)),
            }

        return rfgs.score_glm_for_forecast(
            run_id=run_id,
            valid_local_date=valid_local_date,
            county_fips=county_list,
            calendar_rows=calendar_rows,
            weather_rows=weather_rows,
        )
    except Exception as exc:
        logger.warning("risk_fusion GLM shadow hook failed (non-fatal): %s", exc)
        try:
            rfgs.record_skipped_run(str(exc))
        except Exception:
            pass
        return False
