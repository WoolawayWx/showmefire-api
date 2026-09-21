"""
Wires a live forecast run's grids into the fire_weather_index shadow
(services/fire_weather_index_shadow.py::score_for_forecast). Called from
DailyForecast.py alongside the existing risk_fusion Phase A/B hooks -
read-only, additive, never raises, never affects the public forecast path.

Builds the same per-county rh_min_afternoon/wind_kts_max/vpd_kpa_max/
precip_24h_mm aggregates services/risk_fusion_hook.py already computes for
its own GLM hook - same grids, same reduction logic, just handed to a
different shadow scorer. Not refactored into one shared helper: keeping
each hook independently readable/removable (per the isolation convention
every other guarded shadow already follows) outweighs the small amount of
duplication here.
"""
from __future__ import annotations

import logging

import numpy as np

from core.risk_fusion_county_reference import county_cells
from core.risk_fusion_features import AFTERNOON_LEAD_HOURS, FULL_LEAD_HOURS, vapor_pressure_deficit_kpa
from services import fire_weather_index_shadow as fwis

logger = logging.getLogger(__name__)


def _reduce(grid: np.ndarray, cell_to_fips: dict, reducer) -> float:
    values = [grid[int(k.split(",")[0]), int(k.split(",")[1])] for k in cell_to_fips]
    return float(reducer(np.asarray(values, dtype="float64")))


def run_fire_weather_index_shadow_for_forecast(
    hourly_rh: list,
    hourly_ws_kts: list,
    hourly_temp_c: list,
    hourly_precip_mm: list,
    run_id: str,
    valid_local_date: str,
    lat: np.ndarray = None,
    lon: np.ndarray = None,
) -> bool:
    """
    hourly_rh/hourly_ws_kts/hourly_temp_c/hourly_precip_mm: per-hour 2D
    grids, same indexing as the risk_fusion GLM hook (hours_ahead 4..15 ->
    index 0..11). hourly_precip_mm is that hour's precipitation INTERVAL,
    not cumulative.

    lat/lon: the same grid's coordinates (DailyForecast.py's own mo_bounds
    crop, same shape as hourly_rh[0]) - optional only so this still works
    if ever called without them, but when given, the shadow renders the
    real pixel map (see fire_weather_index_shadow.py::_render_png) instead
    of the county-choropleth fallback.
    """
    try:
        if not fwis.diagnostics()["enabled"]:
            return False

        cells = county_cells()
        if not hourly_rh:
            fwis.record_skipped_run("no hourly forecast grids available")
            return False

        grid_shape = list(np.asarray(hourly_rh[0]).shape)
        if grid_shape != cells["grid_shape"]:
            fwis.record_skipped_run(
                f"grid shape mismatch: forecast grid {grid_shape} != "
                f"vendored county_cells grid {cells['grid_shape']} - county_cells.json "
                "needs rebuilding against this repo's own HRRR crop before this hook can score"
            )
            return False

        afternoon_indices = [h for h in AFTERNOON_LEAD_HOURS if h < len(hourly_rh) + 4]
        full_indices = [h for h in FULL_LEAD_HOURS if h < len(hourly_rh) + 4]
        afternoon_offsets = [h - 4 for h in afternoon_indices if 0 <= h - 4 < len(hourly_rh)]
        full_offsets = [h - 4 for h in full_indices if 0 <= h - 4 < len(hourly_rh)]
        if not full_offsets:
            fwis.record_skipped_run("no leads available in the day-1 aggregation window")
            return False

        vpd_by_hour = [vapor_pressure_deficit_kpa(np.asarray(hourly_temp_c[i]), np.asarray(hourly_rh[i]))
                       for i in full_offsets]

        cell_to_fips = cells["cell_to_fips"]
        county_list = sorted({fips for fips in cell_to_fips.values()})
        weather_rows = {}
        for fips in county_list:
            county_cell_map = {k: v for k, v in cell_to_fips.items() if v == fips}

            def _cell_values(grid):
                return np.asarray([np.asarray(grid)[int(k.split(",")[0]), int(k.split(",")[1])]
                                   for k in county_cell_map], dtype="float64")

            rh_afternoon = [_reduce(hourly_rh[i], county_cell_map, np.nanmin) for i in afternoon_offsets] or \
                           [_reduce(hourly_rh[i], county_cell_map, np.nanmin) for i in full_offsets]
            wind_all_cells = np.concatenate([_cell_values(hourly_ws_kts[i]) for i in full_offsets])
            vpd_full = [float(np.nanmax(_cell_values(grid))) for grid in vpd_by_hour]
            precip_full = [_reduce(hourly_precip_mm[i], county_cell_map, np.nanmean) for i in full_offsets]

            weather_rows[fips] = {
                "rh_min_afternoon": float(np.nanmin(rh_afternoon)),
                "wind_kts_max": float(np.nanmax(wind_all_cells)),
                "vpd_kpa_max": float(np.nanmax(vpd_full)),
                "precip_24h_mm": float(np.nansum(precip_full)),
            }

        # Same four reductions as the per-county loop above, applied
        # elementwise over the full grid instead of each county's cell
        # subset - lets the shadow render the real pixel map (see
        # fire_weather_index_shadow.py::_render_png) instead of just
        # scoring per county. Only built when lat/lon were actually passed,
        # since they're required to render anything with these grids.
        weather_grids = None
        if lat is not None and lon is not None:
            rh_offsets = afternoon_offsets or full_offsets
            weather_grids = {
                "rh_min_afternoon": np.nanmin(
                    np.stack([np.asarray(hourly_rh[i], dtype="float64") for i in rh_offsets]), axis=0),
                "wind_kts_max": np.nanmax(
                    np.stack([np.asarray(hourly_ws_kts[i], dtype="float64") for i in full_offsets]), axis=0),
                "vpd_kpa_max": np.nanmax(
                    np.stack([np.asarray(grid, dtype="float64") for grid in vpd_by_hour]), axis=0),
                "precip_24h_mm": np.nansum(
                    np.stack([np.asarray(hourly_precip_mm[i], dtype="float64") for i in full_offsets]), axis=0),
            }

        return fwis.score_for_forecast(
            run_id=run_id,
            valid_local_date=valid_local_date,
            county_fips=county_list,
            weather_rows=weather_rows,
            weather_grids=weather_grids,
            lat=lat,
            lon=lon,
        )
    except Exception as exc:
        logger.warning("fire_weather_index shadow hook failed (non-fatal): %s", exc)
        try:
            fwis.record_skipped_run(str(exc))
        except Exception:
            pass
        return False
