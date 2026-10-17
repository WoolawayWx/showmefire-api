"""Score the forecast (and the RAWS stations) against RTMA analyses.

Station verification only sees where RAWS stations exist. This adds two
RTMA-based views on top of it:

* station-point: cached hourly RTMA sampled at each station, compared with the
  forecast and with the observation at the same station-hours.
* gridded: the statewide peak fire-danger class raster against the RTMA peak
  raster, which covers the whole domain rather than just the stations.

RTMA has no fuel-moisture field. Fuel moisture and fire danger here use the
uncalibrated RH estimate from ``rtma_peak`` (not tuned to RAWS, so the
comparison stays independent of the stations) and are marked ``derived``.
Temperature, RH and wind are true RTMA analysis fields.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from services.rtma_capture import cache_path_for_hour
from services.rtma_peak import _hours_for_local_date, _lon_lat_meshes, _squeeze2d

logger = logging.getLogger(__name__)

MS_TO_KTS = 1.9438444924406
ESTIMATED_FM_INTERCEPT = 3.0
ESTIMATED_FM_RH_SLOPE = 0.25

CONTINUOUS_VARIABLES = {
    'Temperature (C)': ('temp', 'rtma_temp'),
    'Relative Humidity (%)': ('rh', 'rtma_rh'),
    'Wind Speed (m/s)': ('wind', 'rtma_wind'),
    'Fuel Moisture (%)': ('fm', 'rtma_fm'),
}
DERIVED_VARIABLES = ('Fuel Moisture (%)', 'Fire Danger Index')


def _finite(value) -> bool:
    return isinstance(value, (int, float, np.floating)) and math.isfinite(value)


def sample_rtma_at_points(
    target_date,
    station_coords: dict,
    fire_danger_fn,
    cache_dir: Path | None = None,
) -> pd.DataFrame:
    """One row per station and cached RTMA hour in the verification window.

    ``station_coords`` maps stid -> (lat, lon). ``fire_danger_fn(fm, rh, wind_kts)``
    is the canonical classifier. Hours missing from the 7-day cache are skipped.
    """
    from scipy.spatial import cKDTree

    stids = [s for s, (lat, lon) in station_coords.items() if _finite(lat) and _finite(lon)]
    if not stids:
        return pd.DataFrame()
    points = np.array([[station_coords[s][0], station_coords[s][1]] for s in stids], dtype=float)

    rows = []
    tree = None
    grid_shape = None
    for hour in _hours_for_local_date(target_date):
        path = cache_path_for_hour(hour, cache_dir=cache_dir)
        if not path.is_file():
            continue
        try:
            with xr.open_dataset(path) as ds:
                lon, lat = _lon_lat_meshes(ds['longitude'].values, ds['latitude'].values)
                if tree is None or grid_shape != lon.shape:
                    tree = cKDTree(np.column_stack((lat.ravel(), lon.ravel())))
                    grid_shape = lon.shape
                distance, idx = tree.query(points)
                t2m = _squeeze2d(ds['t2m'].values).ravel()[idx] - 273.15
                rh = _squeeze2d(ds['r2'].values).ravel()[idx]
                u10 = _squeeze2d(ds['u10'].values).ravel()[idx]
                v10 = _squeeze2d(ds['v10'].values).ravel()[idx]
        except Exception:
            logger.exception('Unable to sample RTMA hour %s', path.name)
            continue
        wind = np.hypot(u10, v10)
        fm = ESTIMATED_FM_INTERCEPT + ESTIMATED_FM_RH_SLOPE * rh
        timestamp = pd.Timestamp(hour).tz_convert('UTC')
        for i, stid in enumerate(stids):
            fd = None
            if np.isfinite(fm[i]) and np.isfinite(rh[i]) and np.isfinite(wind[i]):
                fd = fire_danger_fn(float(fm[i]), float(rh[i]), float(wind[i] * MS_TO_KTS))
            rows.append({
                'stid': stids[i],
                'timestamp': timestamp,
                'rtma_temp': float(t2m[i]),
                'rtma_rh': float(rh[i]),
                'rtma_wind': float(wind[i]),
                'rtma_fm': float(fm[i]),
                'rtma_fire_danger': fd,
            })
    return pd.DataFrame(rows)


def _continuous(frame: pd.DataFrame, pred_col: str, ref_col: str) -> dict:
    valid = frame[[pred_col, ref_col]].apply(pd.to_numeric, errors='coerce').dropna()
    if valid.empty:
        return {'mae': None, 'rmse': None, 'bias': None, 'count': 0, 'correlation': None}
    diff = valid[pred_col] - valid[ref_col]
    corr = None
    if len(valid) > 1:
        c = np.corrcoef(valid[ref_col], valid[pred_col])[0, 1]
        corr = round(float(c), 4) if np.isfinite(c) else None
    return {
        'mae': round(float(diff.abs().mean()), 4),
        'rmse': round(float(np.sqrt((diff ** 2).mean())), 4),
        'bias': round(float(diff.mean()), 4),
        'count': int(len(valid)),
        'correlation': corr,
    }


def _categorical(frame: pd.DataFrame, pred_col: str, ref_col: str, categories: int = 5) -> dict:
    valid = frame[[pred_col, ref_col]].apply(pd.to_numeric, errors='coerce').dropna()
    if valid.empty:
        return {'mae': None, 'bias': None, 'count': 0, 'exact_match_rate': None,
                'within_one_category_rate': None, 'metric_type': 'categorical'}
    ref = valid[ref_col].round().astype(int).clip(0, categories - 1)
    pred = valid[pred_col].round().astype(int).clip(0, categories - 1)
    diff = pred - ref
    return {
        'mae': round(float(diff.abs().mean()), 4),
        'bias': round(float(diff.mean()), 4),
        'count': int(len(valid)),
        'exact_match_rate': round(float((diff == 0).mean()), 4),
        'within_one_category_rate': round(float((diff.abs() <= 1).mean()), 4),
        'metric_type': 'categorical',
    }


def station_point_metrics(merged: pd.DataFrame, rtma: pd.DataFrame) -> dict | None:
    """Forecast-vs-RTMA and station-vs-RTMA metrics over matching station-hours.

    ``merged`` is the existing station merge (pred_*/obs_* columns). Bias is
    always first-named minus RTMA, so a positive forecast bias means the
    forecast is higher than RTMA.
    """
    if rtma is None or rtma.empty:
        return None
    joined = pd.merge(merged, rtma, on=['stid', 'timestamp'], how='inner')
    if joined.empty:
        return None

    forecast_vs_rtma, station_vs_rtma = {}, {}
    for label, (key, ref) in CONTINUOUS_VARIABLES.items():
        forecast_vs_rtma[label] = _continuous(joined, f'pred_{key}', ref)
        station_vs_rtma[label] = _continuous(joined, f'obs_{key}', ref)
    forecast_vs_rtma['Fire Danger Index'] = _categorical(joined, 'pred_fire_danger', 'rtma_fire_danger')
    station_vs_rtma['Fire Danger Index'] = _categorical(joined, 'obs_fire_danger', 'rtma_fire_danger')

    return {
        'reference': 'RTMA analysis sampled at station locations (nearest grid cell)',
        'derived_variables': list(DERIVED_VARIABLES),
        'record_count': int(len(joined)),
        'stations_count': int(joined['stid'].nunique()),
        'hours_count': int(joined['timestamp'].nunique()),
        'forecast_vs_rtma': forecast_vs_rtma,
        'station_vs_rtma': station_vs_rtma,
    }


def gridded_peak_metrics(target_date: str, gis_dir: Path | None = None) -> dict | None:
    """Statewide peak-class forecast vs RTMA peak, as agreement/bias over all pixels."""
    from core.config import GIS_DIR
    from scripts.compare_09z_rtma import build_comparison

    gis_dir = Path(gis_dir or GIS_DIR)
    forecast = gis_dir / 'forecast_peak' / 'archive' / f'{target_date}.tif'
    rtma = gis_dir / 'rtma_peak' / 'archive' / f'{target_date}.tif'
    if not forecast.is_file() or not rtma.is_file():
        return None
    try:
        result = build_comparison(forecast, rtma, target_date)
    except Exception:
        logger.exception('Gridded RTMA comparison failed for %s', target_date)
        return None
    agreement = result['fire_danger_category_agreement']
    return {
        'reference': 'RTMA peak fire-danger raster (statewide, all pixels)',
        'valid_pixel_count': result['valid_pixel_count'],
        'fire_danger': {
            'exact_match_rate': round(agreement['agreement_rate'], 4),
            'within_one_category_rate': round(agreement['within_one_rate'], 4),
            'mean_abs_diff': round(agreement['mean_abs_diff'], 4),
            'bias': round(agreement['mean_bias'], 4),
            'rmse': round(agreement['rmse'], 4),
            'forecast_higher_rate': round(agreement['forecast_higher_rate'], 4),
            'forecast_lower_rate': round(agreement['forecast_lower_rate'], 4),
        },
        'confusion_matrix': {**result['confusion_matrix'], 'columns': 'forecast'},
        'rtma_fuel_moisture': result['rtma_fuel_moisture'],
    }


def build_rtma_metrics(merged, station_coords, target_date, fire_danger_fn,
                       cache_dir=None, gis_dir=None, include_gridded=True) -> dict:
    """Assemble the ``rtma_metrics`` report block. Never raises."""
    block = {'station_point': None, 'gridded': None}
    try:
        if isinstance(target_date, str):
            day = datetime.strptime(target_date, '%Y-%m-%d').date()
        else:
            day = target_date
        sampled = sample_rtma_at_points(day, station_coords, fire_danger_fn, cache_dir=cache_dir)
        block['station_point'] = station_point_metrics(merged, sampled)
    except Exception:
        logger.exception('Station-point RTMA verification failed')
    if include_gridded:
        try:
            block['gridded'] = gridded_peak_metrics(str(target_date), gis_dir=gis_dir)
        except Exception:
            logger.exception('Gridded RTMA verification failed')
    block['available'] = bool(block['station_point'] or block['gridded'])
    return block
