"""Sensor-anomaly QC for archived observations used by verification.

Two independent checks run against the hourly-observation DataFrame built by
forecast/endOfDayReport.py::get_observation_dataframe before it is merged with
forecasts or fed into calculate_fire_danger:

- out-of-range: a single reading outside a generous physical envelope (data
  entry / transmission errors).
- stuck sensor: a station reports the exact same value for a given variable
  across every hour in the window (dead/disconnected sensor, not real calm).

Both null the offending value in place rather than dropping the row, so the
station's other variables for that hour are unaffected, and record what was
excluded so nothing disappears from a report silently.
"""
from typing import List, Dict, Any, Tuple

import pandas as pd

STUCK_MIN_READINGS = 4
PLAUSIBLE_RANGES = {
    'obs_temp': (-40.0, 55.0),   # deg C
    'obs_rh': (0.0, 100.0),      # %
    'obs_wind': (0.0, 60.0),     # m/s
    'obs_fm': (0.0, 60.0),       # %
}


def flag_out_of_range(obs_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Null individual values outside PLAUSIBLE_RANGES. Returns (df, exclusions)."""
    df = obs_df.copy()
    exclusions: List[Dict[str, Any]] = []
    if df.empty:
        return df, exclusions

    for col, (low, high) in PLAUSIBLE_RANGES.items():
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors='coerce')
        bad_mask = values.notna() & ((values < low) | (values > high))
        for stid, value in zip(df.loc[bad_mask, 'stid'], values[bad_mask]):
            exclusions.append({
                'stid': stid,
                'variable': col,
                'reason': 'out_of_range',
                'value': float(value),
                'n_readings': 1,
            })
        df.loc[bad_mask, col] = None

    return df, exclusions


def flag_stuck_sensors(obs_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Null a station's entire column for the window if every reading is identical.

    Requires at least STUCK_MIN_READINGS non-null readings for that station/column
    before judging it stuck, so short windows don't get flagged on coincidence.
    """
    df = obs_df.copy()
    exclusions: List[Dict[str, Any]] = []
    if df.empty:
        return df, exclusions

    for col in PLAUSIBLE_RANGES:
        if col not in df.columns:
            continue
        for stid, group in df.groupby('stid'):
            values = pd.to_numeric(group[col], errors='coerce').dropna()
            if len(values) < STUCK_MIN_READINGS:
                continue
            if values.nunique() == 1:
                stuck_value = float(values.iloc[0])
                exclusions.append({
                    'stid': stid,
                    'variable': col,
                    'reason': 'stuck_value',
                    'value': stuck_value,
                    'n_readings': int(len(values)),
                })
                df.loc[df['stid'] == stid, col] = None

    return df, exclusions


def apply_qc(obs_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """Runs both checks and merges their exclusion lists. Returns (cleaned_df, exclusions)."""
    df, range_exclusions = flag_out_of_range(obs_df)
    df, stuck_exclusions = flag_stuck_sensors(df)
    return df, range_exclusions + stuck_exclusions
