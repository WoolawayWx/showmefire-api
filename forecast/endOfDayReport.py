import json
import os
import sys
import argparse
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
import logging

# Add project root to path (api/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from core.fire_danger import calculate_fire_danger as canonical_fire_danger
from core.fire_danger import CATEGORY_LABELS, RULE_SPEC
from core.ignored_stations import get_ignored_stations
from core import observation_qc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

ARCHIVE_DIR = Path(BASE_DIR) / "archive"
FORECAST_DIR = ARCHIVE_DIR / "forecasts"
RAW_DATA_DIR = ARCHIVE_DIR / "raw_data"
REPORTS_DIR = Path(BASE_DIR) / "reports"
REPORTS_DIR.mkdir(exist_ok=True)
PLOTS_DIR = REPORTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# Radius used by the neighborhood/regional verification metric: a prediction
# counts as regionally corroborated if a station within this many miles
# observed a category within one step of it, even if the exact station
# missed. 45mi was picked by checking actual inter-station spacing across the
# live 18-station network (median/mean nearest-neighbor distance ~37-38mi):
# at 30mi, 11/18 stations have zero neighbors (mean 0.44 neighbors); at 45mi,
# only 3/18 are isolated (mean 1.56 neighbors) - override via env var if the
# station network changes materially.
NEIGHBORHOOD_RADIUS_MILES = float(os.getenv("VERIFICATION_NEIGHBORHOOD_RADIUS_MILES", "45"))


def _to_float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def export_verification_history_csv(history, verification_csv_file):
    """Generate compatibility CSV from canonical JSON history."""
    rows = []

    for entry in history:
        date_value = str(entry.get('date', '')).strip()
        if not date_value:
            continue

        # Keep legacy-compatible compact date format in CSV.
        date_compact = date_value.replace('-', '')
        metrics = entry.get('metrics', {})

        temp = metrics.get('Temperature (C)', {})
        rh = metrics.get('Relative Humidity (%)', {})
        wind = metrics.get('Wind Speed (m/s)', {})
        fm = metrics.get('Fuel Moisture (%)', {})

        rows.append({
            'date': date_compact,
            'generated_at': entry.get('generated_at') or datetime.utcnow().isoformat() + 'Z',
            'num_forecasts': 1,
            'num_comparisons': int(entry.get('record_count', 0) or 0),
            'temp_mae_c': _to_float_or_none(temp.get('mae')),
            'temp_bias_c': _to_float_or_none(temp.get('bias')),
            'rh_mae_pct': _to_float_or_none(rh.get('mae')),
            'rh_bias_pct': _to_float_or_none(rh.get('bias')),
            'wind_mae_ms': _to_float_or_none(wind.get('mae')),
            'wind_bias_ms': _to_float_or_none(wind.get('bias')),
            'fm_mae_pct': _to_float_or_none(fm.get('mae')),
            'fm_bias_pct': _to_float_or_none(fm.get('bias')),
            'fm_count': int(fm.get('count', 0) or 0),
        })

    verification_df = pd.DataFrame(rows)
    if verification_df.empty:
        logger.warning("No history rows available to export verification CSV.")
        return

    verification_df = verification_df.sort_values('date')
    verification_df.to_csv(verification_csv_file, index=False)
    logger.info(f"Updated verification history CSV: {verification_csv_file}")

def load_latest_file(directory: Path, prefix: str):
    """Finds the most recent file in a directory matching a prefix."""
    if not directory.exists():
        logger.error(f"Directory not found: {directory}")
        return None
    
    files = list(directory.glob(f"{prefix}*.json"))
    if not files:
        logger.warning(f"No files found with prefix '{prefix}' in {directory}")
        return None
    
    # Sort files by name (which includes date)
    latest_file = sorted(files)[-1]
    logger.info(f"Loaded latest file: {latest_file.name}")
    
    with open(latest_file, 'r') as f:
        return json.load(f), latest_file

def extract_date_token(filename: str):
    """Extract YYYYMMDD token from filename if present."""
    m = re.search(r"(20\d{6})", filename)
    return m.group(1) if m else None


def find_matching_files(forecast_dir, raw_dir, forecast_glob_pattern="station_forecasts_*.json"):
    """
    Attempts to find a forecast file and a raw data file that share the same date.
    Returns (forecast_data, forecast_file, raw_data, raw_file)
    """
    if not forecast_dir.exists() or not raw_dir.exists():
        return None, None, None, None

    fc_files = sorted(list(forecast_dir.glob(forecast_glob_pattern)), reverse=True)
    raw_files = sorted(list(raw_dir.glob("raw_data_*.json")), reverse=True)
    
    if not fc_files or not raw_files:
        return None, None, None, None

    # Try to find a match
    for fc_file in fc_files:
        try:
            fc_date_str = extract_date_token(fc_file.name)
            if fc_date_str:
                
                # Look for raw file containing this date in its name
                for raw_file in raw_files:
                    if fc_date_str in raw_file.name:
                        logger.info(f"Found matching file pair: {fc_file.name} + {raw_file.name}")
                        with open(fc_file, 'r') as f:
                            fc_data = json.load(f)
                        with open(raw_file, 'r') as f:
                            raw_data = json.load(f)
                        return fc_data, fc_file, raw_data, raw_file
        except Exception:
            continue
            
    # Fallback to latest
    logger.warning("No date-matched files found. Falling back to latest files.")
    
    # Use load_latest logic manually here or just take index 0
    fc_file = fc_files[0]
    raw_file = raw_files[0]
    
    logger.info(f"Loading latest: {fc_file.name} + {raw_file.name}")
    
    with open(fc_file, 'r') as f:
        fc_data = json.load(f)
    with open(raw_file, 'r') as f:
        raw_data = json.load(f)
        
    return fc_data, fc_file, raw_data, raw_file

def find_files_for_date(forecast_dir, raw_dir, target_date_str, forecast_glob_pattern="station_forecasts_*.json"):
    """Like find_matching_files, but targets a specific YYYYMMDD date instead
    of always taking the latest files - used for rerunning verification for a
    past date. Kept separate from find_matching_files so the no-`--date` cron
    path's "latest by sort" behavior is untouched and easy to verify.
    Returns (forecast_data, forecast_file, raw_data, raw_file), any of which
    may be None if no match is found for that date.
    """
    if not forecast_dir.exists() or not raw_dir.exists():
        return None, None, None, None

    fc_files = [f for f in forecast_dir.glob(forecast_glob_pattern) if extract_date_token(f.name) == target_date_str]
    raw_files = [f for f in raw_dir.glob("raw_data_*.json") if extract_date_token(f.name) == target_date_str]

    if not fc_files or not raw_files:
        return None, None, None, None

    fc_file = sorted(fc_files)[-1]
    raw_file = sorted(raw_files)[-1]
    logger.info(f"Found files for {target_date_str}: {fc_file.name} + {raw_file.name}")

    with open(fc_file, 'r') as f:
        fc_data = json.load(f)
    with open(raw_file, 'r') as f:
        raw_data = json.load(f)

    return fc_data, fc_file, raw_data, raw_file

def fahrenheit_to_celsius(f):
    return (f - 32) * 5.0/9.0

def mph_to_ms(mph):
    return mph * 0.44704

def knots_to_ms(knots):
    return knots * 0.514444

def ms_to_kts(ms):
    return ms * 1.94384

def calculate_fire_danger(fm, rh, wind_kts):
    """
    Fire Danger Criteria based on ShowMeFire.org:
    Low: FM >= 15%
    Moderate: FM < 15% WITH (RH < 45% OR Wind >= 10 kts)
    Elevated: FM < 9% WITH the canonical dry/breezy combinations
    Critical: FM < 9% WITH (RH < 25% AND Wind >= 15 kts)
    Extreme: FM < 7% WITH (RH < 20% AND Wind >= 25 kts)
    """
    return canonical_fire_danger(fm, rh, wind_kts, missing_category=None)

# Reuses the same monotonic wind-speed rungs that already drive fire-danger
# classification (fire_danger_rules.json), skipping the compound
# elevated_very_dry_wind threshold (an OR-condition with very-low RH, not a
# simple ladder rung) - so a wind category score stays diagnostically tied
# to the bands that actually matter for danger classification, rather than
# an arbitrary new set of breakpoints.
_WIND_THRESHOLDS_KTS = RULE_SPEC["thresholds"]
_WIND_CATEGORY_LADDER = [
    _WIND_THRESHOLDS_KTS["moderate_wind"],
    _WIND_THRESHOLDS_KTS["elevated_wind"],
    _WIND_THRESHOLDS_KTS["critical_wind"],
    _WIND_THRESHOLDS_KTS["extreme_wind"],
]

def wind_speed_ms_to_category(wind_ms):
    """Bucket a wind speed (m/s) into the same Low..Extreme ladder used for
    fire-danger categories, using knots thresholds from fire_danger_rules.json."""
    if wind_ms is None or pd.isna(wind_ms):
        return None
    wind_kts = ms_to_kts(wind_ms)
    category = 0
    for threshold in _WIND_CATEGORY_LADDER:
        if wind_kts >= threshold:
            category += 1
    return category

def haversine_miles(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lon points, in miles."""
    if any(v is None or pd.isna(v) for v in (lat1, lon1, lat2, lon2)):
        return None
    earth_radius_miles = 3958.8
    lat1_r, lon1_r, lat2_r, lon2_r = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
    return 2 * earth_radius_miles * np.arcsin(np.sqrt(a))

def build_station_neighbors(station_coords, radius_miles=NEIGHBORHOOD_RADIUS_MILES):
    """For each station, the list of OTHER stations within radius_miles (haversine)."""
    neighbors = {}
    stids = list(station_coords.keys())
    for stid_a in stids:
        lat_a, lon_a = station_coords[stid_a]
        nearby = []
        for stid_b in stids:
            if stid_b == stid_a:
                continue
            lat_b, lon_b = station_coords[stid_b]
            distance = haversine_miles(lat_a, lon_a, lat_b, lon_b)
            if distance is not None and distance <= radius_miles:
                nearby.append(stid_b)
        neighbors[stid_a] = nearby
    return neighbors

def parse_date(date_str):
    """
    Parses date string into a UTC-aware datetime (ISO8601 with Z suffix).
    Always returns a pandas.Timestamp in UTC.
    """
    ts = pd.Timestamp(date_str)
    if ts.tzinfo is None:
        # Assume naive times are UTC
        ts = ts.tz_localize('UTC')
    else:
        ts = ts.tz_convert('UTC')
    return ts

def get_forecast_dataframe(forecast_data, target_date_start, target_date_end):
    """
    Extracts forecast data into a DataFrame.
    Filters for the relevant target validation window.
    """
    records = []
    
    for stid, data in forecast_data.get("stations", {}).items():
        lat = data.get('lat')
        lon = data.get('lon')
        for fc in data.get("forecasts", []):
            fc_time = parse_date(fc['time'])
            # Filter to relevant window
            if target_date_start <= fc_time <= target_date_end:
                temp_c = fc.get('temp_c')
                wind_ms = fc.get('wind_speed_ms')
                records.append({
                    'stid': stid,
                    'timestamp': fc_time.round('h'),
                    'pred_temp': temp_c,
                    'pred_rh': fc.get('rh'),
                    'pred_wind': wind_ms,
                    'pred_fm': fc.get('fuel_moisture'),
                    'pred_fire_danger': fc.get('fire_danger')
                })
    df = pd.DataFrame(records)
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df['timestamp'] = df['timestamp'].dt.round('h')
    return df

def get_observation_dataframe(raw_data, target_date_start, target_date_end):
    """
    Extracts observation data into a DataFrame.
    Converts units to match forecast (C, m/s).
    Rounds timestamps to nearest hour for comparison.
    """
    records = []

    wind_units = raw_data.get('UNITS', {}).get('wind_speed')
    if wind_units and wind_units.lower() not in ('knots', 'mph', 'm/s', 'meters/second'):
        logger.warning(
            f"Unrecognized wind_speed units '{wind_units}' in raw data; "
            "verify the observed-wind conversion is still correct."
        )

    # Handle both list and dict structures for stations
    stations = raw_data.get('STATION', [])
    if isinstance(stations, dict):
        stations = [stations]

    ignored = get_ignored_stations()
    station_coords = {}

    for station in stations:
        stid = station.get('STID')
        if stid in ignored:
            continue
        try:
            lat = float(station.get('LATITUDE'))
            lon = float(station.get('LONGITUDE'))
            station_coords[stid] = (lat, lon)
        except (TypeError, ValueError):
            pass
        obs = station.get('OBSERVATIONS', {})
        times = obs.get('date_time', [])
        temps = obs.get('air_temp_set_1', []) or obs.get('air_temp', [])
        rhs = obs.get('relative_humidity_set_1', []) or obs.get('relative_humidity', [])
        winds = obs.get('wind_speed_set_1', []) or obs.get('wind_speed', [])
        fms = obs.get('fuel_moisture_set_1', []) or obs.get('fuel_moisture', [])
        min_len = len(times)
        for i in range(min_len):
            try:
                obs_time_str = times[i]
                obs_time = parse_date(obs_time_str)
                base_time = obs_time.round('h')
                if target_date_start <= base_time <= target_date_end:
                    temp_val = temps[i] if i < len(temps) and temps[i] is not None else None
                    if temp_val is not None:
                        temp_val = fahrenheit_to_celsius(temp_val)
                    wind_val = winds[i] if i < len(winds) and winds[i] is not None else None
                    if wind_val is not None:
                        wind_val = knots_to_ms(wind_val)
                    rh_val = rhs[i] if i < len(rhs) and rhs[i] is not None else None
                    fm_val = fms[i] if i < len(fms) and fms[i] is not None else None
                    records.append({
                        'stid': stid,
                        'timestamp': base_time,
                        'obs_temp': temp_val,
                        'obs_rh': rh_val,
                        'obs_wind': wind_val,
                        'obs_fm': fm_val
                    })
            except (ValueError, IndexError, TypeError):
                continue
    df = pd.DataFrame(records)
    qc_exclusions = []
    if not df.empty:
        df = df.groupby(['stid', 'timestamp']).mean(numeric_only=True).reset_index()
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df['timestamp'] = df['timestamp'].dt.round('h')
        df, qc_exclusions = observation_qc.apply_qc(df)
        df['obs_fire_danger'] = df.apply(
            lambda row: calculate_fire_danger(
                row['obs_fm'],
                row['obs_rh'],
                ms_to_kts(row['obs_wind']) if pd.notna(row['obs_wind']) else None
            ), axis=1
        )
    return df, qc_exclusions, station_coords
# --- New function to merge and align forecast and observation data ---
def merge_forecast_and_obs(forecast_df, obs_df):
    """
    Merge forecast and observation dataframes on stid and timestamp (hourly).
    Only keep rows where both have data for the same station and time.
    """
    if forecast_df.empty or obs_df.empty:
        return pd.DataFrame()
    merged = pd.merge(
        forecast_df,
        obs_df,
        on=['stid', 'timestamp'],
        how='inner',  # Only keep matching times and stations
        suffixes=('_fc', '_obs')
    )
    return merged

def calculate_metrics(merged_df, variable_map):
    metrics = {}
    
    for metric_name, (pred_col, obs_col) in variable_map.items():
        # filter valid rows
        valid = merged_df.dropna(subset=[pred_col, obs_col]).copy()
        # Defensive: ensure both columns are numeric and units match
        if not valid.empty:
            valid.loc[:, pred_col] = pd.to_numeric(valid[pred_col], errors='coerce')
            valid.loc[:, obs_col] = pd.to_numeric(valid[obs_col], errors='coerce')
        if valid.empty:
            metrics[metric_name] = {'mae': None, 'rmse': None, 'bias': None, 'count': 0}
            print(f"{metric_name}: No valid data for comparison")
            continue
        y_true = valid[obs_col]
        y_pred = valid[pred_col]
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred)**2))
        bias = np.mean(y_pred - y_true)
        # Calculate R^2 for plots if needed, or simple correlation
        corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0
        metrics[metric_name] = {
            'mae': round(mae, 4),
            'rmse': round(rmse, 4),
            'bias': round(bias, 4),
            'count': len(valid),
            'correlation': round(corr, 4)
        }
        print(f"{metric_name}: MAE={round(mae, 4)}, RMSE={round(rmse, 4)}, Bias={round(bias, 4)}, Count={len(valid)}, Correlation={round(corr, 4)}")
    return metrics

def calculate_categorical_metrics(merged_df, pred_col, obs_col, labels=CATEGORY_LABELS):
    """Metrics appropriate for an ordinal category code (0=Low..N=Extreme), not a
    continuous physical quantity. Pearson correlation and RMSE are omitted: they read
    as physical-magnitude stats but are not meaningful on a handful of ordinal classes.
    The `mae`/`mean_absolute_category_error` and `bias` keys measure category-step
    distance, which IS a legitimate ordinal stat, and are kept (mae duplicated under
    the legacy key for existing consumers).
    """
    valid = merged_df.dropna(subset=[pred_col, obs_col]).copy()
    if valid.empty:
        print(f"{pred_col}/{obs_col}: No valid data for comparison")
        return {'mae': None, 'bias': None, 'count': 0, 'exact_match_rate': None,
                'within_one_category_rate': None, 'metric_type': 'categorical'}

    y_true = pd.to_numeric(valid[obs_col], errors='coerce').round().astype(int).clip(0, len(labels) - 1)
    y_pred = pd.to_numeric(valid[pred_col], errors='coerce').round().astype(int).clip(0, len(labels) - 1)
    diff = y_pred - y_true

    mean_abs_category_error = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    exact_match_rate = float(np.mean(diff == 0))
    within_one_category_rate = float(np.mean(np.abs(diff) <= 1))

    metrics = {
        'mae': round(mean_abs_category_error, 4),
        'mean_absolute_category_error': round(mean_abs_category_error, 4),
        'bias': round(bias, 4),
        'count': len(valid),
        'exact_match_rate': round(exact_match_rate, 4),
        'within_one_category_rate': round(within_one_category_rate, 4),
        'metric_type': 'categorical',
    }
    print(
        f"Fire Danger Index: ExactMatch={metrics['exact_match_rate']}, "
        f"WithinOne={metrics['within_one_category_rate']}, "
        f"MeanCategoryError={metrics['mae']}, Bias={metrics['bias']}, Count={metrics['count']}"
    )
    return metrics

def calculate_confusion_matrix(merged_df, pred_col='pred_fire_danger', obs_col='obs_fire_danger'):
    """Category confusion matrix: observed (rows) vs predicted (cols), Low..Extreme."""
    valid = merged_df.dropna(subset=[pred_col, obs_col]).copy()
    if valid.empty:
        return None
    valid[pred_col] = valid[pred_col].round().astype(int).clip(0, len(CATEGORY_LABELS) - 1)
    valid[obs_col] = valid[obs_col].round().astype(int).clip(0, len(CATEGORY_LABELS) - 1)
    matrix = pd.crosstab(valid[obs_col], valid[pred_col]).reindex(
        index=range(len(CATEGORY_LABELS)), columns=range(len(CATEGORY_LABELS)), fill_value=0
    )
    return {'labels': list(CATEGORY_LABELS), 'matrix': matrix.values.tolist()}

def calculate_neighborhood_metrics(merged_df, pred_col, obs_col, neighbors, labels=CATEGORY_LABELS,
                                    radius_miles=NEIGHBORHOOD_RADIUS_MILES):
    """Regional accuracy: a prediction is a 'hit' if the station itself OR any
    station within radius_miles observed a category within one step of the
    predicted category, at the same hour. Reported alongside (not instead of)
    the strict per-station exact/within-one rates for direct comparison.
    """
    valid = merged_df.dropna(subset=[pred_col, obs_col]).copy()
    if valid.empty:
        return {'metric_type': 'neighborhood', 'radius_miles': radius_miles, 'neighborhood_hit_rate': None,
                'strict_exact_match_rate': None, 'strict_within_one_rate': None,
                'mean_neighbor_count': None, 'count': 0}

    valid[pred_col] = pd.to_numeric(valid[pred_col], errors='coerce').round().astype(int).clip(0, len(labels) - 1)
    valid[obs_col] = pd.to_numeric(valid[obs_col], errors='coerce').round().astype(int).clip(0, len(labels) - 1)

    # Index observed categories by (stid, timestamp) once, for fast neighbor lookups.
    obs_by_station_time = valid.set_index(['stid', 'timestamp'])[obs_col].to_dict()

    def is_hit(row):
        candidates = [row['stid']] + list(neighbors.get(row['stid'], []))
        for candidate_stid in candidates:
            observed = obs_by_station_time.get((candidate_stid, row['timestamp']))
            if observed is not None and abs(observed - row[pred_col]) <= 1:
                return True
        return False

    hits = valid.apply(is_hit, axis=1)
    diff = valid[pred_col] - valid[obs_col]
    neighbor_counts = valid['stid'].map(lambda stid: len(neighbors.get(stid, [])))

    metrics = {
        'metric_type': 'neighborhood',
        'radius_miles': radius_miles,
        'neighborhood_hit_rate': round(float(hits.mean()), 4),
        'strict_exact_match_rate': round(float(np.mean(diff == 0)), 4),
        'strict_within_one_rate': round(float(np.mean(np.abs(diff) <= 1)), 4),
        'mean_neighbor_count': round(float(neighbor_counts.mean()), 2),
        'count': len(valid),
    }
    print(
        f"{pred_col}/{obs_col} neighborhood ({radius_miles}mi): HitRate={metrics['neighborhood_hit_rate']}, "
        f"StrictExact={metrics['strict_exact_match_rate']}, StrictWithinOne={metrics['strict_within_one_rate']}, "
        f"AvgNeighbors={metrics['mean_neighbor_count']}"
    )
    return metrics

def generate_plots(merged_df, variable_map, report_date, report_suffix=""):
    """
    Generates scatter plots for Predicted vs Observed values.
    Saves plots to REPORTS_DIR/plots/{date}/
    """
    # Create daily directory for plots; suffix gets an isolated subfolder.
    daily_plot_dir = REPORTS_DIR / report_date / "plots"
    if report_suffix:
        daily_plot_dir = daily_plot_dir / report_suffix
    daily_plot_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    
    saved_plots = []
    
    for metric_name, (pred_col, obs_col) in variable_map.items():
        valid = merged_df.dropna(subset=[pred_col, obs_col]).copy()
        # Defensive: ensure both columns are numeric and units match
        if not valid.empty:
            valid.loc[:, pred_col] = pd.to_numeric(valid[pred_col], errors='coerce')
            valid.loc[:, obs_col] = pd.to_numeric(valid[obs_col], errors='coerce')
        if valid.empty:
            continue
        plt.figure(figsize=(10, 6))
        # Calculate min/max for dynamic limits that keep aspect ratio roughly 1:1 if possible
        val_min = min(valid[pred_col].min(), valid[obs_col].min())
        val_max = max(valid[pred_col].max(), valid[obs_col].max())
        padding = (val_max - val_min) * 0.05
        # Scatter plot
        sns.scatterplot(x=valid[obs_col], y=valid[pred_col], alpha=0.5)
        # 1:1 Reference Line
        plt.plot([val_min-padding, val_max+padding], [val_min-padding, val_max+padding], 
                 ls='--', c='.3', label='Perfect Forecast')
        plt.title(f"{metric_name}: Forecast vs Observed ({report_date})")
        plt.xlabel("Observed")
        plt.ylabel("Forecast")
        plt.legend()
        clean_name = metric_name.split('(')[0].strip().lower().replace(' ', '_')
        filename = f"{clean_name}_scatter.png"
        filepath = daily_plot_dir / filename
        plt.savefig(filepath)
        plt.close()
        logger.info(f"Generated plot: {filepath}")
        saved_plots.append(str(filepath))
    return saved_plots

def update_history(report, history_file):
    """
    Updates the historical validation JSON file.
    """
    history = []
    if history_file.exists():
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
        except json.JSONDecodeError:
            logger.warning("Could not read history file. Starting fresh.")
            
    # Check if this date already exists to avoid duplicates
    existing_idx = next((i for i, item in enumerate(history) if item['date'] == report['date']), None)
    
    if existing_idx is not None:
        logger.info(f"Updating existing report for {report['date']}")
        history[existing_idx] = report
    else:
        history.append(report)
        
    # Sort by date
    history.sort(key=lambda x: x['date'])
    
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2, default=str)
        
    return history

def get_history_df(history):
    """Converts history list to a DataFrame with datetime index."""
    data = []
    for entry in history:
        row = {'date': entry['date']}
        metrics = entry.get('metrics', {})
        for variable, values in metrics.items():
            # Ensure values is a dict and has count > 0
            if isinstance(values, dict) and values.get('count', 0) > 0:
                # Temperature (C) -> Temperature_MAE
                clean_var = variable.split('(')[0].strip().replace(' ', '_')
                
                if 'mae' in values:
                    row[f"{clean_var}_MAE"] = values['mae']
                # Handle missing RMSE in older records
                if 'rmse' in values:
                    row[f"{clean_var}_RMSE"] = values['rmse']
                    
        data.append(row)
        
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    # Handle mixed date formats (ISO timestamps vs YYYY-MM-DD)
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    df = df.set_index('date').sort_index()
    return df

def generate_history_plots(df, report_date, report_suffix=""):
    """
    Generates rolling average plots for 7, 30, and 60 days.
    """
    if df.empty:
        return []

    history_dir = "history" if not report_suffix else f"history_{report_suffix}"
    plot_dir = REPORTS_DIR / report_date / "plots" / history_dir
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    sns.set_theme(style="whitegrid")
    saved_plots = []
    
    windows = [7, 30, 60]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green
    
    metric_cols = [c for c in df.columns]
    
    for metric in metric_cols:
        plt.figure(figsize=(10, 6))
        
        # Plot Raw Data (faint)
        sns.lineplot(data=df, x=df.index, y=metric, color='gray', alpha=0.3, linewidth=1, label='Daily')
        
        # Plot Rolling Avgs
        for i, window in enumerate(windows):
            if len(df) >= window:
                rolling = df[metric].rolling(window=window).mean()
                if not rolling.dropna().empty:
                    sns.lineplot(x=rolling.index, y=rolling.values, color=colors[i], label=f'{window}-Day Avg')
        
        var_name = metric.replace('_', ' ')
        plt.title(f"Historical {var_name}")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        
        # Save
        filename = f"history_{metric}.png"
        filepath = plot_dir / filename
        plt.savefig(filepath)
        plt.close()
        saved_plots.append(str(filepath))
        
    return saved_plots

def print_rolling_averages(df, window=7):
    """
    Calculates and prints rolling averages for metrics using historical dataframe.
    """
    if df.empty:
        return
    
    # Calculate rolling mean
    rolling = df.rolling(window=window).mean()
    
    if not rolling.empty:
        latest = rolling.iloc[-1]
        print("\n" + "-"*50)
        print(f"Rolling {window}-Day Average Performance")
        print("-" * 50)
        
        # Only print columns that exist (some might be missing if no data ever existed)
        cols = [c for c in latest.index if not pd.isna(latest[c])]
        
        for col in cols:
            print(f"{col:<20} : {latest[col]:.4f}")
        print("="*50 + "\n")

def parse_args():
    parser = argparse.ArgumentParser(description="End-of-day forecast validation report generator")
    parser.add_argument(
        "--forecast-glob",
        default="station_forecasts_*.json",
        help="Glob pattern in archive/forecasts used to select forecast JSON files",
    )
    parser.add_argument(
        "--report-suffix",
        default="",
        help="Optional suffix for report/history output names (example: beta)",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="Rerun verification for a past date (YYYY-MM-DD) instead of today. "
             "Restores that date's archived forecast/observation files from R2 "
             "if they're no longer present locally. Omit for normal cron behavior.",
    )
    return parser.parse_args()


def run_report(date=None, forecast_glob="station_forecasts_*.json", report_suffix=""):
    """Core report-generation logic, callable directly (e.g. from an admin
    endpoint) as well as via the CLI. Raises RuntimeError on any condition
    that should abort the run - callers that need process-exit-on-failure
    CLI behavior (main(), below) are responsible for catching and exiting;
    an API caller should let the exception surface as an HTTP error instead
    of taking down the whole process with sys.exit().  Returns the report dict.
    """
    suffix = report_suffix.strip().lower()
    suffix_tag = f"_{suffix}" if suffix else ""
    history_file = REPORTS_DIR / f"validation_history{suffix_tag}.json"
    verification_csv_file = REPORTS_DIR / f"verification_history{suffix_tag}.csv"

    logger.info("Starting End of Day Validation Report...")

    target_date = None
    if date:
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            raise RuntimeError(f"date must be YYYY-MM-DD, got: {date}")

    # 1. Load Data
    if target_date:
        date_compact = target_date.replace('-', '')
        forecast_data, fc_file, raw_data, raw_file = find_files_for_date(
            FORECAST_DIR, RAW_DATA_DIR, date_compact, forecast_glob_pattern=forecast_glob,
        )
        if not forecast_data or not raw_data:
            logger.info(f"No local archive/forecasts or archive/raw_data files for {target_date}; attempting R2 restore...")
            from services.archive_bundler import restore_and_unpack_date
            restore_and_unpack_date(date_compact)
            forecast_data, fc_file, raw_data, raw_file = find_files_for_date(
                FORECAST_DIR, RAW_DATA_DIR, date_compact, forecast_glob_pattern=forecast_glob,
            )
        if not forecast_data or not raw_data:
            raise RuntimeError(f"No archived data found for {target_date} (checked local disk and R2).")
    else:
        forecast_data, fc_file, raw_data, raw_file = find_matching_files(
            FORECAST_DIR,
            RAW_DATA_DIR,
            forecast_glob_pattern=forecast_glob,
        )

    if not forecast_data or not raw_data:
        raise RuntimeError("Missing data files.")

    # 2. Determine Time Window
    # Use the forecast run_date to determine the validation window
    run_date_str = forecast_data.get('run_date')
    if run_date_str:
        run_date = pd.Timestamp(run_date_str)
        # The forecast is for the day of run_date, starting from 16:00 UTC (10am Central)
        forecast_day = run_date.date()
        start_search = pd.Timestamp(forecast_day, tz='UTC') + pd.Timedelta(hours=16)
        end_search = start_search + pd.Timedelta(hours=11)  # 16:00 to 03:00 next day
    else:
        # fallback
        start_search = pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=1)
        end_search = pd.Timestamp.now(tz='UTC') + pd.Timedelta(days=1)
    
    # 3. Process DataFrames
    logger.info("Processing forecast data...")
    fc_df = get_forecast_dataframe(forecast_data, start_search, end_search)
    
    logger.info("Processing observation data...")
    obs_df, qc_exclusions, station_coords = get_observation_dataframe(raw_data, start_search, end_search)
    if qc_exclusions:
        logger.warning(f"QC excluded {len(qc_exclusions)} station/variable reading(s): {qc_exclusions}")
    
    if fc_df.empty:
        raise RuntimeError("Forecast DataFrame is empty (no relevant timestamps).")

    if obs_df.empty:
        raise RuntimeError("Observation DataFrame is empty.")

    # 4. Merge
    logger.info("Merging forecasts and observations...")
    merged = pd.merge(fc_df, obs_df, on=['stid', 'timestamp'], how='inner')

    if merged.empty:
        raise RuntimeError("No overlapping records found between forecast and observations.")
        
    logger.info(f"Found {len(merged)} overlapping records for validation.")
    
    # 5. Calculate Metrics
    continuous_variable_map = {
        'Temperature (C)': ('pred_temp', 'obs_temp'),
        'Relative Humidity (%)': ('pred_rh', 'obs_rh'),
        'Wind Speed (m/s)': ('pred_wind', 'obs_wind'),
        'Fuel Moisture (%)': ('pred_fm', 'obs_fm'),
    }
    # Kept alongside continuous_variable_map (rather than replacing it) so
    # generate_plots(), which iterates variable_map, still produces the FDI scatter.
    variable_map = dict(continuous_variable_map, **{
        'Fire Danger Index': ('pred_fire_danger', 'obs_fire_danger')
    })

    results = calculate_metrics(merged, continuous_variable_map)
    results['Fire Danger Index'] = calculate_categorical_metrics(
        merged, 'pred_fire_danger', 'obs_fire_danger'
    )
    confusion = calculate_confusion_matrix(merged)

    # 5b. Wind categorical scoring - additive, alongside the continuous
    # Wind Speed (m/s) metrics above, not a replacement.
    merged['pred_wind_cat'] = merged['pred_wind'].apply(wind_speed_ms_to_category)
    merged['obs_wind_cat'] = merged['obs_wind'].apply(wind_speed_ms_to_category)
    results['Wind Speed Category'] = calculate_categorical_metrics(
        merged, 'pred_wind_cat', 'obs_wind_cat'
    )
    wind_confusion = calculate_confusion_matrix(merged, pred_col='pred_wind_cat', obs_col='obs_wind_cat')

    # 5c. Neighborhood/regional verification - additive alongside the strict
    # per-station confusion matrix/categorical metrics above.
    neighbors = build_station_neighbors(station_coords)
    neighborhood_verification = {
        'fire_danger': calculate_neighborhood_metrics(merged, 'pred_fire_danger', 'obs_fire_danger', neighbors),
        'wind_category': calculate_neighborhood_metrics(merged, 'pred_wind_cat', 'obs_wind_cat', neighbors),
    }

    # 6. Output Report
    report_date = target_date or datetime.now().strftime("%Y-%m-%d")
    report = {
        'date': report_date,
        'generated_at': datetime.utcnow().isoformat() + 'Z',
        'report_suffix': suffix or 'default',
        'forecast_glob': forecast_glob,
        'forecast_source': fc_file.name,
        'observation_source': raw_file.name,
        'metrics': results,
        'confusion_matrix': confusion,
        'wind_confusion_matrix': wind_confusion,
        'neighborhood_verification': neighborhood_verification,
        'stations_count': merged['stidnunique'] if 'stidnunique' in dir(merged) else merged['stid'].nunique(),
        'record_count': len(merged),
        'qc_exclusions': qc_exclusions,
    }

    comparison_rows = []
    for _, row in merged.sort_values(['timestamp', 'stid']).iterrows():
        comparison_rows.append({
            'station': row['stid'],
            'timestamp': row['timestamp'].isoformat(),
            'forecast': {
                'temperature_c': _to_float_or_none(row.get('pred_temp')),
                'relative_humidity_pct': _to_float_or_none(row.get('pred_rh')),
                'wind_speed_ms': _to_float_or_none(row.get('pred_wind')),
                'fuel_moisture_pct': _to_float_or_none(row.get('pred_fm')),
                'fire_danger': _to_float_or_none(row.get('pred_fire_danger')),
            },
            'observed': {
                'temperature_c': _to_float_or_none(row.get('obs_temp')),
                'relative_humidity_pct': _to_float_or_none(row.get('obs_rh')),
                'wind_speed_ms': _to_float_or_none(row.get('obs_wind')),
                'fuel_moisture_pct': _to_float_or_none(row.get('obs_fm')),
                'fire_danger': _to_float_or_none(row.get('obs_fire_danger')),
            },
        })
    report['comparison_rows'] = comparison_rows
    report['verification_ai_packet'] = (
        f"reports/{report_date}/verification_ai_packet{suffix_tag}.json"
    )
    try:
        from ai.verification_summary import generate_verification_summary
        recent_history = []
        if history_file.exists():
            try:
                with open(history_file, 'r', encoding='utf-8') as history_handle:
                    loaded_history = json.load(history_handle)
                if isinstance(loaded_history, list):
                    recent_history = loaded_history
            except (json.JSONDecodeError, OSError):
                logger.warning("Could not load prior verification history for AI context.")
        report['ai_summary'] = generate_verification_summary(
            report, comparison_rows, recent_history
        )
    except Exception:
        logger.exception("Unable to generate optional Gemini verification summary")
        report['ai_summary'] = None
    
    # Generate Plots
    logger.info("Generating plots...")
    plot_files = generate_plots(merged, variable_map, report_date, report_suffix=suffix)
    report['plots'] = plot_files
    
    # Update History
    logger.info("Updating validation history...")
    history = update_history(report, history_file)
    export_verification_history_csv(history, verification_csv_file)

    # Console Output
    print("\n" + "="*50)
    print(f"End of Day Report: {report_date}")
    print("="*50)
    print(f"Forecast File: {fc_file.name}")
    print(f"Obs File:      {raw_file.name}")
    print(f"Stations:      {report['stations_count']}")
    print(f"Records:       {report['record_count']}")
    print("-" * 50)
    print(f"{'Variable':<25} | {'MAE':<10} | {'RMSE':<10} | {'Bias':<10}")
    print("-" * 50)
    
    for var, m in results.items():
        if m['count'] > 0:
            rmse_display = m.get('rmse', 'n/a (categorical)')
            print(f"{var:<25} | {m['mae']:<10} | {str(rmse_display):<10} | {m['bias']:<10}")
        else:
            print(f"{var:<25} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10}")
    
    # Prepare History DataFrame
    history_df = get_history_df(history)

    # Print Rolling Averages
    print_rolling_averages(history_df, window=7)

    # Generate History Plots
    if not history_df.empty:
        logger.info("Generating historical trend plots...")
        generate_history_plots(history_df, report_date, report_suffix=suffix)
    
    # Save Daily JSON Report
    # We save this in a date-specific folder alongside plots now, or just the main reports dir?
    # Let's keep the main reports dir for easy access, but also the dated folder.
    daily_report_dir = REPORTS_DIR / report_date
    daily_report_dir.mkdir(parents=True, exist_ok=True)
    
    report_filename = f"validation_summary{suffix_tag}.json"
    report_file = daily_report_dir / report_filename
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Machine-readable input for the Cloudflare AI worker. Keep this separate
    # from the public summary so the worker can generate its own narrative.
    try:
        from ai.verification_summary import write_verification_ai_packet
        packet_file = daily_report_dir / f"verification_ai_packet{suffix_tag}.json"
        write_verification_ai_packet(report, history, comparison_rows, packet_file)
        if not suffix:
            write_verification_ai_packet(
                report,
                history,
                comparison_rows,
                REPORTS_DIR / "verification_ai_packet.json",
            )
        logger.info(f"AI verification packet saved to {packet_file}")
    except Exception:
        logger.exception("Unable to write Cloudflare AI verification packet.")
        
    # Also save to main dir for compat
    legacy_report_file = REPORTS_DIR / f"validation_summary{suffix_tag}_{report_date}.json"
    with open(legacy_report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Report saved to {report_file}")
    return report


def main():
    """Thin CLI wrapper around run_report() - preserves the existing
    process-exits-nonzero-on-failure behavior cron relies on."""
    args = parse_args()
    try:
        run_report(date=args.date, forecast_glob=args.forecast_glob, report_suffix=args.report_suffix)
    except RuntimeError as exc:
        logger.error(str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
