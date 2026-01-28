import json
import os
import sys
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
HISTORY_FILE = REPORTS_DIR / "validation_history.json"

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

def find_matching_files(forecast_dir, raw_dir):
    """
    Attempts to find a forecast file and a raw data file that share the same date.
    Returns (forecast_data, forecast_file, raw_data, raw_file)
    """
    if not forecast_dir.exists() or not raw_dir.exists():
        return None, None, None, None

    fc_files = sorted(list(forecast_dir.glob("station_forecasts_*.json")), reverse=True)
    raw_files = sorted(list(raw_dir.glob("raw_data_*.json")), reverse=True)
    
    if not fc_files or not raw_files:
        return None, None, None, None

    # Try to find a match
    for fc_file in fc_files:
        # Extract date YYYYMMDD from station_forecasts_YYYYMMDD_HH.json
        try:
            parts = fc_file.name.split('_')
            if len(parts) >= 3:
                fc_date_str = parts[2] # 20260122
                
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

def fahrenheit_to_celsius(f):
    return (f - 32) * 5.0/9.0

def mph_to_ms(mph):
    return mph * 0.44704

def ms_to_kts(ms):
    return ms * 1.94384

def calculate_fire_danger(fm, rh, wind_kts):
    """
    Fire Danger Criteria based on ShowMeFire.org:
    Low: FM >= 15%
    Moderate: FM 9-14% WITH (RH < 50% AND Wind >= 10 kts)
    Elevated: FM < 9% WITH (RH < 45% OR Wind >= 10 kts)
    Critical: FM < 9% WITH (RH < 25% AND Wind >= 15 kts)
    Extreme: FM < 7% WITH (RH < 20% AND Wind >= 30 kts)
    """
    # Safety check for None/NaN
    if pd.isna(fm) or pd.isna(rh) or pd.isna(wind_kts):
        return None

    # LOW (Fuels are too wet to carry fire effectively)
    if fm >= 15: 
        return 0 
    
    # 5. EXTREME (The most restrictive)
    if fm < 7 and rh < 20 and wind_kts >= 30:
        return 4
    
    # 4. CRITICAL (High)
    if fm < 9 and rh < 25 and wind_kts >= 15:
        return 3
        
    # 3. ELEVATED
    if fm < 9 and (rh < 45 or wind_kts >= 10):
        return 2
        
    # 2. MODERATE
    # Change to AND logic: FM must be low AND weather must be active
    if (9 <= fm < 15) and (rh < 50 and wind_kts >= 10):
        return 1
        
    # 1. LOW (Default if FM is high or conditions aren't met)
    return 0

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
    
    # Handle both list and dict structures for stations
    stations = raw_data.get('STATION', [])
    if isinstance(stations, dict):
        stations = [stations]
        
    for station in stations:
        stid = station.get('STID')
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
                        wind_val = mph_to_ms(wind_val)
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
    if not df.empty:
        df = df.groupby(['stid', 'timestamp']).mean().reset_index()
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df['timestamp'] = df['timestamp'].dt.round('h')
        df['obs_fire_danger'] = df.apply(
            lambda row: calculate_fire_danger(
                row['obs_fm'], 
                row['obs_rh'], 
                ms_to_kts(row['obs_wind'])
            ), axis=1
        )
    return df
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

def generate_plots(merged_df, variable_map, report_date):
    """
    Generates scatter plots for Predicted vs Observed values.
    Saves plots to REPORTS_DIR/plots/{date}/
    """
    # Create daily directory for plots
    daily_plot_dir = REPORTS_DIR / report_date / "plots"
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

def update_history(report):
    """
    Updates the historical validation JSON file.
    """
    history = []
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, 'r') as f:
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
    
    with open(HISTORY_FILE, 'w') as f:
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

def generate_history_plots(df, report_date):
    """
    Generates rolling average plots for 7, 30, and 60 days.
    """
    if df.empty:
        return []

    plot_dir = REPORTS_DIR / report_date / "plots" / "history"
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
                    sns.lineplot(data=rolling, x=rolling.index, y=rolling, color=colors[i], label=f'{window}-Day Avg')
        
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

def main():
    logger.info("Starting End of Day Validation Report...")
    
    # 1. Load Data (Auto-matching dates)
    forecast_data, fc_file, raw_data, raw_file = find_matching_files(FORECAST_DIR, RAW_DATA_DIR)
    
    if not forecast_data or not raw_data:
        logger.error("Missing data files. Aborting.")
        return

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
    obs_df = get_observation_dataframe(raw_data, start_search, end_search)
    
    if fc_df.empty:
        logger.warning("Forecast DataFrame is empty (no relevant timestamps).")
        return
        
    if obs_df.empty:
        logger.warning("Observation DataFrame is empty.")
        return
        
    # 4. Merge
    logger.info("Merging forecasts and observations...")
    merged = pd.merge(fc_df, obs_df, on=['stid', 'timestamp'], how='inner')
    
    if merged.empty:
        logger.warning("No overlapping records found between forecast and observations.")
        return
        
    logger.info(f"Found {len(merged)} overlapping records for validation.")
    
    # 5. Calculate Metrics
    variable_map = {
        'Temperature (C)': ('pred_temp', 'obs_temp'),
        'Relative Humidity (%)': ('pred_rh', 'obs_rh'),
        'Wind Speed (m/s)': ('pred_wind', 'obs_wind'),
        'Fuel Moisture (%)': ('pred_fm', 'obs_fm'),
        'Fire Danger Index': ('pred_fire_danger', 'obs_fire_danger')
    }
    
    results = calculate_metrics(merged, variable_map)
    
    # 6. Output Report
    report_date = datetime.now().strftime("%Y-%m-%d")
    report = {
        'date': report_date,
        'forecast_source': fc_file.name,
        'observation_source': raw_file.name,
        'metrics': results,
        'stations_count': merged['stidnunique'] if 'stidnunique' in dir(merged) else merged['stid'].nunique(),
        'record_count': len(merged)
    }
    
    # Generate Plots
    logger.info("Generating plots...")
    plot_files = generate_plots(merged, variable_map, report_date)
    report['plots'] = plot_files
    
    # Update History
    logger.info("Updating validation history...")
    history = update_history(report)

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
            print(f"{var:<25} | {m['mae']:<10} | {m['rmse']:<10} | {m['bias']:<10}")
        else:
            print(f"{var:<25} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10}")
    
    # Prepare History DataFrame
    history_df = get_history_df(history)

    # Print Rolling Averages
    print_rolling_averages(history_df, window=7)

    # Generate History Plots
    if not history_df.empty:
        logger.info("Generating historical trend plots...")
        generate_history_plots(history_df, report_date)
    
    # Save Daily JSON Report
    # We save this in a date-specific folder alongside plots now, or just the main reports dir?
    # Let's keep the main reports dir for easy access, but also the dated folder.
    daily_report_dir = REPORTS_DIR / report_date
    daily_report_dir.mkdir(parents=True, exist_ok=True)
    
    report_file = daily_report_dir / "validation_summary.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
        
    # Also save to main dir for compat
    legacy_report_file = REPORTS_DIR / f"validation_summary_{report_date}.json"
    with open(legacy_report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"Report saved to {report_file}")

if __name__ == "__main__":
    main()
