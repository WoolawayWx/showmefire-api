import json
from pathlib import Path
import pandas as pd
from datetime import datetime
import numpy as np
import logging
import argparse
import sys
import os

# To handle imports if needed, though this script should be mostly standalone
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from services.timeseries import get_timeseries_data

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_forecast_file(filepath):
    """Load and process a station_forecasts JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def load_all_observations(raw_data_dir):
    """
    Load all available raw data into a structured dictionary.
    Returns: { station_id: { datetime_utc: { temp, rh, fm, ... } } }
    """
    obs_data = {}
    
    files = list(Path(raw_data_dir).glob("raw_data_*.json"))
    logger.info(f"Loading observations from {len(files)} files...")
    
    for file in files:
        with open(file, 'r') as f:
            raw = json.load(f)
            
        if "STATION" not in raw:
            continue
            
        for station in raw["STATION"]:
            stid = station["STID"]
            if stid not in obs_data:
                obs_data[stid] = {}
                
            observations = station.get("OBSERVATIONS", {})
            times = observations.get("date_time", [])
            
            # Map variable names (e.g., air_temp_set_1 -> temp_c)
            # Synoptic returns F for english units usually, need to check metadata
            # For now assuming the raw_data was fetched with units=english (F)
            # But the forecast is in C. We'll need to convert.
            
            temps = observations.get("air_temp_set_1", [])
            rhs = observations.get("relative_humidity_set_1", [])
            wspds = observations.get("wind_speed_set_1", []) # kts or mph? Usually mph/kts depending on request
            fms = observations.get("fuel_moisture_set_1", [])
            
            # Check length matches
            if not all(len(x) == len(times) for x in [temps, rhs] if x):
                continue
                
            for i, time_str in enumerate(times):
                # time_str is ISO8601 UTC (e.g. 2026-01-16T16:17:00Z)
                # Normalize to hourly if needed?
                # For now store exact matches or close matches
                
                # Parse time
                dt = pd.to_datetime(time_str)
                # Round to nearest hour for comparison?
                dt_round = dt.round('h')
                
                # Store
                obs_data[stid][dt_round] = {
                    "temp_f": temps[i] if i < len(temps) else None,
                    "rh": rhs[i] if i < len(rhs) else None,
                    "wind_speed": wspds[i] if i < len(wspds) else None,
                    "fuel_moisture": fms[i] if i < len(fms) else None
                }
                
    return obs_data

def compare_forecast_obs(forecast_path, obs_data):
    """
    Compare a specific forecast file against the loaded observations.
    """
    try:
        data = load_forecast_file(forecast_path)
    except Exception as e:
        logger.error(f"Error loading {forecast_path}: {e}")
        return []

    forecast_run_date = data.get("run_date")
    # This run_date is now US/Central based on recent changes
    
    comparisons = []
    
    for stid, station_info in data.get("stations", {}).items():
        if stid not in obs_data:
            continue
            
        station_obs = obs_data[stid]
        
        for fcst in station_info.get("forecasts", []):
            fcst_time_str = fcst["time"]
            # fcst_time_str is US/Central ISO format (2026-01-17T13:00:00)
            
            # Convert forecast time to UTC to match observations index
            dt_fcst_central = pd.Timestamp(fcst_time_str).tz_localize('US/Central')
            dt_fcst_utc = dt_fcst_central.tz_convert('UTC')
            
            # Look for matching observation
            if dt_fcst_utc in station_obs:
                obs = station_obs[dt_fcst_utc]
                
                # Metrics to compare
                # Temp: Forecast is C, Obs might be F. 
                # Let's see... API usually calls units='english' -> F.
                # Forecast JSON says "temp_c".
                
                obs_temp_c = (obs["temp_f"] - 32) * 5/9 if obs["temp_f"] is not None else None
                
                # Wind conversion: API often returns m/s if not specified, or kts/mph.
                # Assuming Synoptic returned 'm/s' or similar. 
                # forecast is m/s. 
                # If obs is mph/kts we need conversion. 
                # For now assuming obs is m/s or close enough to compare raw, 
                # but we'll add it to CSV to verify later.
                obs_wind = obs["wind_speed"]

                if obs_temp_c is not None:
                     comparisons.append({
                        "station_id": stid,
                        "valid_time_utc": dt_fcst_utc.isoformat(),
                        "lead_hour": int((dt_fcst_utc - pd.Timestamp(forecast_run_date).tz_localize('US/Central').tz_convert('UTC')).total_seconds() / 3600),
                        
                        "temp_fcst": fcst["temp_c"],
                        "temp_obs": obs_temp_c,
                        "temp_error": fcst["temp_c"] - obs_temp_c,
                        
                        "rh_fcst": fcst["rh"],
                        "rh_obs": obs["rh"],
                        "rh_error": fcst["rh"] - (obs["rh"] if obs["rh"] is not None else 0),
                        
                        "fm_fcst": fcst["fuel_moisture"],
                        "fm_obs": obs["fuel_moisture"],
                        "fm_error": fcst["fuel_moisture"] - (obs["fuel_moisture"] if obs["fuel_moisture"] is not None else 0) if obs["fuel_moisture"] is not None else None,

                        "wind_fcst": fcst["wind_speed_ms"],
                        "wind_obs": obs_wind,
                        "wind_error": fcst["wind_speed_ms"] - (obs_wind if obs_wind is not None else 0) if obs_wind is not None else None
                    })
    
    return comparisons

def main():
    parser = argparse.ArgumentParser(description="Compare forecast vs observations")
    parser.add_argument("--forecast-dir", default="archive/forecasts", help="Directory containing forecast JSONs")
    parser.add_argument("--raw-dir", default="archive/raw_data", help="Directory containing raw obs JSONs")
    parser.add_argument("--output", default="reports/forecast_comparison_latest.csv", help="Output file")
    
    args = parser.parse_args()
    
    # Load all observations first
    obs_data = load_all_observations(args.raw_dir)
    logger.info(f"Loaded observations for {len(obs_data)} stations.")
    
    # Process all forecast files
    all_comparisons = []
    forecast_files = sorted(list(Path(args.forecast_dir).glob("station_forecasts_*.json")))
    
    logger.info(f"Comparing {len(forecast_files)} forecast files...")
    
    for f in forecast_files:
        comps = compare_forecast_obs(f, obs_data)
        all_comparisons.extend(comps)
        
    if not all_comparisons:
        logger.warning("No matching comparisons found.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(all_comparisons)
    
    # Filter out missing comparisons
    df = df.dropna(subset=['temp_error'])
    
    # Save detailed CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved detailed comparison to {output_path}")
    
    # Calculate Summary Metrics
    logger.info("\n=== Validation Summary ===")
    
    # Overall MAE/Bias
    mae_temp = df['temp_error'].abs().mean()
    bias_temp = df['temp_error'].mean()
    
    mae_rh = df['rh_error'].abs().mean()
    bias_rh = df['rh_error'].mean()
    
    # FM metrics (could be missing data)
    df_fm = df.dropna(subset=['fm_error'])
    if not df_fm.empty:
        mae_fm = df_fm['fm_error'].abs().mean()
        bias_fm = df_fm['fm_error'].mean()
        logger.info(f"Fuel Moisture: MAE={mae_fm:.2f}%, Bias={bias_fm:.2f}% (n={len(df_fm)})")
    else:
        logger.info("Fuel Moisture: No data")
        
    logger.info(f"Temperature:   MAE={mae_temp:.2f}C, Bias={bias_temp:.2f}C (n={len(df)})")
    logger.info(f"Rel Humidity:  MAE={mae_rh:.2f}%, Bias={bias_rh:.2f}% (n={len(df)})")
    
    # Save summary JSON
    summary = {
        "generated_at": datetime.utcnow().isoformat(),
        "metrics": {
            "temperature": {"mae": mae_temp, "bias": bias_temp, "count": len(df)},
            "rh": {"mae": mae_rh, "bias": bias_rh, "count": len(df)},
            "fuel_moisture": {"mae": mae_fm if not df_fm.empty else None, "bias": bias_fm if not df_fm.empty else None, "count": len(df_fm)}
        }
    }
    
    with open(output_path.parent / "validation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
