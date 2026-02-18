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
        try:
            with open(file, 'r') as f:
                raw_content = json.load(f)

            # Handle if file is list vs dict
            if isinstance(raw_content, list):
                # Data is a list of station dictionaries
                # Expected format: [{ 'STID': '...', 'OBSERVATIONS': { ... } }, ...]
                stations_iter = raw_content
            elif isinstance(raw_content, dict):
                # If it's a dict, it might be keyed by STID or simply wrapped
                if "STATION" in raw_content:
                     stations_iter = raw_content["STATION"]
                else:
                     # Assume it's { "STID": {...}, ... }
                     # Convert to list format to unify processing
                     stations_iter = [{"STID": k, "OBSERVATIONS": v} for k, v in raw_content.items()]
            else:
                logger.warning(f"Unknown format in {file}")
                continue
                
            for station in stations_iter:
                # Get ID safely
                stid = station.get("STID")
                if not stid:
                    continue
                    
                if stid not in obs_data:
                    obs_data[stid] = {}
                
                # Check where observations are located
                observations = station.get("OBSERVATIONS")
                if not observations:
                    # Sometimes the station dict IS the observation dict minus the STID key
                    # depending on how it was saved.
                    # But typically Synoptic separates them.
                    # If observations is missing, check if keys like "air_temp_set_1" exist directly
                    if "air_temp_set_1" in station:
                        observations = station
                    else:
                        continue
                
                # Extract time array and variable arrays
                times = observations.get("date_time", []) # Synoptic usually uses 'date_time', sometimes 'TIME'
                if not times:
                    times = observations.get("TIME", [])

                temps = observations.get("air_temp_set_1", [])
                rhs = observations.get("relative_humidity_set_1", [])
                wind_speeds = observations.get("wind_speed_set_1", [])
                
                # Add FM manually if available
                # Synoptic often sends "fuel_moisture_set_1"
                fms = observations.get("fuel_moisture_set_1", [])

                # Process this station's timeline
                for i, time_str in enumerate(times):
                    try:
                        dt = pd.to_datetime(time_str)
                        if dt.tzinfo is None:
                            dt = dt.tz_localize('UTC')
                        else:
                            dt = dt.tz_convert('UTC')
                        
                        dt_round = dt.round('h')

                        obs_data[stid][dt_round] = {
                            "temp_c": temps[i] if i < len(temps) else None,
                            "rh": rhs[i] if i < len(rhs) else None,
                            "wind_m_s": wind_speeds[i] if i < len(wind_speeds) else None,
                            "fm": fms[i] if i < len(fms) else None
                        }
                    except (ValueError, IndexError):
                        continue
                        
        except Exception as e:
            logger.error(f"Error loading {file}: {e}")
                
    return obs_data

def compare_forecast_obs(forecast_path, obs_data):
    """
    Compare a specific forecast file against the loaded observations.
    """
    try:
        data = load_forecast_file(forecast_path)
    except Exception as e:
        logger.error(f"Failed to load forecast file {forecast_path}: {e}")
        return []

    forecast_run_date = data.get("run_date")
    # This run_date is now US/Central based on recent changes
    
    comparisons = []
    
    comparisons = []
    
    stations = data.get("stations", {})
    if not stations:
        logger.warning(f"No stations found in forecast file {forecast_path}. Skipping.")
        return []
    
    for stid, station_info in stations.items():
        station_obs = obs_data.get(stid, {})
        
        if not station_obs:
            continue
        
        for fcst in station_info.get("forecasts", []):
            fcst_time_str = fcst["time"]
            
            # Robust forecast time parsing
            try:
                dt_fcst = pd.Timestamp(fcst_time_str)
                if dt_fcst.tzinfo is None:
                    # Assume UTC if naive, as most standardized storage is UTC
                    dt_fcst = dt_fcst.tz_localize('UTC')
                else:
                    dt_fcst = dt_fcst.tz_convert('UTC')
                
                # Round to nearest hour to match observation keys
                dt_fcst_round = dt_fcst.round('h')

                if dt_fcst_round in station_obs:
                    obs = station_obs[dt_fcst_round]
                    obs = station_obs[dt_fcst_round]
                    
                    # Store comparison if we have valid observation data
                    if obs.get("temp_c") is not None:
                        # Convert forecast F to C for comparison if needed, 
                        # OR ensure your forecast is already C. 
                        # Assuming forecast is C based on variable names like 'temp_forecast_c' later.
                        
                        # Check units! If your forecast JSON has temp in F (common in US), convert.
                        # If forecast is Celsius, use directly.
                        # Assuming forecast is in Celsius for now based on 'temp_forecast_c' in dataframe code.
                        fcst_temp = fcst.get("temp")
                        fcst_rh = fcst.get("humidity")
                        fcst_wind = fcst.get("wind_speed")
                        fcst_fm = fcst.get("fuel_moisture_10hr")

                        comparisons.append({
                            "station_id": stid,
                            "run_date": forecast_run_date,
                            "valid_time_utc": dt_fcst_round,
                            "lead_hour": int((dt_fcst_round - pd.Timestamp(forecast_run_date).tz_convert('UTC')).total_seconds() / 3600),
                            "temp_fcst": fcst_temp,
                            "temp_obs": obs.get("temp_c"),
                            "temp_error": fcst_temp - obs.get("temp_c"),
                            "rh_fcst": fcst_rh,
                            "rh_obs": obs.get("rh"),
                            "rh_error": fcst_rh - obs.get("rh"),
                            "wind_fcst": fcst_wind,
                            "wind_obs": obs.get("wind_m_s"),
                            "wind_error": (fcst_wind - obs.get("wind_m_s")) if obs.get("wind_m_s") is not None else None,
                            "fm_fcst": fcst_fm,
                            "fm_obs": obs.get("fm"),
                            "fm_error": (fcst_fm - obs.get("fm")) if obs.get("fm") is not None and fcst_fm is not None else None
                        })
            except Exception as e:
                # logger.debug(f"Skipping forecast point due to error: {e}")
                continue
    
    return comparisons

def main():
    parser = argparse.ArgumentParser(description="Compare forecast vs observations")
    parser.add_argument("--forecast-dir", default="archive/forecasts", help="Directory containing forecast JSONs")
    parser.add_argument("--raw-dir", default="archive/raw_data", help="Directory containing raw obs JSONs")
    parser.add_argument("--output", default="reports/forecast_comparison_latest.csv", help="Output file")
    parser.add_argument("--date", help="Filter forecast files by date (format: YYYY-MM-DD or YYYYMMDD)")
    
    args = parser.parse_args()
    
    # Load all observations first
    obs_data = load_all_observations(args.raw_dir)
    logger.info(f"Loaded observations for {len(obs_data)} stations.")
    
    # Process all forecast files
    all_comparisons = []
    forecast_files = sorted(list(Path(args.forecast_dir).glob("station_forecasts_*.json")))
    
    # Filter by date if specified
    if args.date:
        # Normalize date format (remove hyphens)
        date_filter = args.date.replace('-', '')
        forecast_files = [f for f in forecast_files if date_filter in f.name]
        logger.info(f"Filtering for date: {args.date} (found {len(forecast_files)} matching files)")
    
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
    
    # Determine date for folder organization
    if args.date:
        report_date = args.date.replace('-', '')
    else:
        # Use the date from the forecast files
        if forecast_files:
            # Extract date from first forecast filename (e.g., station_forecasts_20260131_12.json)
            filename = forecast_files[0].name
            report_date = filename.split('_')[2]  # Gets '20260131'
        else:
            report_date = datetime.utcnow().strftime('%Y%m%d')
    
    # Create date-specific output directory
    output_dir = Path(args.output).parent / report_date
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving results to: {output_dir}")
    
    # Save detailed CSV
    output_path = output_dir / "forecast_comparison_detailed.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved detailed comparison to {output_path}")
    
    # ============================================================
    # NEW: Create side-by-side comparison dataframe
    # ============================================================
    sidebyside_df = pd.DataFrame({
        'station_id': df['station_id'],
        'valid_time_utc': df['valid_time_utc'],
        'lead_hour': df['lead_hour'],
        
        # Temperature (Celsius)
        'temp_forecast_c': df['temp_fcst'].round(2),
        'temp_observed_c': df['temp_obs'].round(2),
        'temp_error_c': df['temp_error'].round(2),
        
        # Relative Humidity (%)
        'rh_forecast_pct': df['rh_fcst'].round(1),
        'rh_observed_pct': df['rh_obs'].round(1),
        'rh_error_pct': df['rh_error'].round(1),
        
        # Fuel Moisture (%)
        'fuel_moisture_forecast_pct': df['fm_fcst'].round(2) if 'fm_fcst' in df else None,
        'fuel_moisture_observed_pct': df['fm_obs'].round(2) if 'fm_obs' in df else None,
        'fuel_moisture_error_pct': df['fm_error'].round(2) if 'fm_error' in df else None,
        
        # Wind Speed (m/s)
        'wind_forecast_ms': df['wind_fcst'].round(2) if 'wind_fcst' in df else None,
        'wind_observed_ms': df['wind_obs'].round(2) if 'wind_obs' in df else None,
        'wind_error_ms': df['wind_error'].round(2) if 'wind_error' in df else None
    })
    
    # Save side-by-side CSV
    sidebyside_path = output_dir / "forecast_vs_observed_sidebyside.csv"
    sidebyside_df.to_csv(sidebyside_path, index=False)
    logger.info(f"Saved side-by-side comparison to {sidebyside_path}")
    
    # Also create a JSON version for easier programmatic access
    sidebyside_json = sidebyside_df.to_dict(orient='records')
    sidebyside_json_path = output_dir / "forecast_vs_observed_sidebyside.json"
    with open(sidebyside_json_path, 'w') as f:
        json.dump(sidebyside_json, f, indent=2)
    logger.info(f"Saved side-by-side comparison JSON to {sidebyside_json_path}")
    
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
        mae_fm = None
        bias_fm = None
        
    # NEW: Wind metrics
    df_wind = df.dropna(subset=['wind_error'])
    if not df_wind.empty:
        mae_wind = df_wind['wind_error'].abs().mean()
        bias_wind = df_wind['wind_error'].mean()
        logger.info(f"Wind Speed:    MAE={mae_wind:.2f} m/s, Bias={bias_wind:.2f} m/s (n={len(df_wind)})")
    else:
        logger.info("Wind Speed:    No data")
        mae_wind = bias_wind = None
        
    logger.info(f"Temperature:   MAE={mae_temp:.2f}C, Bias={bias_temp:.2f}C (n={len(df)})")
    logger.info(f"Rel Humidity:  MAE={mae_rh:.2f}%, Bias={bias_rh:.2f}% (n={len(df)})")
    
    # Save summary JSON
    summary = {
        "date": report_date,
        "generated_at": datetime.utcnow().isoformat(),
        "forecast_files": len(forecast_files),
        "metrics": {
            "temperature": {"mae": float(mae_temp), "bias": float(bias_temp), "count": len(df)},
            "rh": {"mae": float(mae_rh), "bias": float(bias_rh), "count": len(df)},
            "fuel_moisture": {"mae": float(mae_fm) if not df_fm.empty else None, "bias": float(bias_fm) if not df_fm.empty else None, "count": len(df_fm)},
            "wind_speed": {"mae": float(mae_wind) if mae_wind is not None else None, "bias": float(bias_wind) if bias_wind is not None else None, "count": len(df_wind)}
        }
    }
    
    with open(output_dir / "validation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # ============================================================
    # Update verification tracking file
    # ============================================================
    verification_file = Path(args.output).parent / "verification_history.csv"
    
    # Load existing verification data or create new
    if verification_file.exists():
        verification_df = pd.read_csv(verification_file)
        # Ensure date column is string type
        verification_df['date'] = verification_df['date'].astype(str)
        # Remove existing entry for this date if present (override)
        verification_df = verification_df[verification_df['date'] != report_date]
    else:
        verification_df = pd.DataFrame()
    
    # Add new verification entry
    new_entry = pd.DataFrame([{
        'date': str(report_date),  # Ensure it's a string
        'generated_at': datetime.utcnow().isoformat(),
        'num_forecasts': len(forecast_files),
        'num_comparisons': len(df),
        'temp_mae_c': round(mae_temp, 2),
        'temp_bias_c': round(bias_temp, 2),
        'rh_mae_pct': round(mae_rh, 2),
        'rh_bias_pct': round(bias_rh, 2),
        'wind_mae_ms': round(mae_wind, 2) if mae_wind is not None else None,
        'wind_bias_ms': round(bias_wind, 2) if bias_wind is not None else None,
        'fm_mae_pct': round(mae_fm, 2) if not df_fm.empty else None,
        'fm_bias_pct': round(bias_fm, 2) if not df_fm.empty else None,
        'fm_count': len(df_fm)
    }])
    
    verification_df = pd.concat([verification_df, new_entry], ignore_index=True)
    verification_df = verification_df.sort_values('date')
    
    # Save updated verification history
    verification_df.to_csv(verification_file, index=False)
    logger.info(f"Updated verification history: {verification_file}")
    logger.info(f"Verification history now contains {len(verification_df)} dates")

if __name__ == "__main__":
    main()