import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path
import logging
import asyncio
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.database import get_db_path
from services.synoptic import fetch_historical_station_data

# Setup logging
LOGS_DIR = Path(__file__).resolve().parent.parent / 'logs'
LOGS_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'verification.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

async def fetch_and_store_observations():
    """Fetches key weather variables from Synoptic for the last 26 hours and stores them."""
    logger.info("Fetching observations for verification...")
    
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=26)
    
    try:
        # Fetch data for MO and neighbors
        # Network 1=RAWS, 2=NWS/FAA (ASOS)
        # We focus on RAWS (1) and NWS (2) for broad coverage, but mainly RAWS has fuel moisture.
        data = await fetch_historical_station_data(
            start_time=start_time, 
            end_time=end_time,
            networks=[1, 2] 
        )
        
        if not data or 'STATION' not in data:
            logger.warning("No station data returned from Synoptic API.")
            return

        conn = sqlite3.connect(get_db_path())
        cursor = conn.cursor()
        
        count = 0
        for station in data['STATION']:
            stid = station.get('STID')
            lat = float(station.get('LATITUDE', 0))
            lon = float(station.get('LONGITUDE', 0))
            obs = station.get('OBSERVATIONS', {})
            
            # Times are usually in list under 'date_time'
            times = obs.get('date_time', [])
            
            # Get variable arrays (handle missing keys safely)
            # Synoptic returns lists of values corresponding to date_time indices
            
            # Helper to get first available set (e.g. set_1)
            def get_vals(key_prefix):
                for k in obs.keys():
                    if k.startswith(key_prefix):
                        return obs[k]
                return [None] * len(times)

            fm_vals = get_vals('fuel_moisture')
            temp_vals = get_vals('air_temp')
            rh_vals = get_vals('relative_humidity')
            wind_vals = get_vals('wind_speed')
            precip_vals = get_vals('precip_accum_one_hour') # or precip_accum
            
            for i, ts in enumerate(times):
                if ts is None: continue
                
                # Format timestamp to match SQLite standard (ISO8601)
                # Synoptic: 2026-01-12T10:00:00Z
                ts_clean = ts.replace('T', ' ').replace('Z', '')
                
                fm = fm_vals[i]
                temp = temp_vals[i]
                rh = rh_vals[i]
                ws = wind_vals[i]
                precip = precip_vals[i]
                
                # Check if we have at least one valid metric to store
                if all(v is None for v in [fm, temp, rh, ws]):
                    continue
                    
                cursor.execute('''
                    INSERT INTO observations (
                        station_id, observation_date, 
                        fuel_moisture_percentage, temp_c, rel_humidity, wind_speed_ms, precip_accum_1h_mm,
                        latitude, longitude
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(station_id, observation_date) DO UPDATE SET
                        fuel_moisture_percentage=excluded.fuel_moisture_percentage,
                        temp_c=excluded.temp_c,
                        rel_humidity=excluded.rel_humidity,
                        wind_speed_ms=excluded.wind_speed_ms,
                        precip_accum_1h_mm=excluded.precip_accum_1h_mm
                ''', (stid, ts_clean, fm, temp, rh, ws, precip, lat, lon))
                count += 1
                
        conn.commit()
        conn.close()
        logger.info(f"Successfully ingested {count} observations into database.")
        
    except Exception as e:
        logger.error(f"Error fetching/storing observations: {e}", exc_info=True)

def analyze_forecast_verification():
    db_path = get_db_path()
    if not db_path.exists():
        logger.error(f"Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    
    # Analyze various metrics
    # We join on nearest hour or exact match. 
    # Since we extract forecast for specific hours, and obs are hourly, exact match on formatted string usually works 
    # if both are YYYY-MM-DD HH:MM:00
    
    query = """
    SELECT 
        f.station_id,
        s.name as station_name,
        f.valid_time,
        f.fuel_moisture as fcst_fm,
        o.fuel_moisture_percentage as obs_fm,
        f.temp_c as fcst_temp,
        o.temp_c as obs_temp,
        f.rel_humidity as fcst_rh,
        o.rel_humidity as obs_rh,
        f.wind_speed_ms as fcst_wind,
        o.wind_speed_ms as obs_wind
    FROM station_forecasts f
    JOIN observations o ON f.station_id = o.station_id 
        AND strftime('%Y-%m-%d %H:%M', f.valid_time) = strftime('%Y-%m-%d %H:%M', o.observation_date)
    LEFT JOIN stations s ON f.station_id = s.id
    WHERE datetime(f.valid_time) >= datetime('now', '-24 hours')
    """
    
    try:
        df = pd.read_sql(query, conn)
    except Exception as e:
        logger.error(f"Error executing verification query: {e}")
        conn.close()
        return

    conn.close()
    
    if df.empty:
        logger.warning("No matching forecast and observation data found for the last 24 hours.")
        return
        
    logger.info(f"=== Daily Forecast Verification (Last 24 Hours) ===")
    logger.info(f"Data points: {len(df)}")
    
    metrics_summary = []

    # Helper for metrics
    def calc_metrics(df, fcst_col, obs_col, label, unit):
        # Filter NaNs
        valid = df.dropna(subset=[fcst_col, obs_col])
        if valid.empty:
            return
            
        error = valid[fcst_col] - valid[obs_col]
        mae = error.abs().mean()
        rmse = np.sqrt((error ** 2).mean())
        bias = error.mean()
        
        logger.info(f"\n{label}:")
        logger.info(f"  MAE:  {mae:.2f} {unit}")
        logger.info(f"  RMSE: {rmse:.2f} {unit}")
        logger.info(f"  Bias: {bias:.2f} {unit}")
        
        metrics_summary.append({
            'variable': label,
            'mae': mae,
            'rmse': rmse,
            'bias': bias,
            'count': len(valid)
        })

    calc_metrics(df, 'fcst_fm', 'obs_fm', 'Fuel Moisture', '%')
    calc_metrics(df, 'fcst_temp', 'obs_temp', 'Temperature', 'C')
    calc_metrics(df, 'fcst_rh', 'obs_rh', 'Relative Humidity', '%')
    calc_metrics(df, 'fcst_wind', 'obs_wind', 'Wind Speed', 'm/s')
    
    # Save Report
    reports_dir = Path(__file__).resolve().parent.parent / 'reports'
    reports_dir.mkdir(exist_ok=True)
    
    # Create a daily folder
    today = datetime.now().strftime("%Y-%m-%d")
    daily_report_dir = reports_dir / today
    daily_report_dir.mkdir(exist_ok=True)
    
    report_file = daily_report_dir / 'verification_data.csv'
    df.to_csv(report_file, index=False)
    logger.info(f"Detailed verification data saved to: {report_file}")
    
    # Save summary text
    summary_file = daily_report_dir / 'verification_summary.txt'
    with open(summary_file, 'w') as f:
        f.write(f"Forecast Verification Report - {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write("==================================================\n\n")
        for m in metrics_summary:
            f.write(f"{m['variable']}:\n")
            f.write(f"  MAE:  {m['mae']:.2f}\n")
            f.write(f"  RMSE: {m['rmse']:.2f}\n")
            f.write(f"  Bias: {m['bias']:.2f}\n")
            f.write(f"  Count: {m['count']}\n\n")
            
    logger.info(f"Summary report saved to: {summary_file}")

if __name__ == "__main__":
    # Run async fetch first
    try:
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(fetch_and_store_observations())
    except Exception as e:
        logger.error(f"Failed to fetch observations: {e}")
        
    # Then run analysis
    analyze_forecast_verification()
