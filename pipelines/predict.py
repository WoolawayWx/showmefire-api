import pandas as pd
import xgboost as xgb
import sqlite3
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import get_db_path

def get_danger_info(row):
    fm = row['predicted_fuel_moisture']
    rh = row['rel_humidity']
    wind = row['wind_speed_ms'] * 2.237  # Convert to mph
    
    # ANSI Color Codes for terminal output
    RED = '\033[91m'
    ORANGE = '\033[38;5;208m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RESET = '\033[0m'

    if fm < 5 or rh < 20 or wind > 25:
        return f"{RED}EXTREME{RESET}"
    elif fm <= 7 or rh <= 30 or wind > 20:
        return f"{ORANGE}HIGH (CRITICAL){RESET}"
    elif fm <= 10 or rh <= 35 or wind > 15:
        return f"{YELLOW}ELEVATED{RESET}"
    elif fm <= 15 or rh <= 45 or wind > 10:
        return f"{BLUE}MODERATE{RESET}"
    else:
        return f"{GREEN}LOW{RESET}"

def run_live_prediction():
    # 1. Load the trained model
    model_path = 'models/fuel_moisture_model.json'
    if not os.path.exists(model_path):
        print("❌ Model artifact not found. Please run training first.")
        return
    
    model = xgb.XGBRegressor()
    model.load_model(model_path)
    
    # 2. Get the most recent weather data from the DB
    # We need at least the last 6 hours to calculate the rolling means (lags)
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    
    # This query fetches the latest weather for all stations
    query = """
    SELECT wf.*, s.lat, s.lon, snap.snapshot_date
    FROM weather_features wf
    JOIN stations s ON wf.station_id = s.id
    JOIN snapshots snap ON wf.snapshot_id = snap.id
    ORDER BY snap.snapshot_date DESC, wf.station_id
    """
    df = pd.read_sql(query, conn)
    conn.close()

    if df.empty:
        print("⚠️ No weather data found in database for prediction.")
        return

    # 3. Feature Engineering (Match the training format exactly)
    df['obs_time'] = pd.to_datetime(df['snapshot_date'])
    df['hour'] = df['obs_time'].dt.hour
    df['month'] = df['obs_time'].dt.month
    df['emc_baseline'] = df['rel_humidity'] / 5.0
    
    # Calculate lags (rolling means)
    df = df.sort_values(['station_id', 'obs_time'])
    for window in [3, 6]:
        df[f'temp_mean_{window}h'] = df.groupby('station_id')['temp_c'].transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[f'rh_mean_{window}h'] = df.groupby('station_id')['rel_humidity'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    
    # Add precipitation features if precip_mm column exists
    if 'precip_mm' in df.columns:
        # Rolling precipitation sums
        for window in [1, 3, 6, 24]:
            df[f'precip_{window}h'] = df.groupby('station_id')['precip_mm'].transform(
                lambda x: x.rolling(window, min_periods=1).sum()
            )
        
        # Hours since last measurable rain (>0.1mm)
        def hours_since_rain(group):
            result = []
            hours_count = 0
            for precip in group:
                if precip > 0.1:
                    hours_count = 0
                else:
                    hours_count += 1
                result.append(hours_count)
            return pd.Series(result, index=group.index)
        
        df['hours_since_rain'] = df.groupby('station_id')['precip_mm'].transform(hours_since_rain)

    # 4. Filter for only the absolute latest timestamp to show current conditions
    latest_time = df['obs_time'].max()
    current_conditions = df[df['obs_time'] == latest_time].copy()

    # 5. Predict
    features = [
        'temp_c', 'rel_humidity', 'wind_speed_ms',
        'hour', 'month', 'emc_baseline', 
        'temp_mean_3h', 'rh_mean_3h', 'temp_mean_6h', 'rh_mean_6h'
    ]
    
    # Add precipitation features if they exist
    precip_features = ['precip_1h', 'precip_3h', 'precip_6h', 'precip_24h', 'hours_since_rain']
    for feat in precip_features:
        if feat in current_conditions.columns:
            features.append(feat)
    
    current_conditions['predicted_fuel_moisture'] = model.predict(current_conditions[features])

    # Apply danger level classification
    current_conditions['danger_level'] = current_conditions.apply(get_danger_info, axis=1)

    # 6. Output Results
    print(f"\n🔮 Live Predictions for {latest_time}:")
    print(current_conditions[['station_id', 'temp_c', 'rel_humidity', 'predicted_fuel_moisture', 'danger_level']])
    
    return current_conditions

if __name__ == "__main__":
    run_live_prediction()