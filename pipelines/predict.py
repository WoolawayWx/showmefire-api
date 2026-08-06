import pandas as pd
import xgboost as xgb
import sqlite3
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.database import get_db_path
from core.fire_danger import CATEGORY_LABELS, calculate_fire_danger, meters_per_second_to_knots
from models.features import build_causal_features, validate_feature_contract
from models.versioning import load_active_model_path
from services.model_shadow import run_shadow

def get_danger_info(row):
    fm = row['predicted_fuel_moisture']
    rh = row['rel_humidity']
    wind = meters_per_second_to_knots(row['wind_speed_ms'])
    
    # ANSI Color Codes for terminal output
    RED = '\033[91m'
    ORANGE = '\033[38;5;208m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    RESET = '\033[0m'

    category = calculate_fire_danger(fm, rh, wind)
    colors = {0: GREEN, 1: BLUE, 2: YELLOW, 3: ORANGE, 4: RED}
    if category is None:
        return "UNAVAILABLE"
    return f"{colors[category]}{CATEGORY_LABELS[category].upper()}{RESET}"

def run_live_prediction():
    # 1. Load the trained (stable) model - see models/versioning.py
    try:
        model_path = load_active_model_path("fuel_moisture")
    except FileNotFoundError:
        print("❌ No stable model registered. Please run training + promote_model.py first.")
        return

    model = xgb.XGBRegressor()
    model.load_model(str(model_path))
    
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

    # 3. Feature Engineering (shared with training and strictly causal)
    df['obs_time'] = pd.to_datetime(df['snapshot_date'], utc=True)
    df = build_causal_features(df)

    # 4. Filter for only the absolute latest timestamp to show current conditions
    latest_time = df['obs_time'].max()
    current_conditions = df[df['obs_time'] == latest_time].copy()

    # 5. Predict
    features = list(model.get_booster().feature_names or [])
    if not features:
        raise ValueError("Active fuel-moisture model has no stored feature contract")
    validate_feature_contract(
        current_conditions,
        {"feature_columns": features, "feature_schema_version": "1.0.0"},
    )
    
    stable_predictions = model.predict(current_conditions[features])
    current_conditions['predicted_fuel_moisture'] = run_shadow(current_conditions, stable_predictions)

    # Apply danger level classification
    current_conditions['danger_level'] = current_conditions.apply(get_danger_info, axis=1)

    # 6. Output Results
    print(f"\n🔮 Live Predictions for {latest_time}:")
    print(current_conditions[['station_id', 'temp_c', 'rel_humidity', 'predicted_fuel_moisture', 'danger_level']])
    
    return current_conditions

if __name__ == "__main__":
    run_live_prediction()
