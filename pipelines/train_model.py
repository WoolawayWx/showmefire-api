import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.versioning import register_trained_model

def train_fuel_moisture_model(channel="beta", bump="patch"):
    # 1. Ensure folders exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    # 2. Load Data
    df = pd.read_csv('data/final_training_data.csv')
    
    # 3. Define Features and Target
    features = [
        'temp_c', 'rel_humidity', 'wind_speed_ms', 'lat', 'lon', 
        'hour', 'month', 'emc_baseline', 
        'temp_mean_3h', 'rh_mean_3h', 'temp_mean_6h', 'rh_mean_6h'
    ]
    
    # Base features (always included)
    features_to_use = [
        'temp_c', 'rel_humidity', 'wind_speed_ms', 
        'hour', 'month', 'emc_baseline', 
        'temp_mean_3h', 'rh_mean_3h', 'temp_mean_6h', 'rh_mean_6h'
    ]
    
    # Add precipitation features if they exist in the dataset
    precip_features = ['precip_1h', 'precip_3h', 'precip_6h', 'precip_24h', 'hours_since_rain']
    available_precip_features = [f for f in precip_features if f in df.columns]
    
    if available_precip_features:
        features_to_use.extend(available_precip_features)
        print(f"✅ Including precipitation features: {available_precip_features}")
    else:
        print("⚠️  No precipitation features found in training data.")
    
    X = df[features_to_use]
    y = df['target_fm']
    
    # 4. Split into Train/Test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Initialize and Train XGBoost
    model = xgb.XGBRegressor(
        n_estimators=200,      # Slightly more trees for stability
        learning_rate=0.1,
        max_depth=5,
        objective='reg:squarederror'
    )
    model.fit(X_train, y_train)
    
    # 6. Evaluate Performance
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"📈 Model Performance:")
    print(f"   - Mean Absolute Error: {mae:.2f}%")
    print(f"   - R-Squared Score: {r2:.2f}")
    
    # 7. Save the model artifact to a scratch path, then hand it to the version registry
    scratch_model_path = Path('models') / f'.scratch_fuel_moisture_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    model.save_model(str(scratch_model_path))

    performance_metrics = {
        "mae": round(mae, 3),
        "r2_score": round(r2, 3),
        "training_samples": len(X_train),
        "test_samples": len(X_test)
    }

    version = register_trained_model(
        model_type="fuel_moisture",
        source_path=scratch_model_path,
        performance=performance_metrics,
        bump=bump,
        channel=channel,
    )
    scratch_model_path.unlink()
    print(f"✅ Registered fuel_moisture model as {channel} version {version}")
    if channel == "beta":
        print(f"   Promote it with: python pipelines/promote_model.py --model fuel_moisture --version {version}")

    # 8. Feature Importance Visualization
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(model)
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the fuel moisture model")
    parser.add_argument("--channel", choices=["beta", "stable"], default="beta",
                         help="Channel to register the trained model under (default: beta)")
    parser.add_argument("--bump", choices=["major", "minor", "patch"], default="patch",
                         help="Version segment to bump (default: patch)")
    args = parser.parse_args()

    train_fuel_moisture_model(channel=args.channel, bump=args.bump)