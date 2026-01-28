import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import json
import shutil
from datetime import datetime
from pathlib import Path

def update_model_config(model_type, model_filename, performance_metrics=None):
    """Update the models/config.json with new model information and maintain history."""
    config_path = 'models/config.json'
    archive_dir = Path('models/archive')
    archive_dir.mkdir(exist_ok=True)
    
    # Load existing config or create default
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {
            "fuel_moisture": {
                "active_version": None,
                "threshold": 0.85,
                "last_updated": None,
                "history": []
            },
            "fire_danger": {
                "active_version": None,
                "history": []
            }
        }
    
    # Archive the current model if it exists
    if model_type in config and config[model_type]["active_version"]:
        current_model = config[model_type]["active_version"]
        model_path = Path('models') / current_model
        
        if model_path.exists():
            # Create archive filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_filename = f"{Path(current_model).stem}_{timestamp}{Path(current_model).suffix}"
            archive_path = archive_dir / archive_filename
            
            # Move to archive
            shutil.move(str(model_path), str(archive_path))
            print(f"📦 Archived previous model: {current_model} → {archive_filename}")
            
            # Add to history
            history_entry = {
                "version": current_model,
                "archived_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "archive_path": str(archive_path),
                "performance": config[model_type].get("performance", {}),
                "last_updated": config[model_type].get("last_updated")
            }
            
            if "history" not in config[model_type]:
                config[model_type]["history"] = []
            config[model_type]["history"].append(history_entry)
            
            # Keep only last 10 entries in history
            if len(config[model_type]["history"]) > 10:
                config[model_type]["history"] = config[model_type]["history"][-10:]
    
    # Update the specific model section with new model
    if model_type in config:
        config[model_type]["active_version"] = model_filename
        config[model_type]["last_updated"] = datetime.now().strftime("%Y-%m-%d")
        
        # Add performance metrics if provided
        if performance_metrics:
            config[model_type]["performance"] = performance_metrics
    
    # Save updated config
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Updated {config_path} with new {model_type} model: {model_filename}")
    print(f"📚 History maintained: {len(config[model_type].get('history', []))} previous versions")

def train_fuel_moisture_model():
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
    
    # --- THIS PART WAS MISSING ---
    # 7. Save the Model Artifact (temporarily with timestamp to avoid archiving conflict)
    import datetime
    temp_model_path = f'models/fuel_moisture_model_temp_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    final_model_path = 'models/fuel_moisture_model.json'
    
    model.save_model(temp_model_path)
    print(f"✅ SUCCESS: Model saved to {temp_model_path}")
    
    # 8. Update model configuration (this will archive the old model if it exists)
    performance_metrics = {
        "mae": round(mae, 3),
        "r2_score": round(r2, 3),
        "training_samples": len(X_train),
        "test_samples": len(X_test)
    }
    
    update_model_config(
        model_type="fuel_moisture",
        model_filename=os.path.basename(final_model_path),
        performance_metrics=performance_metrics
    )
    
    # 9. Now move the temp model to final location
    import shutil
    shutil.move(temp_model_path, final_model_path)
    print(f"✅ Model moved to final location: {final_model_path}")
    # -----------------------------

    # 9. Feature Importance Visualization
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(model)
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png')

if __name__ == "__main__":
    train_fuel_moisture_model()