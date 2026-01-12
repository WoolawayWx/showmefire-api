import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os

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
    # 7. Save the Model Artifact
    model_path = 'models/fuel_moisture_model.json'
    model.save_model(model_path)
    print(f"✅ SUCCESS: Model saved to {model_path}")
    # -----------------------------

    # 8. Feature Importance Visualization
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(model)
    plt.tight_layout()
    plt.savefig('plots/feature_importance.png')

if __name__ == "__main__":
    train_fuel_moisture_model()