"""
forecastverification.py

Machine Learning and Forecast Verification System
- Processes archived weather data for ML training
- Trains fuel moisture prediction models
- Verifies forecast accuracy
- Generates performance reports
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pickle
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns

# ML imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== DATA PROCESSING ====================

def calculate_nelson_emc(rh, temp_f):
    """Calculate equilibrium moisture content using Nelson's equations."""
    if pd.isna(rh) or pd.isna(temp_f):
        return None
    
    temp_c = (temp_f - 32) * 5/9
    
    if rh <= 10:
        emc = 0.03 + 0.2626 * rh - 0.00104 * rh * temp_c
    elif rh <= 50:
        emc = 2.22 - 0.160 * rh + 0.01660 * temp_c
    else:
        emc = 21.06 - 0.4944 * rh + 0.005565 * rh**2 - 0.00063 * rh * temp_c
    
    return max(1, min(40, emc))


def load_archived_data(filepath: Path) -> Dict:
    """Load raw archived JSON data."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"Loaded {filepath.name}")
        return data
    except Exception as e:
        logger.error(f"Error loading {filepath}: {e}")
        return None


def process_timeseries_to_dataframe(api_response: Dict) -> pd.DataFrame:
    """
    Convert Synoptic API timeseries response to pandas DataFrame.
    Creates features for machine learning.
    """
    all_records = []
    
    if not api_response.get("STATION"):
        logger.warning("No stations in API response")
        return pd.DataFrame()
    
    for station in api_response["STATION"]:
        stid = station.get("STID")
        name = station.get("NAME")
        state = station.get("STATE")
        latitude = station.get("LATITUDE")
        longitude = station.get("LONGITUDE")
        elevation = station.get("ELEVATION")
        network = station.get("MNET_SHORTNAME", "Unknown")
        
        observations = station.get("OBSERVATIONS", {})
        times = observations.get("date_time", [])
        
        # Extract time series
        fm_values = observations.get("fuel_moisture_set_1", [])
        rh_values = observations.get("relative_humidity_set_1", [])
        temp_values = observations.get("air_temp_set_1", [])
        wind_values = observations.get("wind_speed_set_1", [])
        gust_values = observations.get("wind_gust_set_1", [])
        solar_values = observations.get("solar_radiation_set_1", [])
        precip_values = observations.get("precip_accum_set_1", [])
        
        # Process each timestep
        for i in range(len(times)):
            timestamp = pd.to_datetime(times[i])
            
            record = {
                'stid': stid,
                'station_name': name,
                'state': state,
                'network': network,
                'timestamp': timestamp,
                'latitude': latitude,
                'longitude': longitude,
                'elevation': elevation,
                'hour': timestamp.hour,
                'day_of_year': timestamp.dayofyear,
                'month': timestamp.month,
            }
            
            # Current observations
            record['rh'] = rh_values[i] if i < len(rh_values) and rh_values[i] is not None else None
            record['temp'] = temp_values[i] if i < len(temp_values) and temp_values[i] is not None else None
            record['wind'] = wind_values[i] if i < len(wind_values) and wind_values[i] is not None else None
            record['wind_gust'] = gust_values[i] if i < len(gust_values) and gust_values[i] is not None else None
            record['solar'] = solar_values[i] if i < len(solar_values) and solar_values[i] is not None else None
            record['precip'] = precip_values[i] if i < len(precip_values) and precip_values[i] is not None else None
            record['fuel_moisture'] = fm_values[i] if i < len(fm_values) and fm_values[i] is not None else None
            
            # Lagged features (previous hours)
            if i > 0:
                record['prev_rh'] = rh_values[i-1] if i-1 < len(rh_values) else None
                record['prev_temp'] = temp_values[i-1] if i-1 < len(temp_values) else None
                record['prev_fm'] = fm_values[i-1] if i-1 < len(fm_values) else None
            
            if i > 2:
                record['rh_3h_avg'] = np.mean([rh_values[j] for j in range(i-3, i) if j < len(rh_values) and rh_values[j] is not None])
                record['temp_3h_avg'] = np.mean([temp_values[j] for j in range(i-3, i) if j < len(temp_values) and temp_values[j] is not None])
            
            all_records.append(record)
    
    df = pd.DataFrame(all_records)
    
    if len(df) > 0:
        # Calculate derived features
        df['temp_c'] = (df['temp'] - 32) * 5/9
        df['emc_simple'] = 3 + 0.25 * df['rh']
        df['emc_nelson'] = df.apply(
            lambda row: calculate_nelson_emc(row['rh'], row['temp']) 
            if pd.notna(row['rh']) and pd.notna(row['temp']) else None,
            axis=1
        )
        
        # Vapor pressure deficit (indicator of drying potential)
        df['vpd'] = df.apply(
            lambda row: calculate_vpd(row['temp'], row['rh']) if pd.notna(row['temp']) and pd.notna(row['rh']) else None,
            axis=1
        )
        
        # Wind chill / heat stress indicators
        df['wind_temp_interaction'] = df['wind'] * df['temp_c']
        
        logger.info(f"Processed {len(df)} observations from {df['stid'].nunique()} stations")
    
    return df


def calculate_vpd(temp_f, rh):
    """Calculate Vapor Pressure Deficit (VPD) in kPa."""
    temp_c = (temp_f - 32) * 5/9
    # Saturation vapor pressure (kPa)
    svp = 0.6108 * np.exp(17.27 * temp_c / (temp_c + 237.3))
    # Actual vapor pressure
    avp = svp * (rh / 100)
    # VPD
    vpd = svp - avp
    return vpd


def load_all_archived_data(archive_dir: str = "archive/raw_data") -> pd.DataFrame:
    """
    Load all archived data files and combine into one DataFrame.
    """
    archive_path = Path(archive_dir)
    
    if not archive_path.exists():
        logger.warning(f"Archive directory does not exist: {archive_dir}")
        return pd.DataFrame()
    
    all_dfs = []
    
    for filepath in sorted(archive_path.glob("raw_data_*.json")):
        logger.info(f"Loading {filepath.name}...")
        data = load_archived_data(filepath)
        if data:
            df = process_timeseries_to_dataframe(data)
            if not df.empty:
                all_dfs.append(df)
    
    if not all_dfs:
        logger.warning("No data loaded from archives")
        return pd.DataFrame()
    
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Remove duplicates (same station, same timestamp)
    combined_df = combined_df.drop_duplicates(subset=['stid', 'timestamp'])
    
    # Sort by timestamp
    combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
    
    logger.info(f"Combined dataset: {len(combined_df)} observations from {combined_df['stid'].nunique()} stations")
    logger.info(f"Date range: {combined_df['timestamp'].min()} to {combined_df['timestamp'].max()}")
    
    return combined_df


# ==================== MACHINE LEARNING ====================

def prepare_training_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare data for ML training.
    Returns: (features_df, targets_df) with only complete cases
    """
    # Filter to only rows with fuel moisture observations
    raws_df = df[df['fuel_moisture'].notna()].copy()
    
    if len(raws_df) == 0:
        logger.error("No fuel moisture observations found!")
        return pd.DataFrame(), pd.DataFrame()
    
    # Define feature columns
    feature_cols = [
        'rh', 'temp', 'temp_c', 'wind', 'solar', 'precip',
        'prev_rh', 'prev_temp', 'prev_fm',
        'rh_3h_avg', 'temp_3h_avg',
        'emc_simple', 'emc_nelson', 'vpd',
        'wind_temp_interaction',
        'hour', 'day_of_year', 'month',
        'latitude', 'longitude', 'elevation'
    ]
    
    # Keep only columns that exist
    available_cols = [col for col in feature_cols if col in raws_df.columns]
    
    # Drop rows with any missing features
    clean_df = raws_df[available_cols + ['fuel_moisture']].dropna()
    
    logger.info(f"Training data: {len(clean_df)} complete observations")
    logger.info(f"Features: {len(available_cols)} columns")
    
    X = clean_df[available_cols]
    y = clean_df['fuel_moisture']
    
    return X, y


def train_fuel_moisture_model(X: pd.DataFrame, y: pd.Series, model_type: str = "random_forest") -> Dict:
    """
    Train a fuel moisture prediction model.
    
    Args:
        X: Feature DataFrame
        y: Target Series (fuel moisture)
        model_type: "random_forest" or "gradient_boosting"
    
    Returns: Dict with model, scaler, metrics, and feature importance
    """
    logger.info(f"Training {model_type} model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    if model_type == "random_forest":
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
    elif model_type == "gradient_boosting":
        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    train_metrics = {
        'mae': mean_absolute_error(y_train, y_train_pred),
        'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'r2': r2_score(y_train, y_train_pred)
    }
    
    test_metrics = {
        'mae': mean_absolute_error(y_test, y_test_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'r2': r2_score(y_test, y_test_pred)
    }
    
    # Cross-validation
    cv_scores = cross_val_score(
        model, X_train_scaled, y_train, 
        cv=5, scoring='neg_mean_absolute_error'
    )
    cv_mae = -cv_scores.mean()
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info(f"Training complete!")
    logger.info(f"  Train MAE: {train_metrics['mae']:.3f}%")
    logger.info(f"  Test MAE:  {test_metrics['mae']:.3f}%")
    logger.info(f"  Test RMSE: {test_metrics['rmse']:.3f}%")
    logger.info(f"  Test R²:   {test_metrics['r2']:.3f}")
    logger.info(f"  CV MAE:    {cv_mae:.3f}%")
    
    return {
        'model': model,
        'scaler': scaler,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'cv_mae': cv_mae,
        'feature_importance': feature_importance,
        'feature_names': list(X.columns)
    }


def save_model(model_dict: Dict, filepath: Path):
    """Save trained model and associated data."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_dict, f)
    
    logger.info(f"Model saved to {filepath}")


def load_model(filepath: Path) -> Dict:
    """Load trained model."""
    if not filepath.exists():
        logger.error(f"Model file not found: {filepath}")
        return None
    
    with open(filepath, 'rb') as f:
        model_dict = pickle.load(f)
    
    logger.info(f"Model loaded from {filepath}")
    return model_dict


def predict_fuel_moisture(model_dict: Dict, features: pd.DataFrame) -> np.ndarray:
    """
    Use trained model to predict fuel moisture.
    
    Args:
        model_dict: Dictionary from train_fuel_moisture_model()
        features: DataFrame with same columns as training data
    
    Returns: Array of predicted fuel moisture values
    """
    model = model_dict['model']
    scaler = model_dict['scaler']
    
    # Ensure features are in correct order
    feature_names = model_dict['feature_names']
    features = features[feature_names]
    
    # Scale and predict
    features_scaled = scaler.transform(features)
    predictions = model.predict(features_scaled)
    
    return predictions


# ==================== FORECAST VERIFICATION ====================

def compare_simple_model_to_ml(df: pd.DataFrame, model_dict: Dict) -> pd.DataFrame:
    """
    Compare simple RH-based model to ML model on test data.
    """
    # Filter to observations with fuel moisture
    raws_df = df[df['fuel_moisture'].notna()].copy()
    
    # Prepare features
    X = raws_df[model_dict['feature_names']].dropna()
    actual_fm = raws_df.loc[X.index, 'fuel_moisture']
    
    # Predictions
    ml_pred = predict_fuel_moisture(model_dict, X)
    simple_pred = raws_df.loc[X.index, 'emc_simple']
    
    # Calculate errors
    results = pd.DataFrame({
        'timestamp': raws_df.loc[X.index, 'timestamp'],
        'stid': raws_df.loc[X.index, 'stid'],
        'actual_fm': actual_fm,
        'simple_pred': simple_pred,
        'ml_pred': ml_pred,
        'simple_error': simple_pred - actual_fm,
        'ml_error': ml_pred - actual_fm,
        'simple_abs_error': np.abs(simple_pred - actual_fm),
        'ml_abs_error': np.abs(ml_pred - actual_fm)
    })
    
    # Summary statistics
    logger.info("\n" + "="*60)
    logger.info("MODEL COMPARISON")
    logger.info("="*60)
    logger.info(f"Simple RH Model:")
    logger.info(f"  MAE:  {results['simple_abs_error'].mean():.3f}%")
    logger.info(f"  RMSE: {np.sqrt((results['simple_error']**2).mean()):.3f}%")
    logger.info(f"  Bias: {results['simple_error'].mean():.3f}%")
    
    logger.info(f"\nML Model:")
    logger.info(f"  MAE:  {results['ml_abs_error'].mean():.3f}%")
    logger.info(f"  RMSE: {np.sqrt((results['ml_error']**2).mean()):.3f}%")
    logger.info(f"  Bias: {results['ml_error'].mean():.3f}%")
    
    improvement = (results['simple_abs_error'].mean() - results['ml_abs_error'].mean())
    logger.info(f"\nImprovement: {improvement:.3f}% MAE reduction")
    logger.info("="*60 + "\n")
    
    return results


def generate_verification_report(results: pd.DataFrame, output_dir: str = "reports"):
    """
    Generate visualizations and report for forecast verification.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # 1. Error distribution comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].hist(results['simple_error'], bins=30, alpha=0.7, label='Simple Model', color='orange')
    axes[0].hist(results['ml_error'], bins=30, alpha=0.7, label='ML Model', color='blue')
    axes[0].axvline(0, color='black', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Error (Predicted - Actual) %')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Error Distribution')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # 2. Scatter plot: Actual vs Predicted
    axes[1].scatter(results['actual_fm'], results['simple_pred'], alpha=0.5, s=10, label='Simple Model', color='orange')
    axes[1].scatter(results['actual_fm'], results['ml_pred'], alpha=0.5, s=10, label='ML Model', color='blue')
    axes[1].plot([0, 30], [0, 30], 'k--', linewidth=1, label='Perfect Prediction')
    axes[1].set_xlabel('Actual Fuel Moisture (%)')
    axes[1].set_ylabel('Predicted Fuel Moisture (%)')
    axes[1].set_title('Actual vs Predicted')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / f'verification_{timestamp}.png', dpi=150, bbox_inches='tight')
    logger.info(f"Saved verification plot to {output_path / f'verification_{timestamp}.png'}")
    plt.close()
    
    # 3. Time series of errors (if timestamps available)
    if len(results) > 10:
        fig, ax = plt.subplots(figsize=(14, 6))
        results_sorted = results.sort_values('timestamp')
        ax.plot(results_sorted['timestamp'], results_sorted['simple_abs_error'], 
                alpha=0.7, label='Simple Model MAE', color='orange')
        ax.plot(results_sorted['timestamp'], results_sorted['ml_abs_error'], 
                alpha=0.7, label='ML Model MAE', color='blue')
        ax.set_xlabel('Time')
        ax.set_ylabel('Absolute Error (%)')
        ax.set_title('Forecast Error Over Time')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / f'error_timeseries_{timestamp}.png', dpi=150, bbox_inches='tight')
        logger.info(f"Saved time series plot to {output_path / f'error_timeseries_{timestamp}.png'}")
        plt.close()
    
    # 4. Generate text report
    report_file = output_path / f'verification_report_{timestamp}.txt'
    with open(report_file, 'w') as f:
        f.write("FORECAST VERIFICATION REPORT\n")
        f.write("="*60 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total observations: {len(results)}\n")
        f.write(f"Stations: {results['stid'].nunique()}\n\n")
        
        f.write("SIMPLE RH-BASED MODEL\n")
        f.write("-"*60 + "\n")
        f.write(f"Mean Absolute Error: {results['simple_abs_error'].mean():.3f}%\n")
        f.write(f"RMSE: {np.sqrt((results['simple_error']**2).mean()):.3f}%\n")
        f.write(f"Mean Bias: {results['simple_error'].mean():.3f}%\n")
        f.write(f"Std Dev: {results['simple_error'].std():.3f}%\n\n")
        
        f.write("MACHINE LEARNING MODEL\n")
        f.write("-"*60 + "\n")
        f.write(f"Mean Absolute Error: {results['ml_abs_error'].mean():.3f}%\n")
        f.write(f"RMSE: {np.sqrt((results['ml_error']**2).mean()):.3f}%\n")
        f.write(f"Mean Bias: {results['ml_error'].mean():.3f}%\n")
        f.write(f"Std Dev: {results['ml_error'].std():.3f}%\n\n")
        
        improvement = results['simple_abs_error'].mean() - results['ml_abs_error'].mean()
        pct_improvement = (improvement / results['simple_abs_error'].mean()) * 100
        f.write("IMPROVEMENT\n")
        f.write("-"*60 + "\n")
        f.write(f"MAE Reduction: {improvement:.3f}% ({pct_improvement:.1f}% improvement)\n")
    
    logger.info(f"Saved text report to {report_file}")


# ==================== MAIN WORKFLOW ====================

def run_daily_training(archive_dir: str = "archive/raw_data", 
                      model_dir: str = "models"):
    """
    Daily workflow: Load all archived data, train model, save results.
    Run this once per day to update the ML model.
    """
    logger.info("="*60)
    logger.info("DAILY MODEL TRAINING")
    logger.info("="*60)
    
    # 1. Load all archived data
    df = load_all_archived_data(archive_dir)
    
    if df.empty:
        logger.error("No data available for training!")
        return None
    
    # 2. Prepare training data
    X, y = prepare_training_data(df)
    
    if X.empty:
        logger.error("No training data prepared!")
        return None
    
    # 3. Train model
    model_dict = train_fuel_moisture_model(X, y, model_type="random_forest")
    
    # 4. Save model
    model_path = Path(model_dir) / f"fuel_moisture_model_{datetime.now().strftime('%Y%m%d')}.pkl"
    save_model(model_dict, model_path)
    
    # Also save as "latest"
    latest_path = Path(model_dir) / "fuel_moisture_model_latest.pkl"
    save_model(model_dict, latest_path)
    
    # 5. Generate verification report
    results = compare_simple_model_to_ml(df, model_dict)
    generate_verification_report(results)
    
    # 6. Print feature importance
    logger.info("\nTop 10 Most Important Features:")
    logger.info("-"*60)
    for idx, row in model_dict['feature_importance'].head(10).iterrows():
        logger.info(f"  {row['feature']:20s}: {row['importance']:.4f}")
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60 + "\n")
    
    return model_dict


def quick_model_test():
    """
    Quick test to see if model improves over simple RH formula.
    """
    logger.info("Running quick model test...")
    
    # Load data
    df = load_all_archived_data()
    
    if df.empty or len(df[df['fuel_moisture'].notna()]) < 100:
        logger.warning("Not enough data for meaningful test")
        return
    
    # Train on subset
    X, y = prepare_training_data(df)
    model_dict = train_fuel_moisture_model(X, y)
    
    # Compare
    results = compare_simple_model_to_ml(df, model_dict)
    
    logger.info("\nQuick test complete!")


# ==================== COMMAND LINE INTERFACE ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fuel Moisture ML Training and Verification')
    parser.add_argument('--train', action='store_true', help='Run daily training workflow')
    parser.add_argument('--test', action='store_true', help='Run quick model test')
    parser.add_argument('--archive-dir', default='archive/raw_data', help='Archive directory')
    parser.add_argument('--model-dir', default='models', help='Model save directory')
    
    args = parser.parse_args()
    
    if args.train:
        run_daily_training(args.archive_dir, args.model_dir)
    elif args.test:
        quick_model_test()
    else:
        # Default: run training
        logger.info("No command specified, running daily training...")
        run_daily_training(args.archive_dir, args.model_dir)