import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from pathlib import Path

# Setup paths
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
reports_dir = project_root / 'reports'
plots_dir = reports_dir / 'plots'

import seaborn as sns

def calculate_fire_danger(fm, rh, wind_ms):
    """
    Replicates Fire Danger Logic.
    Input wind is m/s, needs kts for criteria.
    """
    if pd.isna(fm) or pd.isna(rh) or pd.isna(wind_ms):
        return np.nan
        
    wind_kts = wind_ms * 1.94384
    
    # LOW
    if fm >= 15: 
        return 0 
    
    # EXTREME
    if fm < 7 and rh < 20 and wind_kts >= 30:
        return 4
    
    # CRITICAL
    if fm < 9 and rh < 25 and wind_kts >= 15:
        return 3
        
    # ELEVATED
    if fm < 9 and (rh < 45 or wind_kts >= 10):
        return 2
        
    # MODERATE
    if (9 <= fm < 15) and (rh < 50 and wind_kts >= 10):
        return 1
        
    # LOW default
    return 0

def generate_plots(csv_path):
    print(f"Reading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find report file at {csv_path}")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Use a clean style
    plt.style.use('bmh') # 'bmh' is usually available and looks decent
    
    # Filters
    df_fm = df.dropna(subset=['fm_obs', 'fm_fcst'])
    
    # ==========================================
    # 0. CATEGORICAL ACCURACY (Confusion Matrix)
    # ==========================================
    # We need to construct the classes first
    if 'wind_fcst' in df.columns:
        print("Calculating Fire Danger Classes...")
        df['class_fcst'] = df.apply(lambda x: calculate_fire_danger(x['fm_fcst'], x['rh_fcst'], x['wind_fcst']), axis=1)
        df['class_obs'] = df.apply(lambda x: calculate_fire_danger(x['fm_obs'], x['rh_obs'], x['wind_obs']), axis=1)
        
        df_class = df.dropna(subset=['class_fcst', 'class_obs'])
        
        if not df_class.empty:
            labels = [0, 1, 2, 3, 4]
            label_names = ['Low', 'Moderate', 'Elevated', 'Critical', 'Extreme']
            
            # Create confusion matrix
            cm = pd.crosstab(df_class['class_obs'], df_class['class_fcst']).reindex(index=labels, columns=labels, fill_value=0)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                        xticklabels=label_names, yticklabels=label_names)
            
            plt.title('Fire Danger Prediction Accuracy\n(Observed vs Forecast)', fontsize=14)
            plt.ylabel('Observed Reality', fontsize=12)
            plt.xlabel('Forecasted Danger', fontsize=12)
            
            # Calculate accuracy
            correct = np.trace(cm)
            total = cm.sum().sum()
            acc = correct / total * 100 if total > 0 else 0
            
            plt.figtext(0.5, 0.02, f"Overall Categorical Accuracy: {acc:.1f}%", ha="center", fontsize=12, fontweight='bold')
            
            output_path = plots_dir / '0_categorical_accuracy.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved confusion matrix to {output_path}")
            plt.close()
    
    # ==========================================
    # 1. SCATTER PLOTS (Predicted vs Observed)
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # -- Temperature --
    ax = axes[0,0]
    ax.scatter(df['temp_obs'], df['temp_fcst'], alpha=0.5, edgecolors='w', s=40)
    
    # 1:1 Line
    min_val = min(df['temp_obs'].min(), df['temp_fcst'].min()) - 2
    max_val = max(df['temp_obs'].max(), df['temp_fcst'].max()) + 2
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, label='Perfect Forecast')
    
    ax.set_title(f"Temperature (°C)\nMAE: {df['temp_error'].abs().mean():.2f}°C")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Forecast")
    ax.grid(True, alpha=0.3)
    
    # -- Relative Humidity --
    ax = axes[0,1]
    ax.scatter(df['rh_obs'], df['rh_fcst'], alpha=0.5, edgecolors='w', color='orange', s=40)
    ax.plot([0, 100], [0, 100], 'k--', alpha=0.7)
    
    ax.set_title(f"Relative Humidity (%)\nMAE: {df['rh_error'].abs().mean():.1f}%")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Forecast")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # -- Wind Speed --
    ax = axes[1,0]
    # Check if we have wind data
    if 'wind_fcst' in df.columns:
        df_wind = df.dropna(subset=['wind_obs', 'wind_fcst'])
        if not df_wind.empty:
            ax.scatter(df_wind['wind_obs'], df_wind['wind_fcst'], alpha=0.5, edgecolors='w', color='purple', s=40)
            max_wind = max(df_wind['wind_obs'].max(), df_wind['wind_fcst'].max()) + 2
            ax.plot([0, max_wind], [0, max_wind], 'k--', alpha=0.7)
            
            ax.set_title(f"Wind Speed (m/s)\nMAE: {df_wind['wind_error'].abs().mean():.2f} m/s")
            ax.set_xlabel("Observed")
            ax.set_ylabel("Forecast")
        else:
             ax.text(0.5, 0.5, "No Wind Data", ha='center')
    else:
        ax.text(0.5, 0.5, "No Wind Columns", ha='center')
    ax.grid(True, alpha=0.3)

    # -- Fuel Moisture --
    ax = axes[1,1]
    if not df_fm.empty:
        ax.scatter(df_fm['fm_obs'], df_fm['fm_fcst'], alpha=0.5, edgecolors='w', color='green', s=40)
        
        max_fm = max(df_fm['fm_obs'].max(), df_fm['fm_fcst'].max()) + 2
        ax.plot([0, max_fm], [0, max_fm], 'k--', alpha=0.7)
        
        ax.set_title(f"Fuel Moisture (%)\nMAE: {df_fm['fm_error'].abs().mean():.2f}%")
        ax.set_xlabel("Observed")
        ax.set_ylabel("Forecast")
        ax.set_xlim(0, max_fm)
        ax.set_ylim(0, max_fm)
    else:
        ax.text(0.5, 0.5, "No Fuel Moisture Data", ha='center')
    
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f"Forecast vs Observations: {df['valid_time_utc'].iloc[0][:10]}", fontsize=16)
    plt.tight_layout()
    plt.savefig(plots_dir / '1_scatter_plots.png', dpi=150)
    print(f"Saved scatter plots to {plots_dir / '1_scatter_plots.png'}")
    plt.close()


    # ==========================================
    # 2. ERROR DISTRIBUTION (Histograms)
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # -- Temperature --
    ax = axes[0,0]
    ax.hist(df['temp_error'], bins=20, edgecolor='white', alpha=0.7)
    ax.axvline(0, color='k', linestyle='--', linewidth=1.5)
    ax.axvline(df['temp_error'].mean(), color='r', linestyle='-', linewidth=1.5, label=f"Bias: {df['temp_error'].mean():.2f}")
    
    ax.set_title("Temperature Error Distribution")
    ax.set_xlabel("Error (Forecast - Obs) [°C]")
    ax.legend()
    
    # -- RH --
    ax = axes[0,1]
    ax.hist(df['rh_error'], bins=20, color='orange', edgecolor='white', alpha=0.7)
    ax.axvline(0, color='k', linestyle='--', linewidth=1.5)
    ax.axvline(df['rh_error'].mean(), color='r', linestyle='-', linewidth=1.5, label=f"Bias: {df['rh_error'].mean():.1f}")
    
    ax.set_title("RH Error Distribution")
    ax.set_xlabel("Error (Forecast - Obs) [%]")
    ax.legend()

    # -- Wind --
    ax = axes[1,0]
    if 'wind_error' in df.columns:
         df_wind = df.dropna(subset=['wind_error'])
         if not df_wind.empty:
            ax.hist(df_wind['wind_error'], bins=20, color='purple', edgecolor='white', alpha=0.7)
            ax.axvline(0, color='k', linestyle='--', linewidth=1.5)
            ax.axvline(df_wind['wind_error'].mean(), color='r', linestyle='-', linewidth=1.5, label=f"Bias: {df_wind['wind_error'].mean():.2f}")
            ax.set_title("Wind Speed Error Distribution")
            ax.set_xlabel("Error (Forecast - Obs) [m/s]")
            ax.legend()
    
    # -- FM --
    ax = axes[1,1]
    if not df_fm.empty:
        ax.hist(df_fm['fm_error'], bins=20, color='green', edgecolor='white', alpha=0.7)
        ax.axvline(0, color='k', linestyle='--', linewidth=1.5)
        ax.axvline(df_fm['fm_error'].mean(), color='r', linestyle='-', linewidth=1.5, label=f"Bias: {df_fm['fm_error'].mean():.2f}")
        
        ax.set_title("Fuel Moisture Error Distribution")
        ax.set_xlabel("Error (Forecast - Obs) [%]")
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(plots_dir / '2_error_distributions.png', dpi=150)
    print(f"Saved error histograms to {plots_dir / '2_error_distributions.png'}")
    plt.close()

    # ==========================================
    # 3. RESIDUAL PLOTS (Systematic Bias Check)
    # ==========================================
    # Do we overpredict high values and underpredict low values?
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # -- Temp --
    ax = axes[0,0]
    ax.scatter(df['temp_obs'], df['temp_error'], alpha=0.6)
    ax.axhline(0, color='k', linestyle='--')
    ax.set_title("Bias vs Observed Temperature")
    ax.set_xlabel("Observed Temp (°C)")
    ax.set_ylabel("Error (Bias)")
    
    # -- RH --
    ax = axes[0,1]
    ax.scatter(df['rh_obs'], df['rh_error'], color='orange', alpha=0.6)
    ax.axhline(0, color='k', linestyle='--')
    ax.set_title("Bias vs Observed RH")
    ax.set_xlabel("Observed RH (%)")
    
    # -- Wind --
    ax = axes[1,0]
    if 'wind_obs' in df.columns and 'wind_error' in df.columns:
        df_wind = df.dropna(subset=['wind_obs', 'wind_error'])
        if not df_wind.empty:
            ax.scatter(df_wind['wind_obs'], df_wind['wind_error'], color='purple', alpha=0.6)
            ax.axhline(0, color='k', linestyle='--')
            ax.set_title("Bias vs Observed Wind Speed")
            ax.set_xlabel("Observed Wind (m/s)")
            ax.set_ylabel("Error")

    # -- FM --
    ax = axes[1,1]
    if not df_fm.empty:
        ax.scatter(df_fm['fm_obs'], df_fm['fm_error'], color='green', alpha=0.6)
        ax.axhline(0, color='k', linestyle='--')
        ax.set_title("Bias vs Observed Fuel Moisture")
        ax.set_xlabel("Observed Fuel Moisture (%)")
        
    plt.tight_layout()
    plt.savefig(plots_dir / '3_systematic_bias.png', dpi=150)
    print(f"Saved residual plots to {plots_dir / '3_systematic_bias.png'}")
    plt.close()

if __name__ == "__main__":
    # Default path
    comparison_file = reports_dir / 'forecast_comparison_latest.csv'
    
    # Allow override via args
    if len(sys.argv) > 1:
        comparison_file = Path(sys.argv[1])
        
    generate_plots(comparison_file)
