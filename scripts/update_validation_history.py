import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import logging
import sys

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
reports_dir = project_root / 'reports'
history_file = reports_dir / 'validation_history.json'
summary_file = reports_dir / 'validation_summary.json'
plots_dir = reports_dir / 'plots'

def update_history():
    """
    Reads the latest validation_summary.json and appends it to validation_history.json
    """
    if not summary_file.exists():
        logger.error(f"No summary file found at {summary_file}. Run compare_forecasts.py first.")
        return None

    with open(summary_file, 'r') as f:
        latest = json.load(f)

    # Load existing history
    history = []
    if history_file.exists():
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Could not decode {history_file}, starting fresh.")
            history = []
    
    # Check if this timestamp already exists to avoid duplicates
    latest_ts = latest['generated_at']
    if any(entry['date'] == latest_ts for entry in history):
        logger.info(f"Entry for {latest_ts} already exists. Skipping append.")
    else:
        new_entry = {
            "date": latest_ts,
            "metrics": latest['metrics']
        }
        history.append(new_entry)
        
        # Sort by date
        history.sort(key=lambda x: x['date'])
        
        # Save back
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"Updated history with entry for {latest_ts}")
    
    return history

def calculate_rolling_stats(history):
    """
    Calculates 3, 7, and 30 day rolling averages for MAE and Bias.
    """
    # Convert to DataFrame
    data = []
    for entry in history:
        row = {'date': pd.to_datetime(entry['date'])}
        metrics = entry['metrics']
        
        for var in ['temperature', 'rh', 'fuel_moisture']:
            # Handle potential None values
            m = metrics.get(var, {})
            row[f'{var}_mae'] = m.get('mae')
            row[f'{var}_bias'] = m.get('bias')
        
        data.append(row)
    
    if not data:
        return None
        
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    
    # Calculate Rolling Averages
    windows = [3, 7, 30]
    stats_summary = {}

    for var in ['temperature', 'rh', 'fuel_moisture']:
        stats_summary[var] = {}
        for window in windows:
            if f'{var}_mae' not in df.columns:
                continue
                
            # We want the LAST row's rolling value (current status)
            # Use min_periods=1 to get values even if we don't have full history yet
            roll_mae = df[f'{var}_mae'].rolling(window=window, min_periods=1).mean().iloc[-1]
            roll_bias = df[f'{var}_bias'].rolling(window=window, min_periods=1).mean().iloc[-1]
            
            stats_summary[var][f'{window}d'] = {
                "mae": roll_mae,
                "bias": roll_bias
            }

    print("\n" + "="*40)
    print("   ROLLING PERFORMANCE TRENDS   ")
    print("="*40)
    
    for var, windows_data in stats_summary.items():
        if not windows_data: continue
        print(f"\n{var.replace('_', ' ').upper()}:")
        
        # Table Header
        print(f"  {'Period':<8} | {'MAE':<8} | {'Bias':<8}")
        print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}")
        
        # Table Rows
        print(f"  {'3-Day':<8} | {windows_data['3d']['mae']:<8.2f} | {windows_data['3d']['bias']:<8.2f}")
        print(f"  {'7-Day':<8} | {windows_data['7d']['mae']:<8.2f} | {windows_data['7d']['bias']:<8.2f}")
        print(f"  {'30-Day':<8} | {windows_data['30d']['mae']:<8.2f} | {windows_data['30d']['bias']:<8.2f}")
        
    return df

def plot_trends(df):
    """
    Generates a trend plot for Bias over time.
    """
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup plot style
    plt.style.use('bmh')
    
    vars_to_plot = [('temperature', 'Temperature (°C)'), 
                    ('rh', 'Relative Humidity (%)'), 
                    ('fuel_moisture', 'Fuel Moisture (%)')]
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    for i, (var, label) in enumerate(vars_to_plot):
        col_bias = f'{var}_bias'
        col_mae = f'{var}_mae'
        
        if col_bias not in df.columns or df[col_bias].dropna().empty:
            axes[i].text(0.5, 0.5, f"No Data for {label}", ha='center', va='center')
            continue
            
        ax = axes[i]
        
        # Zero line
        ax.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        # Plot Daily Bias dots
        ax.plot(df.index, df[col_bias], 'o', color='gray', alpha=0.4, label='Daily Bias', markersize=4)
        
        # Plot 7-day Rolling Bias
        roll_bias = df[col_bias].rolling(window=7, min_periods=1).mean()
        ax.plot(df.index, roll_bias, color='tab:red', linewidth=2.5, label='7-Day Avg Bias')
        
        # Plot 30-day Rolling Bias (smoother)
        roll_bias_30 = df[col_bias].rolling(window=30, min_periods=1).mean()
        ax.plot(df.index, roll_bias_30, color='tab:blue', linestyle='--', linewidth=1.5, label='30-Day Avg Bias')

        # Shade region between Bias and Zero to indicate magnitude? 
        # Or maybe confusing. Let's keep it clean.
        
        ax.set_ylabel("Bias (Pred - Obs)")
        ax.set_title(f"{label} Performance Trend")
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(loc='upper right')

    plt.xlabel("Date")
    plt.suptitle("Forecast Bias History (Rolling Averages)", fontsize=16, y=0.98)
    plt.tight_layout()
    
    output_path = plots_dir / '4_bias_trends.png'
    plt.savefig(output_path, dpi=150)
    logger.info(f"Saved trend plot to {output_path}")

if __name__ == "__main__":
    history = update_history()
    if history:
        df = calculate_rolling_stats(history)
        if df is not None:
            plot_trends(df)
