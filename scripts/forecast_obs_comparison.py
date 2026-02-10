#!/usr/bin/env python3
"""
Comprehensive Forecast vs Observation Comparison Script

Analyzes and visualizes forecast skill across multiple metrics:
- Temperature, Relative Humidity, Fuel Moisture, Wind Speed
- Time series plots, scatter plots, and error analysis
- Bias, MAE, RMSE, correlation, and skill score metrics
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
from scipy import stats

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

# Variable configurations
VARIABLES = {
    'temperature': {
        'forecast_col': 'temp_forecast_c',
        'observed_col': 'temp_observed_c',
        'error_col': 'temp_error_c',
        'units': '°C',
        'title': 'Temperature',
        'color': 'red'
    },
    'rh': {
        'forecast_col': 'rh_forecast_pct',
        'observed_col': 'rh_observed_pct',
        'error_col': 'rh_error_pct',
        'units': '%',
        'title': 'Relative Humidity',
        'color': 'blue'
    },
    'fuel_moisture': {
        'forecast_col': 'fuel_moisture_forecast_pct',
        'observed_col': 'fuel_moisture_observed_pct',
        'error_col': 'fuel_moisture_error_pct',
        'units': '%',
        'title': 'Fuel Moisture',
        'color': 'green'
    },
    'wind_speed': {
        'forecast_col': 'wind_forecast_ms',
        'observed_col': 'wind_observed_ms',
        'error_col': 'wind_error_ms',
        'units': 'm/s',
        'title': 'Wind Speed',
        'color': 'orange'
    }
}


class ForecastComparison:
    """Comprehensive forecast vs observation analysis"""
    
    def __init__(self, csv_path, output_dir=None):
        """
        Initialize comparison object
        
        Args:
            csv_path: Path to forecast_vs_observed_sidebyside.csv
            output_dir: Output directory for reports and plots
        """
        self.csv_path = Path(csv_path)
        self.df = pd.read_csv(csv_path)
        
        # Setup output directory
        if output_dir is None:
            output_dir = self.csv_path.parent
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parse datetime columns
        self.df['valid_time_utc'] = pd.to_datetime(self.df['valid_time_utc'])
        
    def calculate_metrics(self, var_key):
        """Calculate comprehensive metrics for a variable"""
        var = VARIABLES[var_key]
        forecast_col = var['forecast_col']
        observed_col = var['observed_col']
        error_col = var['error_col']
        
        # Remove NaN values
        mask = (self.df[forecast_col].notna()) & (self.df[observed_col].notna())
        forecast = self.df.loc[mask, forecast_col].values
        observed = self.df.loc[mask, observed_col].values
        errors = self.df.loc[mask, error_col].values
        
        if len(forecast) == 0:
            return None
        
        # Calculate metrics
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        bias = np.mean(errors)
        correlation = np.corrcoef(forecast, observed)[0, 1]
        
        # Skill score (1 - RMSE/std(obs))
        obs_std = np.std(observed)
        skill = 1 - (rmse / obs_std) if obs_std > 0 else 0
        
        return {
            'n_samples': len(forecast),
            'mean_forecast': float(np.mean(forecast)),
            'mean_observed': float(np.mean(observed)),
            'std_forecast': float(np.std(forecast)),
            'std_observed': float(np.std(observed)),
            'mae': float(mae),
            'rmse': float(rmse),
            'bias': float(bias),
            'correlation': float(correlation),
            'skill_score': float(skill),
            'min_error': float(np.min(errors)),
            'max_error': float(np.max(errors)),
            '25th_percentile_error': float(np.percentile(errors, 25)),
            '75th_percentile_error': float(np.percentile(errors, 75))
        }
    
    def generate_metrics_report(self):
        """Generate comprehensive metrics report"""
        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'csv_file': str(self.csv_path.name),
            'n_records': len(self.df),
            'variables': {}
        }
        
        for var_key in VARIABLES:
            metrics = self.calculate_metrics(var_key)
            if metrics:
                report['variables'][var_key] = metrics
        
        return report
    
    def plot_time_series(self):
        """Create time series comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Forecast vs Observed - Time Series by Station', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for idx, (var_key, ax) in enumerate(zip(VARIABLES.keys(), axes)):
            var = VARIABLES[var_key]
            forecast_col = var['forecast_col']
            observed_col = var['observed_col']
            units = var['units']
            title = var['title']
            color = var['color']
            
            # Get one station as example
            station = self.df['station_id'].iloc[0]
            station_data = self.df[self.df['station_id'] == station].sort_values('valid_time_utc')
            
            mask = station_data[forecast_col].notna() & station_data[observed_col].notna()
            station_data = station_data[mask]
            
            ax.plot(station_data['valid_time_utc'], station_data[forecast_col], 
                   marker='o', label='Forecast', color=color, alpha=0.7, linewidth=2)
            ax.plot(station_data['valid_time_utc'], station_data[observed_col], 
                   marker='s', label='Observed', color='black', alpha=0.7, linewidth=2)
            
            ax.set_title(f'{title} ({station})', fontweight='bold')
            ax.set_xlabel('Valid Time (UTC)')
            ax.set_ylabel(f'{title} ({units})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comparison_timeseries.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: comparison_timeseries.png")
        plt.close()
    
    def plot_scatter_analysis(self):
        """Create scatter plots with regression lines"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Forecast vs Observed - Scatter Analysis', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for idx, (var_key, ax) in enumerate(zip(VARIABLES.keys(), axes)):
            var = VARIABLES[var_key]
            forecast_col = var['forecast_col']
            observed_col = var['observed_col']
            units = var['units']
            title = var['title']
            color = var['color']
            
            # Get data
            mask = self.df[forecast_col].notna() & self.df[observed_col].notna()
            forecast = self.df.loc[mask, forecast_col].values
            observed = self.df.loc[mask, observed_col].values
            
            # Scatter plot
            ax.scatter(observed, forecast, alpha=0.5, s=30, color=color)
            
            # Perfect forecast line
            min_val = min(observed.min(), forecast.min())
            max_val = max(observed.max(), forecast.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect Forecast')
            
            # Regression line
            z = np.polyfit(observed, forecast, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min_val, max_val, 100)
            ax.plot(x_line, p(x_line), 'r-', linewidth=2, alpha=0.8, label='Regression')
            
            # Calculate correlation
            correlation = np.corrcoef(forecast, observed)[0, 1]
            
            ax.set_title(f'{title} (r={correlation:.3f})', fontweight='bold')
            ax.set_xlabel(f'Observed ({units})')
            ax.set_ylabel(f'Forecast ({units})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comparison_scatter.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: comparison_scatter.png")
        plt.close()
    
    def plot_error_distribution(self):
        """Create error distribution plots"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Forecast Error Distribution', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for idx, (var_key, ax) in enumerate(zip(VARIABLES.keys(), axes)):
            var = VARIABLES[var_key]
            error_col = var['error_col']
            units = var['units']
            title = var['title']
            color = var['color']
            
            # Get errors
            errors = self.df[error_col].dropna().values
            
            # Histogram
            ax.hist(errors, bins=30, color=color, alpha=0.7, edgecolor='black')
            
            # Add statistics
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            ax.axvline(mean_error, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_error:.2f}')
            ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            
            ax.set_title(f'{title} Error (σ={std_error:.2f})', fontweight='bold')
            ax.set_xlabel(f'Error ({units})')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comparison_errors.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: comparison_errors.png")
        plt.close()
    
    def plot_error_by_lead_time(self):
        """Analyze error growth with forecast lead time"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Forecast Error by Lead Time', fontsize=16, fontweight='bold')
        
        axes = axes.flatten()
        
        for idx, (var_key, ax) in enumerate(zip(VARIABLES.keys(), axes)):
            var = VARIABLES[var_key]
            error_col = var['error_col']
            units = var['units']
            title = var['title']
            color = var['color']
            
            # Group by lead hour
            lead_errors = self.df.groupby('lead_hour')[error_col].apply(
                lambda x: np.abs(x).mean()
            ).sort_index()
            
            ax.bar(lead_errors.index, lead_errors.values, color=color, alpha=0.7, edgecolor='black')
            ax.plot(lead_errors.index, lead_errors.values, marker='o', color='red', linewidth=2)
            
            ax.set_title(f'{title} - Absolute Mean Error vs Lead Time', fontweight='bold')
            ax.set_xlabel('Forecast Lead Time (hours)')
            ax.set_ylabel(f'Mean Absolute Error ({units})')
            ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comparison_lead_time.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: comparison_lead_time.png")
        plt.close()
    
    def generate_html_report(self, metrics):
        """Generate HTML report with interactive tables"""
        html = f"""
        <html>
        <head>
            <title>Forecast vs Observation Comparison Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                h1 {{ color: #333; border-bottom: 3px solid #0066cc; padding-bottom: 10px; }}
                h2 {{ color: #0066cc; margin-top: 30px; }}
                table {{ border-collapse: collapse; width: 100%; background: white; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                th {{ background: #0066cc; color: white; padding: 12px; text-align: left; }}
                td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
                tr:hover {{ background: #f9f9f9; }}
                .metric-value {{ font-weight: bold; color: #0066cc; }}
                .image {{ margin: 20px 0; text-align: center; }}
                img {{ max-width: 100%; border: 1px solid #ddd; padding: 5px; }}
            </style>
        </head>
        <body>
            <h1>Forecast vs Observation Comparison Report</h1>
            <p><strong>Generated:</strong> {metrics['generated_at']}</p>
            <p><strong>Data Source:</strong> {metrics['csv_file']}</p>
            <p><strong>Total Records:</strong> {metrics['n_records']}</p>
            
            <h2>Summary Metrics</h2>
            <table>
                <tr>
                    <th>Variable</th>
                    <th>Samples</th>
                    <th>MAE</th>
                    <th>RMSE</th>
                    <th>Bias</th>
                    <th>Correlation</th>
                    <th>Skill Score</th>
                </tr>
        """
        
        for var_key, var_metrics in metrics['variables'].items():
            var = VARIABLES[var_key]
            units = var['units']
            title = var['title']
            
            html += f"""
                <tr>
                    <td><strong>{title}</strong></td>
                    <td>{var_metrics['n_samples']}</td>
                    <td class="metric-value">{var_metrics['mae']:.3f} {units}</td>
                    <td class="metric-value">{var_metrics['rmse']:.3f} {units}</td>
                    <td class="metric-value">{var_metrics['bias']:.3f} {units}</td>
                    <td class="metric-value">{var_metrics['correlation']:.3f}</td>
                    <td class="metric-value">{var_metrics['skill_score']:.3f}</td>
                </tr>
            """
        
        html += """
            </table>
            
            <h2>Detailed Variable Analysis</h2>
        """
        
        for var_key, var_metrics in metrics['variables'].items():
            var = VARIABLES[var_key]
            units = var['units']
            title = var['title']
            
            html += f"""
            <h3>{title}</h3>
            <table>
                <tr><td><strong>Mean Forecast</strong></td><td class="metric-value">{var_metrics['mean_forecast']:.2f} {units}</td></tr>
                <tr><td><strong>Mean Observed</strong></td><td class="metric-value">{var_metrics['mean_observed']:.2f} {units}</td></tr>
                <tr><td><strong>Std Dev (Forecast)</strong></td><td class="metric-value">{var_metrics['std_forecast']:.2f} {units}</td></tr>
                <tr><td><strong>Std Dev (Observed)</strong></td><td class="metric-value">{var_metrics['std_observed']:.2f} {units}</td></tr>
                <tr><td><strong>Error 25th %ile</strong></td><td class="metric-value">{var_metrics['25th_percentile_error']:.2f} {units}</td></tr>
                <tr><td><strong>Error 75th %ile</strong></td><td class="metric-value">{var_metrics['75th_percentile_error']:.2f} {units}</td></tr>
                <tr><td><strong>Error Range</strong></td><td class="metric-value">{var_metrics['min_error']:.2f} to {var_metrics['max_error']:.2f} {units}</td></tr>
            </table>
            """
        
        html += """
            <h2>Visualizations</h2>
            
            <h3>Time Series Comparison</h3>
            <div class="image"><img src="comparison_timeseries.png" alt="Time Series Comparison"></div>
            
            <h3>Scatter Plot Analysis</h3>
            <div class="image"><img src="comparison_scatter.png" alt="Scatter Analysis"></div>
            
            <h3>Error Distribution</h3>
            <div class="image"><img src="comparison_errors.png" alt="Error Distribution"></div>
            
            <h3>Error by Lead Time</h3>
            <div class="image"><img src="comparison_lead_time.png" alt="Error by Lead Time"></div>
            
        </body>
        </html>
        """
        
        return html
    
    def generate_all_reports(self):
        """Generate all comparison reports and visualizations"""
        print(f"\n{'='*60}")
        print(f"Forecast vs Observation Comparison Analysis")
        print(f"{'='*60}\n")
        
        # Calculate metrics
        print("Calculating metrics...")
        metrics = self.generate_metrics_report()
        
        # Save JSON metrics
        metrics_path = self.output_dir / 'detailed_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"✓ Saved: detailed_metrics.json")
        
        # Generate plots
        print("\nGenerating visualizations...")
        self.plot_time_series()
        self.plot_scatter_analysis()
        self.plot_error_distribution()
        self.plot_error_by_lead_time()
        
        # Generate HTML report
        print("\nGenerating HTML report...")
        html_content = self.generate_html_report(metrics)
        html_path = self.output_dir / 'forecast_comparison_report.html'
        with open(html_path, 'w') as f:
            f.write(html_content)
        print(f"✓ Saved: forecast_comparison_report.html")
        
        print(f"\n{'='*60}")
        print(f"Analysis complete! All files saved to: {self.output_dir}")
        print(f"{'='*60}\n")
        
        return metrics


def main():
    """Main entry point"""
    # Find latest forecast comparison CSV
    reports_dir = Path(__file__).parent.parent / 'reports'
    
    # Get the most recent date directory
    date_dirs = sorted([d for d in reports_dir.iterdir() if d.is_dir() and d.name.isdigit()], reverse=True)
    
    if not date_dirs:
        print("No forecast comparison files found in reports directory")
        return
    
    latest_dir = date_dirs[0]
    csv_file = latest_dir / 'forecast_vs_observed_sidebyside.csv'
    
    if not csv_file.exists():
        print(f"Forecast comparison file not found: {csv_file}")
        return
    
    print(f"Analyzing: {csv_file}")
    
    # Run comparison
    comparison = ForecastComparison(csv_file, output_dir=latest_dir)
    comparison.generate_all_reports()


if __name__ == '__main__':
    main()
