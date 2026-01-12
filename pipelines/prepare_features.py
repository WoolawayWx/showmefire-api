import pandas as pd
import matplotlib.pyplot as plt
import os

def prepare_features():
    df = pd.read_csv('data/training_set_mo.csv')
    df['obs_time'] = pd.to_datetime(df['obs_time'])
    
    # 1. Temporal Features
    df['hour'] = df['obs_time'].dt.hour
    df['month'] = df['obs_time'].dt.month
    
    # 2. Physics Baseline (Simple EMC approximation)
    # Fuel moisture roughly follows RH / 5
    df['emc_baseline'] = df['rel_humidity'] / 5.0
    
    # 3. Save the enhanced dataset
    df.to_csv('data/ai_features_mo.csv', index=False)
    print(f"✅ Enhanced dataset saved with {len(df)} rows.")
    
    # 4. Quick Visualization for Station ASLM7
    sample = df[df['station_id'] == 'ASLM7'].sort_values('obs_time')
    plt.figure(figsize=(10,6))
    plt.plot(sample['obs_time'], sample['target_fm'], label='Actual FM %', color='red', marker='o')
    plt.plot(sample['obs_time'], sample['emc_baseline'], label='Physics Baseline (EMC)', linestyle='--', color='blue')
    plt.title('Fuel Moisture vs Physics Baseline (Station ASLM7)')
    plt.xlabel('Time')
    plt.ylabel('Moisture %')
    plt.legend()
    plt.savefig('plots/station_preview.png')

def enhance_features_with_lags():
    df = pd.read_csv('data/ai_features_mo.csv')
    df['obs_time'] = pd.to_datetime(df['obs_time'])
    
    # Sort by station and time to ensure rolling windows work correctly
    df = df.sort_values(['station_id', 'obs_time'])
    
    # Calculate rolling averages for key weather metrics
    # This represents the cumulative effect of the last few hours
    for window in [3, 6]:
        df[f'temp_mean_{window}h'] = df.groupby('station_id')['temp_c'].transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[f'rh_mean_{window}h'] = df.groupby('station_id')['rel_humidity'].transform(lambda x: x.rolling(window, min_periods=1).mean())
    
    # Add precipitation features if precip_mm column exists
    if 'precip_mm' in df.columns:
        # Rolling precipitation sums (accumulated precip over time windows)
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
        print(f"✅ Added precipitation features: precip_1h, precip_3h, precip_6h, precip_24h, hours_since_rain")
    else:
        print("⚠️  No precipitation data found in dataset. Skipping precipitation features.")

    # Save the final AI-ready dataset
    df.to_csv('data/final_training_data.csv', index=False)
    print(f"✅ Final training data created with {len(df)} rows and lagged features.")

if __name__ == "__main__":
    if not os.path.exists('plots'): os.makedirs('plots')
    prepare_features()
    enhance_features_with_lags()