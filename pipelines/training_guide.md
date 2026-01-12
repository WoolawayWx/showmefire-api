# Show Me Fire ML Pipeline Guide

This guide explains how the Show Me Fire machine learning pipeline works and how to process new data to retrain the fuel moisture prediction model.

---

## 📋 System Architecture

The pipeline follows this flow:

1. **Observations** → RAWS station data (fuel moisture readings)
2. **HRRR Data** → Weather model data (temperature, humidity, wind, precipitation)
3. **Feature Engineering** → Create ML-ready features with rolling averages and precipitation metrics
4. **Model Training** → XGBoost learns the relationship between weather and fuel moisture
5. **Forecasting** → Predict fuel moisture from HRRR forecasts

---

## 🚀 Complete Pipeline Workflow

### Phase 1: Data Ingestion (One-Time Setup)

**Ingest RAWS Observations**
```bash
python3 pipelines/ingest_obs.py
```
- Reads `archive/raw_data/*.json` files
- Populates `observations` and `stations` tables
- Filters for Missouri stations only

**Index Stations (Optional - Performance Optimization)**
```bash
python3 pipelines/index_stations.py
```
- Pre-calculates HRRR grid indices for each station
- Speeds up extraction by 100x

### Phase 2: Extract Weather Features

**Step 1: Reset Snapshots (If Re-processing)**

If you've updated extraction logic or want to re-process existing data:

```bash
python3 scripts/reset_snapshots.py
```

This will:
- Delete all `weather_features` records
- Set all snapshots to `is_processed=0`
- Allow re-extraction with updated code

**Step 2: Extract HRRR Weather**

```bash
python3 pipelines/extract_hrrr.py
```

**What it does:**
- Finds all unprocessed snapshots (`is_processed=0`)
- Opens matching HRRR NetCDF files from `cache/hrrr/`
- For each station, extracts at nearest grid point:
  - Temperature (°C)
  - Relative Humidity (%)
  - Wind Speed (m/s)
  - Precipitation (mm) - **NEW**
- Saves to `weather_features` table
- Marks snapshots as processed

**Output:** `weather_features` table populated with HRRR data

### Phase 3: Generate Training Dataset

**Step 3: Create Training Set**

```bash
python3 pipelines/generate_training_set.py
```

**What it does:**
- Joins `observations` + `weather_features` + `snapshots` tables
- Matches fuel moisture readings with weather conditions by date and station
- Outputs: `data/training_set_mo.csv`

**Key SQL Logic:**
```sql
SELECT 
    o.fuel_moisture_percentage as target_fm,  -- What we predict
    wf.temp_c, wf.rel_humidity, wf.wind_speed_ms, wf.precip_mm,  -- Features
    s.lat, s.lon
FROM observations o
JOIN weather_features wf ON o.station_id = wf.station_id
JOIN snapshots snap ON wf.snapshot_id = snap.id
WHERE DATE(o.observation_date) = snap.snapshot_date
```

**Step 4: Feature Engineering**

```bash
python3 pipelines/prepare_features.py
```

**What it does:**
- Adds temporal features (hour, month)
- Calculates physics baseline (EMC)
- Calculates rolling means (3h, 6h) for temperature and humidity
- **Adds precipitation features:**
  - `precip_1h`, `precip_3h`, `precip_6h`, `precip_24h` - rolling sums
  - `hours_since_rain` - hours since last measurable rain (>0.1mm)
- Generates preview visualization: `plots/station_preview.png`
- Outputs: 
  - `data/ai_features_mo.csv` (intermediate)
  - `data/final_training_data.csv` (final ML-ready dataset)

### Phase 4: Model Training

**Step 5: Train XGBoost Model**

```bash
python3 pipelines/train_model.py
```

**What it does:**
- Loads `data/final_training_data.csv`
- Automatically includes precipitation features if available
- Trains XGBoost model with 200 trees
- Evaluates performance (MAE, R²)
- Saves model: `models/fuel_moisture_model.json`
- Creates feature importance plot: `plots/feature_importance.png`

**Expected Output:**
```
✅ Including precipitation features: ['precip_1h', 'precip_3h', 'precip_6h', 'precip_24h', 'hours_since_rain']
📈 Model Performance:
   - Mean Absolute Error: 2.50%
   - R-Squared Score: 0.85
✅ SUCCESS: Model saved to models/fuel_moisture_model.json
```

### Phase 5: Update Forecast Script

**Step 6: Enable Precipitation Features in Production**

After retraining with precipitation, update `forecast/forecastedfiredanger.py`:

1. Comment out the old FEATURES list
2. Uncomment the extended FEATURES list with precipitation

```python
# Extended features list for models trained with precipitation
FEATURES = [
    'temp_c', 'rel_humidity', 'wind_speed_ms', 'hour', 'month',
    'emc_baseline', 'temp_mean_3h', 'rh_mean_3h', 'temp_mean_6h', 'rh_mean_6h',
    'precip_1h', 'precip_3h', 'precip_6h', 'precip_24h', 'hours_since_rain'
]
```

### Phase 6: Testing & Validation

**Test Live Predictions**

```bash
python3 pipelines/predict.py
```

Runs predictions on the latest weather data from the database.

**Verify Training Data Quality**

```bash
python3 -c "import pandas as pd; df = pd.read_csv('data/final_training_data.csv'); \
print(f'Total Rows: {len(df)}'); \
print(f'Date Range: {df[\"obs_time\"].min()} to {df[\"obs_time\"].max()}'); \
print(f'Stations: {df[\"station_id\"].nunique()}'); \
print('\\nFeature Correlations:'); \
print(df[['temp_c', 'rel_humidity', 'target_fm']].corr())"
```

---

## 📊 Quick Reference: Pipeline Commands

| Step | Command | Output |
|------|---------|--------|
| **1. Ingest Observations** | `python3 pipelines/ingest_obs.py` | Database: `observations`, `stations` |
| **2. Reset (if needed)** | `python3 scripts/reset_snapshots.py` | Clears `weather_features`, resets flags |
| **3. Extract HRRR** | `python3 pipelines/extract_hrrr.py` | Database: `weather_features` |
| **4. Generate Training Set** | `python3 pipelines/generate_training_set.py` | `data/training_set_mo.csv` |
| **5. Feature Engineering** | `python3 pipelines/prepare_features.py` | `data/final_training_data.csv` + plot |
| **6. Train Model** | `python3 pipelines/train_model.py` | `models/fuel_moisture_model.json` |
| **7. Test Predictions** | `python3 pipelines/predict.py` | Live predictions |

---

## 🔍 Troubleshooting

**"No matching rows found" in training set**
- Check that snapshots exist for dates with observations
- Ensure `extract_hrrr.py` ran successfully
- Verify HRRR files exist in `cache/hrrr/` for those dates

**"No precipitation features found"**
- Re-run `extract_hrrr.py` to get precipitation data
- Check that HRRR files contain APCP variable
- Run `python3 scripts/reset_snapshots.py` first if needed

**"Model feature mismatch"**
- Model was trained with different features than you're using
- Retrain the model after updating training data
- Update FEATURES list in `forecastedfiredanger.py`

**"No unprocessed snapshots"**
- All snapshots already processed
- Add new snapshots to database, or
- Run `python3 scripts/reset_snapshots.py` to re-process

---

## 💡 Tips

- **Always run the full pipeline** after adding new snapshots
- **The CSV files are generated outputs**, not inputs - they're created from the database
- **Precipitation features improve accuracy** during/after rain events
- **Feature importance plots** show which variables matter most to the model
- **Preview graphs** help verify data quality before training

---

## 📁 File Structure

```
api/
├── pipelines/
│   ├── ingest_obs.py           # Load RAWS observations
│   ├── extract_hrrr.py          # Extract weather from HRRR
│   ├── generate_training_set.py # Join obs + weather
│   ├── prepare_features.py      # Feature engineering
│   ├── train_model.py           # Train XGBoost
│   └── predict.py               # Test predictions
├── scripts/
│   └── reset_snapshots.py       # Reset processing flags
├── data/
│   ├── training_set_mo.csv      # Raw joined data
│   ├── ai_features_mo.csv       # With temporal features
│   └── final_training_data.csv  # ML-ready dataset
├── models/
│   └── fuel_moisture_model.json # Trained model
└── plots/
    ├── station_preview.png      # Data quality viz
    └── feature_importance.png   # Model insights
```
