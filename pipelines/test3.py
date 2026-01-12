import xgboost as xgb

model = xgb.XGBRegressor()
model.load_model('models/fuel_moisture_model.json')

# 'gain' = the average improvement in accuracy brought by a feature
importance = model.get_booster().get_score(importance_type='gain')

print("\n💎 FEATURE IMPORTANCE (Ranked by GAIN/QUALITY):")
print("-" * 45)
sorted_gain = sorted(importance.items(), key=lambda x: x[1], reverse=True)

for name, score in sorted_gain:
    print(f"{name:<15} : {score:.2f}")