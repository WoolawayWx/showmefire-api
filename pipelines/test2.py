import xgboost as xgb

# 1. Load the model
model = xgb.XGBRegressor()
model.load_model('models/fuel_moisture_model.json')

# 2. Get the scores
# 'weight' = how many times a feature was used to split data
importance = model.get_booster().get_score(importance_type='weight')

print("\n🔥 FEATURE IMPORTANCE (Ranked by Model Influence):")
print("-" * 45)

# 3. Sort by score descending
sorted_scores = sorted(importance.items(), key=lambda x: x[1], reverse=True)

for name, score in sorted_scores:
    print(f"{name:<15} : {int(score):>5} points")