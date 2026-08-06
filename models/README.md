# Model release workflow

Public forecasts continue to use the `fuel_moisture` stable artifact followed by
the canonical rule in `core/fire_danger.py`. The direct fire-danger model is
advisory only.

1. Capture the current rollback baseline:
   `python scripts/capture_model_baseline.py`
2. Generate causal pairs and features:
   `python pipelines/generate_training_set.py && python pipelines/prepare_features.py`
3. Train a beta candidate:
   `python pipelines/train_model.py`
4. Leave production on stable while shadow records accumulate in
   `logs/model_shadow.jsonl`.
5. After at least 30 days and an Elevated-or-higher sample, attach evidence:
   `python scripts/finalize_shadow_validation.py`
6. Promote only if all gates pass:
   `python pipelines/promote_model.py --model fuel_moisture`
7. Roll back explicitly when needed:
   `python pipelines/promote_model.py --model fuel_moisture --rollback`

The daily validation pipeline runs the seven-day post-promotion monitor and
automatically rolls back material live metric regressions.
