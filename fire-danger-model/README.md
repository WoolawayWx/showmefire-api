# Fire Danger Model (Standalone)

This folder contains a standalone fire danger modeling workflow.

Scope:
- Trains and evaluates a standalone fire danger regression model.
- Maps continuous predictions to categories with calibrated score thresholds.
- Does not integrate with production forecast scripts, scheduler, or API routes.

## Workflow

Single command (prep -> train -> evaluate):

```bash
python3 api/fire-danger-model/run_pipeline.py
```

With custom train params:

```bash
python3 api/fire-danger-model/run_pipeline.py --n-estimators 400 --learning-rate 0.03 --max-depth 6
```

1. Prepare data

```bash
python3 api/fire-danger-model/data_prep.py
```

2. Train model

```bash
python3 api/fire-danger-model/train.py
```

3. Evaluate model

```bash
python3 api/fire-danger-model/evaluate.py
```

4. Run inference

```bash
python3 api/fire-danger-model/infer.py --input-file api/fire-danger-model/data/prepared_test.csv --output-file api/fire-danger-model/reports/sample_predictions.csv
```

## Primary Verification Metric

- Fixed-label Macro F1 on mapped fire-danger categories.

## Evaluation Gates

- Class-support gate: Elevated/Critical/Extreme must each meet minimum test support.
- Primary gate: Macro F1 must exceed configured threshold and class-support gate must pass.
- Baseline gate: Model Macro F1 must not underperform rule baseline beyond allowed degradation.

Gate outcomes are written to evaluation JSON and summarized by run_pipeline.py.

## Leakage Guardrail

- target_fm is intentionally excluded from model feature columns.
- target_fm is used only to create standalone labels for offline experimentation.

## Notes

- Source data default: api/data/final_training_data.csv
- Model artifacts are written under api/fire-danger-model/models
- Reports are written under api/fire-danger-model/reports
- Category logic mirrors current rule-based criteria for comparability
