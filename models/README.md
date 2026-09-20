# Model release workflow

This guide's content has moved to `model-training/docs/model_lifecycle.md`'s §7 ("Troubleshooting" — the `fuel_moisture`-specific server-side sequence: baseline capture, feature generation, beta training, shadow accumulation, evidence finalization, promotion/rollback).

Public forecasts continue to use the `fuel_moisture` stable artifact followed by the canonical rule in `core/fire_danger.py`. The direct fire-danger model is advisory only.
