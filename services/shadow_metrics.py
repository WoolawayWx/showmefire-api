"""Shared vs-stable comparison helpers for shadow/advisory model families
that have a real stable counterpart to compare categorical output against.

Extracted from the copy-pasted category-disagreement block duplicated
between services/v4_shadow.py::record_predictions and
services/v5_shadow.py::record_predictions - same field names, same values,
purely internal refactor. Only usable for families whose predicted
quantity has the same units/meaning as core.fire_danger's category system
(fuel-moisture-derived danger category) - fire_weather_index (a new
continuous score with no rule-based equivalent) and risk_fusion_glm
(predicts fire count/probability, a quantity the live rule doesn't produce
at all) have no stable counterpart and are intentionally NOT handled here;
their diagnostics() stay operational-health-only.
"""
from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np

from core.fire_danger import calculate_fire_danger


def categories_for(fm: Iterable[float], rh: Iterable[float], wind_kts: Iterable[float]) -> List[Optional[int]]:
    """Vectorized-by-zip wrapper around calculate_fire_danger, matching the
    exact call shape already inlined in v4_shadow.py/v5_shadow.py."""
    return [calculate_fire_danger(f, r, w) for f, r, w in zip(fm, rh, wind_kts)]


def category_disagreement_summary(stable_category: List[Optional[int]], beta_category: List[Optional[int]]) -> dict:
    """Row-count summary of stable-vs-beta categorical disagreement.

    `category_disagreements` preserves the exact count the original
    inlined `sum(a != b for a, b in zip(...))` produced (a None on either
    side counts as a disagreement there too) - kept unchanged since
    existing evidence files already carry this field under this exact
    semantics. `disagreement_rate` is a NEW field with no prior behavior
    to preserve, so it's computed correctly from scratch: disagreements
    counted only among rows where both sides are actually available,
    divided by that same comparable-row count - not the raw
    `category_disagreements` count above, which double-counts
    unavailability as disagreement and would give a misleading rate.
    """
    disagreements = sum(a != b for a, b in zip(stable_category, beta_category))
    unavailable = sum(a is None or b is None for a, b in zip(stable_category, beta_category))
    comparable_disagreements = sum(
        a != b for a, b in zip(stable_category, beta_category) if a is not None and b is not None
    )
    total = len(stable_category)
    comparable = total - unavailable
    return {
        "category_disagreements": disagreements,
        "unavailable": unavailable,
        "comparable_rows": comparable,
        "disagreement_rate": round(comparable_disagreements / comparable, 4) if comparable else None,
    }


def continuous_error_summary(stable_values: Iterable[float], beta_values: Iterable[float]) -> dict:
    """MAE/bias between two aligned continuous arrays (e.g. a real observed
    fuel moisture value vs a beta prediction). `stable_values` is really
    just "the reference/ground-truth series" here - used by
    shadow_observation_scoring.py to compare a prediction against a real
    observation, not only against another model's prediction."""
    stable = np.asarray(list(stable_values), dtype=float)
    beta = np.asarray(list(beta_values), dtype=float)
    mask = np.isfinite(stable) & np.isfinite(beta)
    if not mask.any():
        return {"mae": None, "bias": None, "n": 0}
    diff = beta[mask] - stable[mask]
    return {"mae": round(float(np.mean(np.abs(diff))), 4), "bias": round(float(np.mean(diff)), 4), "n": int(mask.sum())}
