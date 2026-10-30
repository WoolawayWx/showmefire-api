"""
Canonical rule plumbing for the outlook: categorize + seasonal dampening + the rule/seasonal state
that goes into the manifest.

Same functions as the day-1 ensemble product (services/ensemble_fire_danger/runtime.py::rule_functions),
but the seasonal state is explicit. `gdd_accum` can be passed in so a backfill/verification run uses the
GDD of its own date instead of the wall clock. One statewide green factor is held for the whole seven
days (GDD state is not gridded and changes slowly); that approximation is recorded in the manifest.
A missing GDD means NO dampening (green factor 0), and the manifest says so rather than hiding it.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

_UNSET = object()


def rule_functions(gdd_accum=_UNSET) -> Tuple[Callable, Callable, dict]:
    from core.fire_danger import (MAX_DEMOTION_FRACTION, RULE_SPEC, RULE_SPEC_SHA256, _green_factor,
                                  calculate_fire_danger)
    from services.ensemble_fire_danger import core as ens_core
    from services.rule_uncertainty import category_vectorized, check_parity

    thresholds = RULE_SPEC["thresholds"]
    check_parity(calculate_fire_danger, thresholds)  # refuse to run on vectorized/scalar drift
    if gdd_accum is _UNSET:
        try:
            from services.seasonal_fuel_state import current_gdd_accum
            gdd_accum = current_gdd_accum()
        except Exception:
            gdd_accum = None
    green = _green_factor(gdd_accum)

    def categorize(fm, rh, wind_kts):
        return category_vectorized(fm, rh, wind_kts, thresholds)

    def dampen(category, fm, rh, wind_kts):
        return ens_core.seasonal_dampening_vectorized(category, fm, rh, wind_kts, green, thresholds,
                                                      MAX_DEMOTION_FRACTION)

    info = {
        "rule_spec_version": RULE_SPEC["version"], "rule_spec_sha256": RULE_SPEC_SHA256,
        "gdd_accum_since_mar1": None if gdd_accum is None else float(gdd_accum),
        "green_factor": float(green), "max_demotion_fraction": MAX_DEMOTION_FRACTION,
        "seasonal_state": "missing_no_dampening" if gdd_accum is None else "applied",
        "seasonal_state_scope": "single statewide green factor held for all 168 hours",
    }
    return categorize, dampen, info
