"""Experimental continuous fire-danger scoring.

The production rule contract in :mod:`core.fire_danger` is intentionally
unchanged - `official_category` below is a byte-for-byte copy of its
branching logic, kept only as the reference to compare against.

Two things differ from that hard-cutoff rule, by design:

1. Each individual comparison (fm/rh/wind vs. its threshold) is a ramp
   centered on the threshold instead of a boolean - membership is 0.5
   exactly at the threshold, approaching 1.0 as conditions get more
   severe and 0.0 as they get milder within `span` units either side.
2. AND still takes the weakest (minimum) component, so a tier's required
   conditions all still have to be genuinely met - AND-combinations flip
   at the identical boundary as production, just through a ramp instead
   of a jump. OR uses a probabilistic sum (the standard fuzzy-logic OR)
   instead of a plain max, so two branches that are each only partially
   satisfied can combine into a genuine qualification. Production's hard
   OR needs one specific branch fully true; this lets several
   simultaneously-marginal paths - e.g. slightly-too-humid AND
   slightly-too-calm on one branch, slightly-too-dry AND slightly-too-
   windy on the other - add up instead of being individually dismissed.

`beta_category` is the highest tier whose combined score clears 0.5 under
this softened combination, so it is genuinely different from
`official_category` at OR boundaries, not just a smoothed restatement of
the same decision.
"""
from __future__ import annotations

import math
from typing import Any

from core.fire_danger import CATEGORY_LABELS, RULE_SPEC


BETA_SCORER_VERSION = "2.0.0"
THRESHOLDS = RULE_SPEC["thresholds"]


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _below(value: float, threshold: float, span: float) -> float:
    """0.5 at the threshold; 1.0 a further span/2 below it; 0.0 a further span/2 above it."""
    return _clip01((threshold - value) / span + 0.5)


def _above(value: float, threshold: float, span: float) -> float:
    """0.5 at the threshold; 1.0 a further span/2 above it; 0.0 a further span/2 below it."""
    return _clip01((value - threshold) / span + 0.5)


def _and(*values: float) -> float:
    return min(values)


def _or(*values: float) -> float:
    """Probabilistic sum (fuzzy T-conorm), not max.

    Several partially-satisfied branches combine into a genuine
    qualification even when no single branch is alone enough - unlike
    max(), which only ever reflects whichever branch is most advanced.
    """
    combined = 0.0
    for value in values:
        combined = combined + value - combined * value
    return _clip01(combined)


def score_fire_danger(
    fuel_moisture: float,
    relative_humidity: float,
    wind_speed_knots: float,
) -> dict[str, Any]:
    """Return the official category plus a softened experimental category and score."""
    values = (fuel_moisture, relative_humidity, wind_speed_knots)
    if any(value is None or not math.isfinite(float(value)) for value in values):
        raise ValueError("fuel_moisture, relative_humidity, and wind_speed_knots must be finite")

    fm, rh, wind = map(float, values)
    moderate_score = _and(
        _below(fm, THRESHOLDS["low_fm"], 6.0),
        _or(
            _below(rh, THRESHOLDS["moderate_rh"], 20.0),
            _above(wind, THRESHOLDS["moderate_wind"], 10.0),
        ),
    )
    elevated_branches = {
        "rh35_wind12": _and(
            _below(rh, THRESHOLDS["elevated_rh"], 10.0),
            _above(wind, THRESHOLDS["elevated_wind"], 8.0),
        ),
        "rh25_wind5": _and(
            _below(rh, THRESHOLDS["elevated_very_dry_rh"], 10.0),
            _above(wind, THRESHOLDS["elevated_very_dry_wind"], 10.0),
        ),
    }
    elevated_score = _and(
        _below(fm, THRESHOLDS["elevated_fm"], 2.0),
        _or(*elevated_branches.values()),
    )
    critical_score = _and(
        _below(fm, THRESHOLDS["elevated_fm"], 2.0),
        _below(rh, THRESHOLDS["critical_rh"], 10.0),
        _above(wind, THRESHOLDS["critical_wind"], 10.0),
    )
    extreme_score = _and(
        _below(fm, THRESHOLDS["extreme_fm"], 2.0),
        _below(rh, THRESHOLDS["extreme_rh"], 10.0),
        _above(wind, THRESHOLDS["extreme_wind"], 15.0),
    )

    # official_category: unchanged canonical hard-threshold decision, kept
    # only as the reference to compare beta_category against.
    if fm >= THRESHOLDS["low_fm"]:
        official_category = 0
    elif fm < THRESHOLDS["extreme_fm"] and rh < THRESHOLDS["extreme_rh"] and wind >= THRESHOLDS["extreme_wind"]:
        official_category = 4
    elif fm < THRESHOLDS["elevated_fm"] and rh < THRESHOLDS["critical_rh"] and wind >= THRESHOLDS["critical_wind"]:
        official_category = 3
    elif fm < THRESHOLDS["elevated_fm"] and (
        (rh < THRESHOLDS["elevated_rh"] and wind >= THRESHOLDS["elevated_wind"])
        or (rh < THRESHOLDS["elevated_very_dry_rh"] and wind >= THRESHOLDS["elevated_very_dry_wind"])
    ):
        official_category = 2
    elif fm < THRESHOLDS["low_fm"] and (
        rh < THRESHOLDS["moderate_rh"] or wind >= THRESHOLDS["moderate_wind"]
    ):
        official_category = 1
    else:
        official_category = 0

    tier_scores = {1: moderate_score, 2: elevated_score, 3: critical_score, 4: extreme_score}
    beta_category = 0
    score = 0.0
    for tier, tier_score in tier_scores.items():
        if tier_score >= 0.5:
            beta_category = tier
            # Rescale the qualifying half of the ramp (0.5-1.0) onto the
            # tier's own point on the 0-4 scale (0.0-1.0).
            score = max(score, (tier - 1) + 2 * (tier_score - 0.5))

    return {
        "scorer_version": BETA_SCORER_VERSION,
        "official_category": official_category,
        "official_label": CATEGORY_LABELS[official_category],
        "beta_category": beta_category,
        "beta_label": CATEGORY_LABELS[beta_category],
        "score": round(min(4.0, score), 4),
        "criteria": {
            "moderate": round(moderate_score, 4),
            "elevated": round(elevated_score, 4),
            "elevated_branches": {
                key: round(value, 4) for key, value in elevated_branches.items()
            },
            "critical": round(critical_score, 4),
            "extreme": round(extreme_score, 4),
        },
    }
