"""Experimental continuous fire-danger scoring.

The production rule contract in :mod:`core.fire_danger` is intentionally
unchanged. This module exposes the same thresholds as continuous scores for
Testbed experiments and beta products only.
"""
from __future__ import annotations

import math
from typing import Any

from core.fire_danger import CATEGORY_LABELS, RULE_SPEC


BETA_SCORER_VERSION = "1.0.0"
THRESHOLDS = RULE_SPEC["thresholds"]


def _clip01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _below(value: float, threshold: float, span: float) -> float:
    return _clip01((threshold - value) / span)


def _above(value: float, threshold: float, span: float) -> float:
    return _clip01((value - threshold) / span)


def _and(*values: float) -> float:
    return min(values)


def _or(*values: float) -> float:
    return max(values)


def score_fire_danger(
    fuel_moisture: float,
    relative_humidity: float,
    wind_speed_knots: float,
) -> dict[str, Any]:
    """Return the official category plus an experimental continuous score.

    Each category occupies one point on a 0–4 scale. A category's fractional
    score is zero at its official threshold and approaches one as conditions
    move deeper into that category. ``AND`` uses the limiting component and
    ``OR`` uses the strongest qualifying branch.
    """
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
    score = moderate_score
    for tier, tier_score in tier_scores.items():
        if official_category >= tier:
            # Elevated/Critical/Extreme begin at 1/2/3 respectively on the
            # experimental scale, even when the threshold was just crossed.
            score = max(score, (tier - 1) + tier_score)
    return {
        "scorer_version": BETA_SCORER_VERSION,
        "official_category": official_category,
        "official_label": CATEGORY_LABELS[official_category],
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

