from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import mean
from typing import Iterable


@dataclass(frozen=True)
class PromotionDecision:
    eligible: bool
    gates: dict[str, bool]
    reasons: tuple[str, ...]


def mae(pairs: Iterable[tuple[float, float]]) -> float | None:
    errors = [abs(float(forecast) - float(observed)) for forecast, observed in pairs if isfinite(float(forecast)) and isfinite(float(observed))]
    return mean(errors) if errors else None


def bias(pairs: Iterable[tuple[float, float]]) -> float | None:
    errors = [float(forecast) - float(observed) for forecast, observed in pairs if isfinite(float(forecast)) and isfinite(float(observed))]
    return mean(errors) if errors else None


def pinball_loss(rows: Iterable[tuple[float, float, float]]) -> float | None:
    losses = []
    for forecast, observed, quantile in rows:
        error = float(observed) - float(forecast)
        q = float(quantile)
        losses.append(max(q * error, (q - 1) * error))
    return mean(losses) if losses else None


def brier_score(pairs: Iterable[tuple[float, bool]]) -> float | None:
    losses = [(float(probability) / 100.0 - float(outcome)) ** 2 for probability, outcome in pairs if isfinite(float(probability))]
    return mean(losses) if losses else None


def interval_coverage(rows: Iterable[tuple[float, float, float]]) -> float | None:
    values = [float(low) <= float(observed) <= float(high) for low, high, observed in rows]
    return 100.0 * mean(values) if values else None


def confidence_reliability(rows: Iterable[tuple[int, bool]], bin_width: int = 10) -> list[dict]:
    bins: dict[int, list[bool]] = {}
    for confidence, correct in rows:
        lower = min(90, max(0, int(confidence) // bin_width * bin_width))
        bins.setdefault(lower, []).append(bool(correct))
    return [
        {"range": [lower, lower + bin_width - 1 if lower < 90 else 100], "count": len(values), "forecastConfidence": lower + bin_width / 2, "observedAccuracy": 100 * mean(values)}
        for lower, values in sorted(bins.items())
    ]


def evaluate_rrfs_refs_promotion(run_metrics: list[dict]) -> PromotionDecision:
    """Apply the documented shadow-to-stable gates to consecutive 12Z runs."""
    recent = run_metrics[-30:]
    complete = len(recent) == 30 and all(run.get("complete") and run.get("cycle_hour") == 12 for run in recent)
    availability = complete and mean(float(run.get("required_hour_availability", 0)) for run in recent) >= 0.95
    core_variables = ("temperature", "relative_humidity", "wind", "fuel_moisture")
    no_regression = complete and all(
        mean(float(run["mae"][variable]) for run in recent) <= mean(float(run["baseline_mae"][variable]) for run in recent) * 1.02
        for variable in core_variables
    )
    high_risk_improvement = complete and all(
        mean(float(run["elevated_plus_mae"][variable]) for run in recent) <= mean(float(run["baseline_elevated_plus_mae"][variable]) for run in recent) * 0.95
        for variable in ("relative_humidity", "wind", "fuel_moisture")
    )
    brier_skill = complete and mean(float(run.get("brier_skill", -1)) for run in recent) > 0
    false_negative = complete and all(float(run.get("max_false_negative_increase", 99)) <= 1 for run in recent)
    gates = {
        "thirtyConsecutiveComplete12ZRuns": complete,
        "requiredHourAvailabilityAtLeast95Percent": availability,
        "coreMaeRegressionNoMoreThan2Percent": no_regression,
        "elevatedPlusRhWindFuelImprovementAtLeast5Percent": high_risk_improvement,
        "positiveBrierSkill": brier_skill,
        "noGreaterThanOneCategoryFalseNegativeIncrease": false_negative,
    }
    return PromotionDecision(all(gates.values()), gates, tuple(name for name, passed in gates.items() if not passed))

