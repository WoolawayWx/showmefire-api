"""
Monte Carlo propagation of forecast uncertainty through the canonical,
UNTOUCHED fire-danger rule (core/fire_danger.py::calculate_fire_danger).

This delivers the rule-derived half of the fire_risk_fusion product -
category probabilities, P(>= Elevated), P(>= Critical), rule stability -
with ZERO fire-occurrence labels, because it only needs the deterministic
rule plus forecast-error statistics that already exist.

Contract-mirror discipline (see api/core/contract_mirrors.json): this
module is byte-identical to model-training/risk_fusion/rule_uncertainty.py
and imports ONLY numpy/stdlib - never core.fire_danger or anything else
repo-specific. calculate_fire_danger is passed in by the CALLER as a
plain callable (see check_parity below), never imported here. That is
what makes byte-identity possible across two repos that otherwise have
no shared import path.

The vectorized category_vectorized() below is a second implementation of
the same threshold logic in core/fire_danger.py::calculate_fire_danger -
exactly the kind of duplication that produced the api/services/v5_scorer.py
drift bug this project has already seen once. check_parity() exists
specifically to catch that class of bug before a bundle ships: every
training run must call it against the real calculate_fire_danger and
refuse to proceed on any mismatch.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

RULE_MC_SEED = 811
DEFAULT_N_DRAWS = 2000
DEFAULT_RHO_FM_RH = -0.6

CATEGORY_LOW = 0
CATEGORY_MODERATE = 1
CATEGORY_ELEVATED = 2
CATEGORY_CRITICAL = 3
CATEGORY_EXTREME = 4
CATEGORY_LABELS = ("low", "moderate", "elevated", "critical", "extreme")
MISSING_CATEGORY = -1


def category_vectorized(fm: np.ndarray, rh: np.ndarray, wind_kts: np.ndarray, thresholds: Dict[str, float]) -> np.ndarray:
    """
    Vectorized reproduction of core/fire_danger.py::calculate_fire_danger.
    fm/rh/wind_kts broadcast together; returns an int array of the same
    broadcast shape, with MISSING_CATEGORY wherever any input is non-finite.
    Overwrite order matters - see module docstring's explanation of why
    moderate -> elevated -> critical -> extreme -> low(override) reproduces
    the scalar if/elif precedence exactly.
    """
    fm = np.asarray(fm, dtype="float64")
    rh = np.asarray(rh, dtype="float64")
    wind_kts = np.asarray(wind_kts, dtype="float64")
    fm, rh, wind_kts = np.broadcast_arrays(fm, rh, wind_kts)

    missing = ~(np.isfinite(fm) & np.isfinite(rh) & np.isfinite(wind_kts))

    category = np.full(fm.shape, CATEGORY_LOW, dtype="int8")

    moderate_cond = (fm < thresholds["low_fm"]) & ((rh < thresholds["moderate_rh"]) | (wind_kts >= thresholds["moderate_wind"]))
    category = np.where(moderate_cond, CATEGORY_MODERATE, category)

    elevated_cond = (fm < thresholds["elevated_fm"]) & (
        ((rh < thresholds["elevated_rh"]) & (wind_kts >= thresholds["elevated_wind"]))
        | ((rh < thresholds["elevated_very_dry_rh"]) & (wind_kts >= thresholds["elevated_very_dry_wind"]))
    )
    category = np.where(elevated_cond, CATEGORY_ELEVATED, category)

    critical_cond = (fm < thresholds["elevated_fm"]) & (rh < thresholds["critical_rh"]) & (wind_kts >= thresholds["critical_wind"])
    category = np.where(critical_cond, CATEGORY_CRITICAL, category)

    extreme_cond = (fm < thresholds["extreme_fm"]) & (rh < thresholds["extreme_rh"]) & (wind_kts >= thresholds["extreme_wind"])
    category = np.where(extreme_cond, CATEGORY_EXTREME, category)

    low_override = fm >= thresholds["low_fm"]
    category = np.where(low_override, CATEGORY_LOW, category)

    category = np.where(missing, MISSING_CATEGORY, category)
    return category


def check_parity(calculate_fire_danger: Callable, thresholds: Dict[str, float], n_samples: int = 512, seed: int = 0) -> None:
    """
    Cross-check category_vectorized against the real scalar
    calculate_fire_danger over random points AND explicit threshold-
    boundary edge cases. Raises AssertionError on any mismatch - this is
    the mandatory guard against the two implementations silently drifting.
    """
    rng = np.random.default_rng(seed)
    fm = rng.uniform(0.0, 40.0, size=n_samples)
    rh = rng.uniform(0.0, 100.0, size=n_samples)
    wind = rng.uniform(0.0, 60.0, size=n_samples)

    boundary_values = [
        thresholds["low_fm"], thresholds["elevated_fm"], thresholds["extreme_fm"],
        thresholds["moderate_rh"], thresholds["elevated_rh"], thresholds["elevated_very_dry_rh"],
        thresholds["critical_rh"], thresholds["extreme_rh"],
        thresholds["moderate_wind"], thresholds["elevated_wind"], thresholds["elevated_very_dry_wind"],
        thresholds["critical_wind"], thresholds["extreme_wind"],
    ]
    for value in boundary_values:
        fm = np.append(fm, value)
        rh = np.append(rh, value)
        wind = np.append(wind, value)
    fm = np.append(fm, [float("nan")])
    rh = np.append(rh, [50.0])
    wind = np.append(wind, [10.0])

    vectorized = category_vectorized(fm, rh, wind, thresholds)
    for i in range(len(fm)):
        expected = calculate_fire_danger(fm[i], rh[i], wind[i], missing_category=MISSING_CATEGORY)
        actual = int(vectorized[i])
        if expected != actual:
            raise AssertionError(
                f"rule_uncertainty parity check failed at index {i}: "
                f"fm={fm[i]}, rh={rh[i]}, wind={wind[i]} -> vectorized={actual}, reference={expected}"
            )


def _draw_correlated_fm_rh(fm: np.ndarray, rh: np.ndarray, fm_sigma: np.ndarray, rh_sigma: np.ndarray,
                            n_draws: int, rho: float, rng: np.random.Generator):
    """Gaussian-copula-coupled FM/RH draws, shape (n_draws, *fm.shape)."""
    shape = (n_draws,) + fm.shape
    z_fm = rng.standard_normal(shape)
    z_indep = rng.standard_normal(shape)
    z_rh = rho * z_fm + np.sqrt(max(0.0, 1.0 - rho ** 2)) * z_indep

    fm_draws = fm[None, ...] + fm_sigma[None, ...] * z_fm
    rh_draws = rh[None, ...] + rh_sigma[None, ...] * z_rh

    fm_draws = np.clip(fm_draws, 0.5, 40.0)
    rh_draws = np.clip(rh_draws, 0.0, 100.0)
    return fm_draws, rh_draws


def _draw_wind(wind_kts: np.ndarray, wind_sigma_log: np.ndarray, n_draws: int, rng: np.random.Generator):
    """Lognormal draws on the already fire-reduced wind, shape (n_draws, *wind_kts.shape)."""
    shape = (n_draws,) + wind_kts.shape
    safe_wind = np.maximum(wind_kts, 0.1)
    z = rng.standard_normal(shape)
    return safe_wind[None, ...] * np.exp(wind_sigma_log[None, ...] * z - 0.5 * wind_sigma_log[None, ...] ** 2)


def sample_category_probabilities(
    fm: np.ndarray,
    rh: np.ndarray,
    wind_kts: np.ndarray,
    fm_sigma: np.ndarray,
    rh_sigma: np.ndarray,
    wind_sigma_log: np.ndarray,
    thresholds: Dict[str, float],
    n_draws: int = DEFAULT_N_DRAWS,
    seed: int = RULE_MC_SEED,
    rho_fm_rh: float = DEFAULT_RHO_FM_RH,
    county_index: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Monte Carlo propagate FM/RH/wind uncertainty through the canonical
    rule. All inputs share a broadcastable shape; output arrays have that
    same shape (draws are reduced out).

    county_index, if given, seeds a per-county-independent RNG stream
    (default_rng(seed + county_index)) so results are order-independent
    across counties - matches the county-day aggregation this feeds.

    Returns a dict with, per input point:
        deterministic_category         - the rule evaluated at point estimates (no noise)
        category_probability_{low,moderate,elevated,critical,extreme}
        probability_at_or_above_elevated / _critical / _extreme
        modal_category                 - the most frequent sampled category
        stability                      - P(sampled category == deterministic_category)
        modal_disagrees                - bool, modal_category != deterministic_category
    """
    fm = np.asarray(fm, dtype="float64")
    rh = np.asarray(rh, dtype="float64")
    wind_kts = np.asarray(wind_kts, dtype="float64")
    fm, rh, wind_kts, fm_sigma, rh_sigma, wind_sigma_log = np.broadcast_arrays(
        fm, rh, wind_kts, np.asarray(fm_sigma, dtype="float64"),
        np.asarray(rh_sigma, dtype="float64"), np.asarray(wind_sigma_log, dtype="float64"),
    )

    seed_offset = int(np.asarray(county_index)) if county_index is not None else 0
    rng = np.random.default_rng(seed + seed_offset)

    deterministic_category = category_vectorized(fm, rh, wind_kts, thresholds)

    fm_draws, rh_draws = _draw_correlated_fm_rh(fm, rh, fm_sigma, rh_sigma, n_draws, rho_fm_rh, rng)
    wind_draws = _draw_wind(wind_kts, wind_sigma_log, n_draws, rng)
    sampled = category_vectorized(fm_draws, rh_draws, wind_draws, thresholds)

    valid = sampled >= 0
    valid_counts = np.maximum(valid.sum(axis=0), 1)

    result: Dict[str, np.ndarray] = {"deterministic_category": deterministic_category}
    for code, label in enumerate(CATEGORY_LABELS):
        result[f"category_probability_{label}"] = (sampled == code).sum(axis=0) / valid_counts

    result["probability_at_or_above_elevated"] = (
        result["category_probability_elevated"] + result["category_probability_critical"] + result["category_probability_extreme"]
    )
    result["probability_at_or_above_critical"] = result["category_probability_critical"] + result["category_probability_extreme"]
    result["probability_at_or_above_extreme"] = result["category_probability_extreme"]

    stacked_probs = np.stack([result[f"category_probability_{label}"] for label in CATEGORY_LABELS], axis=0)
    modal_category = np.argmax(stacked_probs, axis=0)
    result["modal_category"] = modal_category
    result["modal_disagrees"] = modal_category != deterministic_category

    matches = (sampled == deterministic_category[None, ...]) & valid
    result["stability"] = matches.sum(axis=0) / valid_counts

    return result
