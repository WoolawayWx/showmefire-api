"""
Numeric core of the ensemble fire danger product: members -> hourly
categories -> peak category -> exceedance probabilities -> calibrated
probabilities -> categorical forecast.

Contract-mirror discipline (see api/core/contract_mirrors.json, pair
"ensemble_fire_danger_core"): byte-identical to
model-training/ensemble_fire_danger/core.py, and imports only
numpy/scipy/stdlib. The fire-danger rule itself is NEVER reimplemented
here - callers pass their repo's parity-checked vectorized rule
(rule_uncertainty.category_vectorized, itself a mirror pair) as
`categorize`, and the seasonal-dampening constants from their own
core/fire_danger.py. That keeps exactly one definition of the rule per
repo, and api/tests/test_ensemble_fire_danger.py asserts the vectorized
dampening below matches core.fire_danger.seasonal_dampening_adjustment
cell for cell.

Two probability flavours are produced, on purpose:

- POINT probability: weighted fraction of members whose peak category at
  the grid cell is >= k. The categorical forecast is derived from this
  (highest k whose calibrated point probability >= tau_k; tau_k = 0.5 is
  exactly the weighted ensemble median category).
- NEIGHBORHOOD probability (what the four confidence maps show): weighted
  fraction of members reaching >= k anywhere within `radius` of the cell -
  the same "neighborhood maximum ensemble probability" idea HREF and SPC
  use, because a ~10-member convection-allowing ensemble has almost no
  skill at placing a threshold crossing on an exact 3km cell, but real
  skill at "near here".

Both are smoothed with the same sigma=1.5 gaussian the public map uses,
then passed through an optional per-category isotonic calibration (knots
fitted offline by model-training/ensemble_fire_danger/fit.py; identity
when no bundle exists), then forced monotone across k.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

CATEGORY_IDS = (1, 2, 3, 4)  # >= Moderate, >= Elevated, >= Critical, >= Extreme
CATEGORY_KEYS = {1: "moderate", 2: "elevated", 3: "critical", 4: "extreme"}
MISSING = -1
KNOTS_TO_MPS = 0.514444
MPS_TO_KNOTS = 1.9438444924406


# --- seasonal dampening (vectorized) ----------------------------------------

def _clip01(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 1.0)


def severity_fraction_vectorized(category: np.ndarray, fm: np.ndarray, rh: np.ndarray, wind_kts: np.ndarray,
                                 thresholds: Dict[str, float]) -> np.ndarray:
    """Vectorized core/fire_danger.py::_severity_fraction (1.0 for Low/Moderate/missing)."""
    t = thresholds
    fraction = np.ones(np.shape(category), dtype="float64")

    wind_floor = min(t["elevated_wind"], t["elevated_very_dry_wind"])
    elevated = (_clip01((t["elevated_rh"] - rh) / (t["elevated_rh"] - t["critical_rh"]))
                + _clip01((wind_kts - wind_floor) / (t["critical_wind"] - wind_floor))
                + _clip01((t["elevated_fm"] - fm) / t["elevated_fm"])) / 3.0
    critical = (_clip01((t["critical_rh"] - rh) / (t["critical_rh"] - t["extreme_rh"]))
                + _clip01((wind_kts - t["critical_wind"]) / (t["extreme_wind"] - t["critical_wind"]))
                + _clip01((t["elevated_fm"] - fm) / (t["elevated_fm"] - t["extreme_fm"]))) / 3.0
    extreme = (_clip01((t["extreme_rh"] - rh) / t["extreme_rh"])
               + _clip01((wind_kts - t["extreme_wind"]) / 15.0)
               + _clip01((t["extreme_fm"] - fm) / t["extreme_fm"])) / 3.0
    fraction = np.where(category == 2, elevated, fraction)
    fraction = np.where(category == 3, critical, fraction)
    fraction = np.where(category == 4, extreme, fraction)
    return fraction


def seasonal_dampening_vectorized(category: np.ndarray, fm: np.ndarray, rh: np.ndarray, wind_kts: np.ndarray,
                                  green_factor: float, thresholds: Dict[str, float],
                                  max_demotion_fraction: float) -> np.ndarray:
    """Vectorized core/fire_danger.py::seasonal_dampening_adjustment.

    green_factor is the caller's core.fire_danger._green_factor(gdd) (a
    single statewide value - GDD state is not gridded). Demotes an
    Elevated/Critical/Extreme cell one tier only when it is marginal in its
    own tier; never raises; MISSING stays MISSING."""
    category = np.asarray(category)
    if green_factor is None or green_factor <= 0.0:
        return category
    eligible = category >= 2
    fraction = severity_fraction_vectorized(category, fm, rh, wind_kts, thresholds)
    demote = eligible & (fraction < green_factor * max_demotion_fraction)
    return np.where(demote, category - 1, category).astype(category.dtype)


# --- members -> peak category ------------------------------------------------

def peak_category(hourly_fm: np.ndarray, hourly_rh: np.ndarray, hourly_wind_kts: np.ndarray,
                  categorize: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
                  dampen: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]] = None,
                  snow_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """(hours, y, x) inputs -> (y, x) int8 peak category (MISSING where every hour is missing).

    Same per-hour order of operations as DailyForecast.py: canonical rule
    -> seasonal dampening -> snow forces Low; then max over the window."""
    hourly = categorize(hourly_fm, hourly_rh, hourly_wind_kts).astype("int16")
    if dampen is not None:
        hourly = dampen(hourly, hourly_fm, hourly_rh, hourly_wind_kts).astype("int16")
    if snow_mask is not None:
        hourly = np.where(np.broadcast_to(snow_mask, hourly.shape) & (hourly != MISSING), 0, hourly)
    return hourly.max(axis=0).astype("int8")


# --- probabilities -----------------------------------------------------------

def disk_footprint(radius_cells: float) -> np.ndarray:
    r = max(0, int(np.floor(radius_cells)))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (yy ** 2 + xx ** 2) <= radius_cells ** 2 + 1e-9


def _normalized_weights(peaks: np.ndarray, weights: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
    w = np.asarray(weights, dtype="float64").reshape(-1, 1, 1)
    valid = peaks != MISSING
    w_valid = np.where(valid, w, 0.0)
    total = w_valid.sum(axis=0)
    return w_valid, total


def exceedance_probability(peaks: np.ndarray, weights: Sequence[float], k: int,
                           radius_cells: float = 0.0) -> np.ndarray:
    """Weighted fraction of members with peak >= k at (radius=0) or within
    radius of (radius>0) each cell. peaks: (members, y, x). Members missing
    at a cell drop out of that cell's denominator; NaN where none remain."""
    w_valid, total = _normalized_weights(peaks, weights)
    hits = (peaks >= k) & (peaks != MISSING)
    if radius_cells > 0:
        footprint = disk_footprint(radius_cells)
        hits = np.stack([maximum_filter(member.astype("uint8"), footprint=footprint, mode="nearest") > 0
                         for member in hits])
    numerator = (w_valid * hits).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, numerator / total, np.nan)


def nan_gaussian(field: np.ndarray, sigma: float) -> np.ndarray:
    """NaN-aware gaussian smoothing (same weighted-normalization trick as
    DailyForecast's hourly risk smoothing)."""
    if sigma is None or sigma <= 0:
        return field
    valid = np.isfinite(field)
    weight = gaussian_filter(valid.astype("float64"), sigma=sigma)
    smoothed = gaussian_filter(np.where(valid, field, 0.0), sigma=sigma)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = smoothed / weight
    return np.where(valid, out, np.nan)


def apply_calibration(prob: np.ndarray, knots: Optional[dict]) -> np.ndarray:
    """Isotonic knots {"x": [...], "y": [...]} (monotone, 0..1) -> piecewise-linear
    map; identity when knots is None/empty."""
    if not knots or not knots.get("x"):
        return prob
    x = np.asarray(knots["x"], dtype="float64")
    y = np.asarray(knots["y"], dtype="float64")
    out = np.interp(np.clip(prob, 0.0, 1.0), x, y)
    return np.where(np.isfinite(prob), out, np.nan)


def enforce_monotone(stack: np.ndarray) -> np.ndarray:
    """stack[k-1] = P(>= k); force P(>=1) >= P(>=2) >= ... (cumulative minimum)."""
    filled = np.where(np.isfinite(stack), stack, np.inf)
    out = np.minimum.accumulate(filled, axis=0)
    return np.where(np.isfinite(stack), np.clip(out, 0.0, 1.0), np.nan)


def categorical_from_probabilities(point_stack: np.ndarray, thresholds: Dict[int, float]) -> np.ndarray:
    """Highest k whose calibrated point P(>=k) >= tau_k; 0 (Low) otherwise;
    NaN where probabilities are missing."""
    category = np.zeros(point_stack.shape[1:], dtype="float64")
    for k in CATEGORY_IDS:
        category = np.where(point_stack[k - 1] >= thresholds.get(k, 0.5), float(k), category)
    return np.where(np.isfinite(point_stack[0]), category, np.nan)


def probability_products(peaks: np.ndarray, weights: Sequence[float], *, radius_cells: float,
                         smooth_sigma: float, calibration: Optional[dict] = None,
                         categorical_thresholds: Optional[Dict[int, float]] = None,
                         default_threshold: float = 0.5) -> Dict[str, np.ndarray]:
    """Everything the graphics and evidence need from one track's member peaks.

    calibration: {"point": {"1": knots, ...}, "neighborhood": {"1": knots, ...}} or None."""
    calibration = calibration or {}
    raw_point, raw_nbhd = [], []
    for k in CATEGORY_IDS:
        raw_point.append(nan_gaussian(exceedance_probability(peaks, weights, k, 0.0), smooth_sigma))
        raw_nbhd.append(nan_gaussian(exceedance_probability(peaks, weights, k, radius_cells), smooth_sigma))
    raw_point = np.stack(raw_point)
    raw_nbhd = np.stack(raw_nbhd)
    point = enforce_monotone(np.stack([apply_calibration(raw_point[k - 1], (calibration.get("point") or {}).get(str(k)))
                                       for k in CATEGORY_IDS]))
    nbhd = enforce_monotone(np.stack([apply_calibration(raw_nbhd[k - 1], (calibration.get("neighborhood") or {}).get(str(k)))
                                      for k in CATEGORY_IDS]))
    thresholds = {k: float((categorical_thresholds or {}).get(k, default_threshold)) for k in CATEGORY_IDS}
    return {
        "raw_point": raw_point,
        "raw_neighborhood": raw_nbhd,
        "point": point,
        "neighborhood": nbhd,
        "categorical": categorical_from_probabilities(point, thresholds),
        "member_count": np.sum(peaks != MISSING, axis=0).astype("int16"),
    }


# --- synthetic members from ensemble mean/spread (Track E) -------------------

def _coherent_noise(rng: np.random.Generator, shape: Tuple[int, int], sigma: float) -> np.ndarray:
    """Spatially-correlated N(0, 1) field (per cell, across draws).

    Normalized by the ANALYTIC std of gaussian-filtered 2D white noise,
    1 / (2 sqrt(pi) sigma) - not by the field's own spatial std/mean. Per-
    draw standardization was tried first and removes each draw's domain-
    wide offset, which is most of the variance once the correlation length
    is a sizeable fraction of the domain: it under-dispersed the draws by
    ~25% (caught by test_recovers_mean_and_spread)."""
    z = rng.standard_normal(shape)
    if sigma and sigma > 0:
        z = gaussian_filter(z, sigma=sigma, mode="wrap") * (2.0 * np.sqrt(np.pi) * sigma)
    return z


def synthetic_draws(t_mean: np.ndarray, t_sprd: np.ndarray, td_mean: np.ndarray, td_sprd: np.ndarray,
                    w_mean: np.ndarray, w_sprd: np.ndarray, *, n_draws: int, seed: int, rho_t_td: float,
                    spatial_sigma: float):
    """Yield (t2m_K, d2m_K, wind10_ms) arrays shaped (hours, y, x), one draw at a time.

    Each draw uses ONE spatially-coherent standard-normal field per
    variable for every hour (a synthetic member is consistently warm/dry/
    windy through the day, like a real member, instead of independent
    hour-to-hour noise that would wash out peak-window extremes). T and Td
    errors are coupled with correlation rho_t_td; Td is capped at T; wind
    is clipped at 0. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    shape = t_mean.shape[1:]
    rho = float(np.clip(rho_t_td, -0.99, 0.99))
    for _ in range(int(n_draws)):
        z_t = _coherent_noise(rng, shape, spatial_sigma)
        z_i = _coherent_noise(rng, shape, spatial_sigma)
        z_w = _coherent_noise(rng, shape, spatial_sigma)
        z_td = rho * z_t + np.sqrt(1.0 - rho ** 2) * z_i
        t = t_mean + np.nan_to_num(t_sprd) * z_t[None]
        td = np.minimum(td_mean + np.nan_to_num(td_sprd) * z_td[None], t)
        w = np.clip(w_mean + np.nan_to_num(w_sprd) * z_w[None], 0.0, None)
        yield t, td, w


def pooled_mean_spread(means: List[np.ndarray], spreads: List[np.ndarray], weights: Sequence[float]):
    """Mixture mean/spread of several ensembles' (mean, spread) fields:
    total variance = within-ensemble variance + between-ensemble variance."""
    w = np.asarray(weights, dtype="float64")
    w = w / w.sum()
    mean = sum(wi * m for wi, m in zip(w, means))
    var = sum(wi * (s ** 2 + (m - mean) ** 2) for wi, m, s in zip(w, means, spreads))
    return mean, np.sqrt(var)


# --- FM anchoring ------------------------------------------------------------

def anchored_fm(member_xgb_fm: np.ndarray, control_xgb_fm: Optional[np.ndarray],
                production_fm: Optional[np.ndarray], clip: Tuple[float, float]) -> np.ndarray:
    """FM_member = FM_production + (FM_xgb_member - FM_xgb_control), clipped.

    Keeps the operational (RAWS-initialized / spatial-ONNX) fuel moisture
    as the anchor and lets each member perturb it only by how its own
    weather drives the same XGBoost model differently from the control's.
    Falls back to the member's raw XGBoost FM when either anchor is absent."""
    if production_fm is None or control_xgb_fm is None or production_fm.shape != member_xgb_fm.shape:
        return np.clip(member_xgb_fm, *clip)
    return np.clip(production_fm + (member_xgb_fm - control_xgb_fm), *clip)


def county_summary(field: np.ndarray, county_cells: Dict[str, List[Tuple[int, int]]],
                   reducer: str = "max") -> Dict[str, Optional[float]]:
    """county_fips -> reduced value over that county's cells (None if all NaN)."""
    out: Dict[str, Optional[float]] = {}
    for fips, cells in county_cells.items():
        if not cells:
            out[fips] = None
            continue
        rows, cols = zip(*cells)
        values = field[np.asarray(rows), np.asarray(cols)]
        values = values[np.isfinite(values)]
        if not values.size:
            out[fips] = None
        else:
            out[fips] = float(np.max(values) if reducer == "max" else np.mean(values))
    return out
