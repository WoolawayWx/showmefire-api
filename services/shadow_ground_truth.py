"""Ground-truth accuracy comparison for a registry model's beta candidate.

Reuses `forecast/endOfDayReport.py`'s existing, already-trusted nightly
verification output rather than building a parallel observation-fetch/QC/
no-lookahead system: `scripts/validateForecast.sh` scores the stable model's
per-station forecast into `reports/validation_history.json` every night, and
(when a beta fuel_moisture candidate exists) scores the beta model's
grid-sampled per-station "model shadow" forecast the same way into
`reports/validation_history_model_shadow.json`. This module just diffs the
two, once enough distinct days of matched evidence has accumulated.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
STABLE_HISTORY_PATH = REPORTS_DIR / "validation_history.json"
SHADOW_HISTORY_PATH = REPORTS_DIR / "validation_history_model_shadow.json"

FM_METRIC_KEY = "Fuel Moisture (%)"
FIRE_DANGER_METRIC_KEY = "Fire Danger Index"


def _load_history(path):
    path = Path(path)
    if not path.exists():
        return []
    try:
        history = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return history if isinstance(history, list) else []


def _by_date(history):
    return {entry.get("date"): entry for entry in history if entry.get("date")}


def evaluate_ground_truth_accuracy(
    model_type="fuel_moisture",
    minimum_days=14,
    minimum_daily_samples=50,
    mae_tolerance=0.05,
    category_tolerance=0.02,
    stable_history_path=STABLE_HISTORY_PATH,
    shadow_history_path=SHADOW_HISTORY_PATH,
):
    """Compare stable-vs-beta accuracy against real observed outcomes.

    Matches days present in both histories where each side's Fuel Moisture
    metric has at least `minimum_daily_samples` scored records, then averages
    each day's MAE (continuous fuel-moisture error) and exact-match rate
    (categorical fire-danger accuracy) across those matched days. `passed` is
    true only if beta is not meaningfully worse than stable on either axis -
    it does not have to be strictly better.
    """
    stable_by_date = _by_date(_load_history(stable_history_path))
    shadow_by_date = _by_date(_load_history(shadow_history_path))

    matched_days = []
    for date, shadow_entry in shadow_by_date.items():
        stable_entry = stable_by_date.get(date)
        if not stable_entry:
            continue
        stable_fm = (stable_entry.get("metrics") or {}).get(FM_METRIC_KEY) or {}
        shadow_fm = (shadow_entry.get("metrics") or {}).get(FM_METRIC_KEY) or {}
        if (stable_fm.get("count") or 0) < minimum_daily_samples:
            continue
        if (shadow_fm.get("count") or 0) < minimum_daily_samples:
            continue
        stable_fd = (stable_entry.get("metrics") or {}).get(FIRE_DANGER_METRIC_KEY) or {}
        shadow_fd = (shadow_entry.get("metrics") or {}).get(FIRE_DANGER_METRIC_KEY) or {}
        if stable_fm.get("mae") is None or shadow_fm.get("mae") is None:
            continue
        if stable_fd.get("exact_match_rate") is None or shadow_fd.get("exact_match_rate") is None:
            continue
        matched_days.append({
            "date": date,
            "stable_mae": stable_fm["mae"], "beta_mae": shadow_fm["mae"],
            "stable_exact_match_rate": stable_fd["exact_match_rate"],
            "beta_exact_match_rate": shadow_fd["exact_match_rate"],
        })

    evidence = {
        "model_type": model_type,
        "minimum_days": minimum_days,
        "minimum_daily_samples": minimum_daily_samples,
        "matched_days": len(matched_days),
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    if len(matched_days) < minimum_days:
        evidence["passed"] = False
        evidence["reason"] = "insufficient evidence"
        return evidence

    stable_mae = sum(day["stable_mae"] for day in matched_days) / len(matched_days)
    beta_mae = sum(day["beta_mae"] for day in matched_days) / len(matched_days)
    stable_exact_match_rate = sum(day["stable_exact_match_rate"] for day in matched_days) / len(matched_days)
    beta_exact_match_rate = sum(day["beta_exact_match_rate"] for day in matched_days) / len(matched_days)

    passed = (
        beta_mae <= stable_mae * (1 + mae_tolerance)
        and beta_exact_match_rate >= stable_exact_match_rate - category_tolerance
    )
    evidence.update({
        "passed": passed,
        "stable": {"mae": round(stable_mae, 4), "exact_match_rate": round(stable_exact_match_rate, 4)},
        "beta": {"mae": round(beta_mae, 4), "exact_match_rate": round(beta_exact_match_rate, 4)},
        "mae_tolerance": mae_tolerance,
        "category_tolerance": category_tolerance,
    })
    return evidence
