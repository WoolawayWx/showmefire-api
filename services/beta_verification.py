"""Outcome verification for process-isolated Testbed forecasts.

This module reads a completed beta forecast and the matching operational
observation archive, but writes only below ``BETA_ROOT/verification``.  It is
deliberately separate from ``forecast.endOfDayReport`` because that pipeline
updates the production verification history and plots.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.beta_fire_danger import BETA_SCORER_VERSION, score_fire_danger
from core.config import ARCHIVE_RAW_DATA_DIR
from core.fire_danger import CATEGORY_LABELS, meters_per_second_to_knots
from forecast.endOfDayReport import get_forecast_dataframe, get_observation_dataframe
from services.beta_products import BETA_ROOT
from services.forecast_jobs import BETA_FORECAST_ROOT


BETA_VERIFICATION_ROOT = BETA_ROOT / "verification"
BETA_VERIFICATION_HISTORY = BETA_VERIFICATION_ROOT / "history.json"
BETA_VERIFICATION_LATEST = BETA_VERIFICATION_ROOT / "latest.json"
MAX_HISTORY = int(os.getenv("BETA_VERIFICATION_HISTORY_LIMIT", "90"))
MINIMUM_SUPPORT = int(os.getenv("BETA_VERIFICATION_MINIMUM_SUPPORT", "50"))
_DATE_TOKEN = re.compile(r"(20\d{6})")


def _atomic_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _date_token(path: Path) -> str | None:
    match = _DATE_TOKEN.search(path.name)
    return match.group(1) if match else None


def _select_forecast(forecast_dir: Path, target_date: str | None) -> tuple[Path, str]:
    candidates = sorted(Path(forecast_dir).glob("station_forecasts_beta_*.json"))
    if target_date:
        compact = target_date.replace("-", "")
        candidates = [path for path in candidates if _date_token(path) == compact]
    if not candidates:
        detail = f" for {target_date}" if target_date else ""
        raise RuntimeError(f"No completed Testbed forecast is available{detail}.")
    selected = candidates[-1]
    compact = _date_token(selected)
    if not compact:
        raise RuntimeError(f"Could not determine the forecast date from {selected.name}.")
    return selected, compact


def _load_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Unable to read {path.name}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path.name} does not contain a JSON object.")
    return payload


def _ordinal_metrics(predicted: pd.Series, observed: pd.Series) -> dict:
    frame = pd.DataFrame({"predicted": predicted, "observed": observed}).dropna()
    if frame.empty:
        return {
            "mae": None, "bias": None, "exact_match_rate": None,
            "within_one_category_rate": None, "count": 0,
        }
    prediction = pd.to_numeric(frame["predicted"], errors="coerce").to_numpy(float)
    truth = pd.to_numeric(frame["observed"], errors="coerce").to_numpy(float)
    valid = np.isfinite(prediction) & np.isfinite(truth)
    prediction, truth = prediction[valid], truth[valid]
    if not len(prediction):
        return {
            "mae": None, "bias": None, "exact_match_rate": None,
            "within_one_category_rate": None, "count": 0,
        }
    rounded = np.clip(np.rint(prediction), 0, len(CATEGORY_LABELS) - 1)
    truth = np.clip(np.rint(truth), 0, len(CATEGORY_LABELS) - 1)
    continuous_difference = prediction - truth
    rounded_difference = rounded - truth
    return {
        "mae": round(float(np.mean(np.abs(continuous_difference))), 4),
        "bias": round(float(np.mean(continuous_difference)), 4),
        "exact_match_rate": round(float(np.mean(rounded_difference == 0)), 4),
        "within_one_category_rate": round(float(np.mean(np.abs(rounded_difference) <= 1)), 4),
        "count": int(len(prediction)),
    }


def _difference(candidate: float | None, stable: float | None) -> float | None:
    if candidate is None or stable is None:
        return None
    return round(float(candidate) - float(stable), 4)


def load_latest_beta_verification() -> dict | None:
    try:
        report = json.loads(BETA_VERIFICATION_LATEST.read_text(encoding="utf-8"))
        return report if isinstance(report, dict) else None
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def run_beta_verification(
    target_date: str | None = None,
    *,
    forecast_dir: Path = BETA_FORECAST_ROOT / "archive" / "forecasts",
    observations_dir: Path = ARCHIVE_RAW_DATA_DIR,
    output_root: Path = BETA_VERIFICATION_ROOT,
) -> dict:
    """Score beta and stable forecast outputs without modifying production state."""
    forecast_path, compact_date = _select_forecast(Path(forecast_dir), target_date)
    observation_path = Path(observations_dir) / f"raw_data_{compact_date}.json"
    if not observation_path.exists():
        raise RuntimeError(f"The observation archive for {compact_date} is not available yet.")

    forecast_payload = _load_json(forecast_path)
    observation_payload = _load_json(observation_path)
    report_date = f"{compact_date[:4]}-{compact_date[4:6]}-{compact_date[6:]}"
    # The operating window is defined in Central time, so let timezone rules
    # handle CST/CDT instead of hard-coding a UTC hour.
    start = pd.Timestamp(f"{report_date} 10:00", tz="America/Chicago").tz_convert("UTC")
    end = pd.Timestamp(f"{report_date} 21:00", tz="America/Chicago").tz_convert("UTC")
    forecast_frame = get_forecast_dataframe(forecast_payload, start, end)
    observation_frame, qc_exclusions, _ = get_observation_dataframe(observation_payload, start, end)
    if forecast_frame.empty or observation_frame.empty:
        raise RuntimeError("The beta forecast and observation archive have no scoreable rows.")
    merged = pd.merge(forecast_frame, observation_frame, on=["stid", "timestamp"], how="inner")
    if merged.empty:
        raise RuntimeError("The beta forecast and observations do not overlap by station and hour.")

    scores = []
    categories = []
    for row in merged.itertuples(index=False):
        try:
            result = score_fire_danger(
                row.pred_fm,
                row.pred_rh,
                meters_per_second_to_knots(row.pred_wind),
            )
            scores.append(result["score"])
            categories.append(result["official_category"])
        except (TypeError, ValueError):
            scores.append(np.nan)
            categories.append(np.nan)
    merged["beta_score"] = scores
    merged["beta_category"] = categories

    stable = _ordinal_metrics(merged["pred_fire_danger"], merged["obs_fire_danger"])
    beta = _ordinal_metrics(merged["beta_category"], merged["obs_fire_danger"])
    beta_continuous = _ordinal_metrics(merged["beta_score"], merged["obs_fire_danger"])
    elevated = merged[pd.to_numeric(merged["obs_fire_danger"], errors="coerce") >= 2]
    elevated_stable = _ordinal_metrics(elevated["pred_fire_danger"], elevated["obs_fire_danger"])
    elevated_beta = _ordinal_metrics(elevated["beta_category"], elevated["obs_fire_danger"])

    report = {
        "date": report_date,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scorer_version": BETA_SCORER_VERSION,
        "status": "ready" if beta["count"] >= MINIMUM_SUPPORT else "collecting_evidence",
        "minimum_support": MINIMUM_SUPPORT,
        "record_count": int(len(merged)),
        "stations_count": int(merged["stid"].nunique()),
        "sources": {"forecast": forecast_path.name, "observations": observation_path.name},
        "stable": stable,
        "beta": beta,
        "beta_continuous": beta_continuous,
        "delta": {
            "mae": _difference(beta["mae"], stable["mae"]),
            "exact_match_rate": _difference(beta["exact_match_rate"], stable["exact_match_rate"]),
            "within_one_category_rate": _difference(
                beta["within_one_category_rate"], stable["within_one_category_rate"]
            ),
        },
        "elevated_observations": {
            "count": elevated_beta["count"],
            "stable": elevated_stable,
            "beta": elevated_beta,
            "delta_mae": _difference(elevated_beta["mae"], elevated_stable["mae"]),
        },
        "qc_exclusion_count": len(qc_exclusions),
        "isolation": {
            "production_forecast_outputs_modified": False,
            "production_verification_history_modified": False,
            "output_root": "testbed/verification",
        },
    }

    output_root = Path(output_root)
    report_path = output_root / f"{compact_date}.json"
    _atomic_json_write(report_path, report)
    _atomic_json_write(output_root / "latest.json", report)
    history_path = output_root / "history.json"
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
        if not isinstance(history, list):
            history = []
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        history = []
    history = [item for item in history if item.get("date") != report["date"]]
    history.append(report)
    history = sorted(history, key=lambda item: item.get("date", ""))[-MAX_HISTORY:]
    _atomic_json_write(history_path, history)
    return report
