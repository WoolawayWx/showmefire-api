"""Will this incident keep producing detections? A server-side model that needs no human labels.

Question: given only the first WINDOW_MINUTES of an incident (what the satellites saw, where, when, and what the ground
and weather there are like), will it still be producing detections HORIZON_HOURS after its first one? That is a rough
"how serious will this get" signal: planned and field burns tend to be over quickly, wildfires persist. The answer is
already in the history of every past incident, so it trains on its own data (services/incident_models/features.py builds
the same inputs for training and serving).

STATUS: NOT SERVED. Measured on production history (2,593 incidents, newest quarter held out, 2026-11-21) no variant (deep
or shallow XGBoost, logistic regression, with or without weather, all incidents or only those GOES watched) beat just counting
the detections in the first window (AUC 0.67) or the existing rules confidence score (0.68); every learned model landed at
0.59-0.67 with a 95% interval of about +-0.045. The first 30 minutes simply carry little information about whether an incident
persists. Nothing calls this module except the command line, and the gate below keeps it from serving a model that does not
beat those baselines. Re-run `evaluate` once there is more history (or a better target) before wiring it in.

It is advisory: it would be shown beside the confidence score and never used to send or hold an alert.

Honesty built in:
  * Evaluated on the NEWEST quarter of incidents (a time split, never a random one) against baselines: the base rate, the
    count of detections alone, and the existing rules confidence score computed from the same first window.
  * A model is only served if it clears a gate (AUC and Brier score on that holdout). Otherwise the previous good model
    stays, or nothing is served.
  * It only predicts for incidents still inside the horizon; after that the real outcome is shown instead.

Run by hand on the server:   python -m services.incident_models.persistence evaluate     (nothing is saved)
                             python -m services.incident_models.persistence train        (saves if it passes the gate)
and weekly by the `retrain_incident_models` job (core/scheduler.py).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from services.incident_models import features as F

logger = logging.getLogger(__name__)

HORIZON_HOURS = 3.0
MATURE_HOURS = 6.0           # an incident counts as history once its first detection is this old (late data has arrived)
DEFAULT_SINCE = "2026-08-27"  # the incident table starts here
MIN_ROWS = 300
HOLDOUT_FRACTION = 0.25
VALID_FRACTION = 0.15        # of the training part, used only to pick the number of trees
GATE_AUC = 0.60
XGB_PARAMS = dict(
    n_estimators=400, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    reg_lambda=3.0, gamma=0.5, eval_metric="logloss", early_stopping_rounds=30, n_jobs=2, random_state=7, tree_method="hist",
)


def model_dir() -> Path:
    from core.config import DATA_DIR

    return Path(os.getenv("SMF_INCIDENT_MODEL_DIR") or (DATA_DIR / "incident_models")) / "persistence"


# --------------------------------------------------------------------------------------------------- the data
def _connect() -> sqlite3.Connection:
    from core.database import get_db_path

    conn = sqlite3.connect(get_db_path(), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def load_history(conn: sqlite3.Connection, since: str = DEFAULT_SINCE) -> tuple[list[dict], dict[int, list[dict]]]:
    """Every live incident and its satellite detections (one query each, not one per incident)."""
    incidents = [dict(r) for r in conn.execute(
        "SELECT id, public_slug, centroid_latitude, centroid_longitude, first_detected_at, last_detected_at, "
        "detection_count, status, county_fips FROM fire_incidents WHERE status NOT IN ('deleted','merged')")]
    marks = ",".join("?" for _ in F.SATELLITE_SOURCES)
    members: dict[int, list[dict]] = {}
    for row in conn.execute(
        "SELECT id, incident_id, latitude, longitude, occurred_at, frp, source, confidence, detection_confidence_pct, "
        f"footprint_geojson, land_cover, bright_t7, satellite, status FROM fire_events WHERE incident_id IS NOT NULL "
        f"AND source IN ({marks}) AND occurred_at >= ?", (*F.SATELLITE_SOURCES, since)):
        members.setdefault(row["incident_id"], []).append(dict(row))
    return incidents, members


def persisted(sat: list[dict], t0: datetime, hours: float = HORIZON_HOURS) -> bool:
    """True when some detection came at least `hours` after the first one."""
    return any(m["_t"] >= t0 + timedelta(hours=hours) for m in sat)


def build_dataset(conn: sqlite3.Connection, since: str = DEFAULT_SINCE, *, with_weather: bool = True,
                  now: Optional[datetime] = None) -> tuple[pd.DataFrame, dict[int, tuple[dict, list[dict]]]]:
    """One row per mature incident: its first-window features, whether it persisted, and when it began. The second
    return value keeps each incident and its first-window detections for the rules-score baseline."""
    now = now or datetime.now(timezone.utc)
    incidents, members = load_history(conn, since)
    rows: list[dict[str, Any]] = []
    extras: dict[int, tuple[dict, list[dict]]] = {}
    for incident in incidents:
        sat = F.satellite_members(members.get(incident["id"], []))
        t0, window = F.first_window(sat)
        if t0 is None or now - t0 < timedelta(hours=MATURE_HOURS):
            continue
        _, feats = F.persistence_features(incident, sat, conn, with_weather=with_weather)
        rows.append({"incident_id": incident["id"], "t0": t0, "label": int(persisted(sat, t0)), **feats})
        extras[incident["id"]] = (incident, window)
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values("t0").reset_index(drop=True)
    return frame, extras


# --------------------------------------------------------------------------------------------------- evaluating
def _auc(y, score) -> Optional[float]:
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y, score)) if len(set(y)) > 1 else None


def _fit(train: pd.DataFrame, columns: list[str]):
    """XGBoost with the tree count picked on the newest slice of the training part (so it is time-ordered too)."""
    import xgboost as xgb

    cut = int(len(train) * (1 - VALID_FRACTION))
    fit_part, valid_part = train.iloc[:cut], train.iloc[cut:]
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(fit_part[columns], fit_part["label"], eval_set=[(valid_part[columns], valid_part["label"])], verbose=False)
    return model


def _calibration(y: np.ndarray, p: np.ndarray, bins: int = 5) -> list[dict]:
    order = np.argsort(p)
    out = []
    for chunk in np.array_split(order, bins):
        if len(chunk):
            out.append({"mean_predicted": round(float(p[chunk].mean()), 3), "actual_rate": round(float(y[chunk].mean()), 3), "n": int(len(chunk))})
    return out


def evaluate(frame: pd.DataFrame, extras: dict, *, columns: Optional[list[str]] = None) -> dict[str, Any]:
    """Train on the older part, score the newest HOLDOUT_FRACTION, compare with baselines. Saves nothing."""
    from sklearn.metrics import average_precision_score, brier_score_loss

    columns = columns or list(F.PERSISTENCE_FEATURES)
    cut = int(len(frame) * (1 - HOLDOUT_FRACTION))
    train, test = frame.iloc[:cut], frame.iloc[cut:]
    model = _fit(train, columns)
    y = test["label"].to_numpy()
    p = model.predict_proba(test[columns])[:, 1]
    base = float(train["label"].mean())

    baseline_conf = None
    try:
        from services.incident_confidence import score_incident

        scores = []
        for incident_id in test["incident_id"]:
            incident, window = extras[incident_id]
            scores.append(float(score_incident(incident, window)["score"]))
        baseline_conf = _auc(y, scores)
    except Exception:  # noqa: BLE001 - the baseline is informational
        logger.info("rules-score baseline unavailable", exc_info=True)

    importance = sorted(zip(columns, model.feature_importances_), key=lambda kv: -kv[1])[:8]
    return {
        "rows": int(len(frame)), "train": int(len(train)), "holdout": int(len(test)),
        "holdout_from": str(test["t0"].iloc[0]) if len(test) else None,
        "base_rate": round(base, 3), "holdout_rate": round(float(y.mean()), 3) if len(y) else None,
        "auc": _auc(y, p), "average_precision": float(average_precision_score(y, p)) if len(set(y)) > 1 else None,
        "brier": float(brier_score_loss(y, p)), "brier_base_rate": float(brier_score_loss(y, np.full(len(y), base))),
        "baseline_auc_first_window_count": _auc(y, test["n_w"].to_numpy()),
        "baseline_auc_rules_confidence": baseline_conf,
        "calibration": _calibration(y, p),
        "top_features": [{"feature": name, "importance": round(float(value), 3)} for name, value in importance],
        "trees": int(model.best_iteration + 1) if getattr(model, "best_iteration", None) is not None else XGB_PARAMS["n_estimators"],
    }


def weather_ablation(frame: pd.DataFrame, extras: dict) -> Optional[dict[str, Any]]:
    """Does weather help? Only the incidents the RTMA archive covers, with and without the weather columns."""
    covered = frame[frame[list(F.WEATHER_FEATURES)].notna().any(axis=1)].reset_index(drop=True)
    if len(covered) < 150:
        return None
    no_wx = [c for c in F.PERSISTENCE_FEATURES if c not in F.WEATHER_FEATURES]
    with_wx = evaluate(covered, extras)
    without = evaluate(covered, extras, columns=no_wx)
    return {"incidents_with_weather": int(len(covered)), "auc_with_weather": with_wx["auc"], "auc_without_weather": without["auc"]}


BEAT_BASELINES_BY = 0.02


def gate(report: dict) -> bool:
    """A model is only worth serving if it beats the things it is meant to improve on: a floor on AUC, a better Brier score
    than the base rate, AND an AUC above both simple baselines (the number of detections in the first window, and the existing
    rules confidence score) by BEAT_BASELINES_BY. On the data available when this was written (about 2,600 incidents) no
    variant cleared that, which is the honest result: see the module docstring."""
    if report.get("auc") is None or report["auc"] < GATE_AUC or report["brier"] > report["brier_base_rate"]:
        return False
    baselines = [v for v in (report.get("baseline_auc_first_window_count"), report.get("baseline_auc_rules_confidence")) if v is not None]
    return all(report["auc"] >= b + BEAT_BASELINES_BY for b in baselines)


# --------------------------------------------------------------------------------------------------- training
def train(conn: Optional[sqlite3.Connection] = None, *, since: str = DEFAULT_SINCE, with_weather: bool = True,
          save: bool = True, now: Optional[datetime] = None) -> dict[str, Any]:
    """Evaluate on the time-ordered holdout and, if it passes the gate and `save`, refit on everything and save it.
    Returns the report (always written to last_report.json when saving). Never raises."""
    own = conn is None
    conn = conn or _connect()
    try:
        frame, extras = build_dataset(conn, since, with_weather=with_weather, now=now)
        report: dict[str, Any] = {"trained_at": (now or datetime.now(timezone.utc)).isoformat(timespec="seconds"),
                                  "horizon_hours": HORIZON_HOURS, "window_minutes": F.WINDOW_MINUTES}
        if len(frame) < MIN_ROWS or frame["label"].nunique() < 2:
            report.update(status="not_enough_history", rows=int(len(frame)), needed=MIN_ROWS, gate_passed=False)
            return _finish(report, save)
        report.update(evaluate(frame, extras))
        report["gate_passed"] = gate(report)
        report["status"] = "ok" if report["gate_passed"] else "below_gate"
        if report["gate_passed"] and save:
            import xgboost as xgb

            params = dict(XGB_PARAMS)
            params.pop("early_stopping_rounds")
            params["n_estimators"] = max(50, int(report["trees"] * 1.1))
            final = xgb.XGBClassifier(**params)
            final.fit(frame[list(F.PERSISTENCE_FEATURES)], frame["label"], verbose=False)
            folder = model_dir()
            folder.mkdir(parents=True, exist_ok=True)
            version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
            # XGBoost chooses the file format from the extension, so the temporary name must still end in .json
            final.save_model(str(folder / "model.tmp.json"))
            os.replace(folder / "model.tmp.json", folder / "model.json")
            meta = {"version": version, "features": list(F.PERSISTENCE_FEATURES), "trained_on": int(len(frame)),
                    "horizon_hours": HORIZON_HOURS, "holdout_auc": report["auc"], "base_rate": report["base_rate"],
                    "trained_at": report["trained_at"]}
            (folder / "meta.json").write_text(json.dumps(meta, indent=2))
            report["saved_version"] = version
        return _finish(report, save)
    except Exception as error:  # noqa: BLE001
        logger.error("incident persistence training failed: %s", error, exc_info=True)
        return {"status": "error", "error": str(error), "gate_passed": False}
    finally:
        if own:
            conn.close()


def _finish(report: dict, save: bool) -> dict:
    if save:
        folder = model_dir()
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "last_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


# --------------------------------------------------------------------------------------------------- serving
_cache: dict[str, Any] = {"mtime": None, "bundle": None}
_lock = threading.Lock()


def _bundle() -> Optional[tuple[Any, dict]]:
    """(model, meta) for the saved model, reloaded when the file changes; None when there is none to serve."""
    folder = model_dir()
    path, meta_path = folder / "model.json", folder / "meta.json"
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    with _lock:
        if _cache["mtime"] != mtime:
            try:
                import xgboost as xgb

                model = xgb.XGBClassifier()
                model.load_model(str(path))
                _cache["bundle"] = (model, json.loads(meta_path.read_text()))
                _cache["mtime"] = mtime
            except Exception:  # noqa: BLE001
                logger.warning("could not load the persistence model", exc_info=True)
                _cache["bundle"], _cache["mtime"] = None, None
        return _cache["bundle"]


def outlook(incident: dict, members: list[dict], conn: Optional[sqlite3.Connection] = None,
            now: Optional[datetime] = None) -> Optional[dict[str, Any]]:
    """What the incident detail returns as `persistence_outlook`, or None when no model is being served or the incident has
    no timed satellite detection. Inside the horizon it is a prediction; after it, the real outcome. Never raises."""
    try:
        bundle = _bundle()
        if bundle is None:
            return None
        model, meta = bundle
        now = now or datetime.now(timezone.utc)
        sat = F.satellite_members(members)
        t0, _ = F.first_window(sat)
        if t0 is None:
            return None
        base = {"horizon_hours": meta.get("horizon_hours", HORIZON_HOURS), "model": meta["version"],
                "trained_on": meta.get("trained_on"), "holdout_auc": round(float(meta.get("holdout_auc") or 0), 3)}
        if persisted(sat, t0):
            return {**base, "status": "observed", "persisted": True, "probability": None}
        if now - t0 >= timedelta(hours=HORIZON_HOURS):
            return {**base, "status": "observed", "persisted": False, "probability": None}
        own = conn is None
        conn = conn or _connect()
        try:
            _, feats = F.persistence_features(incident, sat, conn)
        finally:
            if own:
                conn.close()
        row = pd.DataFrame([[feats.get(name, np.nan) for name in meta["features"]]], columns=meta["features"])
        return {**base, "status": "predicted", "persisted": None, "probability": round(float(model.predict_proba(row)[0, 1]), 3)}
    except Exception:  # noqa: BLE001 - an extra on the incident page must never break it
        logger.info("persistence outlook failed", exc_info=True)
        return None


# --------------------------------------------------------------------------------------------------- command line
def _print(report: dict) -> None:
    keys = ("status", "rows", "train", "holdout", "holdout_from", "base_rate", "holdout_rate", "auc", "average_precision", "brier",
            "brier_base_rate", "baseline_auc_first_window_count", "baseline_auc_rules_confidence", "gate_passed", "trees", "saved_version")
    for key in keys:
        if key in report:
            value = report[key]
            print(f"  {key:<34} {round(value, 4) if isinstance(value, float) else value}")
    for row in report.get("calibration", []):
        print(f"  calibration  predicted {row['mean_predicted']:.2f}  actual {row['actual_rate']:.2f}  (n={row['n']})")
    for row in report.get("top_features", [])[:6]:
        print(f"  feature      {row['feature']:<18} {row['importance']}")
    if "weather" in report:
        print(f"  weather ablation: {report['weather']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("evaluate", "train"))
    parser.add_argument("--since", default=DEFAULT_SINCE)
    parser.add_argument("--no-weather", action="store_true")
    args = parser.parse_args()
    conn = _connect()
    try:
        if args.command == "evaluate":
            frame, extras = build_dataset(conn, args.since, with_weather=not args.no_weather)
            report = evaluate(frame, extras) if len(frame) >= MIN_ROWS else {"status": "not_enough_history", "rows": len(frame)}
            report["gate_passed"] = gate(report) if "auc" in report else False
            if not args.no_weather and len(frame) >= MIN_ROWS:
                report["weather"] = weather_ablation(frame, extras)
        else:
            report = train(conn, since=args.since, with_weather=not args.no_weather, save=True)
        _print(report)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
