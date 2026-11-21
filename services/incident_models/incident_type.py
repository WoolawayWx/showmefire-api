"""Is this incident a wildfire, or a planned burn? A server-side model that learns from every reviewed incident.

Labels come from people, never from the model: a staff review that labelled an incident's detections wildfire or
prescribed (controlled burn), an approved public answer of "confirmed fire" or "controlled burn", and a "yes" to the
cropland-burn question (a planned burn). Where those disagree the incident is left out. "Not a fire" is left out too: it
is a different question (is there a fire at all?) and there are almost no examples.

Because labels are scarce (on the order of a hundred incidents when this was written), it is a small, heavily regularised
logistic regression refit every day from ALL labels, so every new answer is used the next morning. It is checked with
grouped cross-validation (incidents from the same day stay together, so neighbours cannot leak between train and test) and
it is only SHOWN once that check clears a gate. Until then the detail says it is still collecting labels and gives no number.

STATUS: NOT SERVED. Measured on production labels (50 incidents, 25 wildfire / 25 planned burn, 2026-11-21): grouped
cross-validation AUC 0.78-0.79 (95% interval 0.64-0.91), but cropland share ALONE scores 0.86, canopy alone 0.83, and the rule
"wildfire if cropland share < 50%" is right 80% of the time against this model's 68% at its 0.5 cutoff. So the 19-input model
adds noise to "what is the land cover?". The existing rules confidence score cannot tell the two apart either (AUC 0.56; it
answers "is it a real fire?"). Most misses were prescribed burns in forested counties (Stone, Sullivan, Dade, Barry, Pike),
which look like wildfires by cover. The gate now also requires beating the best single input, so nothing is served; the
module is kept as an evaluation tool (`python -m services.incident_models.incident_type evaluate`) for when there are more
labels or better inputs (permits, burn bans, dispatch).

It would be advisory: shown beside the confidence score and the cropland-burn estimate, and never used to send or hold an alert.
The two answer different questions: the cropland model asks "is it a field burn?", this one "is it planned at all?".

The number it returns is a relative `score` (the model is trained on a balanced set of reviewed incidents), NOT the real-world
probability that an incident is a wildfire.

Run by hand on the server:  python -m services.incident_models.incident_type evaluate | train
and daily by the `retrain_incident_models` job.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import threading
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from services.incident_models import features as F

logger = logging.getLogger(__name__)

WILDFIRE, PLANNED = 1, 0
MIN_LABELS = 40
MIN_PER_CLASS = 15
GATE_AUC = 0.70
BEAT_SINGLE_INPUT_BY = 0.02  # and it must beat the best ONE input on its own (e.g. cropland share) by this much
GATE_AUC_LOW = 0.60          # the lower end of the 95% interval must also clear this, so a lucky small sample cannot pass
CV_FOLDS, CV_REPEATS = 5, 5
SHOW_WILDFIRE_ABOVE, SHOW_PLANNED_BELOW = 0.65, 0.35

COUNT_COLUMNS = ("n", "scans", "frp_max", "frp_mean", "n_ngfs", "n_viirs", "n_near_24h", "n_near_7d", "span_km", "duration_h")
FEATURES = (
    "n", "scans", "duration_h", "frp_max", "frp_mean", "n_ngfs", "n_viirs", "det_conf_max", "span_km",
    "crop", "pasture_grass", "forest", "canopy", "hour_sin", "hour_cos", "month", "weekend", "n_near_24h", "n_near_7d",
)


def model_dir() -> Path:
    from core.config import DATA_DIR

    return Path(os.getenv("SMF_INCIDENT_MODEL_DIR") or (DATA_DIR / "incident_models")) / "type"


def _connect() -> sqlite3.Connection:
    from core.database import get_db_path

    conn = sqlite3.connect(get_db_path(), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


# --------------------------------------------------------------------------------------------------- labels
def labeled_incidents(conn: sqlite3.Connection) -> dict[int, int]:
    """incident id -> WILDFIRE or PLANNED, from every human source; conflicting incidents are dropped."""
    votes: dict[int, set[int]] = defaultdict(set)
    cause: dict[int, Counter] = defaultdict(Counter)
    for row in conn.execute("SELECT incident_id, cause_category FROM fire_events WHERE incident_id IS NOT NULL "
                            "AND cause_category IN ('wildfire','prescribed')"):
        cause[row["incident_id"]][row["cause_category"]] += 1
    for incident_id, counts in cause.items():
        if counts["wildfire"] != counts["prescribed"]:
            votes[incident_id].add(WILDFIRE if counts["wildfire"] > counts["prescribed"] else PLANNED)
    for row in conn.execute("SELECT incident_id, classification FROM fire_incident_feedback WHERE status = 'approved' "
                            "AND classification IN ('confirmed_fire','controlled_burn')"):
        votes[row["incident_id"]].add(WILDFIRE if row["classification"] == "confirmed_fire" else PLANNED)
    try:
        for row in conn.execute("SELECT incident_id FROM cropland_burn_labels WHERE answer = 1"):
            votes[row["incident_id"]].add(PLANNED)
    except sqlite3.OperationalError:
        pass  # the cropland table is created on its first answer
    return {incident_id: next(iter(v)) for incident_id, v in votes.items() if len(v) == 1}


# --------------------------------------------------------------------------------------------------- inputs
def type_features(incident: dict, members: list[dict], conn=None) -> tuple[Optional[datetime], dict[str, float]]:
    """Whole-incident inputs (everything the incident has so far), so a label applied hours or days later is judged on what
    a person could have seen. (None, {}) when there is no timed satellite detection."""
    sat = F.satellite_members(members)
    if not sat:
        return None, {}
    t0 = sat[0]["_t"]
    feats = F.window_features(t0, sat, suffix="")
    feats["duration_h"] = (sat[-1]["_t"] - t0).total_seconds() / 3600
    feats.update(F.cover_for(incident, sat))
    feats.update(F.neighbor_counts(conn, incident, t0))
    return t0, {name: feats.get(name, F.NAN) for name in FEATURES}


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    """Counts get log1p so one huge incident does not dominate a linear model."""
    out = frame[list(FEATURES)].astype(float).copy()
    for column in COUNT_COLUMNS:
        if column in out:
            out[column] = np.log1p(out[column].clip(lower=0))
    return out


def build_dataset(conn: sqlite3.Connection) -> pd.DataFrame:
    labels = labeled_incidents(conn)
    if not labels:
        return pd.DataFrame()
    from core.database import list_fire_incident_members

    rows = []
    for incident_id, label in labels.items():
        incident = conn.execute("SELECT id, centroid_latitude, centroid_longitude, first_detected_at FROM fire_incidents "
                                "WHERE id = ?", (incident_id,)).fetchone()
        if incident is None:
            continue
        members = list_fire_incident_members(incident_id)
        t0, feats = type_features(dict(incident), members, conn)
        if t0 is None:
            continue
        rows.append({"incident_id": incident_id, "label": label, "day": t0.date().isoformat(), **feats})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------- fitting
def make_model():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(),
                         LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000))


def cross_validate(frame: pd.DataFrame) -> dict[str, Any]:
    """Repeated grouped cross-validation: incidents that began on the same day always fall in the same fold. Reports the mean
    AUC over the repeats and, because the sample is small, a bootstrap 95% interval for the AUC of the averaged out-of-fold
    scores (how far the number could plausibly be from the truth)."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold

    x, y, groups = prepare(frame), frame["label"].to_numpy(), frame["day"].to_numpy()
    aucs, total, seen = [], np.zeros(len(y)), np.zeros(len(y))
    for repeat in range(CV_REPEATS):
        splitter = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=repeat)
        scores = np.full(len(y), np.nan)
        for train_index, test_index in splitter.split(x, y, groups):
            model = make_model().fit(x.iloc[train_index], y[train_index])
            scores[test_index] = model.predict_proba(x.iloc[test_index])[:, 1]
        valid = ~np.isnan(scores)
        if valid.sum() and len(set(y[valid])) > 1:
            aucs.append(float(roc_auc_score(y[valid], scores[valid])))
            total[valid] += scores[valid]
            seen[valid] += 1
    out: dict[str, Any] = {"cv_auc": float(np.mean(aucs)) if aucs else None, "cv_auc_sd": float(np.std(aucs)) if aucs else None,
                           "cv_runs": len(aucs), "cv_auc_low": None, "cv_auc_high": None}
    if aucs:
        pooled = np.divide(total, seen, out=np.zeros(len(y)), where=seen > 0)
        rng, boots = np.random.default_rng(11), []
        for _ in range(400):
            index = rng.integers(0, len(y), len(y))
            if len(set(y[index])) > 1:
                boots.append(roc_auc_score(y[index], pooled[index]))
        if boots:
            out["cv_auc_low"], out["cv_auc_high"] = (float(v) for v in np.percentile(boots, [2.5, 97.5]))
    return out


def best_single_input(frame: pd.DataFrame) -> dict[str, Any]:
    """The simplest possible competitor: the one input that, used alone, separates the labels best (direction chosen on the
    data, so this is a slightly generous baseline). A learned model that cannot beat it adds nothing but complexity."""
    from sklearn.metrics import roc_auc_score

    y = frame["label"].to_numpy()
    best_name, best = None, 0.5
    for name in FEATURES:
        values = frame[name].astype(float)
        values = values.fillna(values.median()) if values.notna().any() else None
        if values is None or values.nunique() < 2:
            continue
        auc = float(roc_auc_score(y, values))
        auc = max(auc, 1 - auc)
        if auc > best:
            best_name, best = name, auc
    return {"best_single_input": best_name, "best_single_input_auc": best}


def passes_gate(report: dict[str, Any]) -> bool:
    """Accurate enough (mean and the low end of the interval) AND better than the best single input by a margin."""
    if report.get("cv_auc") is None or report["cv_auc"] < GATE_AUC or (report.get("cv_auc_low") or 0) < GATE_AUC_LOW:
        return False
    return report["cv_auc"] >= (report.get("best_single_input_auc") or 0.5) + BEAT_SINGLE_INPUT_BY


def evaluate(frame: pd.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {"labeled": int(len(frame)), "wildfire": int((frame.get("label") == WILDFIRE).sum()) if len(frame) else 0,
                              "planned": int((frame.get("label") == PLANNED).sum()) if len(frame) else 0}
    if len(frame) < MIN_LABELS or min(report["wildfire"], report["planned"]) < MIN_PER_CLASS:
        report.update(status="collecting_labels", gate_passed=False, needed=f"{MIN_LABELS} labelled incidents, {MIN_PER_CLASS} of each kind")
        return report
    report.update(cross_validate(frame))
    report.update(best_single_input(frame))
    report["gate_passed"] = passes_gate(report)
    report["status"] = "ok" if report["gate_passed"] else (
        "no_better_than_one_input" if (report.get("cv_auc") or 0) >= GATE_AUC else "below_gate")
    model = make_model().fit(prepare(frame), frame["label"])
    coefficients = model[-1].coef_[0]
    names = list(model[0].get_feature_names_out(list(FEATURES))) if hasattr(model[0], "get_feature_names_out") else list(FEATURES)
    report["top_features"] = [{"feature": names[i], "weight": round(float(coefficients[i]), 3)}
                              for i in np.argsort(-np.abs(coefficients))[:6]]
    return report


def train(conn: Optional[sqlite3.Connection] = None, *, save: bool = True) -> dict[str, Any]:
    """Refit from all labels; save (and start serving) only if cross-validation clears the gate. Never raises."""
    own = conn is None
    conn = conn or _connect()
    try:
        frame = build_dataset(conn)
        report = evaluate(frame)
        report["trained_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        folder = model_dir()
        if save:
            folder.mkdir(parents=True, exist_ok=True)
        if report["gate_passed"] and save:
            import joblib

            model = make_model().fit(prepare(frame), frame["label"])
            version = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
            joblib.dump(model, folder / "model.joblib.tmp")
            os.replace(folder / "model.joblib.tmp", folder / "model.joblib")
            (folder / "meta.json").write_text(json.dumps({"version": version, "features": list(FEATURES), "labeled": report["labeled"],
                                                          "wildfire": report["wildfire"], "planned": report["planned"],
                                                          "cv_auc": report["cv_auc"], "cv_auc_low": report.get("cv_auc_low"),
                                                          "trained_at": report["trained_at"]}, indent=2))
            report["saved_version"] = version
        elif save and not report["gate_passed"]:
            # The gate failed (or labels are too few): stop serving anything stale rather than show a number nobody checked.
            for name in ("model.joblib", "meta.json"):
                (folder / name).unlink(missing_ok=True)
        if save:
            (folder / "last_report.json").write_text(json.dumps(report, indent=2, default=str))
        return report
    except Exception as error:  # noqa: BLE001
        logger.error("incident type training failed: %s", error, exc_info=True)
        return {"status": "error", "error": str(error), "gate_passed": False}
    finally:
        if own:
            conn.close()


# --------------------------------------------------------------------------------------------------- serving
_cache: dict[str, Any] = {"mtime": None, "bundle": None}
_lock = threading.Lock()


def _bundle() -> Optional[tuple[Any, dict]]:
    path = model_dir() / "model.joblib"
    try:
        mtime = path.stat().st_mtime
    except OSError:
        with _lock:
            _cache["mtime"], _cache["bundle"] = None, None
        return None
    with _lock:
        if _cache["mtime"] != mtime:
            try:
                import joblib

                _cache["bundle"] = (joblib.load(path), json.loads((model_dir() / "meta.json").read_text()))
                _cache["mtime"] = mtime
            except Exception:  # noqa: BLE001
                logger.warning("could not load the incident type model", exc_info=True)
                _cache["bundle"], _cache["mtime"] = None, None
        return _cache["bundle"]


def _last_report() -> Optional[dict]:
    try:
        return json.loads((model_dir() / "last_report.json").read_text())
    except (OSError, ValueError):
        return None


def assess(incident: dict, members: list[dict], conn: Optional[sqlite3.Connection] = None) -> Optional[dict[str, Any]]:
    """What the incident detail returns as `incident_type`. With a served model: a probability and a plain label. Without
    one: how many labels it has and needs, so the page can say "still learning" and no number is shown. Never raises."""
    try:
        bundle = _bundle()
        if bundle is None:
            report = _last_report() or {}
            return {"status": report.get("status") or "collecting_labels", "labeled": report.get("labeled"),
                    "wildfire": report.get("wildfire"), "planned": report.get("planned"),
                    "needs": f"{MIN_LABELS} labelled incidents, {MIN_PER_CLASS} of each kind, and a result better than the best single input"}
        model, meta = bundle
        own = conn is None
        conn = conn or _connect()
        try:
            t0, feats = type_features(incident, members, conn)
        finally:
            if own:
                conn.close()
        if t0 is None:
            return None
        frame = pd.DataFrame([[feats.get(name, np.nan) for name in meta["features"]]], columns=meta["features"])
        score = float(model.predict_proba(prepare(frame))[0, 1])
        label = "wildfire" if score >= SHOW_WILDFIRE_ABOVE else "planned burn" if score <= SHOW_PLANNED_BELOW else "unclear"
        return {"status": "predicted", "score": round(score, 3), "label": label, "model": meta["version"], "labeled": meta.get("labeled"),
                "cv_auc": round(float(meta.get("cv_auc") or 0), 3), "cv_auc_low": round(float(meta.get("cv_auc_low") or 0), 3),
                "note": "A relative score learned from reviewed incidents (balanced between wildfires and planned burns), not a calibrated probability."}
    except Exception:  # noqa: BLE001 - an extra on the incident page must never break it
        logger.info("incident type assessment failed", exc_info=True)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("evaluate", "train"))
    args = parser.parse_args()
    conn = _connect()
    try:
        report = evaluate(build_dataset(conn)) if args.command == "evaluate" else train(conn, save=True)
        for key, value in report.items():
            print(f"  {key:<14} {round(value, 4) if isinstance(value, float) else value}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
