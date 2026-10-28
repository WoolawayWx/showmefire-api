"""Performance tracking for the fire-danger models, side by side, against real fire occurrence by county and day.

Nothing here changes any model or any public output. It copies each model's county-day predictions out of its evidence
files (and the public rule's county table) into fire_danger_model_scores, then measures how well each one ranks
county-days with a confirmed fire above the rest.

Models tracked (those that store a county-level result):
  fire_weather_index   continuous score + 5 categories   (model-shadow/fire-weather-index)
  risk_fusion_glm      P(>= 1 fire) per county           (model-shadow/risk-fusion-glm)
  ensemble:<track>     categorical + P(>= Elevated)      (model-shadow/ensemble-fire-danger; daytime-peak cycles only)
  public_rule          the public rule-based category    (county_forecast_days)
Not tracked: fire_weather_ml (it stores map summaries, no county result).

There is no production "blend" of these models: each scores on its own. The "consensus" row is an analysis only (the mean
category across model families), there to show whether combining them would help; nothing serves it.

Outcome: a county-day counts as a fire day when an APPROVED fire event at or above min_tier (default admin_reviewed) falls
on it in America/Chicago local time, excluding prescribed, agricultural and not_a_fire causes (the same convention as the
risk-fusion training labels). A day with no event is NOT proof of no fire: reports and reviews lag, so only dates at least
`lag_days` old are scored, and every result carries its counts so thin data reads as thin.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from core import database

logger = logging.getLogger(__name__)

CHICAGO = ZoneInfo("America/Chicago")
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
EVIDENCE_DIRS = {
    "fire_weather_index": Path(os.getenv("SMF_FIRE_WEATHER_INDEX_EVIDENCE_ROOT") or DATA_DIR / "model-shadow" / "fire-weather-index"),
    "risk_fusion_glm": Path(os.getenv("SMF_RISK_FUSION_GLM_EVIDENCE_ROOT") or DATA_DIR / "model-shadow" / "risk-fusion-glm"),
    "ensemble": Path(os.getenv("SMF_ENSEMBLE_FD_EVIDENCE_ROOT") or DATA_DIR / "model-shadow" / "ensemble-fire-danger"),
}
# A run only counts as a forecast for its valid date if it was produced before this local hour that day; later reruns
# have seen part of the day. Runs excluded by it are reported, never silently dropped.
CUTOFF_HOUR_LOCAL = int(os.getenv("FIRE_DANGER_TRACKING_CUTOFF_HOUR", "14"))
# Ensemble cycles whose peak is that local day (the daytime run); other cycles' valid day is ambiguous and is skipped.
ENSEMBLE_DAY_CYCLES_UTC = range(9, 16)
EXCLUDED_CAUSES = ("prescribed", "agricultural", "not_a_fire")
ELEVATED = 2  # category index at/above which a model is "alerting" (0 Low .. 4 Extreme)
MIN_EVENTS_TO_CONCLUDE = 10
MIN_DATES_FOR_CI = 10
LABELS = {"fire_weather_index": "Fire Weather Severity Index (shadow, HRRR only)",
          "fire_weather_index:blend": "Fire Weather Severity Index (shadow, HRRR+RRFS+FV3 blend)", "risk_fusion_glm": "Risk fusion GLM (shadow)",
          "public_rule": "Public rule-based Peak Fire Danger"}
BLEND_MODEL = "fire_weather_index:blend"  # same family as fire_weather_index, so the consensus counts it once
PROBABILITY_OF_FIRE = {"risk_fusion_glm"}  # models whose probability column means P(>= 1 fire in the county that day)


def model_label(model: str) -> str:
    if model.startswith("ensemble:"):
        return f"Ensemble fire danger ({model.split(':', 1)[1]} track, shadow)"
    return LABELS.get(model, model)


# ------------------------------------------------------------------ ingest

def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or database.get_db_path(), timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def _local_date(ts: datetime) -> str:
    return ts.astimezone(CHICAGO).strftime("%Y-%m-%d")


def _parse_utc(value) -> Optional[datetime]:
    if value is None or value == "":
        return None
    try:
        t = pd.to_datetime(str(value).replace("Z", "+00:00"), utc=True)
        return t.to_pydatetime()
    except Exception:
        return None


def _has_run(conn, model: str, run_id: str, valid_date: str) -> bool:
    return conn.execute("SELECT 1 FROM fire_danger_model_scores WHERE model = ? AND run_id = ? AND valid_date = ? LIMIT 1",
                        (model, run_id, valid_date)).fetchone() is not None


def _insert(conn, rows: list[tuple]) -> int:
    before = conn.total_changes
    conn.executemany(
        '''INSERT OR IGNORE INTO fire_danger_model_scores
           (model, run_id, valid_date, county_fips, score, category, probability, produced_at, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''', rows)
    return conn.total_changes - before


def _num(v):
    try:
        f = float(v)
        return None if f != f else f
    except (TypeError, ValueError):
        return None


def _cat(v):
    f = _num(v)
    return None if f is None else int(round(f))


def ingest_scores(db_path: Optional[Path] = None, dirs: Optional[dict] = None) -> dict:
    """Copy any evidence not yet in the table. Idempotent; never raises on one bad file. Returns counts by model."""
    dirs = {**EVIDENCE_DIRS, **(dirs or {})}
    out: dict[str, Any] = {"added": {}, "skipped_files": 0, "unreadable": 0}
    conn = _connect(db_path)

    def add(model, n):
        out["added"][model] = out["added"].get(model, 0) + n

    try:
        for path in sorted(Path(dirs["fire_weather_index"]).glob("*.fire_weather_index_score.json")):
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
                run_id, valid = str(rec["run_id"]), str(rec["valid_local_date"])
                if _has_run(conn, "fire_weather_index", run_id, valid):
                    out["skipped_files"] += 1
                    continue
                rows = [("fire_weather_index", run_id, valid, str(f), _num(sc), _cat(c), None, rec.get("recorded_at"), path.name)
                        for f, sc, c in zip(rec["county_fips"], rec["score"], rec["category"])]
                add("fire_weather_index", _insert(conn, rows))
            except Exception as exc:
                out["unreadable"] += 1
                logger.warning("fire danger tracking: %s unreadable: %s", path.name, exc)

        for path in sorted(Path(dirs["fire_weather_index"]).glob("*.fire_weather_index_blend_score.json")):
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
                run_id, valid = str(rec["run_id"]), str(rec["valid_local_date"])
                if _has_run(conn, BLEND_MODEL, run_id, valid):
                    out["skipped_files"] += 1
                    continue
                rows = [(BLEND_MODEL, run_id, valid, str(f), _num(sc), _cat(c), None, rec.get("recorded_at"), path.name)
                        for f, sc, c in zip(rec["county_fips"], rec["score"], rec["category"])]
                add(BLEND_MODEL, _insert(conn, rows))
            except Exception as exc:
                out["unreadable"] += 1
                logger.warning("fire danger tracking: %s unreadable: %s", path.name, exc)

        for path in sorted(Path(dirs["risk_fusion_glm"]).glob("*.glm_score.json")):
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
                run_id, valid = str(rec["run_id"]), str(rec["valid_local_date"])
                if _has_run(conn, "risk_fusion_glm", run_id, valid):
                    out["skipped_files"] += 1
                    continue
                rows = [("risk_fusion_glm", run_id, valid, str(f), _num(lam), None, _num(p), rec.get("recorded_at"), path.name)
                        for f, lam, p in zip(rec["county_fips"], rec["lam"], rec["p_ge1_fire"])]
                add("risk_fusion_glm", _insert(conn, rows))
            except Exception as exc:
                out["unreadable"] += 1
                logger.warning("fire danger tracking: %s unreadable: %s", path.name, exc)

        for path in sorted(Path(dirs["ensemble"]).glob("*.ensemble_fire_danger.json")):
            try:
                rec = json.loads(path.read_text(encoding="utf-8"))
                anchor = _parse_utc(rec.get("anchor_cycle_utc"))
                if anchor is None or anchor.hour not in ENSEMBLE_DAY_CYCLES_UTC:
                    out["skipped_files"] += 1
                    continue
                valid = _local_date(anchor)
                for name, track in (rec.get("tracks") or {}).items():
                    cats = track.get("county_max_categorical")
                    if not cats:
                        continue
                    model, run_id = f"ensemble:{name}", str(rec["run_id"])
                    if _has_run(conn, model, run_id, valid):
                        continue
                    probs = (track.get("county_max_neighborhood_probability") or {}).get("elevated") or {}
                    rows = [(model, run_id, valid, str(f), _num(probs.get(f)), _cat(c), _num(probs.get(f)), rec.get("recorded_at"), path.name)
                            for f, c in cats.items()]
                    add(model, _insert(conn, rows))
            except Exception as exc:
                out["unreadable"] += 1
                logger.warning("fire danger tracking: %s unreadable: %s", path.name, exc)

        try:
            groups = conn.execute(
                '''SELECT forecast_date, forecast_run_id, published_at FROM county_forecast_days
                   GROUP BY forecast_date, forecast_run_id, published_at''').fetchall()
        except sqlite3.OperationalError:
            groups = []
        for g in groups:
            run_id = f"{g['forecast_run_id'] or 'rule'}@{g['published_at']}"
            if _has_run(conn, "public_rule", run_id, g["forecast_date"]):
                continue
            rows = [("public_rule", run_id, r["forecast_date"], str(r["county_fips"]), None, int(r["danger_level"]), None,
                     r["published_at"], "county_forecast_days")
                    for r in conn.execute('''SELECT forecast_date, county_fips, danger_level, published_at FROM county_forecast_days
                                             WHERE forecast_date = ? AND forecast_run_id = ? AND published_at = ?''',
                                          (g["forecast_date"], g["forecast_run_id"], g["published_at"]))]
            add("public_rule", _insert(conn, rows))
        conn.commit()
        return out
    finally:
        conn.close()


# ------------------------------------------------------------------ outcomes

def fire_county_days(conn, min_tier: str = "admin_reviewed") -> pd.DataFrame:
    """(county_fips, date) with the number of qualifying approved events and the sources involved."""
    from core.fire_events import TIER_RANK
    tiers = [t for t, r in TIER_RANK.items() if r >= TIER_RANK[min_tier]]
    marks = ",".join("?" for _ in tiers)
    ex = ",".join("?" for _ in EXCLUDED_CAUSES)
    rows = conn.execute(
        f'''SELECT county_fips, occurred_at, source, verification_tier FROM fire_events
            WHERE status = 'approved' AND verification_tier IN ({marks}) AND county_fips IS NOT NULL AND county_fips != ''
              AND COALESCE(cause_category, '') NOT IN ({ex})''', (*tiers, *EXCLUDED_CAUSES)).fetchall()
    recs = []
    for r in rows:
        t = _parse_utc(r["occurred_at"])
        if t is not None:
            recs.append({"county_fips": str(r["county_fips"]), "date": _local_date(t), "source": r["source"]})
    if not recs:
        return pd.DataFrame(columns=["county_fips", "date", "events", "sources"])
    df = pd.DataFrame(recs)
    g = df.groupby(["county_fips", "date"])
    return g.agg(events=("source", "size"), sources=("source", lambda s: ",".join(sorted(set(s))))).reset_index()


# ------------------------------------------------------------------ metrics

def _latest_runs(conn, since: Optional[str], until: Optional[str]) -> tuple[pd.DataFrame, dict]:
    """One run per (model, valid date): the latest produced before the local cutoff that day. Also counts the runs the
    cutoff excluded."""
    clauses, params = [], []
    if since:
        clauses.append("valid_date >= ?")
        params.append(since)
    if until:
        clauses.append("valid_date <= ?")
        params.append(until)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    df = pd.read_sql_query(
        f"SELECT model, run_id, valid_date, county_fips, score, category, probability, produced_at FROM fire_danger_model_scores {where}",
        conn, params=params)
    stats = {"runs_after_cutoff_excluded": {}}
    if df.empty:
        return df, stats
    produced = pd.to_datetime(df["produced_at"], utc=True, errors="coerce")
    cutoff = pd.to_datetime(df["valid_date"] + f" {CUTOFF_HOUR_LOCAL:02d}:00", errors="coerce").dt.tz_localize(CHICAGO, ambiguous="NaT", nonexistent="NaT")
    ok = produced.isna() | cutoff.isna() | (produced <= cutoff)
    runs = df.assign(ok=ok).groupby(["model", "run_id", "valid_date"])["ok"].first().reset_index()
    for model, g in runs[~runs["ok"]].groupby("model"):
        stats["runs_after_cutoff_excluded"][model] = int(len(g))
    df = df[ok.to_numpy()]
    if df.empty:
        return df, stats
    df = df.assign(_p=pd.to_datetime(df["produced_at"], utc=True, errors="coerce").astype("int64"))
    latest = df.groupby(["model", "valid_date"])[["_p"]].transform("max")
    keep = df["_p"] == latest["_p"]
    return df[keep].drop(columns=["_p"]).reset_index(drop=True), stats


def _auc_ap(y, s):
    from sklearn.metrics import average_precision_score, roc_auc_score
    if len(np.unique(y)) < 2:
        return None, None
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def _date_bootstrap_auc(y, s, dates, n_boot: int = 200, seed: int = 0):
    from sklearn.metrics import roc_auc_score
    uniq = np.unique(dates)
    if len(uniq) < MIN_DATES_FOR_CI:
        return None
    idx = {d: np.flatnonzero(dates == d) for d in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx[d] for d in pick])
        if len(np.unique(y[rows])) < 2:
            continue
        vals.append(roc_auc_score(y[rows], s[rows]))
    if len(vals) < 20:
        return None
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def _rises(rows: list[dict]) -> Optional[bool]:
    """Does the fire rate go up with category? A drop between neighbouring categories is allowed when it is within sampling
    error (about 1.64 standard errors of the difference), so a single fire in a low category cannot fail the check.
    Categories with fewer than 20 county-days are ignored; None when fewer than 3 remain."""
    use = [r for r in rows if r["county_days"] >= 20]
    if len(use) < 3:
        return None
    for a, b in zip(use, use[1:]):
        drop = a["fire_rate"] - b["fire_rate"]
        if drop <= 0:
            continue
        pooled = (a["fire_days"] + b["fire_days"]) / (a["county_days"] + b["county_days"])
        se = (pooled * (1 - pooled) * (1 / a["county_days"] + 1 / b["county_days"])) ** 0.5
        if drop > 1.64 * se:
            return False
    return True


def _model_metrics(df: pd.DataFrame, model: str, ci: bool = True) -> dict:
    """df: rows for one model with columns score, category, probability, y (0/1), valid_date."""
    y = df["y"].to_numpy(dtype=int)
    out: dict[str, Any] = {"n_county_days": int(len(df)), "dates": int(df["valid_date"].nunique()), "events": int(y.sum()),
                           "base_rate": _r(y.mean()) if len(y) else None}
    if out["events"] < MIN_EVENTS_TO_CONCLUDE:
        out["note"] = f"only {out['events']} fire county-days in this window; too few to conclude (need >= {MIN_EVENTS_TO_CONCLUDE})"
    cat = df["category"]
    if cat.notna().any():
        c = cat.fillna(-1).to_numpy()
        rows = []
        for k in range(5):
            m = c == k
            if m.any():
                rows.append({"category": k, "county_days": int(m.sum()), "fire_days": int(y[m].sum()),
                             "fire_rate": _r(y[m].mean())})
        out["by_category"] = rows
        out["rate_rises_with_category"] = _rises(rows)
        alert = c >= ELEVATED
        hits = int((alert & (y == 1)).sum())
        out["elevated_or_higher"] = {"alert_share": _r(alert.mean()), "fire_days_caught": hits,
                                     "hit_rate": _r(hits / y.sum()) if y.sum() else None,
                                     "precision": _r(hits / alert.sum()) if alert.any() else None,
                                     "lift_vs_base_rate": _r((hits / alert.sum()) / y.mean()) if alert.any() and y.mean() > 0 else None}
    score = df["score"] if df["score"].notna().any() else df["probability"] if df["probability"].notna().any() else cat
    usable = score.notna().to_numpy()
    if usable.sum() >= 30 and out["events"] >= 5:
        sv = score.to_numpy(dtype=float)[usable]
        auc, ap = _auc_ap(y[usable], sv)
        out["auc"], out["average_precision"] = _r(auc), _r(ap)
        if ci:
            out["auc_ci95_by_date"] = _date_bootstrap_auc(y[usable], sv, df["valid_date"].to_numpy()[usable])
    if model in PROBABILITY_OF_FIRE and df["probability"].notna().any():
        p = df["probability"].fillna(0).clip(0, 1).to_numpy()
        out["brier"] = _r(float(np.mean((p - y) ** 2)))
        out["brier_prevalence_only"] = _r(float(np.mean((y.mean() - y) ** 2)))
        out["mean_predicted_vs_observed"] = [_r(p.mean()), _r(y.mean())]
    return out


def _consensus(frames: dict[str, pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Mean category across model FAMILIES (one track per ensemble) per county-day, only where >= 2 families have one."""
    fams: dict[str, str] = {}
    for m, f in frames.items():
        if f["category"].notna().any():
            fam = m.split(":", 1)[0]
            if fam not in fams or len(f) > len(frames[fams[fam]]):
                fams[fam] = m
    if len(fams) < 2:
        return None
    parts = [frames[m][["valid_date", "county_fips", "category"]].rename(columns={"category": fam}) for fam, m in fams.items()]
    merged = parts[0]
    for p in parts[1:]:
        merged = merged.merge(p, on=["valid_date", "county_fips"], how="outer")
    cols = list(fams)
    merged["n_models"] = merged[cols].notna().sum(axis=1)
    merged = merged[merged["n_models"] >= 2].copy()
    merged["category"] = merged[cols].mean(axis=1)
    merged["families"] = ",".join(cols)
    return merged


def evaluate(window_days: int = 90, lag_days: int = 3, min_tier: str = "admin_reviewed", today: Optional[str] = None,
             db_path: Optional[Path] = None) -> dict:
    """Metrics for every tracked model over the last `window_days` evaluable dates and over all time, plus a same-rows
    comparison and the consensus analysis. Reads the table; call ingest_scores() first (the nightly job does)."""
    conn = _connect(db_path)
    try:
        today_d = datetime.strptime(today, "%Y-%m-%d").date() if today else datetime.now(CHICAGO).date()
        last = (today_d - timedelta(days=lag_days)).strftime("%Y-%m-%d")
        fire = fire_county_days(conn, min_tier)
        runs, run_stats = _latest_runs(conn, None, last)
        result: dict[str, Any] = {
            "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "outcome": {"definition": f"approved fire events at tier >= {min_tier}, excluding {', '.join(EXCLUDED_CAUSES)}, "
                                      "counted on the America/Chicago date; a day with no event is not proof of no fire",
                        "min_tier": min_tier, "lag_days": lag_days, "latest_scored_date": last,
                        "fire_county_days_total": int(len(fire)),
                        "by_source": (fire["sources"].str.split(",").explode().value_counts().to_dict() if len(fire) else {})},
            "cutoff": {"hour_local": CUTOFF_HOUR_LOCAL, **run_stats},
            "not_tracked": {"fire_weather_ml": "stores map-level summaries only, no county result"},
            "notes": ["Models were not all running over the same dates; compare them on the 'same_rows' block, not the totals.",
                      "There is no production blend of these models; 'consensus' is an analysis of the mean category."],
        }
        if runs.empty:
            result.update(status="no_scores", message="No model scores are stored yet. Run the ingest once forecasts have written evidence.",
                          windows={})
            return result
        runs["county_fips"] = runs["county_fips"].astype(str)
        fire_set = {(r.county_fips, r.date): int(r.events) for r in fire.itertuples()} if len(fire) else {}
        runs["y"] = [1 if (f, d) in fire_set else 0 for f, d in zip(runs["county_fips"], runs["valid_date"])]
        dates_sorted = sorted(runs["valid_date"].unique())
        window_start = dates_sorted[max(0, len(dates_sorted) - window_days)]
        result["coverage"] = {m: {"first_date": g["valid_date"].min(), "last_date": g["valid_date"].max(),
                                  "dates": int(g["valid_date"].nunique()), "county_days": int(len(g)),
                                  "label": model_label(m)} for m, g in runs.groupby("model")}
        windows = {}
        for name, start in (("recent", window_start), ("all_time", None)):
            sub = runs if start is None else runs[runs["valid_date"] >= start]
            frames = {m: g.reset_index(drop=True) for m, g in sub.groupby("model")}
            models = {m: {"label": model_label(m), **_model_metrics(g, m)} for m, g in frames.items()}
            block: dict[str, Any] = {"from": sub["valid_date"].min(), "to": sub["valid_date"].max(), "models": models}

            # same rows for everyone: only county-days every tracked model scored, so totals are comparable
            keys = None
            for g in frames.values():
                k = set(zip(g["valid_date"], g["county_fips"]))
                keys = k if keys is None else keys & k
            if keys and len(frames) > 1:
                same = {}
                for m, g in frames.items():
                    mask = [(d, f) in keys for d, f in zip(g["valid_date"], g["county_fips"])]
                    same[m] = {"label": model_label(m), **_model_metrics(g[mask].reset_index(drop=True), m, ci=False)}
                block["same_rows"] = {"county_days": len(keys), "models": same}
            else:
                block["same_rows"] = {"county_days": 0, "note": "no county-days were scored by every model yet" if len(frames) > 1 else "only one model has scores"}

            cons = _consensus(frames)
            if cons is not None and len(cons):
                cons["y"] = [1 if (f, d) in fire_set else 0 for f, d in zip(cons["county_fips"], cons["valid_date"])]
                ck = set(zip(cons["valid_date"], cons["county_fips"]))
                cons = cons.assign(score=cons["category"], probability=np.nan)
                cons_m = _model_metrics(cons, "consensus", ci=False)
                fams = cons["families"].iloc[0].split(",")
                singles = {}
                for fam in fams:
                    m = next(mm for mm in frames if mm.split(":", 1)[0] == fam)
                    g = frames[m]
                    mask = [(d, f) in ck for d, f in zip(g["valid_date"], g["county_fips"])]
                    gg = g[mask].reset_index(drop=True)
                    singles[fam] = {"auc": _model_metrics(gg, m, ci=False).get("auc"),
                                    "elevated_or_higher": _model_metrics(gg, m, ci=False).get("elevated_or_higher"), "county_days": int(len(gg))}
                block["consensus"] = {"label": "Consensus (mean category across model families) - analysis only, not served",
                                      "families": fams, "county_days": int(len(cons)), "consensus": cons_m, "each_family_on_the_same_rows": singles}
            windows[name] = block
        result["windows"] = windows
        result["status"] = "ok"
        return result
    finally:
        conn.close()


# ------------------------------------------------------------------ scheduler entry

def tracking_job() -> dict:
    """Nightly: ingest new evidence. Metrics are computed on request from the table. Never raises."""
    try:
        out = ingest_scores()
        logger.info("fire danger tracking ingest: %s", out)
        return out
    except Exception as exc:
        logger.warning("fire danger tracking ingest failed: %s", exc, exc_info=True)
        return {"error": str(exc)}
