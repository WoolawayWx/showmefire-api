"""
Verification records for the seven-day outlook (the input of model-training/fire_outlook).

Two stages, both under <worker root>/verification/:

  forecast/{run_id}.(parquet|csv)   written by the worker with the run: one row per window x station with the
                                    forecast side only (agreement, member distributions at the station cell).
  records/{run_id}.(parquet|csv)    written by `build_records` once windows have matured: the same rows plus the
                                    later observations, observed events and quality flags. Idempotent; rewritten as
                                    more windows mature. Copy this folder to the training machine
                                    ($SMF_DATA_ROOT/fire_outlook/verification).

Columns match model-training/fire_outlook/contract.py. Window statistics (forecast member distribution AND
observation use the same definition):
    fm10_*      window-minimum 10-hour fuel moisture, % (forecast: MODELED; obs: RAWS)
    t2m_c_*     window-maximum temperature, degC
    rh_pct_*    window-minimum relative humidity, %
    wind_ms_*   window-maximum 10 m wind speed, m/s (forecast: before any rule wind factor)
    precip_mm_* window total precipitation, mm
    *_mean/_p10/_p90   mean and 10th/90th percentile over ALL members (NaN if any member is missing)
    *_obs       observed statistic; NaN unless >= MIN_OBS_HOURS of the window's 6 hours have data
    *_persist   last observation at or BEFORE issue time (within PERSIST_MAX_AGE_H), never later
obs_elevated / obs_critical apply the canonical rule + the same seasonal state as the run to hours where fuel
moisture, RH and wind were all observed; they need >= MIN_OBS_HOURS such concurrent hours, else NaN (and qc_ok
is False). Observed wind is not wind-factor-adjusted. Missing observations are never zero-filled.
Stations: RAWS stations that reported 10-hour fuel moisture recently (other stations cannot verify the rule).
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from services.fire_outlook import point, rules, windows

logger = logging.getLogger(__name__)

MIN_OBS_HOURS = 4          # of 6 hours in a window
PERSIST_MAX_AGE_H = 3
LATENCY_HOURS = 6          # wait this long after a window ends before verifying it (late observations)
KTS_PER_MS = 1.9438444924406
WEATHER = ("t2m_c", "rh_pct", "wind_ms", "precip_mm")
# forecast sample key, window statistic of the observed hourly series
SAMPLE_FOR = {"t2m_c": "temp_max_c", "rh_pct": "rh_min_pct", "wind_ms": "wind_max_kts", "precip_mm": "precip_6h_mm"}


# --- tables ---------------------------------------------------------------------------------------------

def write_table(df: pd.DataFrame, stem: Path) -> Path:
    """Parquet when pyarrow is installed, else CSV. Atomic. Returns the written path."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pyarrow  # noqa: F401
        path = stem.with_suffix(".parquet")
        tmp = path.with_name(path.name + ".tmp")
        df.to_parquet(tmp, index=False)
    except ImportError:
        path = stem.with_suffix(".csv")
        tmp = path.with_name(path.name + ".tmp")
        df.to_csv(tmp, index=False)
    tmp.replace(path)
    return path


def read_table(stem: Path) -> Optional[pd.DataFrame]:
    for suffix, reader in ((".parquet", pd.read_parquet), (".csv", pd.read_csv)):
        path = stem.with_suffix(suffix)
        if path.exists():
            return reader(path)
    return None


def verification_dir(root: Path) -> Path:
    return Path(root) / "verification"


# --- stations / observations ---------------------------------------------------------------------------

class SqliteObservations:
    """Reads the `observations` table (station_id, observation_date, fuel_moisture_percentage, temp_c,
    rel_humidity, wind_speed_ms, precip_accum_1h_mm, latitude, longitude)."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def _connect(self):
        return sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)

    def stations(self, since: datetime) -> List[dict]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT station_id, AVG(latitude), AVG(longitude) FROM observations "
                "WHERE substr(observation_date,1,10) >= ? AND fuel_moisture_percentage IS NOT NULL "
                "AND latitude IS NOT NULL AND longitude IS NOT NULL GROUP BY station_id ORDER BY station_id",
                (since.strftime("%Y-%m-%d"),)).fetchall()
        return [{"station_id": str(s), "lat": float(a), "lon": float(o)} for s, a, o in rows]

    def fuel_moisture_payload(self, cycle: datetime, max_age_hours: float = 3.0) -> dict:
        """RAWS-style payload for conditioning.causal_raws_points, built ONLY from observations at or before `cycle`
        (the newest 10-h fuel-moisture reading per station within max_age_hours). Stations carry state "" because the
        table has no state column; use condition_initial_state(raws_states=None) so every in-domain station anchors."""
        cycle = cycle.astimezone(timezone.utc)
        with self._connect() as con:
            df = pd.read_sql_query(
                "SELECT station_id, observation_date AS time, fuel_moisture_percentage AS fm10, latitude, longitude "
                "FROM observations WHERE substr(observation_date,1,10) BETWEEN ? AND ? "
                "AND fuel_moisture_percentage IS NOT NULL AND latitude IS NOT NULL AND longitude IS NOT NULL",
                con, params=[(cycle - timedelta(hours=max_age_hours + 24)).strftime("%Y-%m-%d"), cycle.strftime("%Y-%m-%d")])
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce", format="mixed")
        df = df.dropna(subset=["time"])
        pcycle = pd.Timestamp(cycle)
        df = df[(df["time"] <= pcycle) & (df["time"] >= pcycle - pd.Timedelta(hours=max_age_hours))]
        latest = df.sort_values("time").groupby("station_id").last()
        return {"stations": [
            {"stid": str(sid), "state": "", "latitude": float(r["latitude"]), "longitude": float(r["longitude"]),
             "observations": {"fuel_moisture": {"value": float(r["fm10"]), "time": r["time"].isoformat().replace("+00:00", "Z")}}}
            for sid, r in latest.iterrows()]}

    def hourly(self, station_ids: List[str], start: datetime, end: datetime) -> pd.DataFrame:
        """Last observation in each clock hour, per station, for [start, end]. Columns: station_id, hour (UTC),
        fm10, temp_c, rh, wind_ms, precip_mm."""
        if not station_ids:
            return pd.DataFrame(columns=["station_id", "hour", "fm10", "temp_c", "rh", "wind_ms", "precip_mm"])
        marks = ",".join("?" * len(station_ids))
        with self._connect() as con:
            df = pd.read_sql_query(
                "SELECT station_id, observation_date AS time, fuel_moisture_percentage AS fm10, temp_c, "
                f"rel_humidity AS rh, wind_speed_ms AS wind_ms, precip_accum_1h_mm AS precip_mm FROM observations "
                f"WHERE station_id IN ({marks}) AND substr(observation_date,1,10) BETWEEN ? AND ?",
                con, params=[*station_ids, (start - timedelta(days=1)).strftime("%Y-%m-%d"),
                             (end + timedelta(days=1)).strftime("%Y-%m-%d")])
        # format="mixed": without it pandas infers ONE format from the first row and silently NaT-drops every row
        # written differently (ISO 'T'/'Z' vs 'YYYY-MM-DD HH:MM:SS'), which loses whole stations.
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce", format="mixed")
        df = df.dropna(subset=["time"]).sort_values("time")
        df["hour"] = df["time"].dt.floor("h")
        df = df.groupby(["station_id", "hour"], as_index=False).last().drop(columns="time")
        lo, hi = pd.Timestamp(start), pd.Timestamp(end)
        return df[(df["hour"] >= lo) & (df["hour"] <= hi)].reset_index(drop=True)


def station_cells(stations: List[dict], grid: dict) -> tuple[List[dict], List[tuple[int, int]]]:
    """Stations inside the grid, with their (row, col) cells (stations outside are dropped)."""
    kept, cells = [], []
    for s in stations:
        cell = point.cell_for(s["lat"], s["lon"], grid)
        if cell is not None:
            kept.append(s)
            cells.append(cell)
    return kept, cells


# --- forecast side -----------------------------------------------------------------------------------------

def _stats(samples: np.ndarray) -> Dict[str, np.ndarray]:
    """(members, windows, stations) -> mean/p10/p90 (windows, stations); NaN if any member is NaN there."""
    bad = np.isnan(samples).any(axis=0)
    with np.errstate(all="ignore"):
        p10, p90 = np.percentile(samples, [10, 90], axis=0)
        mean = samples.mean(axis=0)
    return {k: np.where(bad, np.nan, v) for k, v in (("mean", mean), ("p10", p10), ("p90", p90))}


def forecast_rows(manifest: dict, result: dict, stations: List[dict], cells: List[tuple[int, int]],
                  reference: Optional[Dict[str, np.ndarray]] = None) -> pd.DataFrame:
    """Forecast-side rows (one per window x station) from the ensemble result's station samples.
    reference: optional NBM window statistics {var: (28, S)}; adds `<var>_nbm` columns (absent, not zero, when None)."""
    n = int(manifest["member_count"])
    samples = result["station_samples"]
    factor = float((manifest.get("driver") or {}).get("wind_factor", 1.0)) or 1.0
    rows_idx = np.array([c[0] for c in cells])
    cols_idx = np.array([c[1] for c in cells])
    elev = result["agreement_elevated"][:, rows_idx, cols_idx]
    crit = result["agreement_critical"][:, rows_idx, cols_idx]
    fm = _stats_fm(samples["fm10_min"])
    stat = {
        "t2m_c": _stats(samples["temp_max_c"]),
        "rh_pct": _stats(samples["rh_min_pct"]),
        "wind_ms": _stats(samples["wind_max_kts"] / KTS_PER_MS / factor),
        "precip_mm": _stats(samples["precip_6h_mm"]),
    }
    cycle = pd.Timestamp(manifest["cycle_time_utc"])
    frames = []
    for s, station in enumerate(stations):
        data = {
            "run_id": manifest["run_id"], "issue_time": cycle, "window_idx": np.arange(windows.N_WINDOWS),
            "station_id": station["station_id"], "lat": station["lat"], "lon": station["lon"], "n_members": n,
            "agreement_elevated": elev[:, s], "agreement_critical": crit[:, s],
            "fm10_p10": fm["p10"][:, s], "fm10_p50": fm["p50"][:, s], "fm10_p90": fm["p90"][:, s],
            "conditioning_level": (manifest.get("conditioning") or {}).get("level"),
            "rule_spec_version": (manifest.get("rules") or {}).get("rule_spec_version"),
        }
        for v in WEATHER:
            for k in ("mean", "p10", "p90"):
                data[f"{v}_{k}"] = stat[v][k][:, s]
            if reference is not None and v in reference:
                data[f"{v}_nbm"] = reference[v][:, s]
        frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _stats_fm(samples: np.ndarray) -> Dict[str, np.ndarray]:
    bad = np.isnan(samples).any(axis=0)
    with np.errstate(all="ignore"):
        p10, p50, p90 = np.percentile(samples, [10, 50, 90], axis=0)
    return {k: np.where(bad, np.nan, v) for k, v in (("p10", p10), ("p50", p50), ("p90", p90))}


def write_forecast_side(root: Path, manifest: dict, result: dict, stations: List[dict],
                        cells: List[tuple[int, int]], reference: Optional[Dict[str, np.ndarray]] = None) -> Optional[Path]:
    """Called by the worker after a run is written. Returns the file path (None if there are no stations)."""
    if "station_samples" not in result or not stations:
        return None
    df = forecast_rows(manifest, result, stations, cells, reference)
    return write_table(df, verification_dir(root) / "forecast" / manifest["run_id"])


# --- observation side ---------------------------------------------------------------------------------------

def _window_obs(hourly: pd.DataFrame, issue: pd.Timestamp, categorize: Callable, dampen: Callable,
                matured: List[int]) -> pd.DataFrame:
    """Per (station, window) observed statistics, events and coverage for the matured windows."""
    h = hourly.copy()
    h["window_idx"] = ((h["hour"] - issue) // pd.Timedelta(hours=windows.WINDOW_HOURS)).astype(int)
    h = h[h["window_idx"].isin(matured)]
    if h.empty:
        return pd.DataFrame()
    kts = h["wind_ms"] * KTS_PER_MS
    concurrent = h[["fm10", "rh", "wind_ms"]].notna().all(axis=1)
    cat = pd.Series(-1, index=h.index, dtype="int16")
    if concurrent.any():
        fm = h.loc[concurrent, "fm10"].to_numpy(float)
        rh = h.loc[concurrent, "rh"].to_numpy(float)
        w = kts[concurrent].to_numpy(float)
        c0 = categorize(fm, rh, w).astype("int16")
        cat.loc[concurrent] = dampen(c0, fm, rh, w).astype("int16")
    h["cat"], h["concurrent"] = cat, concurrent
    out = []
    for (station, w), g in h.groupby(["station_id", "window_idx"]):
        def stat(col, fn, need=MIN_OBS_HOURS):
            vals = g[col].dropna()
            return float(fn(vals)) if len(vals) >= need else np.nan
        n_conc = int(g["concurrent"].sum())
        ok = n_conc >= MIN_OBS_HOURS
        out.append({
            "station_id": station, "window_idx": int(w), "fm10_obs": stat("fm10", np.min),
            "t2m_c_obs": stat("temp_c", np.max), "rh_pct_obs": stat("rh", np.min),
            "wind_ms_obs": stat("wind_ms", np.max), "precip_mm_obs": stat("precip_mm", np.sum),
            "obs_elevated": bool((g.loc[g["concurrent"], "cat"] >= 2).any()) if ok else np.nan,
            "obs_critical": bool((g.loc[g["concurrent"], "cat"] >= 3).any()) if ok else np.nan,
            "obs_concurrent_hours": n_conc, "obs_hours_fm10": int(g["fm10"].notna().sum()),
        })
    return pd.DataFrame(out)


def _persistence(hourly: pd.DataFrame, issue: pd.Timestamp) -> pd.DataFrame:
    """Last observation at or before issue time (and no older than PERSIST_MAX_AGE_H), per station."""
    h = hourly[(hourly["hour"] <= issue) & (hourly["hour"] >= issue - pd.Timedelta(hours=PERSIST_MAX_AGE_H))]
    if h.empty:
        return pd.DataFrame(columns=["station_id"])
    last = h.sort_values("hour").groupby("station_id").last()
    return pd.DataFrame({"station_id": last.index, "fm10_init_obs": last["fm10"].to_numpy(),
                         "t2m_c_persist": last["temp_c"].to_numpy(), "rh_pct_persist": last["rh"].to_numpy(),
                         "wind_ms_persist": last["wind_ms"].to_numpy()})


def build_records(root: Path, run_id: str, obs, now: datetime, *, latency_hours: float = LATENCY_HOURS,
                  categorize: Optional[Callable] = None, dampen: Optional[Callable] = None) -> Optional[Path]:
    """Join a run's forecast rows with observations for windows that ended >= latency_hours ago.
    Returns the records file, or None when nothing has matured / the run has no forecast-side file."""
    root = Path(root)
    fc = read_table(verification_dir(root) / "forecast" / run_id)
    manifest = json.loads((root / "runs" / run_id / "manifest.json").read_text(encoding="utf-8")) \
        if (root / "runs" / run_id / "manifest.json").exists() else None
    if fc is None or manifest is None:
        return None
    issue = pd.Timestamp(manifest["cycle_time_utc"])
    now_ts = pd.Timestamp(now.astimezone(timezone.utc))
    matured = [w for w in range(windows.N_WINDOWS)
               if issue + pd.Timedelta(hours=(w + 1) * windows.WINDOW_HOURS + latency_hours) <= now_ts]
    if not matured:
        return None
    if categorize is None or dampen is None:
        categorize, dampen, _ = rules.rule_functions(
            gdd_accum=(manifest.get("rules") or {}).get("gdd_accum_since_mar1"))
    station_ids = sorted(fc["station_id"].astype(str).unique())
    hourly = obs.hourly(station_ids, (issue - pd.Timedelta(hours=PERSIST_MAX_AGE_H)).to_pydatetime(),
                        (issue + pd.Timedelta(hours=windows.HORIZON_HOURS + windows.WINDOW_HOURS)).to_pydatetime())
    fc = fc[fc["window_idx"].isin(matured)].copy()
    fc["station_id"] = fc["station_id"].astype(str)
    wobs = _window_obs(hourly, issue, categorize, dampen, matured)
    df = fc.merge(wobs, on=["station_id", "window_idx"], how="left") if not wobs.empty else fc.copy()
    for col in ("fm10_obs", "t2m_c_obs", "rh_pct_obs", "wind_ms_obs", "precip_mm_obs", "obs_elevated",
                "obs_critical", "obs_concurrent_hours", "obs_hours_fm10"):
        if col not in df:
            df[col] = np.nan
    pers = _persistence(hourly, issue)
    df = df.merge(pers, on="station_id", how="left") if len(pers) else df.assign(
        fm10_init_obs=np.nan, t2m_c_persist=np.nan, rh_pct_persist=np.nan, wind_ms_persist=np.nan)
    df["qc_ok"] = df["fm10_obs"].notna() & df["obs_elevated"].notna()
    df["obs_elevated"] = df["obs_elevated"].astype("boolean")
    df["obs_critical"] = df["obs_critical"].astype("boolean")
    df["matured_through_window"] = max(matured)
    return write_table(df, verification_dir(root) / "records" / run_id)


def build_all(root: Path, obs, now: Optional[datetime] = None, max_age_days: float = 10.0, **kwargs) -> Dict[str, Optional[str]]:
    """Update records for every run whose windows could still be maturing (cycle within max_age_days)."""
    root = Path(root)
    now = now or datetime.now(timezone.utc)
    out: Dict[str, Optional[str]] = {}
    runs_dir = root / "runs"
    if not runs_dir.is_dir():
        return out
    for run in sorted(runs_dir.iterdir()):
        if run.name.startswith(".") or not (run / "manifest.json").exists():
            continue
        cycle = datetime.strptime(run.name, "%Y%m%dT%HZ").replace(tzinfo=timezone.utc)
        if (now - cycle) > timedelta(days=max_age_days):
            continue
        try:
            path = build_records(root, run.name, obs, now, **kwargs)
            out[run.name] = None if path is None else str(path)
        except Exception as error:  # noqa: BLE001 - one bad run must not stop the others
            logger.warning("verification records for %s failed: %s", run.name, error)
            out[run.name] = f"error: {error}"
    return out
