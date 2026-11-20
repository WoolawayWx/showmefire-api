"""Forecast snapshots and scoring for the Day 2 / Day 3 beta (what makes an adaptive HRRR/RRFS blend possible).

Nothing here changes a forecast. After each run, ExtendedForecast saves what every model (HRRR, RRFS, and the blend)
said at the observation stations for the mapped hours. Once those hours are over and late observations have arrived,
this module scores each snapshot against the station observations, keeps the scores, and summarises how each model has
been doing lately. The summary also says what blend weights that record WOULD pick; they are reported, never applied
(`status: "shadow"`), until a few weeks of scores show they are stable and beat the fixed weights.

Storage follows the project's own compression process, so the record is small enough to keep:
  * snapshots use hrrr_hourly's writer: every value rounded to its element's step, packed as int16, zlib-compressed
    (a few KB per run), under the same element names and steps as the hourly product;
  * score tables go through fire_outlook.verification.write_table (Parquet, CSV if pyarrow is missing).
Both are written atomically, and snapshots older than KEEP_DAYS are pruned.

    <root>/forecast/day{N}_{YYYYMMDD}_{HH}z.nc     (model, lead_hour, station) packed fields, all models in ONE file
    <root>/scores/day{N}_{YYYYMMDD}_{HH}z.parquet  one row per model x station x variable

All models of a run share one file because the fixed per-file overhead dominates at this size: three models in one
file measured 47 KB against 114 KB as three files (about 120 stations x 12 hours x 5 fields).
    <root>/summary.json                                    trailing metrics and the shadow weights

Window statistics, for the whole window and for the peak block (scope window / peak); forecast and observation use the same definition: max temperature, min relative humidity, max 10 m
wind speed, total rain. An observed statistic needs MIN_OBS_HOURS of the window's hours; nothing is zero-filled.
"""
from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger("extended_forecast.verify")

KEEP_DAYS = int(os.getenv("SMF_EXTENDED_VERIFY_KEEP_DAYS", "60"))
LATENCY_HOURS = 6              # wait this long after a window ends before scoring it (late observations)
MIN_OBS_FRACTION = 2 / 3       # of the window's hours an observed statistic needs
MAX_STATION_KM = 20.0          # a station farther than this from every model cell is not scored
WINDOW_DAYS = 21               # trailing record the summary uses
MIN_PAIRS = 150                # station-values (and MIN_RUN_DAYS distinct run days) before a weight is proposed
MIN_RUN_DAYS = 7
WEIGHT_PRIOR = 0.7             # HRRR share when there is no usable record
WEIGHT_MIN, WEIGHT_MAX = 0.3, 0.7
MPH_PER_MS = 2.236936

# Element names and steps are hrrr_hourly's, so the packing writer can be reused unchanged.
SNAPSHOT_VARS = ("temp_f", "rh_pct", "wind_speed_mph", "wind_gust_mph", "precip_mm_1h")
VARIABLES = {                  # score variable -> (snapshot variable, reducer over the window)
    "temp_max_f": ("temp_f", "max"),
    "rh_min_pct": ("rh_pct", "min"),
    "wind_max_mph": ("wind_speed_mph", "max"),
    "precip_mm": ("precip_mm_1h", "sum"),
}
_STEM = re.compile(r"^day(\d+)_(\d{8})_(\d{2})z$")


def default_root() -> Path:
    override = os.getenv("SMF_EXTENDED_VERIFY_DIR")
    if override:
        return Path(override)
    from services.beta_products import BETA_ROOT
    return Path(BETA_ROOT) / "verification" / "extended"


def stem_for(day: int, cycle: datetime) -> str:
    return f"day{day}_{cycle:%Y%m%d_%H}z"


# --- snapshots --------------------------------------------------------------------------------------------------
def nearest_cells(lon: np.ndarray, lat: np.ndarray, stations: list[dict], max_km: float = MAX_STATION_KM):
    """(station index list, (row, col) list) for stations within `max_km` of a model cell, nearest cell each."""
    kept, cells = [], []
    flat_lon, flat_lat = lon.ravel(), lat.ravel()
    for index, station in enumerate(stations):
        dlat = (flat_lat - station["lat"]) * 111.0
        dlon = (flat_lon - station["lon"]) * 111.0 * np.cos(np.radians(station["lat"]))
        distance = np.hypot(dlat, dlon)
        best = int(np.nanargmin(distance))
        if distance[best] <= max_km:
            kept.append(index)
            cells.append(np.unravel_index(best, lon.shape))
    return kept, cells


def snapshot(ds: xr.Dataset, window_leads: tuple[int, int], cycle: datetime, stations: list[dict], *, day: int,
             model: str, lon: np.ndarray, lat: np.ndarray, precip_increments: Callable, precip_name: Callable,
             ) -> xr.Dataset | None:
    """Station values for the mapped hours from a model's dataset (kelvin/percent/m/s/run-total rain in, display units out).

    Returns None when no station is within range of the grid."""
    kept, cells = nearest_cells(lon, lat, stations)
    if not kept:
        return None
    rows = np.array([c[0] for c in cells])
    cols = np.array([c[1] for c in cells])
    base = pd.Timestamp(cycle.replace(tzinfo=None))
    times = pd.DatetimeIndex(ds.time.values)
    leads = np.array([int(round((t - base) / pd.Timedelta(hours=1))) for t in times])
    wanted = np.where((leads >= window_leads[0]) & (leads <= window_leads[1]))[0]
    wanted = wanted[np.argsort(leads[wanted])]
    if wanted.size == 0:
        return None

    def at_stations(array3d):
        return np.asarray(array3d, dtype="float64")[:, rows, cols]

    temp_c = at_stations(ds.t2m.values)[wanted] - 273.15
    ws_ms = np.hypot(at_stations(ds.u10.values)[wanted], at_stations(ds.v10.values)[wanted])
    rain = precip_name(ds)
    increments = precip_increments(np.asarray(ds[rain].values, dtype="float64")) if rain else None
    gust_name = next((n for n in ("gust", "i10fg", "fg10") if n in ds), None)
    out = {
        "temp_f": temp_c * 9 / 5 + 32,
        "rh_pct": at_stations(ds.r2.values)[wanted],
        "wind_speed_mph": ws_ms * MPH_PER_MS,
        "wind_gust_mph": at_stations(ds[gust_name].values)[wanted] * MPH_PER_MS if gust_name else np.full(temp_c.shape, np.nan),
        "precip_mm_1h": increments[wanted][:, rows, cols] if increments is not None else np.full(temp_c.shape, np.nan),
    }
    dims = ("lead_hour", "station")
    data = xr.Dataset({name: (dims, values.astype("float64")) for name, values in out.items()},
                      coords={"lead_hour": leads[wanted],
                              "station_id": ("station", [str(stations[i]["station_id"]) for i in kept]),
                              "latitude": ("station", [float(stations[i]["lat"]) for i in kept]),
                              "longitude": ("station", [float(stations[i]["lon"]) for i in kept])})
    data.attrs.update(day=day, model=model, cycle_utc=cycle.isoformat().replace("+00:00", "Z"),
                      window_start_lead=int(window_leads[0]), window_end_lead=int(window_leads[1]))
    return data


def combine(snapshots: dict[str, xr.Dataset | None], attrs: dict | None = None) -> xr.Dataset | None:
    """One dataset with a `model` dimension, on the stations every model has. None if no model produced a snapshot.

    `attrs` are stored with it (the peak block's first/last lead, so scoring can judge the same hours the map shows)."""
    present = {model: data for model, data in snapshots.items() if data is not None}
    if not present:
        return None
    common = set.intersection(*(set(map(str, d.station_id.values)) for d in present.values()))
    pieces = []
    for model, data in present.items():
        keep = [i for i, s in enumerate(data.station_id.values) if str(s) in common]
        piece = data.isel(station=keep)
        pieces.append(piece.drop_vars(["latitude", "longitude"]).expand_dims(model=[model]))
    first = present[next(iter(present))]
    keep = [i for i, s in enumerate(first.station_id.values) if str(s) in common]
    out = xr.concat(pieces, dim="model")
    out = out.assign_coords(latitude=("station", first.latitude.values[keep]), longitude=("station", first.longitude.values[keep]))
    out.attrs = {**{k: v for k, v in first.attrs.items() if k != "model"}, **(attrs or {})}   # concat copies the first piece's attrs, model included
    return out


def write_snapshot(data: xr.Dataset, root: Path, day: int, cycle: datetime) -> Path:
    """int16-packed + zlib, via hrrr_hourly's writer (the project's compression process)."""
    from services.hrrr_hourly.store import write_netcdf
    path = Path(root) / "forecast" / f"{stem_for(day, cycle)}.nc"
    write_netcdf(data, path)
    return path


def read_snapshot(path: Path) -> xr.Dataset:
    with xr.open_dataset(path) as handle:       # scale_factor / _FillValue are applied on read; NaN comes back as NaN
        return handle.load()


def prune_snapshots(root: Path, now: datetime, keep_days: int = KEEP_DAYS) -> int:
    removed = 0
    cutoff = (now - timedelta(days=keep_days)).strftime("%Y%m%d")
    for folder, pattern in ((Path(root) / "forecast", "*.nc"), (Path(root) / "scores", "*.*")):
        if not folder.is_dir():
            continue
        for path in folder.glob(pattern):
            match = _STEM.match(path.stem)
            if match and match.group(2) < cutoff:
                path.unlink(missing_ok=True)
                removed += 1
    return removed


def record(snapshots: dict[str, xr.Dataset | None], root: Path, day: int, cycle: datetime,
           attrs: dict | None = None) -> list[Path]:
    """Write one file for the run, holding every model that produced a snapshot. [] if none did."""
    combined = combine(snapshots, attrs)
    return [] if combined is None else [write_snapshot(combined, root, day, cycle)]


# --- observations -----------------------------------------------------------------------------------------------
def observed_stats(hourly: pd.DataFrame, station_ids: list[str], hours: pd.DatetimeIndex) -> dict[str, dict[str, float]]:
    """Window statistics per station from `hourly` (station_id, hour UTC, temp_c, rh, wind_ms, precip_mm)."""
    need = max(1, int(np.ceil(MIN_OBS_FRACTION * len(hours))))
    frame = hourly[hourly["hour"].isin(hours)] if len(hourly) else hourly
    result: dict[str, dict[str, float]] = {}
    for station in station_ids:
        part = frame[frame["station_id"] == station] if len(frame) else frame
        stats = {name: np.nan for name in VARIABLES}
        for name, column, reducer, factor in (("temp_max_f", "temp_c", "max", None), ("rh_min_pct", "rh", "min", 1.0),
                                              ("wind_max_mph", "wind_ms", "max", MPH_PER_MS), ("precip_mm", "precip_mm", "sum", 1.0)):
            values = pd.to_numeric(part[column], errors="coerce").dropna() if len(part) else pd.Series(dtype=float)
            # A rain total with missing hours would read low, so it needs every hour; the extremes need most of them.
            if len(values) < (len(hours) if reducer == "sum" else need):
                continue
            value = float(getattr(values, reducer)())
            stats[name] = value * 9 / 5 + 32 if name == "temp_max_f" else value * factor
        result[station] = stats
    return result


def forecast_stats(data: xr.Dataset) -> dict[str, dict[str, float]]:
    stations = [str(s) for s in data.station_id.values]
    out: dict[str, dict[str, float]] = {s: {} for s in stations}
    for name, (variable, reducer) in VARIABLES.items():
        array = np.asarray(data[variable].values, dtype="float64")
        with np.errstate(all="ignore"):
            if reducer == "sum":
                value = np.where(np.isfinite(array).all(axis=0), np.nansum(array, axis=0), np.nan)
            else:
                value = np.where(np.isfinite(array).any(axis=0), getattr(np, f"nan{reducer}")(array, axis=0), np.nan)
        for station, number in zip(stations, value):
            out[station][name] = float(number)
    return out


def score_snapshot(data: xr.Dataset, hourly_provider: Callable[[list[str], datetime, datetime], pd.DataFrame],
                   now: datetime, *, latency_hours: float = LATENCY_HOURS) -> pd.DataFrame | None:
    """Rows (one per model x station x variable) for a snapshot whose window has matured; None if it has not."""
    cycle = datetime.fromisoformat(str(data.attrs["cycle_utc"]).replace("Z", "+00:00"))
    first = cycle + timedelta(hours=int(data.lead_hour.values.min()))
    last = cycle + timedelta(hours=int(data.lead_hour.values.max()))
    if now < last + timedelta(hours=latency_hours):
        return None
    stations = [str(s) for s in data.station_id.values]
    hourly = hourly_provider(stations, first, last)
    # Two scopes: the whole mapped window (steadier, what the shadow weights use) and the peak block the maps show.
    scopes = [("window", data)]
    if "peak_start_lead" in data.attrs and "peak_end_lead" in data.attrs:
        scopes.append(("peak", data.sel(lead_hour=slice(int(data.attrs["peak_start_lead"]), int(data.attrs["peak_end_lead"])))))
    rows = []
    for scope, part in scopes:
        if part.sizes["lead_hour"] == 0:
            continue
        hours = pd.DatetimeIndex([pd.Timestamp(cycle + timedelta(hours=int(h))) for h in part.lead_hour.values])
        observed = observed_stats(hourly, stations, hours)
        for model in part.model.values:
            forecast = forecast_stats(part.sel(model=model))
            for station in stations:
                for name in VARIABLES:
                    f, o = forecast[station].get(name, np.nan), observed[station].get(name, np.nan)
                    rows.append({"day": int(data.attrs["day"]), "model": str(model), "scope": scope,
                                 "cycle_utc": data.attrs["cycle_utc"], "valid_date": f"{first:%Y-%m-%d}",
                                 "station_id": station, "variable": name, "forecast": f, "observed": o,
                                 "error": f - o if np.isfinite(f) and np.isfinite(o) else np.nan})
    return pd.DataFrame(rows)


def score_matured(root: Path, hourly_provider: Callable, now: datetime | None = None) -> int:
    """Score every snapshot that has matured and has no score yet. Returns how many were scored."""
    from services.fire_outlook.verification import read_table, write_table
    now = now or datetime.now(timezone.utc)
    scored = 0
    for path in sorted((Path(root) / "forecast").glob("*.nc")):
        stem = Path(root) / "scores" / path.stem
        if read_table(stem) is not None:
            continue
        try:
            rows = score_snapshot(read_snapshot(path), hourly_provider, now)
        except Exception as error:
            logger.warning("could not score %s: %s", path.name, error)
            continue
        if rows is None:
            continue
        write_table(rows, stem)
        scored += 1
    return scored


# --- summary and shadow weights ---------------------------------------------------------------------------------
def load_scores(root: Path, now: datetime, window_days: int = WINDOW_DAYS) -> pd.DataFrame:
    from services.fire_outlook.verification import read_table
    frames = []
    cutoff = (now - timedelta(days=window_days)).strftime("%Y%m%d")
    for path in sorted((Path(root) / "scores").glob("*.*")):
        match = _STEM.match(path.stem)
        if not match or match.group(2) < cutoff:
            continue
        table = read_table(path.with_suffix(""))
        if table is not None and len(table):
            frames.append(table)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def with_scope(scores: pd.DataFrame) -> pd.DataFrame:
    """Scores written before peak blocks existed have no scope column; they were whole-window scores."""
    return scores if "scope" in scores.columns else scores.assign(scope="window")


def metrics(scores: pd.DataFrame) -> list[dict]:
    """MAE / RMSE / bias per (day, model, scope, variable) over the rows where forecast and observation both exist."""
    if scores.empty:
        return []
    good = with_scope(scores).dropna(subset=["error"])
    out = []
    for (day, model, scope, variable), part in good.groupby(["day", "model", "scope", "variable"]):
        error = part["error"].to_numpy(dtype=float)
        out.append({"day": int(day), "model": model, "scope": scope, "variable": variable, "n": int(error.size),
                    "runs": int(part["cycle_utc"].nunique()),
                    "mae": round(float(np.mean(np.abs(error))), 3), "rmse": round(float(np.sqrt(np.mean(error ** 2))), 3),
                    "bias": round(float(np.mean(error)), 3)})
    return out


def propose_weights(scores: pd.DataFrame, day: int = 2, *, primary: str = "hrrr", partner: str = "rrfs") -> dict:
    """The primary model's share per variable, by inverse squared error on station-values both models forecast.

    Clamped to [WEIGHT_MIN, WEIGHT_MAX]; a variable with too few paired values keeps the prior. Rain is never tuned."""
    result = {"prior": WEIGHT_PRIOR, "bounds": [WEIGHT_MIN, WEIGHT_MAX], "per_variable": {}, "overall": WEIGHT_PRIOR,
              "status": "shadow", "applied": False}
    if scores.empty:
        result["reason"] = "no scored runs yet"
        return result
    # Whole-window scores only: a 2-hour peak block is too few hours per station to rank models steadily.
    scores = with_scope(scores)
    day_rows = scores[(scores["day"] == day) & (scores["scope"] == "window")
                      & scores["variable"].isin(["temp_max_f", "rh_min_pct", "wind_max_mph"])].dropna(subset=["error"])
    a = day_rows[day_rows["model"] == primary][["cycle_utc", "station_id", "variable", "error", "valid_date"]]
    b = day_rows[day_rows["model"] == partner][["cycle_utc", "station_id", "variable", "error"]]
    paired = a.merge(b, on=["cycle_utc", "station_id", "variable"], suffixes=("_a", "_b"))
    shares = []
    for variable, part in paired.groupby("variable"):
        n, days = len(part), part["valid_date"].nunique()
        entry = {"pairs": int(n), "run_days": int(days)}
        if n >= MIN_PAIRS and days >= MIN_RUN_DAYS:
            mse_a, mse_b = float(np.mean(part["error_a"] ** 2)), float(np.mean(part["error_b"] ** 2))
            share = mse_b / (mse_a + mse_b) if (mse_a + mse_b) > 0 else WEIGHT_PRIOR
            entry.update(mse_primary=round(mse_a, 3), mse_partner=round(mse_b, 3),
                         primary_share=round(float(np.clip(share, WEIGHT_MIN, WEIGHT_MAX)), 3))
            shares.append(entry["primary_share"])
        else:
            entry["primary_share"] = None
        result["per_variable"][variable] = entry
    if shares:
        result["overall"] = round(float(np.mean(shares)), 3)
    else:
        result["reason"] = f"fewer than {MIN_PAIRS} paired station-values over {MIN_RUN_DAYS} run days"
    return result


def write_summary(root: Path, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    scores = load_scores(root, now)
    summary = {"generated_at": now.isoformat().replace("+00:00", "Z"), "window_days": WINDOW_DAYS,
               "metrics": metrics(scores), "weights_day2": propose_weights(scores, 2)}
    path = Path(root) / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return summary


def station_list(db_path=None, since_days: int = 30) -> list[dict]:
    """RAWS stations that reported recently (the observation source scoring uses)."""
    from services.fire_outlook.verification import SqliteObservations
    if db_path is None:
        from core.database import get_db_path
        db_path = get_db_path()
    return SqliteObservations(Path(db_path)).stations(datetime.now(timezone.utc) - timedelta(days=since_days))


def hourly_provider(db_path=None) -> Callable:
    from services.fire_outlook.verification import SqliteObservations
    if db_path is None:
        from core.database import get_db_path
        db_path = get_db_path()
    observations = SqliteObservations(Path(db_path))
    return lambda ids, start, end: observations.hourly(ids, start, end)
