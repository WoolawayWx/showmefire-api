from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from core.database import get_db_path
from .adapters import SourceCube
from .contracts import HORIZON_HOURS, weights_for_lead


TOLERANCES = {
    "temperature_2m": 3.0,
    "relative_humidity_2m": 15.0,
    "wind_speed_10m": 4.0,
    "wind_gust_10m": 5.0,
}
COMPONENT_WEIGHTS = {
    "model_agreement": 0.30,
    "ensemble_spread": 0.25,
    "cycle_consistency": 0.20,
    "rolling_verification": 0.15,
}


@dataclass(frozen=True)
class ConfidenceInputs:
    components: dict[str, xr.DataArray]
    available_components: tuple[str, ...]


def _score(error: xr.DataArray, tolerance: float) -> xr.DataArray:
    return (100.0 * np.exp(-0.5 * (error / tolerance) ** 2)).clip(0, 100).astype("float32")


def _field(cube: SourceCube, variable: str) -> xr.DataArray | None:
    if variable == "wind_speed_10m":
        if "wind_u_10m" not in cube.dataset or "wind_v_10m" not in cube.dataset:
            return None
        values = np.hypot(cube.dataset.wind_u_10m, cube.dataset.wind_v_10m)
    elif variable in cube.dataset:
        values = cube.dataset[variable]
    else:
        return None
    return values


def _median_scores(scores: list[xr.DataArray], template: xr.DataArray) -> xr.DataArray:
    if not scores:
        return xr.full_like(template, np.nan, dtype="float32")
    return xr.concat([score.broadcast_like(template) for score in scores], dim="confidence_variable").median(
        "confidence_variable", skipna=True
    ).astype("float32")


def model_agreement(cubes: dict[str, SourceCube], template: xr.DataArray) -> xr.DataArray:
    variable_scores: list[xr.DataArray] = []
    for variable, tolerance in TOLERANCES.items():
        hourly: list[xr.DataArray] = []
        for lead in range(template.sizes["time"]):
            expected = set(weights_for_lead(min(lead, HORIZON_HOURS)))
            candidates = []
            for model, cube in cubes.items():
                if model not in expected or lead >= cube.dataset.sizes.get("time", 0):
                    continue
                field = _field(cube, variable)
                if field is None:
                    continue
                value = field.mean("member", skipna=True).isel(time=lead) if "member" in field.dims else field.isel(time=lead)
                if bool(value.notnull().any()):
                    candidates.append(value)
            if len(candidates) < 2:
                hourly.append(xr.full_like(template.isel(time=lead), np.nan, dtype="float32"))
            else:
                stacked = xr.concat(candidates, dim="confidence_source")
                spread = stacked.std("confidence_source", skipna=True).where(
                    stacked.count("confidence_source") >= 2
                )
                hourly.append(_score(spread, tolerance))
        variable_scores.append(xr.concat(hourly, dim="time").assign_coords(time=template.time))
    return _median_scores(variable_scores, template)


def ensemble_spread(cubes: dict[str, SourceCube], template: xr.DataArray) -> xr.DataArray:
    variable_scores: list[xr.DataArray] = []
    for variable, tolerance in TOLERANCES.items():
        candidates = []
        for model in ("refs", "gefs"):
            cube = cubes.get(model)
            if cube is None or cube.dataset.sizes.get("member", 1) < 2:
                continue
            field = _field(cube, variable)
            if field is None:
                continue
            spread = field.std("member", skipna=True).where(field.count("member") >= 2).reindex(time=template.time)
            candidates.append(_score(spread, tolerance))
        if candidates:
            variable_scores.append(xr.concat(candidates, dim="confidence_ensemble").median("confidence_ensemble", skipna=True))
    return _median_scores(variable_scores, template)


def cycle_consistency(current: xr.Dataset, previous: xr.Dataset | None) -> xr.DataArray:
    template = current.temperature_2m
    if previous is None:
        return xr.full_like(template, np.nan, dtype="float32")
    scores = []
    for variable, tolerance in TOLERANCES.items():
        if variable not in current or variable not in previous:
            continue
        prior = previous[variable].reindex(time=current.time)
        scores.append(_score(abs(current[variable] - prior), tolerance))
    return _median_scores(scores, template)


def load_previous_forecast(
    cycle: datetime, db_path: str | Path | None = None,
) -> xr.Dataset | None:
    database = Path(db_path or get_db_path())
    if not database.exists():
        return None
    with sqlite3.connect(database) as connection:
        row = connection.execute(
            """SELECT a.local_path FROM forecast_assets a
               JOIN forecast_runs r ON r.run_id=a.run_id
               WHERE r.status IN ('complete','superseded') AND r.cycle_time_utc<?
                 AND a.kind='cube' AND a.local_path IS NOT NULL
               ORDER BY r.cycle_time_utc DESC LIMIT 1""",
            (cycle.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),),
        ).fetchone()
    if not row or not Path(row[0]).is_file():
        return None
    with xr.open_dataset(row[0]) as dataset:
        variables = [name for name in TOLERANCES if name in dataset]
        return dataset[variables].load() if variables else None


def rolling_verification(
    template: xr.DataArray, cycle: datetime, db_path: str | Path | None = None,
) -> xr.DataArray:
    database = Path(db_path or get_db_path())
    if not database.exists():
        return xr.full_like(template, np.nan, dtype="float32")
    cutoff = (cycle.astimezone(timezone.utc) - timedelta(days=30)).isoformat().replace("+00:00", "Z")
    with sqlite3.connect(database) as connection:
        eligible_buckets = {
            bucket for bucket, samples, stations in connection.execute(
                """SELECT lead_bucket,COUNT(*) AS samples,COUNT(DISTINCT station_id) AS stations
                   FROM forecast_verification
                   WHERE qc_eligible=1 AND valid_time_utc>=? AND valid_time_utc<?
                     AND variable IN ('temperature_2m','relative_humidity_2m','wind_speed_10m','wind_gust_10m')
                   GROUP BY lead_bucket""",
                (cutoff, cycle.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")),
            ) if samples >= 30 and stations >= 3
        }
        rows = connection.execute(
            """SELECT variable,lead_bucket,AVG(ABS(error)) AS mae
               FROM forecast_verification
               WHERE qc_eligible=1 AND valid_time_utc>=? AND valid_time_utc<?
                 AND variable IN ('temperature_2m','relative_humidity_2m','wind_speed_10m','wind_gust_10m')
               GROUP BY variable,lead_bucket""",
            (cutoff, cycle.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")),
        ).fetchall()
    by_bucket: dict[str, list[float]] = {}
    for variable, bucket, mae in rows:
        if bucket not in eligible_buckets or mae is None:
            continue
        by_bucket.setdefault(bucket, []).append(float(100.0 * np.exp(-0.5 * (float(mae) / TOLERANCES[variable]) ** 2)))
    values = np.full(template.sizes["time"], np.nan, dtype=np.float32)
    for lead in range(len(values)):
        bucket = "0-24" if lead <= 24 else "25-48" if lead <= 48 else "49-72"
        if by_bucket.get(bucket):
            values[lead] = float(np.median(by_bucket[bucket]))
    return xr.DataArray(values, dims="time", coords={"time": template.time}).broadcast_like(template)


def build_confidence_inputs(
    cubes: dict[str, SourceCube], current: xr.Dataset, cycle: datetime,
    *, previous: xr.Dataset | None = None, db_path: str | Path | None = None,
) -> ConfidenceInputs:
    template = current.temperature_2m
    components = {
        "model_agreement": model_agreement(cubes, template),
        "ensemble_spread": ensemble_spread(cubes, template),
        "cycle_consistency": cycle_consistency(current, previous),
        "rolling_verification": rolling_verification(template, cycle, db_path),
    }
    available = tuple(name for name, values in components.items() if bool(values.notnull().any()))
    return ConfidenceInputs(components, available)
