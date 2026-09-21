"""DB-backed registry of forecast_v1 source models beyond the core four
(hrrr/rrfs/refs/gefs, whose weights live in contracts.BLEND_WEIGHTS).

This is what the admin "add/promote/schedule/performance" interface reads and
writes. Adding a row here does not add code - the adapter class and its
Herbie fetch parameters (ADAPTER_ACQUISITION in this module) still have to
exist for a given adapter_key. What the registry controls is whether an
already-coded adapter is acquired at all (status), how much it counts in the
live blend (weight, only once 'active'), how often it's attempted
(schedule_minutes), and where its accuracy numbers get logged.

CORE_MODEL_KEYS are excluded from the weight overlay so promoting/demoting a
non-core model can never touch the proven hrrr/rrfs/refs/gefs weighting in
contracts.BLEND_WEIGHTS.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone

from core.database import get_db_path
from .acquisition import AcquisitionSpec
from .contracts import HORIZON_HOURS

CORE_MODEL_KEYS = frozenset({"hrrr", "rrfs", "refs", "gefs"})

# Herbie fetch parameters for adapters the registry can enable beyond the
# core four. Keyed by adapter_key (must also exist in adapters.ADAPTERS).
ADAPTER_ACQUISITION: dict[str, dict] = {
    "fv3hires": dict(
        herbie_model="hiresw", product="fv3_2p5km", domain="conus",
        leads=tuple(range(0, 49)), members=(None,),
    ),
}


@contextmanager
def _db():
    db = sqlite3.connect(get_db_path(), timeout=30)
    db.row_factory = sqlite3.Row
    try:
        with db:
            yield db
    finally:
        db.close()


def _log_event(db, model_key: str, action: str, detail: str = "") -> None:
    db.execute(
        "INSERT INTO forecast_source_model_events(id,model_key,action,detail) VALUES (?,?,?,?)",
        (str(uuid.uuid4()), model_key, action, detail),
    )


def list_models() -> list[dict]:
    with _db() as db:
        rows = db.execute("SELECT * FROM forecast_source_models ORDER BY key").fetchall()
    return [dict(row) for row in rows]


def get_model(key: str) -> dict | None:
    with _db() as db:
        row = db.execute("SELECT * FROM forecast_source_models WHERE key=?", (key,)).fetchone()
    return dict(row) if row else None


def add_model(key: str, display_name: str, adapter_key: str, notes: str = "") -> dict:
    if adapter_key not in ADAPTER_ACQUISITION:
        raise ValueError(f"no acquisition parameters registered for adapter '{adapter_key}'")
    with _db() as db:
        existing = db.execute("SELECT 1 FROM forecast_source_models WHERE key=?", (key,)).fetchone()
        if existing:
            raise ValueError(f"a model with key '{key}' already exists")
        db.execute(
            "INSERT INTO forecast_source_models(key,display_name,adapter_key,status,notes) VALUES (?,?,?,'disabled',?)",
            (key, display_name, adapter_key, notes),
        )
        _log_event(db, key, "added", display_name)
    return get_model(key)


def set_status(key: str, status: str, weight: float | None = None) -> dict:
    """Move a model between disabled/shadow/active.

    'active' requires an explicit weight (0-1) - there is no sane default
    weight for a model nobody has validated yet, so promotion refuses to
    guess one.
    """
    if status not in {"disabled", "shadow", "active"}:
        raise ValueError(f"invalid status: {status}")
    if key in CORE_MODEL_KEYS:
        raise ValueError(f"'{key}' is a core model; its weighting is fixed in contracts.BLEND_WEIGHTS")
    model = get_model(key)
    if not model:
        raise ValueError(f"no such model: {key}")
    if status == "active":
        if weight is None:
            weight = json.loads(model["weight_profile_json"]).get("weight") if model["weight_profile_json"] else None
        if weight is None or not (0 < weight <= 1):
            raise ValueError("promoting to 'active' requires an explicit weight in (0, 1]")
    weight_profile_json = json.dumps({"weight": weight}) if status == "active" else model["weight_profile_json"]
    with _db() as db:
        db.execute(
            """UPDATE forecast_source_models
               SET status=?, weight_profile_json=?, updated_at=CURRENT_TIMESTAMP,
                   promoted_at=CASE WHEN ?='active' THEN CURRENT_TIMESTAMP ELSE promoted_at END
               WHERE key=?""",
            (status, weight_profile_json, status, key),
        )
        action = "promoted" if status == "active" else ("demoted" if model["status"] == "active" else "disabled" if status == "disabled" else "updated")
        _log_event(db, key, action, f"{model['status']} -> {status}" + (f" weight={weight}" if weight else ""))
    return get_model(key)


def set_schedule(key: str, schedule_minutes: int | None) -> dict:
    with _db() as db:
        db.execute(
            "UPDATE forecast_source_models SET schedule_minutes=?, updated_at=CURRENT_TIMESTAMP WHERE key=?",
            (schedule_minutes, key),
        )
        _log_event(db, key, "updated", f"schedule_minutes={schedule_minutes}")
    return get_model(key)


def mark_acquired(key: str) -> None:
    with _db() as db:
        db.execute(
            "UPDATE forecast_source_models SET last_acquired_at=CURRENT_TIMESTAMP WHERE key=?", (key,),
        )


def record_metric(model_key: str, cycle_time: datetime, variable: str, lead_hour: int,
                   available: bool, mean_abs_diff_from_blend: float | None = None) -> None:
    record_metrics_bulk([(model_key, cycle_time, variable, lead_hour, available, mean_abs_diff_from_blend)])


def record_metrics_bulk(rows: list[tuple[str, datetime, str, int, bool, float | None]]) -> None:
    """Log one row per (model, variable, lead) so performance_summary's AVG()
    can compute per-cycle availability/error without engine.py pre-aggregating.
    Called once per blend_sources() run with every row batched, not per-insert,
    since a single run can produce (variables x lead hours) rows per model.
    """
    if not rows:
        return
    with _db() as db:
        db.executemany(
            """INSERT INTO forecast_source_model_metrics
               (model_key,cycle_time,variable,lead_hour,available,mean_abs_diff_from_blend)
               VALUES (?,?,?,?,?,?)""",
            [
                (model_key, cycle_time.astimezone(timezone.utc).isoformat(), variable, lead_hour,
                 1 if available else 0, mean_abs_diff_from_blend)
                for model_key, cycle_time, variable, lead_hour, available, mean_abs_diff_from_blend in rows
            ],
        )


def performance_summary(model_key: str, limit_cycles: int = 20) -> dict:
    with _db() as db:
        rows = db.execute(
            """SELECT cycle_time, variable,
                      AVG(available) AS availability, AVG(mean_abs_diff_from_blend) AS mean_abs_diff
               FROM forecast_source_model_metrics
               WHERE model_key=? AND cycle_time IN (
                   SELECT DISTINCT cycle_time FROM forecast_source_model_metrics
                   WHERE model_key=? ORDER BY cycle_time DESC LIMIT ?
               )
               GROUP BY cycle_time, variable ORDER BY cycle_time DESC""",
            (model_key, model_key, limit_cycles),
        ).fetchall()
    return {"model_key": model_key, "samples": [dict(row) for row in rows]}


def _due(model: dict, now: datetime) -> bool:
    schedule_minutes = model["schedule_minutes"]
    if not schedule_minutes:
        return True
    if not model["last_acquired_at"]:
        return True
    last = datetime.fromisoformat(str(model["last_acquired_at"]).replace("Z", "+00:00"))
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    return (now.astimezone(timezone.utc) - last).total_seconds() >= schedule_minutes * 60


def extra_specs(now: datetime | None = None) -> tuple[AcquisitionSpec, ...]:
    """AcquisitionSpecs for enabled non-core models, throttled by schedule_minutes."""
    now = now or datetime.now(timezone.utc)
    specs = []
    for model in list_models():
        if model["key"] in CORE_MODEL_KEYS or model["status"] == "disabled":
            continue
        params = ADAPTER_ACQUISITION.get(model["adapter_key"])
        if not params or not _due(model, now):
            continue
        specs.append(AcquisitionSpec(
            model["key"], params["herbie_model"], params["product"], params["leads"],
            params["members"], domain=params.get("domain"), required=False,
            role="shadow" if model["status"] == "shadow" else "operational",
        ))
    return tuple(specs)


def weight_overlay(base_weights: dict[str, float], lead_hour: int) -> dict[str, float]:
    """Merge in 'active' non-core models' flat weight, renormalizing to 1.0.

    Only called from engine.blend_sources - contracts.BLEND_WEIGHTS/weights_for_lead
    stay untouched so the proven core weighting is never at risk from this table.
    """
    if not (0 <= lead_hour <= HORIZON_HOURS):
        return base_weights
    extra: dict[str, float] = {}
    for model in list_models():
        if model["key"] in CORE_MODEL_KEYS or model["status"] != "active" or not model["weight_profile_json"]:
            continue
        weight = json.loads(model["weight_profile_json"]).get("weight")
        if weight:
            extra[model["key"]] = float(weight)
    if not extra:
        return base_weights
    reserved = sum(extra.values())
    if reserved >= 1:
        reserved = 0.99
    scale = (1 - reserved) / sum(base_weights.values())
    merged = {name: value * scale for name, value in base_weights.items()}
    merged.update(extra)
    return merged
