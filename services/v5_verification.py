"""Attach mature station observations to immutable V5 shadow predictions."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from core.fire_danger import calculate_fire_danger
from services.v5_shadow import EVIDENCE_ROOT, attach_observations

RAW_ROOT = Path(os.getenv("SMF_RAW_OBSERVATION_ROOT", "archive/raw_data"))
TOLERANCE_MINUTES = 60


def _values(observations, *names):
    for name in names:
        value = observations.get(name)
        if value: return value
    return []


def load_observations(raw_root=RAW_ROOT):
    records = []
    for path in sorted(Path(raw_root).glob("raw_data_*.json")):
        try: payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception: continue
        stations = payload.get("STATION", []); stations = [stations] if isinstance(stations, dict) else stations
        for station in stations:
            observations = station.get("OBSERVATIONS", {}); times = observations.get("date_time", [])
            fm = _values(observations, "fuel_moisture_set_1", "fuel_moisture")
            rh = _values(observations, "relative_humidity_set_1", "relative_humidity")
            wind = _values(observations, "wind_speed_set_1", "wind_speed")
            for index, value in enumerate(times):
                if index >= len(fm) or index >= len(rh) or index >= len(wind): continue
                if any(series[index] is None for series in (fm, rh, wind)): continue
                records.append({"station_id": str(station.get("STID")), "observation_time": pd.to_datetime(value, utc=True),
                                "target_fm": float(fm[index]), "target_rh": float(rh[index]),
                                "target_wind_ms": float(wind[index]) * .44704})
    return pd.DataFrame(records)


def verify_pending(evidence_root=EVIDENCE_ROOT, raw_root=RAW_ROOT, now=None):
    now = pd.Timestamp(now or datetime.now(timezone.utc)); observations = load_observations(raw_root)
    if observations.empty: return {"pending": 0, "attached": 0, "matched_rows": 0}
    attached = matched = pending = 0
    for prediction_path in sorted(Path(evidence_root).glob("*.prediction.json")):
        run_id = prediction_path.name.removesuffix(".prediction.json")
        if prediction_path.with_name(f"{run_id}.observation.json").exists(): continue
        prediction = json.loads(prediction_path.read_text()); result = {}
        recorded_at = pd.to_datetime(prediction.get("recorded_at"), utc=True, errors="coerce")
        mature = True
        for row_key in prediction["row_keys"]:
            try: _, station_id, valid_string = str(row_key).split("|", 2); valid = pd.to_datetime(valid_string, utc=True)
            except Exception: continue
            if now < valid + pd.Timedelta(minutes=TOLERANCE_MINUTES): mature = False; continue
            candidates = observations[observations.station_id == station_id].copy()
            if candidates.empty: continue
            candidates["age"] = (candidates.observation_time - valid).abs().dt.total_seconds() / 60
            selected = candidates.loc[candidates.age.idxmin()]
            if selected.age > TOLERANCE_MINUTES: continue
            if pd.isna(recorded_at) or selected.observation_time <= recorded_at:
                # A verification target must have arrived after its immutable forecast.
                continue
            # One timestamp supplies FM, RH, and wind; no cross-variable joins.
            actual_category = calculate_fire_danger(selected.target_fm, selected.target_rh,
                                                     selected.target_wind_ms * 1.9438444924406)
            result[str(row_key)] = {"row_key": str(row_key), "available": actual_category is not None,
                "valid_time": valid.isoformat(), "observation_time": selected.observation_time.isoformat(),
                "match_age_minutes": float(selected.age), "source_station_id": station_id,
                "target_fm": float(selected.target_fm), "target_rh": float(selected.target_rh),
                "target_wind_ms": float(selected.target_wind_ms), "actual_category": actual_category}
        if not mature: pending += 1; continue
        attach_observations(run_id, result, evidence_root); attached += 1; matched += len(result)
    return {"pending": pending, "attached": attached, "matched_rows": matched}


if __name__ == "__main__": print(json.dumps(verify_pending(), indent=2))
