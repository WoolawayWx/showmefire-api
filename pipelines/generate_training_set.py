"""Build auditable, causal observation/weather pairs for model training."""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.database import get_db_path
from core.precipitation import PRECIPITATION_CONTRACT_SHA256, PRECIPITATION_CONTRACT_VERSION

DEFAULT_TOLERANCE_MINUTES = 60
DEFAULT_MIN_COVERAGE = 0.80


def align_observations_to_weather(observations, weather, tolerance_minutes=60):
    obs, wx = observations.copy(), weather.copy()
    obs["obs_time"] = pd.to_datetime(obs["obs_time"], errors="coerce", utc=True)
    wx["weather_time"] = pd.to_datetime(wx["weather_time"], errors="coerce", utc=True)
    if obs["obs_time"].isna().any() or wx["weather_time"].isna().any():
        raise ValueError("Invalid timestamps found in observation or weather data")
    obs = obs.drop_duplicates(["station_id", "obs_time"], keep="last")
    wx = wx.drop_duplicates(["station_id", "weather_time"], keep="last")
    obs = obs.sort_values(["obs_time", "station_id"], kind="stable")
    wx = wx.sort_values(["weather_time", "station_id"], kind="stable")
    matched = pd.merge_asof(
        obs, wx, left_on="obs_time", right_on="weather_time", by="station_id",
        direction="backward", tolerance=pd.Timedelta(minutes=tolerance_minutes),
        allow_exact_matches=True,
    )
    matched["match_age_minutes"] = (
        matched["obs_time"] - matched["weather_time"]
    ).dt.total_seconds().div(60)
    if (matched["match_age_minutes"].dropna() < 0).any():
        raise ValueError("Causality violation: future weather matched to an observation")
    return matched


def _read_sources(connection):
    observations = pd.read_sql_query("""
        SELECT o.id AS observation_id, o.station_id,
               o.observation_date AS obs_time,
               o.fuel_moisture_percentage AS target_fm, s.lat, s.lon
        FROM observations o JOIN stations s ON s.id = o.station_id
        WHERE o.fuel_moisture_percentage IS NOT NULL
    """, connection)
    weather = pd.read_sql_query("""
        SELECT wf.id AS weather_feature_id, wf.snapshot_id, wf.station_id,
               snap.snapshot_date AS weather_time, wf.temp_c,
               wf.rel_humidity, wf.wind_speed_ms,
               COALESCE(wf.precip_interval_mm, wf.precip_mm) AS precip_mm,
               wf.precip_mm AS precip_accum_mm,
               wf.precip_interval_hours
        FROM weather_features wf JOIN snapshots snap ON snap.id = wf.snapshot_id
    """, connection)
    return observations, weather


def generate_training_set(output_path="data/training_set_mo.csv",
                          metadata_path="data/training_set_mo_meta.json",
                          tolerance_minutes=DEFAULT_TOLERANCE_MINUTES,
                          min_coverage=DEFAULT_MIN_COVERAGE):
    db_path = get_db_path()
    with sqlite3.connect(db_path) as connection:
        observations, weather = _read_sources(connection)
    if observations.empty or weather.empty:
        raise ValueError("Observations and weather features are both required")
    paired = align_observations_to_weather(observations, weather, tolerance_minutes)
    matched = paired[paired["weather_time"].notna()].copy()
    coverage = len(matched) / len(observations)
    if coverage < min_coverage:
        raise ValueError(f"Weather match coverage {coverage:.1%} is below required {min_coverage:.1%}")
    output, metadata = Path(output_path), Path(metadata_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(output, index=False)
    metadata.write_text(json.dumps({
        "database": str(db_path),
        "data_match_policy": {"direction": "nearest_prior", "tolerance_minutes": tolerance_minutes,
                              "minimum_coverage": min_coverage},
        "observation_count": len(observations), "matched_count": len(matched),
        "coverage": coverage, "max_match_age_minutes": float(matched["match_age_minutes"].max()),
        "precipitation_contract_version": PRECIPITATION_CONTRACT_VERSION,
        "precipitation_contract_sha256": PRECIPITATION_CONTRACT_SHA256,
    }, indent=2), encoding="utf-8")
    print(f"Created {len(matched)} causal training pairs at {output} ({coverage:.1%} coverage)")
    return matched


def main():
    parser = argparse.ArgumentParser(description="Generate causal fuel-moisture training pairs")
    parser.add_argument("--output", default="data/training_set_mo.csv")
    parser.add_argument("--metadata", default="data/training_set_mo_meta.json")
    parser.add_argument("--tolerance-minutes", type=int, default=60)
    parser.add_argument("--min-coverage", type=float, default=0.80)
    args = parser.parse_args()
    generate_training_set(args.output, args.metadata, args.tolerance_minutes, args.min_coverage)


if __name__ == "__main__":
    main()
