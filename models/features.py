"""Shared causal feature engineering and model feature contracts."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


FEATURE_SCHEMA_VERSION = "2.0.0"
DEFAULT_FEATURES = [
    "temp_c", "rel_humidity", "wind_speed_ms", "hour", "month",
    "emc_baseline", "temp_mean_3h", "rh_mean_3h", "temp_mean_6h",
    "rh_mean_6h", "precip_1h", "precip_3h", "precip_6h", "precip_24h",
    "hours_since_rain", "hour_sin", "hour_cos", "day_of_year_sin",
    "day_of_year_cos",
]
LEGACY_FEATURES = DEFAULT_FEATURES[:15]
DEFAULT_CLIPS = {
    "temp_c": (-60.0, 60.0),
    "rel_humidity": (0.0, 100.0),
    "wind_speed_ms": (0.0, 75.0),
    "precip_mm": (0.0, 500.0),
}


@dataclass(frozen=True)
class FeatureContract:
    columns: tuple[str, ...]
    schema_version: str = FEATURE_SCHEMA_VERSION
    max_age_minutes: int = 60

    def as_metadata(self) -> dict:
        return {
            "feature_columns": list(self.columns),
            "feature_schema_version": self.schema_version,
            "max_feature_age_minutes": self.max_age_minutes,
        }


def build_causal_features(
    frame: pd.DataFrame,
    *,
    station_col: str = "station_id",
    time_col: str = "obs_time",
    clip_ranges: Mapping[str, tuple[float, float]] = DEFAULT_CLIPS,
) -> pd.DataFrame:
    """Build timestamp-based features using current and prior observations only."""
    required = {station_col, time_col, "temp_c", "rel_humidity", "wind_speed_ms"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing feature source columns: {missing}")

    df = frame.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
    if df[time_col].isna().any():
        raise ValueError("Feature timestamps contain invalid or missing values")
    df = df.sort_values([station_col, time_col], kind="stable")
    df = df.rename(columns={time_col: "obs_time"}) if time_col != "obs_time" else df

    out_of_range = pd.Series(False, index=df.index)
    for column, (lower, upper) in clip_ranges.items():
        if column in df:
            numeric = pd.to_numeric(df[column], errors="coerce")
            out_of_range |= numeric.notna() & ~numeric.between(lower, upper)
            df[column] = numeric.clip(lower, upper)

    df["hour"] = df["obs_time"].dt.hour
    df["month"] = df["obs_time"].dt.month
    day_of_year = df["obs_time"].dt.dayofyear
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    df["emc_baseline"] = df["rel_humidity"] / 5.0
    if "precip_mm" not in df:
        df["precip_mm"] = 0.0
    df["feature_out_of_range"] = out_of_range.reindex(df.index, fill_value=False)

    indexed = df.set_index("obs_time")
    for hours in (3, 6):
        rolling = indexed.groupby(station_col, sort=False)[["temp_c", "rel_humidity"]].rolling(
            f"{hours}h", min_periods=1, closed="both"
        ).mean().reset_index(level=0, drop=True)
        indexed[f"temp_mean_{hours}h"] = rolling["temp_c"]
        indexed[f"rh_mean_{hours}h"] = rolling["rel_humidity"]
    for hours in (1, 3, 6, 24):
        rolling_precip = indexed.groupby(station_col, sort=False)["precip_mm"].rolling(
            f"{hours}h", min_periods=1, closed="both"
        ).sum().reset_index(level=0, drop=True)
        indexed[f"precip_{hours}h"] = rolling_precip
    df = indexed.reset_index()
    rain_time = df["obs_time"].where(df["precip_mm"].fillna(0) > 0.1)
    last_rain = rain_time.groupby(df[station_col], sort=False).ffill()
    df["hours_since_rain"] = (
        (df["obs_time"] - last_rain).dt.total_seconds().div(3600).clip(lower=0)
    ).fillna(24.0).clip(upper=24.0)
    return df


def validate_feature_contract(frame: pd.DataFrame, metadata: Mapping) -> list[str]:
    expected = metadata.get("feature_columns") or metadata.get("feature_names")
    if not isinstance(expected, list) or not expected:
        raise ValueError("Model metadata does not contain a feature column contract")
    schema = metadata.get("feature_schema_version")
    if schema and schema not in {"1.0.0", FEATURE_SCHEMA_VERSION}:
        raise ValueError(f"Unsupported feature schema version: {schema}")
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required model features: {missing}")
    return expected


def feature_ranges(frame: pd.DataFrame, columns: Iterable[str]) -> dict:
    ranges = {}
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        ranges[column] = {"min": float(values.min()), "max": float(values.max())}
    return ranges
