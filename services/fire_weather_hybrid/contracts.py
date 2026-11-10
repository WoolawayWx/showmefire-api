"""Shared grid, clock, units and feature contracts; no application dependencies."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
from zoneinfo import ZoneInfo

import numpy as np

LOCAL_TZ = ZoneInfo("America/Chicago")
LABELS = ("Low", "Moderate", "Elevated", "Critical", "Extreme")
COLORS = ("#90EE90", "#FFED4E", "#FFA500", "#FF0000", "#8B0000")
FUELS = (1, 10, 100, 1000)
WEATHER = ("temp_c", "dewpoint_c", "wind_u_ms", "wind_v_ms", "gust_ms", "precip_mm", "solar_wm2")
STATIC = ("elevation_m", "slope_deg", "aspect_sin", "aspect_cos", "canopy_cover_pct", "canopy_height_m", "fuel_model_code", "burnable_fraction")
FEATURES = (*WEATHER, "rh_pct", "wind_speed_ms", "lead_hour", "hour_sin", "hour_cos", "season_sin", "season_cos", *STATIC,
            "fm1_pct", "fm10_pct", "fm100_pct", "fm1000_pct", "hdw", "haines", "observation_age_hours",
            "wind_20ft_ms", "gfdi", "office_wind_threshold_ms", "office_is_dvn", "fuel_ready")
BEHAVIOR = ("ros_m_min", "intensity_kw_m")
UNITS = {"temp_c": "degC", "dewpoint_c": "degC", "rh_pct": "%", "wind_u_ms": "m/s", "wind_v_ms": "m/s",
         "wind_speed_ms": "m/s", "wind_20ft_ms": "m/s", "gust_ms": "m/s", "precip_mm": "mm", "solar_wm2": "W/m2",
         "ros_m_min": "m/min", "intensity_kw_m": "kW/m", "flame_length_m": "m", "danger_category": "category",
         "red_flag_probability": "%", "red_flag_criteria": "0/1", "hdw": "hPa m/s", "haines": "2-6",
         **{f"fm{fuel}_pct": "% dry weight" for fuel in FUELS}}


@dataclass(frozen=True)
class Grid:
    id: str = "mo-buffer-utm15n-1km-v1"
    crs: str = "EPSG:32615"
    width: int = 801
    height: int = 792
    resolution_m: int = 1000
    west: float = 150000.0
    north: float = 4647000.0

    def __post_init__(self):
        if min(self.width, self.height, self.resolution_m) <= 0:
            raise ValueError("grid dimensions and resolution must be positive")

    @property
    def x(self):
        return self.west + (np.arange(self.width) + .5) * self.resolution_m

    @property
    def y(self):
        return self.north - (np.arange(self.height) + .5) * self.resolution_m

    @property
    def transform(self):
        from rasterio.transform import from_origin
        return from_origin(self.west, self.north, self.resolution_m, self.resolution_m)

    @property
    def fingerprint(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def cell(self, latitude: float, longitude: float):
        from pyproj import Transformer
        x, y = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True).transform(longitude, latitude)
        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError("point outside grid")
        col, row = int(np.floor((x - self.west) / self.resolution_m)), int(np.floor((self.north - y) / self.resolution_m))
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise ValueError("point outside grid")
        return row, col


GRID = Grid()


def utc(value) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00")) if not isinstance(value, datetime) else value
    if parsed.tzinfo is None:
        raise ValueError("timestamps require an explicit timezone")
    return parsed.astimezone(timezone.utc)


def iso(value) -> str:
    return utc(value).isoformat().replace("+00:00", "Z")


def day_hours(day: date, days: int = 1) -> list[datetime]:
    if days not in (1, 2):
        raise ValueError("supported days: 1 or 2")
    start = datetime.combine(day, datetime.min.time(), LOCAL_TZ).astimezone(timezone.utc)
    end = datetime.combine(day + timedelta(days=days), datetime.min.time(), LOCAL_TZ).astimezone(timezone.utc)
    return [start + timedelta(hours=i) for i in range(int((end-start).total_seconds() // 3600))]


def feature_matrix(fields: dict, valid_time, issued_at, shape) -> np.ndarray:
    """C-order cell matrix with identical columns for training and inference."""
    valid, issued = utc(valid_time), utc(issued_at)
    local = valid.astimezone(LOCAL_TZ)
    values = {**fields, "lead_hour": (valid-issued).total_seconds()/3600,
              "hour_sin": np.sin(local.hour * np.pi/12), "hour_cos": np.cos(local.hour * np.pi/12),
              "season_sin": np.sin(local.timetuple().tm_yday * 2*np.pi/365.2425),
              "season_cos": np.cos(local.timetuple().tm_yday * 2*np.pi/365.2425)}
    return np.stack([np.broadcast_to(np.asarray(values.get(name, np.nan)), shape).ravel() for name in FEATURES], axis=1).astype("float32")

