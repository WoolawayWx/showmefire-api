from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final


@dataclass(frozen=True)
class GridDefinition:
    id: str
    crs: str
    width: int
    height: int
    resolution_m: int
    west: float
    north: float

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (
            self.west,
            self.north - self.height * self.resolution_m,
            self.west + self.width * self.resolution_m,
            self.north,
        )


# Missouri plus the existing one-degree research buffer, snapped to 3 km.
PUBLIC_GRID: Final = GridDefinition(
    id="mo-buffer-utm15n-3km-v1",
    crs="EPSG:32615",
    width=267,
    height=264,
    resolution_m=3000,
    west=150_000.0,
    north=4_647_000.0,
)

HORIZON_HOURS: Final = 72
TIME_COUNT: Final = 73
# The v1 extended forecast owns only the second and third forecast days.
# Day 1 is produced by the operational forecast pipeline and must not be
# replaced by an extended-run promotion.
EXTENDED_DAILY_INDICES: Final = (2, 3)
LOCAL_TIMEZONE: Final = "America/Chicago"
CONVECTION_ALLOWING_MODELS: Final = frozenset({"hrrr", "rrfs"})
DETERMINISTIC_MODELS: Final = frozenset({"hrrr", "rrfs"})

REQUIRED_VARIABLES: Final = (
    "temperature_2m",
    "relative_humidity_2m",
    "wind_u_10m",
    "wind_v_10m",
    "wind_gust_10m",
    "precipitation_increment",
    "shortwave_down",
    "cloud_cover",
    "mixing_height",
    "soil_moisture",
    "snow_water_equivalent",
)
OPTIONAL_VARIABLES: Final = (
    "dewpoint_2m",
    "temperature_850hpa",
    "temperature_700hpa",
    "wind_u_850hpa",
    "wind_v_850hpa",
    "wind_u_700hpa",
    "wind_v_700hpa",
)

VARIABLE_UNITS: Final = {
    "temperature_2m": "degC",
    "dewpoint_2m": "degC",
    "relative_humidity_2m": "%",
    "wind_u_10m": "m s-1",
    "wind_v_10m": "m s-1",
    "wind_speed_10m": "m s-1",
    "wind_gust_10m": "m s-1",
    "precipitation_increment": "mm",
    "precipitation_accumulated": "mm",
    "shortwave_down": "W m-2",
    "cloud_cover": "%",
    "mixing_height": "m",
    "soil_moisture": "1",
    "snow_water_equivalent": "mm",
    "fuel_moisture_p10": "%",
    "fuel_moisture_p50": "%",
    "fuel_moisture_p90": "%",
    "fire_danger": "category",
    "meteorological_confidence": "%",
    "category_confidence": "%",
    "probability_rh_le_25": "%",
    "probability_gust_ge_30mph": "%",
    "probability_concurrent": "%",
}


@dataclass(frozen=True)
class ArchiveEncoding:
    dtype: str
    scale_factor: float | None
    fill_value: int | float


ARCHIVE_ENCODINGS: Final = {
    "temperature_2m": ArchiveEncoding("int16", 0.01, -32768),
    "dewpoint_2m": ArchiveEncoding("int16", 0.01, -32768),
    "relative_humidity_2m": ArchiveEncoding("uint16", 0.1, 65535),
    "cloud_cover": ArchiveEncoding("uint16", 0.1, 65535),
    "fuel_moisture_p10": ArchiveEncoding("uint16", 0.1, 65535),
    "fuel_moisture_p50": ArchiveEncoding("uint16", 0.1, 65535),
    "fuel_moisture_p90": ArchiveEncoding("uint16", 0.1, 65535),
    "wind_u_10m": ArchiveEncoding("int16", 0.01, -32768),
    "wind_v_10m": ArchiveEncoding("int16", 0.01, -32768),
    "wind_speed_10m": ArchiveEncoding("uint16", 0.01, 65535),
    "wind_gust_10m": ArchiveEncoding("uint16", 0.01, 65535),
    "precipitation_increment": ArchiveEncoding("uint16", 0.1, 65535),
    "precipitation_accumulated": ArchiveEncoding("uint16", 0.1, 65535),
    "shortwave_down": ArchiveEncoding("uint16", 1.0, 65535),
    "mixing_height": ArchiveEncoding("uint16", 1.0, 65535),
    "soil_moisture": ArchiveEncoding("uint16", 0.0001, 65535),
    "snow_water_equivalent": ArchiveEncoding("uint16", 0.1, 65535),
    "fire_danger": ArchiveEncoding("uint8", None, 255),
    "meteorological_confidence": ArchiveEncoding("uint8", None, 255),
    "category_confidence": ArchiveEncoding("uint8", None, 255),
    "probability_rh_le_25": ArchiveEncoding("uint8", None, 255),
    "probability_gust_ge_30mph": ArchiveEncoding("uint8", None, 255),
    "probability_concurrent": ArchiveEncoding("uint8", None, 255),
    "source_mask": ArchiveEncoding("uint16", None, 65535),
    "quality_mask": ArchiveEncoding("uint16", None, 65535),
}

BLEND_WEIGHTS: Final = (
    (0, 18, {"hrrr": 0.50, "rrfs": 0.40, "refs": 0.10}),
    (19, 36, {"hrrr": 0.40, "rrfs": 0.50, "refs": 0.10}),
    (37, 48, {"hrrr": 0.25, "rrfs": 0.60, "refs": 0.15}),
    (49, 60, {"rrfs": 0.75, "refs": 0.25}),
    (61, 72, {"rrfs": 0.65, "refs": 0.35}),
)

CORRECTION_LIMITS: Final = {
    "temperature_2m": 2.8,
    "relative_humidity_2m": 8.0,
    "wind_u_10m": 2.25,
    "wind_v_10m": 2.25,
    "wind_gust_10m": 2.25,
}

SOURCE_BITS: Final = {"hrrr": 1, "rrfs": 2, "refs": 4, "gefs": 8, "rtma": 16, "observations": 32}
QUALITY_BITS: Final = {
    "source_degraded": 1,
    "optional_field_missing": 2,
    "fuel_spatial_init": 4,
    "fuel_physics_init": 8,
    "correction_capped": 16,
    "insufficient_confidence": 32,
    "coarse_synoptic_fallback": 64,
}

PUBLIC_LAYER_STYLES: Final = {
    "temperature_2m": {"colormap": "turbo", "rescale": [-20.0, 45.0]},
    "relative_humidity_2m": {"colormap": "blues", "rescale": [0.0, 100.0]},
    "wind_speed_10m": {"colormap": "viridis", "rescale": [0.0, 20.0]},
    "wind_gust_10m": {"colormap": "magma", "rescale": [0.0, 30.0]},
    "precipitation_increment": {"colormap": "blues", "rescale": [0.0, 15.0]},
    "precipitation_accumulated": {"colormap": "blues", "rescale": [0.0, 50.0]},
    "fuel_moisture_p10": {"colormap": "rdylgn", "rescale": [0.0, 30.0]},
    "fuel_moisture_p50": {"colormap": "rdylgn", "rescale": [0.0, 30.0]},
    "fuel_moisture_p90": {"colormap": "rdylgn", "rescale": [0.0, 30.0]},
    "fire_danger": {"colormap": "fire_danger", "rescale": [0.0, 4.0]},
    "meteorological_confidence": {"colormap": "viridis", "rescale": [0.0, 100.0]},
    "category_confidence": {"colormap": "viridis", "rescale": [0.0, 100.0]},
    "probability_rh_le_25": {"colormap": "magma", "rescale": [0.0, 100.0]},
    "probability_gust_ge_30mph": {"colormap": "magma", "rescale": [0.0, 100.0]},
    "probability_concurrent": {"colormap": "magma", "rescale": [0.0, 100.0]},
}


def weights_for_lead(lead_hour: int) -> dict[str, float]:
    for start, end, weights in BLEND_WEIGHTS:
        if start <= lead_hour <= end:
            return dict(weights)
    raise ValueError(f"lead hour outside 0-{HORIZON_HOURS}: {lead_hour}")


def correction_multiplier(lead_hour: int) -> float:
    if lead_hour <= 24:
        return 1.0
    if lead_hour <= 48:
        return 0.75
    return 0.5


def utc_rfc3339(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def run_id_for_cycle(cycle_time: datetime) -> str:
    if cycle_time.tzinfo is None:
        cycle_time = cycle_time.replace(tzinfo=timezone.utc)
    return cycle_time.astimezone(timezone.utc).strftime("%Y%m%dT%H%MZ") + "-v1"
