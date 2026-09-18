"""Detection-pattern feature engineering for the per-detection confidence
model. Deliberately excludes any fire-weather/RAWS data - only intrinsic
properties of the satellite detection itself (brightness, FRP, viewing
geometry, land cover, source) are used.

NOTE: services/detection_confidence.py (the live-process scorer) keeps its
own small copy of this logic rather than importing this module directly,
to avoid a sys.path module-name collision with other model-training
packages (see api/fire-danger-model, and services/model_shadow.py which
loads that package's model the same direct way). Keep the two in sync.
"""
import math

FEATURE_NAMES = [
    "frp_log",
    "bright_diff",
    "bright_t7_norm",
    "pixel_area_log",
    "quality_flag",
    "raw_confidence",
    "solar_zenith_norm",
    "satellite_zenith_norm",
    "is_day",
    "cropland_frac",
    "water_frac",
    "is_ngfs",
    "is_viirs",
    "is_modis",
]

NAN = float("nan")


def _num(value):
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def raw_confidence_to_prior(value) -> float:
    text = str(value or "").lower()
    if text == "nominal":
        return 0.82
    if text in {"high", "90"}:
        return 0.92
    if text in {"medium", "probable"}:
        return 0.68
    if text in {"low", "0", "false"}:
        return 0.42
    try:
        return max(0.0, min(1.0, float(value) / 100.0))
    except (TypeError, ValueError):
        return 0.50


def land_cover_fraction(land_cover, *keywords) -> float:
    """land_cover looks like 'Cropland:91,Developed:6,Trees:1,Grass/Herbs:1,Water:1'"""
    if not land_cover:
        return 0.0
    total = 0.0
    for part in str(land_cover).split(","):
        name, _, pct = part.partition(":")
        if any(keyword.lower() in name.strip().lower() for keyword in keywords):
            try:
                total += float(pct) / 100.0
            except ValueError:
                continue
    return total


def build_feature_vector(row: dict) -> list:
    frp = _num(row.get("frp"))
    bright_t7 = _num(row.get("bright_t7"))
    bright_t13 = _num(row.get("bright_t13"))
    pixel_area = _num(row.get("pixel_area"))
    quality_flag = _num(row.get("quality_flag"))
    solar_zenith = _num(row.get("solar_zenith_angle"))
    sat_zenith = _num(row.get("satellite_zenith_angle"))
    daynight = str(row.get("daynight") or "").upper()
    source = str(row.get("source") or "").lower()
    land_cover = row.get("land_cover")

    return [
        math.log1p(frp) if frp is not None and frp > 0 else 0.0,
        (bright_t13 - bright_t7) if bright_t13 is not None and bright_t7 is not None else NAN,
        ((bright_t7 - 273.0) / 50.0) if bright_t7 is not None else NAN,
        math.log1p(pixel_area) if pixel_area is not None and pixel_area > 0 else NAN,
        quality_flag if quality_flag is not None else NAN,
        raw_confidence_to_prior(row.get("confidence")),
        (solar_zenith / 180.0) if solar_zenith is not None else NAN,
        (sat_zenith / 90.0) if sat_zenith is not None else NAN,
        1.0 if daynight == "D" else (0.0 if daynight == "N" else NAN),
        land_cover_fraction(land_cover, "Cropland", "Agricult"),
        land_cover_fraction(land_cover, "Water"),
        1.0 if source == "ngfs" else 0.0,
        1.0 if source == "viirs" else 0.0,
        1.0 if source == "modis" else 0.0,
    ]
