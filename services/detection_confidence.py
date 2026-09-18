"""Per-detection ML confidence (0-100%): scores individual satellite fire
detections using only intrinsic detection-pattern features (brightness
temps, FRP, land cover, viewing geometry, source) - no fire-weather/RAWS
data. Complementary to, and independent from, fire_confidence.py's
incident-cluster confidence below it in the ingest pipeline.

Loads the trained model (api/detection-confidence-model/models/
detection_confidence_model.json, produced by that package's train.py)
directly by path, the same way services/model_shadow.py loads the
fire-danger-model's booster - deliberately not importing that package's
own modules, since two model-training packages under api/ both define a
module named "features"/"config" and Python only keeps one on sys.path
per name. Keep FEATURE_NAMES/build_feature_vector below in sync with
detection-confidence-model/features.py if that logic changes.
"""
import logging
import math
from pathlib import Path

from core.database import list_detection_events_for_scoring, update_detection_confidence

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parent.parent / "detection-confidence-model" / "models" / "detection_confidence_model.json"

FEATURE_NAMES = [
    "frp_log", "bright_diff", "bright_t7_norm", "pixel_area_log",
    "quality_flag", "raw_confidence", "solar_zenith_norm",
    "satellite_zenith_norm", "is_day", "cropland_frac", "water_frac",
    "is_ngfs", "is_viirs", "is_modis",
]
NAN = float("nan")

_booster = None
_booster_load_attempted = False


def _num(value):
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _raw_confidence_to_prior(value) -> float:
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


def _land_cover_fraction(land_cover, *keywords) -> float:
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
        _raw_confidence_to_prior(row.get("confidence")),
        (solar_zenith / 180.0) if solar_zenith is not None else NAN,
        (sat_zenith / 90.0) if sat_zenith is not None else NAN,
        1.0 if daynight == "D" else (0.0 if daynight == "N" else NAN),
        _land_cover_fraction(land_cover, "Cropland", "Agricult"),
        _land_cover_fraction(land_cover, "Water"),
        1.0 if source == "ngfs" else 0.0,
        1.0 if source == "viirs" else 0.0,
        1.0 if source == "modis" else 0.0,
    ]


def _heuristic_probability(features: list) -> float:
    """Calibrated fallback used until enough admin-reviewed detections
    exist to train a real model - same graceful-degradation shape as
    fire_confidence.py's incident-level scorer."""
    (frp_log, bright_diff, _bright_t7_norm, _pixel_area_log, quality_flag, raw_confidence,
     _solar_zenith_norm, _sat_zenith_norm, _is_day, cropland_frac, water_frac,
     _is_ngfs, _is_viirs, _is_modis) = features

    score = 0.40 * raw_confidence + 0.20 * min(1.0, frp_log / 6.0)
    if not math.isnan(bright_diff):
        score += 0.15 * max(0.0, min(1.0, bright_diff / 20.0))
    if quality_flag is not None and not math.isnan(quality_flag):
        score += 0.15 if quality_flag <= 1 else (0.02 if quality_flag >= 3 else 0.08)
    else:
        score += 0.08
    score -= 0.15 * cropland_frac
    score -= 0.10 * water_frac
    return max(0.0, min(1.0, score))


def _load_booster():
    global _booster, _booster_load_attempted
    if _booster_load_attempted:
        return _booster
    _booster_load_attempted = True
    if not MODEL_PATH.exists():
        return None
    try:
        import xgboost as xgb
        booster = xgb.Booster()
        booster.load_model(str(MODEL_PATH))
        _booster = booster
    except Exception as exc:
        logger.warning("detection_confidence: failed to load model at %s: %s", MODEL_PATH, exc)
        _booster = None
    return _booster


def score_detection(row: dict) -> int:
    """Return a 0-100 wildfire-vs-false-positive confidence percentage for
    a single fire_events row's detection-pattern features."""
    features = build_feature_vector(row)
    booster = _load_booster()
    probability = None
    if booster is not None:
        try:
            import xgboost as xgb
            dmat = xgb.DMatrix([features], feature_names=FEATURE_NAMES, missing=NAN)
            probability = float(booster.predict(dmat)[0])
        except Exception as exc:
            logger.warning("detection_confidence: inference failed, using heuristic fallback: %s", exc)
    if probability is None:
        probability = _heuristic_probability(features)
    return round(max(0.0, min(1.0, probability)) * 100)


def refresh_detection_confidence(limit: int = 2000) -> dict:
    rows = list_detection_events_for_scoring(limit=limit)
    scored = 0
    for row in rows:
        try:
            pct = score_detection(row)
            update_detection_confidence(row["id"], pct)
            scored += 1
        except Exception as exc:
            logger.error("detection_confidence: failed to score event %s: %s", row.get("id"), exc)

    logger.info("detection_confidence: scored %d/%d candidate detections", scored, len(rows))
    return {"scored": scored, "candidates": len(rows)}
