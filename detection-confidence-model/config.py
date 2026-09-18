from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

LATEST_MODEL_PATH = MODELS_DIR / "detection_confidence_model.json"
LATEST_MODEL_META_PATH = MODELS_DIR / "detection_confidence_model_meta.json"

RANDOM_STATE = 42

# Below this many admin-reviewed/official-confirmed detections, there isn't
# enough signal to fit a model that won't just memorize noise - callers
# (services/detection_confidence.py) fall back to a calibrated heuristic
# until this threshold is crossed.
MIN_TRAINING_SAMPLES = 30
