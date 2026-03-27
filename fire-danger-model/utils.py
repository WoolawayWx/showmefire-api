import json
from datetime import datetime
from pathlib import Path

import numpy as np

from config import ALL_CATEGORY_IDS, DEFAULT_CATEGORY_THRESHOLDS


def calculate_fire_danger_category(fm, rh, wind_kts):
    """Mirror current rule-based fire danger categories in standalone tooling."""
    if fm >= 15:
        return 0
    if fm < 7 and rh < 20 and wind_kts >= 25:
        return 4
    if fm < 9 and rh < 25 and wind_kts >= 15:
        return 3
    if fm < 9 and ((rh < 35 and wind_kts >= 12) or (rh < 25 and wind_kts >= 5)):
        return 2
    if fm < 15 and (rh < 45 or wind_kts >= 10):
        return 1
    return 0


def calibrate_category_thresholds(pred_scores, true_categories):
    """Derive monotonic score thresholds from train predictions and true classes."""
    pred_scores = np.asarray(pred_scores, dtype=float)
    true_categories = np.asarray(true_categories, dtype=int)

    class_centroids = []
    for class_id in ALL_CATEGORY_IDS:
        class_values = pred_scores[true_categories == class_id]
        if class_values.size == 0:
            class_centroids.append(float(class_id))
        else:
            class_centroids.append(float(np.median(class_values)))

    class_centroids = np.maximum.accumulate(np.asarray(class_centroids, dtype=float))
    thresholds = []
    for i in range(len(class_centroids) - 1):
        thresholds.append(float((class_centroids[i] + class_centroids[i + 1]) / 2.0))

    # Enforce strictly increasing thresholds to keep digitize stable.
    for i in range(1, len(thresholds)):
        if thresholds[i] <= thresholds[i - 1]:
            thresholds[i] = thresholds[i - 1] + 1e-6

    return thresholds


def map_score_to_category(scores, thresholds=None):
    """Map continuous model score to 0-4 classes using configured thresholds."""
    use_thresholds = DEFAULT_CATEGORY_THRESHOLDS if thresholds is None else thresholds
    bins = np.asarray(use_thresholds, dtype=float)
    classes = np.digitize(np.asarray(scores, dtype=float), bins=bins, right=False)
    return np.clip(classes, 0, 4).astype(int)


def category_support(series):
    counts = {int(class_id): 0 for class_id in ALL_CATEGORY_IDS}
    value_counts = series.astype(int).value_counts().to_dict()
    for class_id, count in value_counts.items():
        if class_id in counts:
            counts[int(class_id)] = int(count)
    return counts


def ensure_dirs(*dirs):
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def utc_timestamp():
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
