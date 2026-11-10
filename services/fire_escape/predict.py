"""Load a fire_escape bundle and turn feature rows into calibrated probabilities."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


def load_bundle(directory: Path) -> dict:
    """bundle.json plus a loaded booster per target. The bundle names its own features, so an older bundle
    keeps scoring after features are added to the code."""
    bundle = json.loads((Path(directory) / "bundle.json").read_text(encoding="utf-8"))
    for spec in bundle["targets"].values():
        spec["booster"] = xgb.Booster(model_file=str(Path(directory) / spec["model"]))
    return bundle


def predict(bundle: dict, frame: pd.DataFrame) -> dict[str, np.ndarray]:
    """{target: P(final size >= threshold | a fire starts)} for each row. Missing features stay missing."""
    names = bundle["features"]
    matrix = xgb.DMatrix(frame[names].to_numpy(dtype="float32"), feature_names=names)
    out = {}
    for target, spec in bundle["targets"].items():
        score = spec["booster"].predict(matrix, iteration_range=(0, spec["trees"]), output_margin=True)
        a, b = spec["platt"]
        out[target] = 1.0 / (1.0 + np.exp(-(a * score + b)))
    return out
