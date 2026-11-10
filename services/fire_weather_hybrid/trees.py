"""Native JSON boosted tasks and a strict, checksummed bundle loader."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import xgboost as xgb
from .contracts import FEATURES, GRID


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def fit_task(x, y, *, classifier=False, seed=42, rounds=250):
    y = np.asarray(y, float)
    valid = np.isfinite(y)
    if valid.sum() < 30:
        raise ValueError("at least 30 labeled examples required")
    if classifier and len(np.unique(y[valid])) != 2:
        raise ValueError("classifier requires both observed classes")
    cls = xgb.XGBClassifier if classifier else xgb.XGBRegressor
    model = cls(n_estimators=rounds, max_depth=5, learning_rate=.04, subsample=.8, colsample_bytree=.85,
                n_jobs=2, random_state=seed, tree_method="hist", objective="binary:logistic" if classifier else "reg:squarederror")
    model.fit(np.asarray(x, "float32")[valid], y[valid])
    return model


class Bundle:
    def __init__(self, directory):
        self.root = Path(directory).resolve()
        self.manifest = json.loads((self.root / "bundle.json").read_text())
        if self.manifest.get("schema") != "fire-weather-hybrid-v1" or self.manifest.get("advisory_only") is not True:
            raise ValueError("invalid experimental bundle contract")
        if self.manifest.get("features") != list(FEATURES):
            raise ValueError("feature contract mismatch")
        if self.manifest.get("grid_fingerprint") != GRID.fingerprint:
            raise ValueError("bundle grid fingerprint mismatch")
        allowed = {"ros_m_min", "intensity_kw_m", "temp_c_residual", "rh_pct_residual", "wind_speed_ms_residual",
                   "fm1_pct_residual", "fm10_pct_residual", "fm100_pct_residual", "fm1000_pct_residual", "red_flag"}
        if not set(self.manifest["tasks"]) <= allowed or not {"ros_m_min","intensity_kw_m"} <= set(self.manifest["tasks"]):
            raise ValueError("invalid or missing model tasks")
        for filename, digest in self.manifest["checksums"].items():
            path = (self.root/filename).resolve()
            if self.root not in path.parents or sha256(path) != digest:
                raise ValueError(f"bundle checksum/path mismatch: {filename}")
        required = {"spatial.onnx", "risk_scale.json", *[f"{name}.json" for name in self.manifest["tasks"]]}
        if not required <= set(self.manifest["checksums"]):
            raise ValueError("bundle omits checksums for required assets")
        self.scale = json.loads((self.root/"risk_scale.json").read_text())
        self.models = {}
        for name, spec in self.manifest["tasks"].items():
            model = xgb.XGBClassifier() if spec.get("classifier") else xgb.XGBRegressor()
            model.load_model(self.root/f"{name}.json")
            self.models[name] = model

    def predict(self, x, tasks=None):
        result = {}
        for name, model in self.models.items():
            if tasks is not None and name not in tasks:
                continue
            spec = self.manifest["tasks"][name]
            values = model.predict_proba(x)[:, 1] if spec.get("classifier") else model.predict(x)
            if spec.get("calibration") and spec.get("classifier"):
                values=np.interp(values,spec["calibration"]["x"],spec["calibration"]["y"])
            result[name] = np.maximum(np.expm1(values), 0) if spec.get("log1p") else values
        return result

