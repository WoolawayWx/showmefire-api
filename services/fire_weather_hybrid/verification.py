"""Truth-aware metrics; unsupported measurements are deferred, never passed."""
from __future__ import annotations
import numpy as np


def continuous(actual, predicted):
    y, p = np.broadcast_arrays(np.asarray(actual, float), np.asarray(predicted, float))
    valid = np.isfinite(y) & np.isfinite(p)
    if not valid.any():
        return {"status": "deferred", "samples": 0}
    error = p[valid]-y[valid]
    return {"status": "evaluated", "samples": int(valid.sum()), "mae": float(np.mean(abs(error))),
            "bias": float(np.mean(error)), "rmse": float(np.sqrt(np.mean(error**2)))}


def binary(actual, probability):
    y, p = np.broadcast_arrays(np.asarray(actual, float), np.asarray(probability, float))
    valid = np.isfinite(y) & np.isfinite(p)
    if not valid.any():
        return {"status": "deferred", "samples": 0}
    y, p = y[valid], p[valid]
    if not np.isin(y, [0, 1]).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("binary labels/probabilities out of range")
    tp, fp, fn = sum((p >= .5) & (y == 1)), sum((p >= .5) & (y == 0)), sum((p < .5) & (y == 1))
    return {"status": "evaluated", "samples": len(y), "brier": float(np.mean((p-y)**2)),
            "precision": float(tp/(tp+fp)) if tp+fp else None, "recall": float(tp/(tp+fn)) if tp+fn else None,
            "reliability": [{"lower": lo/10, "upper": (lo+1)/10,
                             "samples": int(np.sum((p >= lo/10) & ((p < (lo+1)/10) | ((lo == 9) & (p == 1))))) } for lo in range(10)]}

