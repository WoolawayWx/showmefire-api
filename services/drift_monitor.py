"""Diagnostics-surface wrapper for services/drift.py.

Mirrors services/model_shadow.py's _state + diagnostics() pattern: a
module-level dict updated whenever run_drift_check() executes, read back
via diagnostics() (see api/routers/spatial_model.py's other diagnostics
routes for the same shape).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from services import drift

logger = logging.getLogger(__name__)

# Every model family with existing shadow evidence (see services/drift.py's
# module docstring for what each one reads).
DRIFT_MODEL_TYPES = ("fuel_moisture", "v5", "risk_fusion_glm")
DRIFT_EVIDENCE_ROOT = Path(os.getenv("DATA_DIR", "data")) / "model-shadow" / "drift"

_state = {"last_run": None, "results": {}}


def diagnostics() -> dict:
    return dict(_state)


def run_drift_check() -> dict:
    """Evaluate drift for every shadow-tracked model type.

    A failure in one model type's drift check is isolated (logged, recorded
    as an error entry) and never blocks the others - same convention as
    api/core/scheduler.py's other jobs.
    """
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results = {}
    for model_type in DRIFT_MODEL_TYPES:
        try:
            report = drift.evaluate_drift(model_type)
            drift.write_drift_report(report, DRIFT_EVIDENCE_ROOT, run_id)
            if report.get("flags"):
                logger.warning("Drift flagged for %s: %s", model_type, report["flags"])
            results[model_type] = report
        except Exception as error:
            logger.error("Drift check failed for %s: %s", model_type, error, exc_info=True)
            results[model_type] = {"model_type": model_type, "error": str(error)}
    _state["last_run"] = run_id
    _state["results"] = results
    return dict(_state)
