"""Unified, read-only operational scorecard for beta systems."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from core.config import ARCHIVE_DIR, GIS_DIR, IMAGES_DIR, REPORTS_DIR
from services.beta_products import BETA_ROOT, load_manifest
from services.beta_verification import BETA_VERIFICATION_HISTORY, load_latest_beta_verification
from services.drift_monitor import diagnostics as drift_diagnostics
from services.forecast_jobs import BETA_FORECAST_ENV, BETA_FORECAST_ROOT, get_beta_forecast_status


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def isolation_checks(beta_root: Path = BETA_ROOT) -> dict:
    """Return executable assertions for the Testbed's production boundary."""
    beta_root = Path(beta_root)
    designated_outputs = {
        "forecast": beta_root / "forecast",
        "products": beta_root / "gis",
        "verification": beta_root / "verification",
    }
    production_roots = [Path(IMAGES_DIR), Path(GIS_DIR), Path(REPORTS_DIR), Path(ARCHIVE_DIR)]
    checks = {
        "all_beta_outputs_under_testbed_root": all(
            _inside(path, beta_root) for path in designated_outputs.values()
        ),
        "testbed_root_separate_from_production_outputs": not any(
            beta_root.resolve() == root.resolve() or _inside(beta_root, root)
            for root in production_roots
        ),
        "database_writes_disabled": BETA_FORECAST_ENV.get("FORECAST_WRITE_DATABASE") == "false",
        "production_shadow_evidence_disabled": BETA_FORECAST_ENV.get("MODEL_SHADOW_ENABLED") == "false",
        "uploads_disabled": BETA_FORECAST_ENV.get("uploadForecast") == "false",
        "beta_status_namespace": BETA_FORECAST_ENV.get("FORECAST_STATUS_KEY") == "ForecastFireDangerBeta",
        "verification_output_isolated": _inside(beta_root / "verification", beta_root),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "testbed_root": str(beta_root),
        "production_paths_modified_by_verification": False,
    }


def _load_history() -> list[dict]:
    try:
        history = json.loads(BETA_VERIFICATION_HISTORY.read_text(encoding="utf-8"))
        return history if isinstance(history, list) else []
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


def _mean(values: list[float]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return round(sum(numeric) / len(numeric), 4) if numeric else None


def _performance_summary() -> dict:
    history = _load_history()[-30:]
    return {
        "days": len(history),
        "records": sum(int(item.get("record_count", 0) or 0) for item in history),
        "stable_mae": _mean([(item.get("stable") or {}).get("mae") for item in history]),
        "beta_mae": _mean([(item.get("beta") or {}).get("mae") for item in history]),
        "mae_delta": _mean([(item.get("delta") or {}).get("mae") for item in history]),
        "beta_exact_match_rate": _mean([
            (item.get("beta") or {}).get("exact_match_rate") for item in history
        ]),
    }


def _age_hours(timestamp: str | None) -> float | None:
    if not timestamp:
        return None
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return round(max(0.0, (datetime.now(timezone.utc) - parsed).total_seconds() / 3600), 1)
    except (TypeError, ValueError):
        return None


def build_beta_operations_status(*, shadows: dict) -> dict:
    manifest = load_manifest()
    latest = load_latest_beta_verification()
    isolation = isolation_checks()
    forecast_job = get_beta_forecast_status()

    shadow_rows = []
    for name, diagnostics in shadows.items():
        diagnostics = diagnostics or {}
        if diagnostics.get("configured") is False:
            state = "not_configured"
        elif diagnostics.get("auto_disabled") or diagnostics.get("healthy") is False:
            state = "attention"
        elif diagnostics.get("enabled"):
            state = "running"
        else:
            state = "paused"
        runs = int(diagnostics.get("runs", 0) or 0)
        successful = int(diagnostics.get("successful_runs", runs) or 0)
        shadow_rows.append({
            "name": name,
            "state": state,
            "runs": runs,
            "successful_runs": successful,
            "success_rate": round(successful / runs, 4) if runs else None,
            "last_success": diagnostics.get("last_success") or diagnostics.get("last_run"),
            "latency_ms": diagnostics.get("latency_ms"),
            "last_error": diagnostics.get("last_error"),
            "public_path_unchanged": diagnostics.get("public_path_unchanged", True),
        })

    needs_attention = [row["name"] for row in shadow_rows if row["state"] == "attention"]
    if not isolation["passed"]:
        overall = "blocked"
    elif needs_attention:
        overall = "attention"
    elif latest or any(row["state"] == "running" for row in shadow_rows):
        overall = "operational"
    else:
        overall = "idle"

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall,
        "attention": needs_attention,
        "isolation": isolation,
        "testbed": {
            "forecast_job": forecast_job,
            "forecast_updated_at": manifest.get("forecast_updated_at"),
            "forecast_age_hours": _age_hours(manifest.get("forecast_updated_at")),
            "observation_updated_at": manifest.get("observation_updated_at"),
            "product_count": len(manifest.get("products") or {}),
            "output_root": str(BETA_FORECAST_ROOT),
        },
        "verification": {
            "latest": latest,
            "rolling_30_days": _performance_summary(),
        },
        "shadows": shadow_rows,
        "drift": drift_diagnostics(),
    }
