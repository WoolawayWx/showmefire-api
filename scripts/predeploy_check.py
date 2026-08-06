"""Read-only production preflight for the Show Me Fire API release."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import xgboost as xgb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.database import get_db_path
from core.fire_danger import RULE_SPEC_SHA256, RULE_SPEC_VERSION
from core.precipitation import PRECIPITATION_CONTRACT_SHA256, PRECIPITATION_CONTRACT_VERSION
from models.versioning import load_active_model_path
from services.v5_shadow import BUNDLE_ENV as V5_BUNDLE_ENV, validate_bundle as validate_v5_bundle


def main() -> int:
    checks: dict[str, dict] = {}

    def record(name: str, passed: bool, detail: str) -> None:
        checks[name] = {"pass": bool(passed), "detail": detail}

    record("production_mode", os.getenv("ENVIRONMENT", "").lower() == "production", "ENVIRONMENT=production")
    for variable in ("JWT_SECRET", "ADMIN_EMAIL", "ADMIN_PASSWORD_HASH"):
        value = os.getenv(variable, "").strip()
        secure = bool(value) and (variable != "JWT_SECRET" or value != "CHANGE-THIS-TO-A-RANDOM-SECRET-KEY")
        record(variable.lower(), secure, "configured" if secure else "missing or insecure")

    db_path = get_db_path()
    record("data_directory", db_path.parent.is_dir() and os.access(db_path.parent, os.W_OK), str(db_path))

    try:
        model_path = load_active_model_path("fuel_moisture", auto_rollback=False)
        model = xgb.Booster()
        model.load_model(str(model_path))
        record("stable_fuel_moisture", True, f"{model_path} ({len(model.feature_names or [])} features)")
    except Exception as exc:
        record("stable_fuel_moisture", False, str(exc))

    record("fire_danger_contract", True, f"{RULE_SPEC_VERSION} sha256={RULE_SPEC_SHA256}")
    record(
        "precipitation_contract",
        True,
        f"{PRECIPITATION_CONTRACT_VERSION} sha256={PRECIPITATION_CONTRACT_SHA256}",
    )

    v5_bundle = os.getenv(V5_BUNDLE_ENV, "").strip()
    if v5_bundle:
        try:
            validate_v5_bundle(v5_bundle)
            record("v5_shadow_bundle", True, v5_bundle)
        except Exception as exc:
            record("v5_shadow_bundle", False, str(exc))
    else:
        record("v5_shadow_bundle", True, "not configured (stable production remains authoritative)")

    passed = all(item["pass"] for item in checks.values())
    print(json.dumps({"pass": passed, "checks": checks}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
