"""Seven-day post-promotion guardrail with automatic registry rollback."""
import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

API_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(API_DIR))
from models.versioning import get_model_entry, rollback


def monitor(model_type="fuel_moisture", history_path=API_DIR / "reports" / "validation_history.json",
            max_mae_regression=0.10, max_bias_regression=0.5, forecast_job_failed=False):
    entry = get_model_entry(model_type); stable = entry.get("stable")
    if not stable:
        return {"action": "none", "reason": "no stable model"}
    try:
        promoted = datetime.fromisoformat(stable["promoted_at"])
    except (KeyError, TypeError, ValueError):
        return {"action": "none", "reason": "promotion timestamp unavailable"}
    if datetime.now() - promoted > timedelta(days=7):
        return {"action": "none", "reason": "post-promotion window complete"}
    previous = next((record for record in reversed(entry.get("history", []))
                     if record.get("channel") == "stable" and record.get("version") != stable.get("version")), None)
    if forecast_job_failed:
        target = rollback(model_type)
        return {"action": "rollback", "version": target, "reason": "forecast job failure"}
    path = Path(history_path)
    if not previous or not path.exists():
        return {"action": "none", "reason": "insufficient comparison evidence"}
    history = json.loads(path.read_text(encoding="utf-8"))
    latest = (history[-1].get("metrics") or {}).get("fuel_moisture") if history else None
    baseline = previous.get("performance") or {}
    if not latest or baseline.get("mae") is None or baseline.get("bias") is None:
        return {"action": "none", "reason": "metrics unavailable"}
    regressed = (latest["mae"] > baseline["mae"] * (1 + max_mae_regression) or
                 abs(latest["bias"]) > abs(baseline["bias"]) + max_bias_regression)
    if regressed:
        target = rollback(model_type)
        return {"action": "rollback", "version": target, "reason": "live metric regression"}
    return {"action": "monitor", "reason": "metrics within guardrails"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--forecast-job-failed", action="store_true")
    parser.add_argument("--max-mae-regression", type=float, default=0.10)
    parser.add_argument("--max-bias-regression", type=float, default=0.5)
    args = parser.parse_args()
    print(json.dumps(monitor(max_mae_regression=args.max_mae_regression,
                             max_bias_regression=args.max_bias_regression,
                             forecast_job_failed=args.forecast_job_failed), indent=2))
