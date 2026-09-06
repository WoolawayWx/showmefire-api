"""Seven-day post-promotion guardrail with automatic registry rollback."""
import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

API_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(API_DIR))
from models.versioning import get_model_entry, rollback

# Keys used in reports/validation_history.json's "metrics" dict (see
# forecast/endOfDayReport.py's continuous_variable_map / calculate_categorical_metrics)
# do not match the model registry's model_type strings - map between them here.
METRIC_KEY_BY_MODEL_TYPE = {
    "fuel_moisture": "Fuel Moisture (%)",
    "fire_danger": "Fire Danger Index",
}


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
    metric_key = METRIC_KEY_BY_MODEL_TYPE.get(model_type, model_type)
    latest = (history[-1].get("metrics") or {}).get(metric_key) if history else None
    baseline = previous.get("performance") or {}
    if not latest or baseline.get("mae") is None or baseline.get("bias") is None:
        return {"action": "none", "reason": "metrics unavailable"}
    regressed = (latest["mae"] > baseline["mae"] * (1 + max_mae_regression) or
                 abs(latest["bias"]) > abs(baseline["bias"]) + max_bias_regression)
    if regressed:
        target = rollback(model_type)
        return {"action": "rollback", "version": target, "reason": "live metric regression"}
    return {"action": "monitor", "reason": "metrics within guardrails"}


def monitor_all(model_types=tuple(METRIC_KEY_BY_MODEL_TYPE), **kwargs):
    """Run the guardrail for every model type with a known validation-history metric key."""
    return {model_type: monitor(model_type=model_type, **kwargs) for model_type in model_types}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-type", default=None,
                        help="Model type to monitor (default: run all of %s)" % ", ".join(METRIC_KEY_BY_MODEL_TYPE))
    parser.add_argument("--forecast-job-failed", action="store_true")
    parser.add_argument("--max-mae-regression", type=float, default=0.10)
    parser.add_argument("--max-bias-regression", type=float, default=0.5)
    args = parser.parse_args()
    if args.model_type:
        result = monitor(model_type=args.model_type, max_mae_regression=args.max_mae_regression,
                         max_bias_regression=args.max_bias_regression, forecast_job_failed=args.forecast_job_failed)
    else:
        result = monitor_all(max_mae_regression=args.max_mae_regression,
                             max_bias_regression=args.max_bias_regression,
                             forecast_job_failed=args.forecast_job_failed)
    print(json.dumps(result, indent=2))
