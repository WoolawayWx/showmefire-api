"""Capture a read-only production model baseline for rollback/auditing."""
import hashlib
import json
import sys
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from models.versioning import API_DIR, get_model_entry
from core.database import get_db_path


def checksum(path):
    digest = hashlib.sha256(); digest.update(path.read_bytes()); return digest.hexdigest()


def capture(output=None):
    payload = {"captured_utc": datetime.now(timezone.utc).isoformat(), "models": {}}
    for model_type in ("fuel_moisture", "fire_danger", "fuel_moisture_spatial"):
        stable = get_model_entry(model_type).get("stable")
        if stable:
            path = API_DIR / stable["file"]
            payload["models"][model_type] = {**stable,
                                              "feature_columns": (stable.get("metadata") or {}).get("feature_columns"),
                                              "exists": path.exists(),
                                              "actual_sha256": checksum(path) if path.exists() else None}
    reports = API_DIR / "reports"
    for name in ("validation_history.json", "forecast_comparison_latest.csv"):
        path = reports / name
        if path.exists():
            payload.setdefault("validation_reports", {})[name] = {"sha256": checksum(path), "size": path.stat().st_size}
    history_path = reports / "validation_history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        payload["validation_metrics"] = {"last_7": history[-7:], "last_30": history[-30:]}
    db_path = get_db_path()
    if Path(db_path).exists():
        try:
            with sqlite3.connect(db_path) as connection:
                connection.row_factory = sqlite3.Row
                rows = connection.execute(
                    "SELECT station_id, valid_time, forecast_run_time, temp_c, rel_humidity, "
                    "wind_speed_ms, precip_mm, fuel_moisture FROM station_forecasts "
                    "ORDER BY forecast_run_time DESC, valid_time DESC LIMIT 20"
                ).fetchall()
            payload["output_sample"] = [dict(row) for row in rows]
        except sqlite3.Error as exc:
            payload["output_sample_error"] = str(exc)
    output = Path(output) if output else reports / f"model_baseline_{datetime.now():%Y%m%d_%H%M%S}.json"
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(output); return payload


if __name__ == "__main__":
    capture(sys.argv[1] if len(sys.argv) > 1 else None)
