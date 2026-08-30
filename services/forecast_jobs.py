"""Process-isolated beta forecast jobs for the Testbed."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from core.beta_fire_danger import score_fire_danger
from core.fire_danger import meters_per_second_to_knots
from services.beta_products import BETA_ROOT, load_manifest, save_manifest


BETA_FORECAST_ROOT = BETA_ROOT / "forecast"
JOB_STATE_PATH = BETA_ROOT / "forecast_job.json"
FORECAST_SCRIPT = Path(__file__).resolve().parent.parent / "forecast" / "DailyForecast_ModelFD.py"
_job_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_job() -> dict | None:
    try:
        return json.loads(JOB_STATE_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _write_job(job: dict) -> None:
    JOB_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = JOB_STATE_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, JOB_STATE_PATH)


def _update_manifest(job: dict) -> None:
    manifest = load_manifest()
    image_dir = BETA_FORECAST_ROOT / "images"
    files = [
        "mo-forecastfiredanger-beta.png",
        "mo-forecastfuelmoisture-beta.png",
        "mo-forecastminrh-beta.png",
        "mo-forecastmaxwind-beta.png",
        "mo-forecastmaxtemp-beta.png",
        "mo-forecastrainfall-beta.png",
        "mo-forecastswe-beta.png",
    ]
    products = manifest.setdefault("products", {})
    for filename in files:
        if (image_dir / filename).exists():
            products[f"forecast_{filename.removeprefix('mo-').removesuffix('-beta.png')}"] = {
                "kind": "image",
                "path": f"forecast/images/{filename}",
                "generated_at": job["finished_at"],
            }
    gis_dir = BETA_FORECAST_ROOT / "gis"
    for filename, name in (
        ("peak_fire_danger.tif", "forecast_peak_tif"),
        ("peak_fire_danger_polygons.geojson", "forecast_peak_polygons"),
        ("peak_fire_danger_points.geojson", "forecast_peak_points"),
    ):
        if (gis_dir / filename).exists():
            products[name] = {
                "kind": "file",
                "path": f"forecast/gis/{filename}",
                "generated_at": job["finished_at"],
            }
    comparison_path = _write_forecast_comparison()
    if comparison_path:
        products["forecast_vs_beta"] = {
            "kind": "geojson",
            "path": "forecast/gis/forecast_vs_beta.geojson",
            "generated_at": job["finished_at"],
        }
    manifest.update({
        "forecast_status": job["status"],
        "forecast_job_id": job["job_id"],
        "forecast_updated_at": job["finished_at"],
        "forecast_model_run": job.get("model_run"),
    })
    save_manifest(manifest)


def _write_forecast_comparison() -> bool:
    """Write one beta-scored peak point per station from the forecast JSON."""
    archive_dir = BETA_FORECAST_ROOT / "archive" / "forecasts"
    files = sorted(archive_dir.glob("station_forecasts_beta_*.json"))
    if not files:
        return False
    try:
        payload = json.loads(files[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    features = []
    for station_id, station in (payload.get("stations") or {}).items():
        best = None
        for forecast in station.get("forecasts") or []:
            try:
                result = score_fire_danger(
                    forecast["fuel_moisture"],
                    forecast["rh"],
                    meters_per_second_to_knots(forecast["wind_speed_ms"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
            candidate = {**forecast, "beta": result}
            if best is None or candidate["beta"]["score"] > best["beta"]["score"]:
                best = candidate
        if not best or station.get("lat") is None or station.get("lon") is None:
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [station["lon"], station["lat"]]},
            "properties": {
                "station_id": station_id,
                "valid_time": best.get("time"),
                "fuel_moisture": best.get("fuel_moisture"),
                "relative_humidity": best.get("rh"),
                "wind_speed_ms": best.get("wind_speed_ms"),
                "official_category": best["beta"]["official_category"],
                "official_label": best["beta"]["official_label"],
                "beta_score": best["beta"]["score"],
                "criteria": best["beta"]["criteria"],
            },
        })
    output = BETA_FORECAST_ROOT / "gis" / "forecast_vs_beta.geojson"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps({
        "type": "FeatureCollection",
        "name": "Beta forecast station peak comparison",
        "metadata": {"scorer_version": "1.0.0", "generated_at": _now()},
        "features": features,
    }, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return True


def _run_beta_forecast(job: dict) -> None:
    BETA_FORECAST_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = BETA_FORECAST_ROOT / "forecast.log"
    job["status"] = "running"
    job["started_at"] = _now()
    _write_job(job)
    environment = os.environ.copy()
    environment.update({
        "FORECAST_OUTPUT_ROOT": str(BETA_FORECAST_ROOT),
        "FORECAST_CACHE_DIR": str(BETA_FORECAST_ROOT / "cache" / "hrrr"),
        "FORECAST_ARCHIVE_FORECASTS_DIR": str(BETA_FORECAST_ROOT / "archive" / "forecasts"),
        "FORECAST_STATUS_KEY": "ForecastFireDangerBeta",
        "FORECAST_WRITE_DATABASE": "false",
        "uploadForecast": "false",
    })
    try:
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                [sys.executable, str(FORECAST_SCRIPT)],
                cwd=str(FORECAST_SCRIPT.parent.parent),
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=int(os.getenv("TESTBED_FORECAST_TIMEOUT_SECONDS", "3600")),
                check=False,
            )
        job["status"] = "completed" if completed.returncode == 0 else "failed"
        job["return_code"] = completed.returncode
        job["model_run"] = _read_model_run()
        if job["status"] == "completed":
            _update_manifest({**job, "finished_at": _now()})
    except subprocess.TimeoutExpired:
        job["status"] = "failed"
        job["error"] = "Beta forecast exceeded its timeout."
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
    job["finished_at"] = _now()
    _write_job(job)


def _read_model_run() -> str | None:
    status_path = BETA_FORECAST_ROOT / "status.json"
    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
        return (status.get("ForecastFireDangerBeta") or {}).get("model_run")
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def trigger_beta_forecast(requested_by: str) -> dict:
    with _job_lock:
        existing = _read_job()
        if existing and existing.get("status") in {"queued", "running"}:
            raise RuntimeError("A beta forecast is already running.")
        job = {
            "job_id": uuid.uuid4().hex,
            "status": "queued",
            "requested_by": requested_by,
            "requested_at": _now(),
        }
        _write_job(job)
        threading.Thread(target=_run_beta_forecast, args=(job,), daemon=True).start()
        return job


def get_beta_forecast_status() -> dict:
    return _read_job() or {"status": "idle"}

