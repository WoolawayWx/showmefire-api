import json
from pathlib import Path

from scripts import compare_09z_12z as comparison


def _write_forecast(path: Path, forecasts: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "stations": {
            "TEST": {"lat": 38.5, "lon": -92.5, "forecasts": forecasts},
        }
    }), encoding="utf-8")
    return path


def test_comparison_uses_only_matching_station_hours(tmp_path):
    file_09 = _write_forecast(tmp_path / "station_forecasts_20260904_09.json", [
        {"time": "2026-09-04T14:00:00", "temp_c": 99, "rh": 99, "wind_speed_ms": 99, "fuel_moisture": 99, "fire_danger": 4},
        {"time": "2026-09-04T16:00:00", "temp_c": 22, "rh": 30, "wind_speed_ms": 5, "fuel_moisture": 8, "fire_danger": 2},
        {"time": "2026-09-04T17:00:00", "temp_c": 26, "rh": 28, "wind_speed_ms": 7, "fuel_moisture": 7, "fire_danger": 3},
    ])
    file_12 = _write_forecast(tmp_path / "station_forecasts_20260904_12.json", [
        {"time": "2026-09-04T16:00:00", "temp_c": 20, "rh": 32, "wind_speed_ms": 4, "fuel_moisture": 9, "fire_danger": 2},
        {"time": "2026-09-04T17:00:00", "temp_c": 22, "rh": 30, "wind_speed_ms": 6, "fuel_moisture": 8, "fire_danger": 2},
    ])

    geojson, summary = comparison.build_comparison(file_09, file_12)

    assert summary["record_count"] == 2
    assert summary["station_count"] == 1
    assert summary["metrics"]["temp_c"]["count"] == 2
    assert summary["metrics"]["temp_c"]["mean_bias"] == 3
    assert summary["metrics"]["temp_c"]["mean_abs_diff"] == 3
    assert summary["fire_danger_category_agreement"]["agreement_rate"] == 0.5
    assert summary["fire_danger_category_agreement"]["within_one_rate"] == 1
    assert geojson["features"][0]["properties"]["matched_hours"] == 2


def test_main_archives_dated_and_latest_outputs(tmp_path, monkeypatch):
    forecast_dir = tmp_path / "forecasts"
    _write_forecast(forecast_dir / "station_forecasts_beta_20990101_12.json", [])
    _write_forecast(forecast_dir / "station_forecasts_20260904_09.json", [
        {"time": "2026-09-04T16:00:00", "temp_c": 20, "rh": 30, "wind_speed_ms": 5, "fuel_moisture": 8, "fire_danger": 2},
    ])
    _write_forecast(forecast_dir / "station_forecasts_20260904_12.json", [
        {"time": "2026-09-04T16:00:00", "temp_c": 21, "rh": 31, "wind_speed_ms": 4, "fuel_moisture": 9, "fire_danger": 2},
    ])
    monkeypatch.setattr(comparison, "FORECAST_ARCHIVE_DIR", forecast_dir)
    monkeypatch.setattr(comparison, "METRICS_ARCHIVE_DIR", tmp_path / "metrics")
    monkeypatch.setattr(comparison, "GIS_ARCHIVE_DIR", tmp_path / "gis-archive")
    monkeypatch.setattr(comparison, "LATEST_SUMMARY_PATH", tmp_path / "latest.json")
    monkeypatch.setattr(comparison, "LATEST_GEOJSON_PATH", tmp_path / "latest.geojson")

    assert comparison.main() == 0
    assert (tmp_path / "metrics" / "2026-09-04.json").exists()
    assert (tmp_path / "gis-archive" / "2026-09-04.geojson").exists()
    assert (tmp_path / "latest.json").exists()
    assert (tmp_path / "latest.geojson").exists()
