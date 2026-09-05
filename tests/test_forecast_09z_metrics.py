import asyncio
import json

from routers import forecast_09z_metrics


def test_metrics_history_and_station_geojson(tmp_path, monkeypatch):
    metrics_dir = tmp_path / "metrics"
    stations_dir = tmp_path / "stations"
    metrics_dir.mkdir()
    stations_dir.mkdir()
    summary = {
        "date": "2026-09-04",
        "station_count": 1,
        "record_count": 12,
        "metrics": {"temp_c": {"mean_abs_diff": 1.5}},
        "fire_danger_category_agreement": {"agreement_rate": 0.75},
    }
    (metrics_dir / "2026-09-04.json").write_text(json.dumps(summary), encoding="utf-8")
    (stations_dir / "2026-09-04.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": []}), encoding="utf-8"
    )
    monkeypatch.setattr(forecast_09z_metrics, "METRICS_DIR", metrics_dir)
    monkeypatch.setattr(forecast_09z_metrics, "STATIONS_DIR", stations_dir)

    history = asyncio.run(forecast_09z_metrics.get_09z_metrics_history(limit=90))
    stations = asyncio.run(forecast_09z_metrics.get_09z_station_metrics("2026-09-04"))

    assert history["dates"][0]["record_count"] == 12
    assert stations["type"] == "FeatureCollection"


def test_observed_and_rtma_histories_are_normalized(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    observed = {
        "date": "2026-09-03",
        "generated_at": "2026-09-04T04:30:00+00:00",
        "stations_count": 2,
        "record_count": 24,
        "metrics": {
            "Temperature (C)": {"count": 24, "mae": 1.2, "bias": -0.3, "rmse": 1.4},
            "Fire Danger Index": {
                "count": 24,
                "exact_match_rate": 0.75,
                "within_one_category_rate": 1.0,
                "mae": 0.25,
                "bias": 0.1,
            },
        },
    }
    history_file = reports_dir / "validation_history_09z.json"
    history_file.write_text(json.dumps([observed]), encoding="utf-8")
    rtma_dir = tmp_path / "rtma"
    rtma_dir.mkdir()
    (rtma_dir / "2026-09-03.json").write_text(json.dumps({
        "date": "2026-09-03",
        "generated_at": "2026-09-04T04:30:00+00:00",
        "valid_pixel_count": 100,
        "fire_danger_category_agreement": {"agreement_rate": 0.8},
    }), encoding="utf-8")
    monkeypatch.setattr(forecast_09z_metrics, "OBSERVED_HISTORY_FILE", history_file)
    monkeypatch.setattr(forecast_09z_metrics, "RTMA_METRICS_DIR", rtma_dir)

    station_history = asyncio.run(forecast_09z_metrics.get_09z_observed_history(limit=90))
    rtma_history = asyncio.run(forecast_09z_metrics.get_09z_rtma_history(limit=90))

    station = station_history["dates"][0]
    assert station["metrics"]["temp_c"]["mean_abs_diff"] == 1.2
    assert station["fire_danger_category_agreement"]["agreement_rate"] == 0.75
    assert rtma_history["dates"][0]["valid_pixel_count"] == 100
