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
