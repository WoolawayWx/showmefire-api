import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from scripts import compare_09z_rtma as comparison


def _write_tif(path: Path, values: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="uint8",
        crs="EPSG:32615",
        transform=from_origin(500000, 4300000, 3000, 3000),
        nodata=255,
    ) as destination:
        destination.write(values.astype("uint8"), 1)
    return path


def test_build_comparison_scores_only_overlapping_valid_pixels(tmp_path):
    forecast = _write_tif(tmp_path / "forecast.tif", np.array([[0, 2], [4, 255]]))
    rtma = _write_tif(tmp_path / "rtma.tif", np.array([[0, 1], [2, 3]]))

    summary = comparison.build_comparison(forecast, rtma, "2026-09-04")
    agreement = summary["fire_danger_category_agreement"]

    assert summary["valid_pixel_count"] == 3
    assert agreement["matches"] == 1
    assert agreement["within_one"] == 2
    assert agreement["mean_bias"] == 1
    assert agreement["mean_abs_diff"] == 1
    assert summary["confusion_matrix"]["matrix"][1][2] == 1
    assert summary["confusion_matrix"]["matrix"][2][4] == 1
    assert summary["rtma_fuel_moisture"]["mode"] == "unknown"


def test_build_comparison_exposes_rtma_fuel_moisture_method(tmp_path):
    forecast = _write_tif(tmp_path / "forecast.tif", np.array([[1]]))
    rtma = _write_tif(tmp_path / "rtma.tif", np.array([[1]]))
    rtma.with_suffix(".json").write_text(
        json.dumps({
            "fuel_moisture": {
                "mode": "rh_estimate_calibrated_with_raws",
                "measured_hours": 8,
            }
        }),
        encoding="utf-8",
    )

    summary = comparison.build_comparison(forecast, rtma, "2026-09-04")
    assert summary["rtma_fuel_moisture"]["mode"] == "rh_estimate_calibrated_with_raws"
    assert summary["rtma_fuel_moisture"]["measured_hours"] == 8


def test_main_writes_dated_and_latest_summaries(tmp_path, monkeypatch):
    forecast_dir = tmp_path / "forecast"
    rtma_dir = tmp_path / "rtma"
    _write_tif(forecast_dir / "2026-09-04.tif", np.array([[0, 1]]))
    _write_tif(rtma_dir / "2026-09-04.tif", np.array([[0, 2]]))
    monkeypatch.setattr(comparison, "FORECAST_DIR", forecast_dir)
    monkeypatch.setattr(comparison, "RTMA_DIR", rtma_dir)
    monkeypatch.setattr(comparison, "METRICS_DIR", tmp_path / "metrics")
    monkeypatch.setattr(comparison, "LATEST_PATH", tmp_path / "latest.json")

    assert comparison.main("2026-09-04") == 0
    dated = json.loads((tmp_path / "metrics" / "2026-09-04.json").read_text())
    assert dated["comparison"] == "09z_minus_rtma"
    assert (tmp_path / "latest.json").exists()
