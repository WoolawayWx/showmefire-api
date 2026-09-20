"""Basic load_bundle() coverage for fire_weather_index_shadow.py - this
module had zero test coverage before Phase 2's registry migration touched
its read side (_resolve_bundle_files), so this pins the pre-existing
directory-based behavior in addition to the new registry-fallback path
(see test_fire_weather_index_registry_migration.py)."""
import json
import tempfile
import unittest
from pathlib import Path

from services import fire_weather_index_shadow as fwis


def _write_bundle(directory: Path, *, version="0.0.1-beta.1") -> None:
    (directory / "factor_weights.json").write_text(json.dumps({
        "schema": "fire-weather-index-factor-weights-v1",
        "weights": {"rh": 1.0, "wind": 1.0, "vpd": 1.0, "precip_relief": 1.0},
        "raw_score_ceiling": {"value": 1.0},
        "ramp_anchors": {
            "rh": {"benign": 60.0, "extreme": 10.0},
            "wind": {"benign": 5.0, "extreme": 30.0},
            "vpd": {"benign": 0.5, "extreme": 4.0},
            "precip_relief": {"benign": 0.0, "extreme": 10.0},
        },
    }))
    (directory / "category_thresholds.json").write_text(json.dumps({
        "schema": "fire-weather-index-category-thresholds-v1",
        "thresholds": [0.2, 0.4, 0.6, 0.8],
        "category_labels": list(fwis.CATEGORY_LABELS),
    }))
    if version is not None:
        (directory / "registered_version.json").write_text(
            json.dumps({"model_type": "fire_weather_index", "version": version}))


class LoadBundleTests(unittest.TestCase):
    def test_missing_directory_raises(self):
        with self.assertRaises(FileNotFoundError):
            fwis.load_bundle(Path("/nonexistent/path/for/sure"))

    def test_valid_bundle_loads_and_computes_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_bundle(directory)
            bundle = fwis.load_bundle(directory)
        self.assertEqual(bundle["version"], "0.0.1-beta.1")
        self.assertEqual(len(bundle["bundle_checksum"]), 64)
        self.assertIn("weights", bundle["factor_weights"])

    def test_wrong_factor_weights_schema_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_bundle(directory)
            data = json.loads((directory / "factor_weights.json").read_text())
            data["schema"] = "wrong-schema"
            (directory / "factor_weights.json").write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                fwis.load_bundle(directory)

    def test_missing_registered_version_file_is_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_bundle(directory, version=None)
            bundle = fwis.load_bundle(directory)
        self.assertIsNone(bundle["version"])


class ScoreCountyDayTests(unittest.TestCase):
    def test_dry_windy_scores_higher_than_wet_calm(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_bundle(directory)
            bundle = fwis.load_bundle(directory)
        dry_windy = fwis.score_county_day(bundle, {
            "rh_min_afternoon": 12.0, "wind_kts_max": 28.0, "vpd_kpa_max": 3.8, "precip_24h_mm": 0.0,
        })
        wet_calm = fwis.score_county_day(bundle, {
            "rh_min_afternoon": 58.0, "wind_kts_max": 6.0, "vpd_kpa_max": 0.6, "precip_24h_mm": 9.0,
        })
        self.assertGreater(dry_windy["score"], wet_calm["score"])
        self.assertGreaterEqual(dry_windy["category"], wet_calm["category"])


if __name__ == "__main__":
    unittest.main()
