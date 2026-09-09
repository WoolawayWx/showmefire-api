import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import xgboost as xgb

from services import fire_weather_ml_shadow as fwms


def _write_bundle(directory: Path, *, advisory_only=True, model_family="xgboost_regressor",
                  feature_columns=None, version="0.0.1-beta.1") -> None:
    feature_columns = list(feature_columns if feature_columns is not None else fwms.EXPECTED_FEATURE_COLUMNS)
    (directory / "contract.json").write_text(json.dumps({
        "advisory_only": advisory_only,
        "model_family": model_family,
        "feature_columns": feature_columns,
    }))
    (directory / "fire_weather_ml_metadata.json").write_text(json.dumps({"feature_columns": feature_columns}))
    if version is not None:
        (directory / "registered_version.json").write_text(
            json.dumps({"model_type": "fire_weather_ml", "version": version}))

    rng = np.random.default_rng(0)
    n = 50
    train = pd.DataFrame({column: rng.uniform(0, 30, size=n) for column in fwms.EXPECTED_FEATURE_COLUMNS})
    label = train["wind_ms"] - train["fm10_pct"] * 0.5
    booster = xgb.train({"objective": "reg:squarederror"}, xgb.DMatrix(train, label=label), num_boost_round=5)
    booster.save_model(str(directory / "fire_weather_ml_model.json"))


def _synthetic_static_and_moisture():
    shape = (2, 2)
    static = {
        "lat": np.full(shape, 38.5), "lon": np.full(shape, -92.5),
        "slope_deg": np.full(shape, 5.0), "aspect_sin": np.zeros(shape), "aspect_cos": np.ones(shape),
        "canopy_cover_pct": np.full(shape, 10.0), "canopy_height_m": np.full(shape, 3.0),
        "valid_mask": np.array([[True, True], [True, False]]),
    }
    moisture = {
        "temp_c": np.full(shape, 28.0), "rh": np.full(shape, 30.0), "wind_ms": np.full(shape, 6.0),
        "precip_mm": np.zeros(shape), "fm1_pct": np.full(shape, 6.0), "fm10_pct": np.full(shape, 8.0),
        "fm100_pct": np.full(shape, 12.0),
    }
    return static, moisture


class LoadBundleTests(unittest.TestCase):
    def test_missing_directory_raises(self):
        with self.assertRaises(FileNotFoundError):
            fwms.load_bundle(Path("/nonexistent/path/for/sure"))

    def test_valid_bundle_loads_and_computes_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_bundle(directory)
            bundle = fwms.load_bundle(directory)
            self.assertEqual(bundle["contract"]["model_family"], "xgboost_regressor")
            self.assertIsInstance(bundle["bundle_checksum"], str)
            self.assertEqual(bundle["version"], "0.0.1-beta.1")

    def test_missing_registered_version_file_is_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_bundle(directory, version=None)
            bundle = fwms.load_bundle(directory)
            self.assertIsNone(bundle["version"])

    def test_non_advisory_bundle_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_bundle(directory, advisory_only=False)
            with self.assertRaises(ValueError):
                fwms.load_bundle(directory)

    def test_wrong_model_family_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_bundle(directory, model_family="glm")
            with self.assertRaises(ValueError):
                fwms.load_bundle(directory)

    def test_feature_columns_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_bundle(directory, feature_columns=list(fwms.EXPECTED_FEATURE_COLUMNS)[::-1])
            with self.assertRaises(ValueError):
                fwms.load_bundle(directory)


class ScoreGridTests(unittest.TestCase):
    def test_predicts_only_within_the_valid_mask(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_bundle(directory)
            bundle = fwms.load_bundle(directory)
            static, moisture = _synthetic_static_and_moisture()
            predictions = fwms.score_grid(bundle, static, moisture)
        self.assertEqual(predictions.shape, (2, 2))
        self.assertTrue(np.isfinite(predictions[0, 0]))
        self.assertTrue(np.isfinite(predictions[0, 1]))
        self.assertTrue(np.isfinite(predictions[1, 0]))
        self.assertTrue(np.isnan(predictions[1, 1]))  # masked invalid cell

    def test_never_predicts_a_negative_spread_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_bundle(directory)
            bundle = fwms.load_bundle(directory)
            # Cold, near-saturated, low-wind conditions - exactly the kind
            # of near-zero-spread cell that produced a small negative raw
            # prediction on real data (see docs/fire_weather_ml_plan.md).
            static, _ = _synthetic_static_and_moisture()
            moisture = {
                "temp_c": np.full((2, 2), 5.0), "rh": np.full((2, 2), 95.0), "wind_ms": np.full((2, 2), 0.5),
                "precip_mm": np.zeros((2, 2)), "fm1_pct": np.full((2, 2), 28.0),
                "fm10_pct": np.full((2, 2), 28.0), "fm100_pct": np.full((2, 2), 28.0),
            }
            predictions = fwms.score_grid(bundle, static, moisture)
        finite = predictions[np.isfinite(predictions)]
        self.assertTrue((finite >= 0.0).all(), finite)


class ClassifyRosChPerHTests(unittest.TestCase):
    def test_negative_or_nan_values_are_nodata(self):
        classes = fwms._classify_ros_ch_per_h(np.array([-1.0, np.nan]))
        self.assertTrue((classes == fwms.NODATA_CLASS).all())

    def test_buckets_match_class_bounds(self):
        classes = fwms._classify_ros_ch_per_h(np.array([0.0, 3.0, 100.0]))
        self.assertEqual(list(classes), [0, 1, 4])  # Very Low, Low, Very High


class CompareTests(unittest.TestCase):
    def test_reports_zero_compared_cells_when_nothing_overlaps(self):
        predicted = np.full((2, 2), np.nan)
        real = np.full((2, 2), np.nan)
        result = fwms._compare(predicted, real)
        self.assertEqual(result["compared_cells"], 0)

    def test_computes_stats_over_the_overlap(self):
        predicted = np.array([[1.0, 2.0], [np.nan, 4.0]])
        real = np.array([[1.5, 2.5], [3.0, np.nan]])
        result = fwms._compare(predicted, real)
        self.assertEqual(result["compared_cells"], 2)
        self.assertAlmostEqual(result["mean_absolute_error_ch_per_h"], 0.5, places=6)


class RenderPngTests(unittest.TestCase):
    def test_writes_a_real_image_file_labeled_with_model_and_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_bundle(directory)
            bundle = fwms.load_bundle(directory)
            static, moisture = _synthetic_static_and_moisture()
            predicted = fwms.score_grid(bundle, static, moisture)
            comparison = fwms._compare(predicted, np.array([[1.0, 2.0], [3.0, np.nan]]))
            out_path = directory / "shadow.png"
            fwms._render_png(predicted, static, bundle, comparison, out_path)
            self.assertTrue(out_path.is_file())
            self.assertGreater(out_path.stat().st_size, 0)


class UpdateManifestTests(unittest.TestCase):
    def test_writes_a_fire_weather_ml_shadow_product_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            beta_root = Path(tmp)
            manifest_path = beta_root / "manifest.json"
            image_path = beta_root / "images" / fwms.IMAGE_FILENAME
            image_path.parent.mkdir(parents=True)
            image_path.write_bytes(b"fake-png")

            from services import beta_products
            with patch.object(beta_products, "BETA_ROOT", beta_root), \
                 patch.object(beta_products, "BETA_MANIFEST_PATH", manifest_path):
                fwms._update_manifest(image_path, {"version": "0.0.1-beta.1"},
                                      {"compared_cells": 4}, "2026-09-09T00:00:00Z")
                manifest = beta_products.load_manifest()

        entry = manifest["products"]["fire_weather_ml_shadow"]
        self.assertTrue(entry["not_for_operations"])
        self.assertEqual(entry["model_version"], "0.0.1-beta.1")
        self.assertEqual(entry["preview"], f"images/{fwms.IMAGE_FILENAME}")


class ScoreForSpreadRateTests(unittest.TestCase):
    def setUp(self):
        self.bundle_dir = tempfile.TemporaryDirectory()
        _write_bundle(Path(self.bundle_dir.name))

        self.requested_patch = patch.object(fwms, "_requested", return_value=True)
        self.requested_patch.start()
        self.configured_patch = patch.object(fwms, "_configured", return_value=True)
        self.configured_patch.start()

        self.state_dir = tempfile.TemporaryDirectory()
        self.state_path_patch = patch.object(fwms, "STATE_PATH", Path(self.state_dir.name) / "shadow-state.json")
        self.state_path_patch.start()

        # So the render/manifest step (best-effort inside score_for_spread_rate)
        # actually runs against a throwaway location instead of the real
        # testbed data dir, letting tests verify it isn't silently swallowed.
        self.beta_root_dir = tempfile.TemporaryDirectory()
        from services import beta_products
        self.beta_root_patch = patch.object(beta_products, "BETA_ROOT", Path(self.beta_root_dir.name))
        self.beta_root_patch.start()
        self.beta_manifest_patch = patch.object(
            beta_products, "BETA_MANIFEST_PATH", Path(self.beta_root_dir.name) / "manifest.json")
        self.beta_manifest_patch.start()

        fwms._state.update(
            enabled=True, consecutive_failures=0, last_error=None, runs=0, successful_runs=0,
            healthy=True, auto_disabled=False, cells_scored=0,
        )

    def tearDown(self):
        self.beta_manifest_patch.stop()
        self.beta_root_patch.stop()
        self.beta_root_dir.cleanup()
        self.state_path_patch.stop()
        self.configured_patch.stop()
        self.requested_patch.stop()
        self.bundle_dir.cleanup()
        self.state_dir.cleanup()

    def test_disabled_by_default_returns_false(self):
        self.requested_patch.stop()
        static, moisture = _synthetic_static_and_moisture()
        with patch.object(fwms, "_requested", return_value=False):
            result = fwms.score_for_spread_rate(
                static, moisture, {"ros_ch_per_h": np.zeros((2, 2))}, bundle_dir=Path(self.bundle_dir.name))
        self.requested_patch.start()
        self.assertFalse(result)

    def test_writes_immutable_evidence_and_updates_state(self):
        static, moisture = _synthetic_static_and_moisture()
        real_grids = {"ros_ch_per_h": np.array([[1.0, 2.0], [3.0, np.nan]])}
        evidence_root = Path(self.state_dir.name)
        result = fwms.score_for_spread_rate(
            static, moisture, real_grids,
            bundle_dir=Path(self.bundle_dir.name), evidence_root=evidence_root,
        )
        self.assertTrue(result)

        written = list(evidence_root.glob("*.fire_weather_ml_score.json"))
        self.assertEqual(len(written), 1)
        record = json.loads(written[0].read_text())
        self.assertIn("comparison", record)

        with self.assertRaises(FileExistsError):
            written[0].open("x")

        state = fwms.diagnostics()
        self.assertTrue(state["healthy"])
        self.assertEqual(state["successful_runs"], 1)
        self.assertEqual(state["model_version"], "0.0.1-beta.1")

        # The graphic + manifest step is best-effort but should actually
        # succeed here (not just silently swallow an error) - verify the
        # real files it should have produced.
        image_path = Path(self.beta_root_dir.name) / "images" / fwms.IMAGE_FILENAME
        self.assertTrue(image_path.is_file(), "shadow PNG was not written")
        self.assertEqual(state["last_image"], str(image_path))

        from services import beta_products
        manifest = beta_products.load_manifest()
        self.assertIn("fire_weather_ml_shadow", manifest.get("products", {}))
        self.assertTrue(manifest["products"]["fire_weather_ml_shadow"]["not_for_operations"])

    def test_auto_disables_after_max_consecutive_failures(self):
        static, moisture = _synthetic_static_and_moisture()
        evidence_root = Path(self.state_dir.name)
        bad_bundle_dir = Path(tempfile.mkdtemp())  # empty - load_bundle will raise
        for _ in range(fwms.MAX_FAILURES):
            result = fwms.score_for_spread_rate(
                static, moisture, {"ros_ch_per_h": np.zeros((2, 2))},
                bundle_dir=bad_bundle_dir, evidence_root=evidence_root,
            )
            self.assertFalse(result)
        state = fwms.diagnostics()
        self.assertTrue(state["auto_disabled"])
        self.assertFalse(state["enabled"])


class RecordSkippedRunTests(unittest.TestCase):
    def setUp(self):
        self.requested_patch = patch.object(fwms, "_requested", return_value=True)
        self.requested_patch.start()
        self.configured_patch = patch.object(fwms, "_configured", return_value=True)
        self.configured_patch.start()
        self.state_dir = tempfile.TemporaryDirectory()
        self.state_path_patch = patch.object(fwms, "STATE_PATH", Path(self.state_dir.name) / "shadow-state.json")
        self.state_path_patch.start()
        fwms._state.update(enabled=True, consecutive_failures=0, last_error=None, runs=0, healthy=True, auto_disabled=False)

    def tearDown(self):
        self.state_path_patch.stop()
        self.configured_patch.stop()
        self.requested_patch.stop()
        self.state_dir.cleanup()

    def test_returns_false_when_disabled(self):
        with patch.object(fwms, "_requested", return_value=False):
            self.assertFalse(fwms.record_skipped_run("no data"))

    def test_records_a_failure_when_enabled(self):
        self.assertTrue(fwms.record_skipped_run("no data"))
        state = fwms.diagnostics()
        self.assertEqual(state["last_error"], "no data")
        self.assertFalse(state["healthy"])


if __name__ == "__main__":
    unittest.main()
