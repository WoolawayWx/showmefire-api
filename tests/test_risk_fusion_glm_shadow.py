import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from services import risk_fusion_glm_shadow as rfgs


def _real_feature_module_sha256() -> str:
    return rfgs._sha256_file(rfgs.FEATURES_MODULE_PATH)


def _write_bundle(directory: Path, *, advisory_only=True, model_family="glm", feature_hash=None,
                  county_fips=("29001",)) -> None:
    feature_hash = feature_hash if feature_hash is not None else _real_feature_module_sha256()
    zero_params_climatology = [0.0] + [0.0] * len(rfgs.MONTH_DUMMY_COLUMNS)
    zero_params_residual = [0.0] + [0.0] * len(rfgs.FAST_WEATHER_FEATURES)
    (directory / "contract.json").write_text(json.dumps({
        "advisory_only": advisory_only,
        "model_family": model_family,
        "feature_module_sha256": feature_hash,
    }))
    (directory / "glm_climatology.json").write_text(json.dumps({
        "feature_columns": rfgs.MONTH_DUMMY_COLUMNS,
        "offset_column": "log_effort_scaled",
        "params": zero_params_climatology,
        "means": {column: 0.0 for column in rfgs.MONTH_DUMMY_COLUMNS},
        "stds": {column: 1.0 for column in rfgs.MONTH_DUMMY_COLUMNS},
    }))
    (directory / "glm_residual.json").write_text(json.dumps({
        "feature_columns": rfgs.FAST_WEATHER_FEATURES,
        "offset_column": "_base_eta",
        "params": zero_params_residual,
        "means": {column: 0.0 for column in rfgs.FAST_WEATHER_FEATURES},
        "stds": {column: 1.0 for column in rfgs.FAST_WEATHER_FEATURES},
    }))
    (directory / "effort.json").write_text(json.dumps({
        "rate_table": [
            {"county_fips": fips, "events": 10.0, "exposure_km2_days": 1.0,
             "raw_rate": 0.001, "reporting_rate_shrunk": 0.001}
            for fips in county_fips
        ],
        "state_mean_rate": 0.001,
        "shrinkage_k": 5.0,
        "n_days_observed": 100,
        "effort_exponent": 1.0,
    }))
    (directory / "county_reference.json").write_text(json.dumps({
        fips: {"burnable_area_km2": 1000.0} for fips in county_fips
    }))


class LoadBundleTests(unittest.TestCase):
    def test_missing_directory_raises(self):
        with self.assertRaises(FileNotFoundError):
            rfgs.load_bundle(Path("/nonexistent/path/for/sure"))

    def test_valid_bundle_loads_and_computes_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_bundle(directory)
            bundle = rfgs.load_bundle(directory)
            self.assertEqual(bundle["contract"]["model_family"], "glm")
            self.assertIsInstance(bundle["bundle_checksum"], str)

    def test_non_advisory_bundle_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_bundle(directory, advisory_only=False)
            with self.assertRaises(ValueError):
                rfgs.load_bundle(directory)

    def test_non_glm_model_family_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_bundle(directory, model_family="gbm")
            with self.assertRaises(ValueError):
                rfgs.load_bundle(directory)

    def test_feature_module_hash_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_bundle(directory, feature_hash="0" * 64)
            with self.assertRaises(ValueError):
                rfgs.load_bundle(directory)


class MonthDummiesTests(unittest.TestCase):
    def test_january_is_the_reference_level(self):
        row = rfgs.month_dummies(1)
        self.assertTrue(all(value == 0.0 for value in row.values()))

    def test_other_months_set_exactly_one_flag(self):
        row = rfgs.month_dummies(7)
        self.assertEqual(row["month_7"], 1.0)
        self.assertEqual(sum(row.values()), 1.0)


class ScoreCountyDayTests(unittest.TestCase):
    def test_all_zero_coefficients_give_deterministic_lam(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_bundle(directory)
            bundle = rfgs.load_bundle(directory)
            calendar_row = {**rfgs.month_dummies(1), "is_weekend": False}
            weather_row = {
                "rh_mean": 40.0, "rh_min_afternoon": 25.0, "wind_kts_max": 15.0,
                "wind_kts_p90": 12.0, "vpd_kpa_max": 2.0, "precip_24h_mm": 0.0,
            }
            # burnable_area_km2=1000, reporting_rate_shrunk=0.001 -> log_effort = log(1000*0.001) = log(1) = 0.
            # All fitted coefficients are zero, so lam = exp(0) = 1.0 exactly.
            result = rfgs.score_county_day(bundle, "29001", calendar_row, weather_row)
            self.assertAlmostEqual(result["lam"], 1.0, places=9)
            self.assertAlmostEqual(result["p_ge1_fire"], 1.0 - 2.718281828459045 ** -1.0, places=6)

    def test_unknown_county_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            _write_bundle(directory)
            bundle = rfgs.load_bundle(directory)
            calendar_row = {**rfgs.month_dummies(1), "is_weekend": False}
            weather_row = {column: 0.0 for column in rfgs.FAST_WEATHER_FEATURES}
            with self.assertRaises(KeyError):
                rfgs.score_county_day(bundle, "99999", calendar_row, weather_row)


class ScoreGlmForForecastTests(unittest.TestCase):
    def setUp(self):
        self.bundle_dir = tempfile.TemporaryDirectory()
        _write_bundle(Path(self.bundle_dir.name))

        self.requested_patch = patch.object(rfgs, "_requested", return_value=True)
        self.requested_patch.start()
        self.configured_patch = patch.object(rfgs, "_configured", return_value=True)
        self.configured_patch.start()

        self.state_dir = tempfile.TemporaryDirectory()
        self.state_path_patch = patch.object(rfgs, "STATE_PATH", Path(self.state_dir.name) / "shadow-state.json")
        self.state_path_patch.start()

        rfgs._state.update(
            enabled=True, consecutive_failures=0, last_error=None, runs=0, successful_runs=0, healthy=True,
            auto_disabled=False, counties_scored=0, county_days_recorded=0,
        )

    def tearDown(self):
        self.state_path_patch.stop()
        self.configured_patch.stop()
        self.requested_patch.stop()
        self.bundle_dir.cleanup()
        self.state_dir.cleanup()

    def test_disabled_by_default_returns_false(self):
        self.requested_patch.stop()
        with patch.object(rfgs, "_requested", return_value=False):
            result = rfgs.score_glm_for_forecast(
                run_id="run1", valid_local_date="2026-07-01", county_fips=["29001"],
                calendar_rows={"29001": {**rfgs.month_dummies(7), "is_weekend": False}},
                weather_rows={"29001": {column: 0.0 for column in rfgs.FAST_WEATHER_FEATURES}},
                bundle_dir=Path(self.bundle_dir.name),
            )
        self.requested_patch.start()
        self.assertFalse(result)

    def test_writes_immutable_evidence_and_updates_state(self):
        evidence_root = Path(self.state_dir.name)
        result = rfgs.score_glm_for_forecast(
            run_id="run1", valid_local_date="2026-07-01", county_fips=["29001"],
            calendar_rows={"29001": {**rfgs.month_dummies(7), "is_weekend": False}},
            weather_rows={"29001": {
                "rh_mean": 40.0, "rh_min_afternoon": 25.0, "wind_kts_max": 15.0,
                "wind_kts_p90": 12.0, "vpd_kpa_max": 2.0, "precip_24h_mm": 0.0,
            }},
            bundle_dir=Path(self.bundle_dir.name),
            evidence_root=evidence_root,
        )
        self.assertTrue(result)
        record = json.loads((evidence_root / "run1.glm_score.json").read_text())
        self.assertEqual(record["county_fips"], ["29001"])
        self.assertEqual(len(record["lam"]), 1)

        with self.assertRaises(FileExistsError):
            (evidence_root / "run1.glm_score.json").open("x")

        state = rfgs.diagnostics()
        self.assertTrue(state["healthy"])
        self.assertEqual(state["successful_runs"], 1)

    def test_auto_disables_after_max_consecutive_failures(self):
        evidence_root = Path(self.state_dir.name)
        for _ in range(rfgs.MAX_FAILURES):
            result = rfgs.score_glm_for_forecast(
                run_id="bad", valid_local_date="2026-07-01", county_fips=["99999"],
                calendar_rows={"99999": {**rfgs.month_dummies(7), "is_weekend": False}},
                weather_rows={"99999": {column: 0.0 for column in rfgs.FAST_WEATHER_FEATURES}},
                bundle_dir=Path(self.bundle_dir.name),
                evidence_root=evidence_root,
            )
            self.assertFalse(result)
        state = rfgs.diagnostics()
        self.assertTrue(state["auto_disabled"])
        self.assertFalse(state["enabled"])


class RecordSkippedRunTests(unittest.TestCase):
    def setUp(self):
        self.requested_patch = patch.object(rfgs, "_requested", return_value=True)
        self.requested_patch.start()
        self.configured_patch = patch.object(rfgs, "_configured", return_value=True)
        self.configured_patch.start()
        self.state_dir = tempfile.TemporaryDirectory()
        self.state_path_patch = patch.object(rfgs, "STATE_PATH", Path(self.state_dir.name) / "shadow-state.json")
        self.state_path_patch.start()
        rfgs._state.update(enabled=True, consecutive_failures=0, last_error=None, runs=0,
                           healthy=True, auto_disabled=False)

    def tearDown(self):
        self.state_path_patch.stop()
        self.configured_patch.stop()
        self.requested_patch.stop()
        self.state_dir.cleanup()

    def test_records_reason_and_marks_unhealthy(self):
        result = rfgs.record_skipped_run("no leads available")
        self.assertTrue(result)
        state = rfgs.diagnostics()
        self.assertFalse(state["healthy"])
        self.assertEqual(state["last_error"], "no leads available")


if __name__ == "__main__":
    unittest.main()
