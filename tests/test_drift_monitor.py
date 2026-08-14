import json
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

API_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(API_DIR))

from services import drift


class ComputePsiTests(unittest.TestCase):
    def test_identical_distributions_have_near_zero_psi(self):
        rng = np.random.default_rng(0)
        reference = rng.normal(0, 1, size=2000)
        current = rng.normal(0, 1, size=2000)
        self.assertLess(drift.compute_psi(reference, current), 0.05)

    def test_shifted_distribution_is_flagged(self):
        rng = np.random.default_rng(1)
        reference = rng.normal(0, 1, size=2000)
        current = rng.normal(4, 1, size=2000)
        self.assertGreater(drift.compute_psi(reference, current), drift.DRIFT_PSI_ALERT)

    def test_insufficient_reference_data_returns_zero_not_error(self):
        self.assertEqual(drift.compute_psi([1, 2], [1, 2, 3], bins=10), 0.0)


class EvaluateDriftFuelMoistureTests(unittest.TestCase):
    def _write_jsonl(self, path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")

    def test_flags_shifted_prediction_distribution(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model_shadow.jsonl"
            now = datetime.now(timezone.utc)
            rng = np.random.default_rng(4)
            rows = []
            # Reference window: 30-8 days ago, several runs/day, stable small differences.
            for day in range(30, 8, -1):
                for run in range(5):
                    ts = (now - timedelta(days=day, hours=run)).isoformat()
                    rows.append({"timestamp": ts, "failed": False,
                                "mean_absolute_difference": float(rng.normal(0.1, 0.02))})
            # Current window: last 7 days, much larger differences (drift).
            for day in range(6, -1, -1):
                for run in range(5):
                    ts = (now - timedelta(days=day, hours=run)).isoformat()
                    rows.append({"timestamp": ts, "failed": False,
                                "mean_absolute_difference": float(rng.normal(5.0, 0.2))})
            self._write_jsonl(path, rows)

            report = drift.evaluate_drift("fuel_moisture", evidence_root=path)
            self.assertIsNotNone(report["prediction_psi"])
            self.assertIn("prediction", report["flags"])
            self.assertEqual(report["model_type"], "fuel_moisture")

    def test_no_evidence_file_returns_empty_report_not_error(self):
        with tempfile.TemporaryDirectory() as directory:
            report = drift.evaluate_drift("fuel_moisture", evidence_root=Path(directory) / "missing.jsonl")
            self.assertIsNone(report["prediction_psi"])
            self.assertEqual(report["flags"], [])

    def test_unknown_model_type_raises(self):
        with self.assertRaises(ValueError):
            drift.evaluate_drift("not_a_real_model_type")


class WriteDriftReportTests(unittest.TestCase):
    def test_write_once_second_write_for_same_run_id_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            report = {"model_type": "fuel_moisture", "generated_at": "now", "flags": []}
            first = drift.write_drift_report(report, directory, "run1")
            self.assertTrue(first.exists())
            with self.assertRaises(FileExistsError):
                drift.write_drift_report(report, directory, "run1")


class DriftMonitorIsolationTests(unittest.TestCase):
    def test_one_model_type_failure_does_not_block_others(self):
        from services import drift_monitor

        original_evaluate = drift.evaluate_drift

        def flaky_evaluate(model_type, *args, **kwargs):
            if model_type == "v5":
                raise RuntimeError("simulated failure reading V5 evidence")
            return original_evaluate(model_type, *args, **kwargs)

        with tempfile.TemporaryDirectory() as directory:
            drift.evaluate_drift = flaky_evaluate
            drift_monitor.DRIFT_EVIDENCE_ROOT = Path(directory)
            try:
                result = drift_monitor.run_drift_check()
            finally:
                drift.evaluate_drift = original_evaluate

            self.assertIn("error", result["results"]["v5"])
            self.assertNotIn("error", result["results"]["fuel_moisture"])
            self.assertNotIn("error", result["results"]["risk_fusion_glm"])


if __name__ == "__main__":
    unittest.main()
