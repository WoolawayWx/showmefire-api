import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from services import beta_verification
from services.beta_operations import isolation_checks


class BetaVerificationTests(unittest.TestCase):
    def _write_inputs(self, root: Path):
        forecasts = root / "testbed" / "forecast" / "archive" / "forecasts"
        observations = root / "production" / "archive" / "raw_data"
        output = root / "testbed" / "verification"
        forecasts.mkdir(parents=True)
        observations.mkdir(parents=True)

        forecast_payload = {
            "run_date": "2026-07-11T12:00:00Z",
            "stations": {
                "TEST": {
                    "lat": 38.0,
                    "lon": -92.0,
                    "forecasts": [
                        {
                            "time": "2026-07-11T16:00:00Z", "temp_c": 25.0,
                            "rh": 60.0, "wind_speed_ms": 2.0,
                            "fuel_moisture": 20.0, "fire_danger": 0,
                        },
                        {
                            "time": "2026-07-11T17:00:00Z", "temp_c": 30.0,
                            "rh": 24.0, "wind_speed_ms": 8.23111,
                            "fuel_moisture": 8.0, "fire_danger": 3,
                        },
                    ],
                }
            },
        }
        observation_payload = {
            "STATION": [{
                "STID": "TEST", "LATITUDE": "38.0", "LONGITUDE": "-92.0",
                "OBSERVATIONS": {
                    "date_time": ["2026-07-11T15:53:00Z", "2026-07-11T16:53:00Z"],
                    "air_temp_set_1": [77.0, 86.0],
                    "relative_humidity_set_1": [60.0, 24.0],
                    "wind_speed_set_1": [3.9, 16.0],
                    "fuel_moisture_set_1": [20.0, 8.0],
                },
            }]
        }
        (forecasts / "station_forecasts_beta_20260711_12.json").write_text(
            json.dumps(forecast_payload), encoding="utf-8"
        )
        (observations / "raw_data_20260711.json").write_text(
            json.dumps(observation_payload), encoding="utf-8"
        )
        return forecasts, observations, output

    def test_report_scores_beta_and_writes_only_to_testbed_output(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            forecasts, observations, output = self._write_inputs(root)
            production_sentinel = root / "production" / "validation_history.json"
            production_sentinel.write_text("unchanged", encoding="utf-8")

            with patch.object(beta_verification, "MINIMUM_SUPPORT", 1):
                report = beta_verification.run_beta_verification(
                    forecast_dir=forecasts,
                    observations_dir=observations,
                    output_root=output,
                )

            self.assertEqual(report["date"], "2026-07-11")
            self.assertEqual(report["record_count"], 2)
            self.assertEqual(report["status"], "ready")
            self.assertEqual(report["stable"]["count"], 2)
            self.assertEqual(report["beta"]["count"], 2)
            self.assertEqual(report["stable"]["exact_match_rate"], 1.0)
            self.assertEqual(report["beta"]["exact_match_rate"], 1.0)
            self.assertLess(report["beta_continuous"]["exact_match_rate"], 1.0)
            self.assertFalse(report["isolation"]["production_verification_history_modified"])
            self.assertTrue((output / "20260711.json").exists())
            self.assertTrue((output / "latest.json").exists())
            self.assertTrue((output / "history.json").exists())
            self.assertEqual(production_sentinel.read_text(encoding="utf-8"), "unchanged")

    def test_missing_observations_waits_without_writing_report(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            forecasts, observations, output = self._write_inputs(root)
            (observations / "raw_data_20260711.json").unlink()
            with self.assertRaisesRegex(RuntimeError, "not available yet"):
                beta_verification.run_beta_verification(
                    forecast_dir=forecasts,
                    observations_dir=observations,
                    output_root=output,
                )
            self.assertFalse(output.exists())


class BetaIsolationTests(unittest.TestCase):
    def test_default_style_testbed_root_passes_all_checks(self):
        result = isolation_checks(Path("data/testbed"))
        self.assertTrue(result["passed"])
        self.assertTrue(all(result["checks"].values()))

    def test_beta_root_inside_production_reports_is_rejected(self):
        result = isolation_checks(Path("reports/testbed"))
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["testbed_root_separate_from_production_outputs"])


if __name__ == "__main__":
    unittest.main()
