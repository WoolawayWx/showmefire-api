import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from core.beta_fire_danger import score_fire_danger
from core.fire_danger import calculate_fire_danger
from services import beta_products
from services import forecast_jobs
try:
    from routers import forecast_admin
except ModuleNotFoundError:
    # The minimal local test environment may not include API auth extras.
    forecast_admin = None


class BetaFireDangerTests(unittest.TestCase):
    def test_official_category_matches_canonical_rules(self):
        samples = [
            (20, 70, 5),
            (14, 44, 5),
            (8, 34, 12),
            (8, 24, 5),
            (8, 24, 15),
            (6, 19, 25),
        ]
        for fm, rh, wind in samples:
            result = score_fire_danger(fm, rh, wind)
            self.assertEqual(result["official_category"], calculate_fire_danger(fm, rh, wind))

    def test_score_increases_beyond_an_or_branch_threshold(self):
        near = score_fire_danger(8, 34.9, 12.1)
        deeper = score_fire_danger(8, 25, 20)
        self.assertGreater(deeper["score"], near["score"])
        self.assertGreater(deeper["criteria"]["elevated_branches"]["rh35_wind12"], 0)

    def test_higher_categories_start_at_their_scale_boundary(self):
        self.assertEqual(score_fire_danger(8.9, 34.9, 12)["score"], 1)
        self.assertEqual(score_fire_danger(8.9, 24.9, 15)["score"], 2)
        self.assertEqual(score_fire_danger(6.9, 19.9, 25)["score"], 3)

    def test_score_is_zero_for_low_conditions(self):
        result = score_fire_danger(20, 70, 5)
        self.assertEqual(result["official_label"], "Low")
        self.assertEqual(result["score"], 0)

    def test_invalid_values_are_rejected(self):
        with self.assertRaises(ValueError):
            score_fire_danger(float("nan"), 40, 10)

    def test_beta_category_matches_official_away_from_or_boundaries(self):
        samples = [(20, 70, 5), (14, 44, 5), (8, 34, 12), (8, 24, 5), (8, 24, 15), (6, 19, 25)]
        for fm, rh, wind in samples:
            result = score_fire_danger(fm, rh, wind)
            self.assertEqual(result["beta_category"], result["official_category"])

    def test_beta_category_can_diverge_when_two_or_branches_are_each_partly_satisfied(self):
        # Neither elevated branch's hard AND is satisfied here (branch1's wind is
        # far short of 12; branch2's rh sits just above 25 and its wind is only
        # moderately past 5), so the canonical rule stays at Moderate. Beta's
        # soft-OR lets the two partially-satisfied branches combine past 0.5.
        result = score_fire_danger(8, 25.2, 8.4)
        self.assertEqual(calculate_fire_danger(8, 25.2, 8.4), 1)
        self.assertEqual(result["official_category"], 1)
        self.assertEqual(result["official_label"], "Moderate")
        self.assertEqual(result["beta_category"], 2)
        self.assertEqual(result["beta_label"], "Elevated")
        branches = result["criteria"]["elevated_branches"]
        self.assertLess(branches["rh35_wind12"], 0.5)
        self.assertLess(branches["rh25_wind5"], 0.5)
        self.assertGreaterEqual(result["criteria"]["elevated"], 0.5)

    def test_observation_products_are_written_to_beta_paths(self):
        station = {
            "stid": "TEST",
            "name": "Test Station",
            "state": "MO",
            "longitude": -92,
            "latitude": 38,
            "observations": {
                "relative_humidity": {"value": 30},
                "wind_speed": {"value": 15},
            },
        }
        raw_station = {
            "stid": "TEST",
            "observations": {"fuel_moisture": {"value": 8}},
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(beta_products, "BETA_GIS_DIR", root / "gis"), \
                    patch.object(beta_products, "BETA_MANIFEST_PATH", root / "manifest.json"), \
                    patch.object(beta_products, "BETA_OBSERVATION_STATE_PATH", root / "state.json"):
                result = beta_products.refresh_observation_products(
                    {"stations": [station]},
                    {"stations": [raw_station]},
                )
                self.assertEqual(result["products"]["realtime_current"]["features"][0]["properties"]["stid"], "TEST")
                self.assertTrue((root / "gis" / "realtime_current.geojson").exists())
                self.assertEqual(result["manifest"]["scorer_version"], "2.0.0")

    def test_beta_forecast_prepares_clean_output_tree(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "forecast"
            with patch.object(forecast_jobs, "BETA_FORECAST_ROOT", root):
                forecast_jobs._prepare_beta_directories()
            for relative in ("images", "gis", "logs", "cache/hrrr", "archive/forecasts"):
                self.assertTrue((root / relative).is_dir(), relative)

    def test_failed_subprocess_records_admin_visible_reason(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "forecast"
            job_state = Path(directory) / "forecast_job.json"

            def fail_run(*args, **kwargs):
                kwargs["stdout"].write("forecast setup\nfirst image write failed\n")
                kwargs["stdout"].flush()
                return SimpleNamespace(returncode=2)

            job = {"job_id": "test", "status": "queued"}
            with patch.object(forecast_jobs, "BETA_FORECAST_ROOT", root), \
                    patch.object(forecast_jobs, "JOB_STATE_PATH", job_state), \
                    patch.object(forecast_jobs.subprocess, "run", side_effect=fail_run), \
                    patch.object(forecast_jobs, "_read_model_run", return_value=None):
                forecast_jobs._run_beta_forecast(job)

            stored = forecast_jobs.json.loads(job_state.read_text(encoding="utf-8"))
            self.assertEqual(stored["status"], "failed")
            self.assertEqual(stored["return_code"], 2)
            self.assertIn("exited with code 2", stored["error"])
            self.assertIn("first image write failed", stored["error_detail"])
            self.assertTrue((root / "images").is_dir())

    def test_stale_running_job_does_not_block_rerun(self):
        stale = {
            "job_id": "interrupted",
            "status": "running",
            "started_at": "2000-01-01T00:00:00+00:00",
        }
        with patch.object(forecast_jobs, "_read_job", return_value=stale), \
                patch.object(forecast_jobs, "_write_job"), \
                patch.object(forecast_jobs.threading.Thread, "start"):
            job = forecast_jobs.trigger_beta_forecast("admin@example.org")
        self.assertEqual(job["status"], "queued")
        self.assertEqual(job["replaces_stale_job_id"], "interrupted")

    def test_recent_running_job_still_blocks_duplicate(self):
        active = {
            "job_id": "active",
            "status": "running",
            "started_at": forecast_jobs._now(),
        }
        with patch.object(forecast_jobs, "_read_job", return_value=active):
            with self.assertRaisesRegex(RuntimeError, "already running"):
                forecast_jobs.trigger_beta_forecast("admin@example.org")

    @unittest.skipIf(forecast_admin is None, "API authentication dependencies are not installed")
    def test_forecast_trigger_requires_admin(self):
        with patch.object(forecast_admin, "verify_token", return_value=None):
            with self.assertRaises(Exception) as context:
                forecast_admin._require_admin()
        self.assertEqual(getattr(context.exception, "status_code", None), 401)

    @unittest.skipIf(forecast_admin is None, "API authentication dependencies are not installed")
    def test_forecast_trigger_accepts_admin(self):
        with patch.object(forecast_admin, "verify_token", return_value="admin@example.org"):
            self.assertEqual(forecast_admin._require_admin(), "admin@example.org")


if __name__ == "__main__":
    unittest.main()
