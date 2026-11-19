"""The post-promotion guardrail rolls back a bad promotion once - and never ping-pongs after a rollback."""
import json
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from models import versioning
from scripts import monitor_model_rollout as monitor


def stamp(days_ago=0):
    return (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d %H:%M:%S")


class RolloutMonitorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        patcher = patch.multiple(versioning, API_DIR=self.root, MODELS_DIR=self.root / "models", CONFIG_PATH=self.root / "models" / "config.json",
                                 VERSIONS_DIR=self.root / "models" / "versions")
        patcher.start()
        self.addCleanup(patcher.stop)
        (self.root / "models" / "versions").mkdir(parents=True)
        for name in ("good.json", "bad.json"):
            (self.root / "models" / "versions" / name).write_text(name)
        good = {"version": "1.0.0", "file": "models/versions/good.json", "performance": {"mae": 2.0, "bias": 0.1}, "promoted_at": stamp(30)}
        bad = {"version": "1.0.1", "file": "models/versions/bad.json", "performance": {"mae": 2.1, "bias": 0.1}, "promoted_at": stamp(1)}
        versioning._save_config({"fuel_moisture": {"stable": bad, "beta": None, "history": [{**good, "channel": "stable", "recorded_at": stamp(1)}]}})
        self.history = self.root / "validation_history.json"

    def write_live(self, mae, bias=0.0):
        self.history.write_text(json.dumps([{"metrics": {"Fuel Moisture (%)": {"mae": mae, "bias": bias}}}]))

    def run_monitor(self, **kwargs):
        with patch.object(versioning, "_announce_live_change"):                  # no GitHub in tests
            return monitor.monitor("fuel_moisture", history_path=self.history, **kwargs)

    def test_a_regression_after_a_promotion_rolls_back_once(self):
        self.write_live(mae=2.9)                                                  # worse than the previous stable's 2.0 by far
        result = self.run_monitor()
        self.assertEqual(result["action"], "rollback")
        self.assertEqual(versioning.get_model_entry("fuel_moisture")["stable"]["version"], "1.0.0")

    def test_after_a_rollback_the_guardrail_does_not_restart_or_roll_back_again(self):
        self.write_live(mae=2.9)
        self.assertEqual(self.run_monitor()["action"], "rollback")
        again = self.run_monitor()                                                # the same bad metrics, a fresh promoted_at
        self.assertEqual(again["action"], "none")
        self.assertIn("rollback target", again["reason"])
        self.assertEqual(versioning.get_model_entry("fuel_moisture")["stable"]["version"], "1.0.0")      # no ping-pong back to 1.0.1

    def test_a_forecast_job_failure_does_not_trigger_a_second_rollback_either(self):
        self.write_live(mae=2.9)
        self.run_monitor()
        self.assertEqual(self.run_monitor(forecast_job_failed=True)["action"], "none")

    def test_metrics_within_the_guardrails_just_keep_monitoring(self):
        self.write_live(mae=2.05)
        self.assertEqual(self.run_monitor()["action"], "monitor")
        self.assertEqual(versioning.get_model_entry("fuel_moisture")["stable"]["version"], "1.0.1")

    def test_an_old_promotion_is_outside_the_window(self):
        config = versioning._load_config()
        config["fuel_moisture"]["stable"]["promoted_at"] = stamp(10)
        versioning._save_config(config)
        self.write_live(mae=9.0)
        self.assertIn("window complete", self.run_monitor()["reason"])


if __name__ == "__main__":
    unittest.main()
