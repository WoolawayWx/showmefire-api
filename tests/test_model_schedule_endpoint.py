"""Covers Phase 5's read-only schedule visibility: GET /api/admin/models/schedule.
Calls the router's async endpoint function directly with a minimal fake
Request exposing app.state.scheduler, same direct-call pattern used by
test_model_admin_settings.py and test_model_admin_activate.py."""
import asyncio
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import routers.model_admin as model_admin


def _run(coro):
    return asyncio.run(coro)


class _FakeJob:
    def __init__(self, job_id, trigger, next_run_time=None):
        self.id = job_id
        self.trigger = trigger
        self.next_run_time = next_run_time


class _FakeScheduler:
    def __init__(self, jobs):
        self._jobs = jobs

    def get_jobs(self):
        return self._jobs


def _fake_request(scheduler):
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(scheduler=scheduler)))


class ModelScheduleEndpointTests(unittest.TestCase):
    def setUp(self):
        self._admin_patcher = patch.object(model_admin, "_require_admin", return_value="tester@example.com")
        self._admin_patcher.start()

    def tearDown(self):
        self._admin_patcher.stop()

    def test_returns_scheduler_not_running_when_state_has_no_scheduler(self):
        result = _run(model_admin.get_model_schedule(_fake_request(None)))
        self.assertEqual(result, {"scheduler_running": False, "jobs": []})

    def test_filters_to_the_curated_allowlist_and_drops_unrelated_jobs(self):
        next_run = datetime(2026, 9, 21, 9, 0, tzinfo=timezone.utc)
        jobs = [
            _FakeJob("run_scheduled_beta_forecast", "cron[hour='9']", next_run),
            _FakeJob("some_unrelated_log_purge_job", "interval[1 day, 0:00:00]", next_run),
        ]
        result = _run(model_admin.get_model_schedule(_fake_request(_FakeScheduler(jobs))))
        self.assertTrue(result["scheduler_running"])
        ids = [row["id"] for row in result["jobs"]]
        self.assertIn("run_scheduled_beta_forecast", ids)
        self.assertNotIn("some_unrelated_log_purge_job", ids)

    def test_job_rows_carry_category_description_cadence_and_next_run(self):
        next_run = datetime(2026, 9, 21, 9, 0, tzinfo=timezone.utc)
        jobs = [_FakeJob("verify_v4_shadow", "interval[3:00:00]", next_run)]
        result = _run(model_admin.get_model_schedule(_fake_request(_FakeScheduler(jobs))))
        row = result["jobs"][0]
        self.assertEqual(row["id"], "verify_v4_shadow")
        self.assertEqual(row["category"], "shadow_verification")
        self.assertIn("V4", row["description"])
        self.assertEqual(row["cadence"], "interval[3:00:00]")
        self.assertEqual(row["next_run_time"], next_run.isoformat())

    def test_job_with_no_next_run_time_returns_none(self):
        jobs = [_FakeJob("drift_check", "cron[hour='2']", None)]
        result = _run(model_admin.get_model_schedule(_fake_request(_FakeScheduler(jobs))))
        self.assertIsNone(result["jobs"][0]["next_run_time"])

    def test_jobs_are_sorted_by_category_then_id(self):
        jobs = [
            _FakeJob("update_seasonal_fuel_state", "cron[hour='1']"),
            _FakeJob("rtma_spread_rate_pipeline", "interval[0:10:00]"),
            _FakeJob("drift_check", "cron[hour='2']"),
        ]
        result = _run(model_admin.get_model_schedule(_fake_request(_FakeScheduler(jobs))))
        categories = [row["category"] for row in result["jobs"]]
        self.assertEqual(categories, sorted(categories))


if __name__ == "__main__":
    unittest.main()
