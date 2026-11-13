import os
import unittest
from unittest.mock import patch

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from pytz import timezone

from core import job_groups
from core.job_groups import ALL_GROUPS, API, CONDITIONAL_FLAGS, HEAVY, JOB_GROUPS, GroupFilteredScheduler, groups_from_env


def _register(groups):
    with patch.dict(os.environ, CONDITIONAL_FLAGS):
        from core.scheduler import register_jobs

        scheduler = AsyncIOScheduler(timezone=timezone("America/Chicago"))
        return register_jobs(scheduler, groups)


class JobGroupTests(unittest.TestCase):
    def test_every_registered_job_has_a_group_and_no_group_names_a_job_that_does_not_exist(self):
        registered = _register(set(ALL_GROUPS))
        self.assertEqual(registered.unclassified, [], "add these job ids to core/job_groups.py")
        self.assertEqual(set(JOB_GROUPS) - set(registered.registered), set(), "stale ids in core/job_groups.py")
        self.assertEqual(len(registered.registered), len(set(registered.registered)), "a job id was registered twice")

    def test_the_two_groups_split_the_jobs_with_nothing_lost_or_shared(self):
        everything = set(_register(set(ALL_GROUPS)).registered)
        api_only = set(_register({API}).registered)
        heavy_only = set(_register({HEAVY}).registered)
        self.assertEqual(api_only & heavy_only, set())
        self.assertEqual(api_only | heavy_only, everything)
        self.assertIn("run_forecast_v1_shadow", heavy_only)
        self.assertIn("hrrr_hourly_forecast", heavy_only)
        self.assertIn("fetch_synoptic", api_only)          # feeds the API serves from memory stay in the API
        self.assertIn("forecast_watchdog", api_only)       # still alerts if the worker is dead
        self.assertIn("drift_check", api_only)

    def test_model_schedule_jobs_all_exist(self):
        from core.scheduler import MODEL_RELEVANT_JOBS

        self.assertEqual(set(MODEL_RELEVANT_JOBS) - set(JOB_GROUPS), set())

    def test_groups_from_env(self):
        self.assertEqual(groups_from_env(""), set(ALL_GROUPS))            # nothing set: everything (unsplit)
        self.assertEqual(groups_from_env("heavy"), {HEAVY})
        self.assertEqual(groups_from_env(" API , heavy "), set(ALL_GROUPS))
        self.assertEqual(groups_from_env("bogus"), set(ALL_GROUPS))        # never "run nothing" by typo
        with patch.dict(os.environ, {"SMF_SCHEDULER_GROUPS": "api"}):
            self.assertEqual(groups_from_env(), {API})

    def test_the_filter_runs_unclassified_jobs_rather_than_dropping_them(self):
        calls = []

        class Fake:
            def add_job(self, func, *args, **kwargs):
                calls.append(kwargs.get("id"))

        wrapper = GroupFilteredScheduler(Fake(), {HEAVY})
        wrapper.add_job(lambda: None, "interval", minutes=1, id="brand_new_job")
        wrapper.add_job(lambda: None, "interval", minutes=1, id="fetch_synoptic")
        self.assertEqual(calls, [])                      # both are api-group, this process only runs heavy
        self.assertEqual(wrapper.unclassified, ["brand_new_job"])

    def test_describe_all_jobs_lists_every_group_even_with_the_flags_off(self):
        from core.scheduler import describe_all_jobs

        with patch.dict(os.environ, {"SMF_FORECAST_V1_ENABLED": "false"}):
            jobs = {job["id"]: job for job in describe_all_jobs()}
            self.assertEqual(jobs["run_forecast_v1_shadow"]["group"], HEAVY)
            self.assertEqual(jobs["fetch_synoptic"]["group"], API)
            self.assertEqual(os.environ["SMF_FORECAST_V1_ENABLED"], "false")   # restored


if __name__ == "__main__":
    unittest.main()
