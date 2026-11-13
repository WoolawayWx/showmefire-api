import asyncio
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_SUBMITTED

from core import scheduler_worker
from services import scheduler_heartbeat


class WorkerTests(unittest.TestCase):
    def test_list_prints_only_the_heavy_jobs(self):
        ids = scheduler_worker.list_jobs({"heavy"})
        self.assertIn("hrrr_hourly_forecast", ids)
        self.assertNotIn("fetch_synoptic", ids)
        self.assertEqual(ids, sorted(ids))

    def test_running_jobs_follow_the_scheduler_events(self):
        running = scheduler_worker.RunningJobs()
        running._submitted(SimpleNamespace(job_id="a", code=EVENT_JOB_SUBMITTED))
        running._submitted(SimpleNamespace(job_id="b", code=EVENT_JOB_SUBMITTED))
        self.assertEqual(running.ids, ["a", "b"])
        running._finished(SimpleNamespace(job_id="a", code=EVENT_JOB_EXECUTED))
        self.assertEqual(running.ids, ["b"])

    def test_a_stop_waits_for_running_jobs_then_gives_up_at_the_deadline(self):
        async def scenario():
            running = scheduler_worker.RunningJobs()
            running._submitted(SimpleNamespace(job_id="forecast", code=EVENT_JOB_SUBMITTED))

            async def finish_soon():
                await asyncio.sleep(0.05)
                running._finished(SimpleNamespace(job_id="forecast", code=EVENT_JOB_EXECUTED))

            task = asyncio.create_task(finish_soon())
            clean = await scheduler_worker.drain(running, 2, poll_seconds=0.01)
            await task
            self.assertTrue(clean)

            running._submitted(SimpleNamespace(job_id="stuck", code=EVENT_JOB_SUBMITTED))
            self.assertFalse(await scheduler_worker.drain(running, 0.05, poll_seconds=0.01))

        asyncio.run(scenario())


class HeartbeatTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.beat = root / "scheduler_heavy.json"
        self.patches = [
            patch.object(scheduler_heartbeat, "HEARTBEAT_PATH", self.beat),
            patch.object(scheduler_heartbeat, "STATE_PATH", root / "state.json"),
            patch.dict(os.environ, {"SMF_SCHEDULER_GROUPS": "api"}),   # this process is the API: the worker is elsewhere
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in self.patches:
            item.stop()
        self.temporary.cleanup()

    def _age(self, minutes):
        when = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).timestamp()
        os.utime(self.beat, (when, when))

    def test_heartbeat_round_trip_and_status(self):
        self.assertIsNone(scheduler_heartbeat.read_heartbeat())
        self.assertIsNone(scheduler_heartbeat.worker_status())
        scheduler_heartbeat.write_heartbeat({"groups": ["heavy"], "forecast_v1_enabled": True, "jobs": []})
        self.assertEqual(scheduler_heartbeat.worker_status()["forecast_v1_enabled"], True)
        self._age(10)
        self.assertIsNone(scheduler_heartbeat.worker_status())        # stale: nothing is running the heavy jobs

    def test_the_api_alerts_once_when_the_worker_goes_quiet_and_again_on_recovery(self):
        scheduler_heartbeat.write_heartbeat({"jobs": []})
        with patch.object(scheduler_heartbeat, "_send") as send:
            self.assertEqual(scheduler_heartbeat.run_watch(), [])
            self._age(12)
            first = scheduler_heartbeat.run_watch()
            self.assertEqual([message for message, _ in first], ["Heavy scheduler worker is stale"])
            self.assertEqual(scheduler_heartbeat.run_watch(), [])          # no repeat inside the reminder window
            self.assertEqual(send.call_count, 1)
            scheduler_heartbeat.write_heartbeat({"jobs": []})
            self.assertEqual([m for m, _ in scheduler_heartbeat.run_watch()], ["Heavy scheduler worker recovered"])

    def test_no_alert_before_a_worker_has_ever_run_or_when_this_process_runs_the_heavy_jobs(self):
        with patch.object(scheduler_heartbeat, "_send") as send:
            self.assertEqual(scheduler_heartbeat.run_watch(), [])           # never reported
            scheduler_heartbeat.write_heartbeat({"jobs": []})
            self._age(30)
            with patch.dict(os.environ, {"SMF_SCHEDULER_GROUPS": "api,heavy"}):
                self.assertEqual(scheduler_heartbeat.run_watch(), [])       # unsplit: the API is the scheduler
            send.assert_not_called()

    def test_health_check_reports_a_silent_worker(self):
        from routers import health

        scheduler = SimpleNamespace(running=True, get_jobs=lambda: [1, 2])
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(scheduler=scheduler)))
        scheduler_heartbeat.write_heartbeat({"jobs": []})
        self.assertIn("alive", health.check_scheduler(request).message)
        self._age(20)
        result = health.check_scheduler(request)
        self.assertIn("silent", result.message)


if __name__ == "__main__":
    unittest.main()
