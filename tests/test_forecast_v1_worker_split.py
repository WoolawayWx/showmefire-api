import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from services import forecast_v1_job as job_module


class WorkerSplitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.db = root / "test.db"
        self.patches = [
            patch.object(job_module, "FORECAST_V1_DIR", root / "forecast-v1"),
            patch.object(job_module, "JOB_STATE_PATH", root / "forecast-v1" / "admin-job.json"),
            patch.object(job_module, "_execution_lock", job_module._RunLock()),
        ]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in self.patches:
            item.stop()
        self.temporary.cleanup()

    def test_the_run_lock_excludes_other_holders_and_is_visible_to_other_processes(self):
        if job_module.fcntl is None:
            self.skipTest("fcntl not available")
        mine, other = job_module._RunLock(), job_module._RunLock()   # two locks = two processes as far as flock cares
        self.assertFalse(other.locked())
        self.assertTrue(mine.acquire())
        self.assertTrue(other.locked())             # "execution_active" is true for the API while the worker runs
        self.assertFalse(other.acquire())
        self.assertFalse(mine.acquire())            # not re-entrant
        mine.release()
        self.assertFalse(other.locked())
        self.assertTrue(other.acquire())
        other.release()

    def test_admin_run_now_only_queues_and_the_worker_starts_it(self):
        job = job_module.trigger_forecast_v1("staff@example.org")
        self.assertEqual((job["status"], job["requested_by"]), ("queued", "staff@example.org"))
        with self.assertRaises(RuntimeError):
            job_module.trigger_forecast_v1("someone@example.org")   # already queued
        started = []
        with patch.object(job_module, "_run_admin_forecast", side_effect=lambda queued: started.append(queued["job_id"])):
            result = job_module.run_requested_admin_forecast()
        self.assertEqual((result["started"], started), (True, [job["job_id"]]))

    def test_the_worker_waits_while_a_scheduled_run_holds_the_lock_and_drops_old_requests(self):
        job_module.trigger_forecast_v1("staff@example.org")
        started = []
        with patch.object(job_module, "_run_admin_forecast", side_effect=lambda queued: started.append(1)):
            self.assertTrue(job_module._execution_lock.acquire())
            try:
                self.assertEqual(job_module.run_requested_admin_forecast(), {"started": False, "busy": True})
            finally:
                job_module._execution_lock.release()
            self.assertEqual(started, [])
            stale = job_module._read_job()
            stale["requested_at"] = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat().replace("+00:00", "Z")
            job_module._write_job(stale)
            self.assertTrue(job_module.run_requested_admin_forecast().get("expired"))
        self.assertEqual(job_module._read_job()["status"], "failed")

    def test_scheduler_requests_and_idle_state_are_ignored_by_the_request_runner(self):
        job_module._write_job({"status": "idle"})
        self.assertEqual(job_module.run_requested_admin_forecast(), {"started": False})
        job_module._write_job({"status": "queued", "requested_by": "scheduler", "requested_at": job_module._now()})
        self.assertEqual(job_module.run_requested_admin_forecast(), {"started": False})

    def test_reconcile_fails_an_orphaned_running_job_and_its_staging_run_but_not_a_live_one(self):
        from forecast_v1 import repository

        repository.ensure_schema(self.db)
        with closing(sqlite3.connect(self.db)) as connection, connection:
            connection.execute(
                "INSERT INTO forecast_runs (run_id, cycle_time_utc, status, grid_id, horizon_hours, schema_version, config_version, "
                "model_version, created_at_utc) VALUES ('r1', '2026-11-12T12:00:00Z', 'staging', 'g', 72, 's', 'c', 'm', '2026-11-12T18:00:00Z')")
        job_module._write_job({"status": "running", "started_at": job_module._now(), "events": []})
        with patch.object(repository, "get_db_path", return_value=self.db):
            if job_module.fcntl is not None:
                self.assertTrue(job_module._execution_lock.acquire())      # a live run holds the lock: leave everything alone
                try:
                    self.assertEqual(job_module.reconcile_orphaned_run(), {"reconciled": False})
                finally:
                    job_module._execution_lock.release()
            result = job_module.reconcile_orphaned_run()
        self.assertTrue(result["reconciled"])
        self.assertEqual(job_module._read_job()["status"], "failed")
        with closing(sqlite3.connect(self.db)) as connection:
            self.assertEqual(connection.execute("SELECT status FROM forecast_runs").fetchone()[0], "failed")


if __name__ == "__main__":
    unittest.main()
