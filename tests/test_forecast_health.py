import asyncio
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

from services import forecast_health as fh

DAY = datetime(2026, 10, 27, tzinfo=timezone.utc)


def at(hour, minute=0):
    return DAY.replace(hour=hour, minute=minute)


class ForecastHealthTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        patcher = patch.object(fh, "STATE_PATH", Path(folder.name) / "forecast_health.json")
        patcher.start()
        self.addCleanup(patcher.stop)
        alert = patch("services.system_alerts.notify_system_issue")
        self.alert = alert.start()
        self.addCleanup(alert.stop)

    def test_ok_stage_does_not_alert(self):
        fh.record("fuel_moisture_observations", "ok", "20 stations", job="forecasts_12z", now=at(13))
        self.alert.assert_not_called()

    def test_degraded_and_failed_stages_alert_with_the_job_as_source(self):
        fh.record("fuel_moisture_observations", "degraded", "used 6 h old data", job="forecasts_12z", now=at(13))
        fh.record("Running Daily Forecast", "failed", "exit 1", job="forecasts_12z", detail="Traceback", now=at(14))
        (first, second) = self.alert.call_args_list
        self.assertEqual(first.args[0], "forecast:forecasts_12z")
        self.assertIn("degraded", first.args[1])
        self.assertIn("FAILED", second.args[1])
        self.assertEqual(second.kwargs["detail"], "Traceback")

    def test_run_status_is_the_worst_stage_and_finish_sends_no_second_alert(self):
        fh.begin_run("forecasts_12z", now=at(13))
        fh.record("a", "degraded", "x", job="forecasts_12z", now=at(13, 5))
        self.alert.reset_mock()
        self.assertEqual(fh.finish_run("forecasts_12z", now=at(14)), "degraded")
        self.alert.assert_not_called()
        self.assertEqual(fh.summary(at(14, 1))["jobs"]["forecasts_12z"]["latest"]["status"], "degraded")

    def test_begin_run_archives_the_previous_run(self):
        fh.begin_run("forecasts_12z", now=at(13))
        fh.record("a", "failed", "x", job="forecasts_12z", alert=False, now=at(13, 1))
        fh.finish_run("forecasts_12z", now=at(13, 2))
        fh.begin_run("forecasts_12z", now=at(13) + timedelta(days=1))
        recent = fh.summary(at(13) + timedelta(days=1, minutes=1))["jobs"]["forecasts_12z"]["recent"]
        self.assertEqual(recent, ["failed"])

    def test_watchdog_flags_a_run_that_never_started_once_per_day(self):
        self.assertEqual(fh.check_overdue(at(15, 29)), [])  # 09z is not due until 15:30Z, 12z until 17:30Z
        reasons = fh.check_overdue(at(17, 31))
        self.assertTrue(any("no forecasts_12z run has started today" in r for r in reasons))
        self.assertEqual(fh.check_overdue(at(18)), [])  # already alerted today
        self.assertEqual(self.alert.call_count, 2)  # 09z and 12z, one alert each

    def test_watchdog_flags_an_unfinished_run_but_not_a_finished_one(self):
        fh.begin_run("forecasts_12z", now=at(13))
        fh.begin_run("forecasts_09z", now=at(12, 5))
        fh.finish_run("forecasts_09z", now=at(13, 30))
        reasons = fh.check_overdue(at(17, 45))
        self.assertEqual(len(reasons), 1)
        self.assertIn("has not finished", reasons[0])

    def test_yesterdays_finished_run_does_not_satisfy_todays_deadline(self):
        yesterday = at(13) - timedelta(days=1)
        fh.begin_run("forecasts_12z", now=yesterday)
        fh.finish_run("forecasts_12z", now=yesterday + timedelta(hours=1))
        self.assertTrue(any("12z" in r for r in fh.check_overdue(at(18))))

    def test_summary_reports_overdue_as_failed(self):
        self.assertEqual(fh.summary(at(12))["status"], "ok")
        self.assertEqual(fh.summary(at(18))["status"], "failed")

    def test_cli_stage_attaches_the_log_tail(self):
        log = Path(fh.STATE_PATH).with_name("run.log")
        log.write_text("\n".join(f"line {n}" for n in range(40)), encoding="utf-8")
        fh.main(["stage", "Running Daily Forecast", "failed", "exited with code 1", "--job", "forecasts_12z",
                 "--log", str(log)])
        detail = self.alert.call_args.kwargs["detail"]
        self.assertIn("line 39", detail)
        self.assertNotIn("line 0\n", detail)


class FuelMoistureWindowTests(unittest.TestCase):
    """fetch_fuel_moisture_at_time keeps the 1 h rule by default and reaches further back only when asked."""

    TARGET = datetime(2026, 10, 27, 13, tzinfo=timezone.utc)

    def _fetch(self, observed_hours_before, **kwargs):
        from services import synoptic

        stamp = (self.TARGET - timedelta(hours=observed_hours_before)).isoformat()
        station = {"STID": "T1", "ID": "1", "NAME": "T", "STATE": "MO", "LATITUDE": "38", "LONGITUDE": "-92",
                   "OBSERVATIONS": {"date_time": [stamp], "fuel_moisture_value_1": [9.5]}}
        calls = []

        async def fake_history(**call):
            calls.append(call)
            return {"STATION": [station]}

        with patch.object(synoptic, "fetch_historical_station_data", fake_history):
            result = asyncio.run(synoptic.fetch_fuel_moisture_at_time(target_time=self.TARGET, **kwargs))
        return result, calls[0]

    @staticmethod
    def _count(result):
        stations = result.get("stations") or result.get("data", {}).get("stations") or []
        return len(stations)

    def test_default_ignores_an_observation_three_hours_old(self):
        result, _ = self._fetch(3)
        self.assertEqual(self._count(result), 0)

    def test_wider_window_accepts_it_and_widens_the_query(self):
        result, call = self._fetch(3, max_age_hours=6)
        self.assertEqual(self._count(result), 1)
        self.assertEqual(call["start_time"], self.TARGET - timedelta(hours=6))

    def test_default_query_window_is_unchanged(self):
        _, call = self._fetch(0.5)
        self.assertEqual(call["start_time"], self.TARGET - timedelta(hours=1))


if __name__ == "__main__":
    unittest.main()
