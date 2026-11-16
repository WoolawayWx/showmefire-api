"""Run-in-progress marks, duration statistics and the overrun alert (services/job_runs.py)."""
import time

import pytest

from services import job_runs


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setattr(job_runs, "_table_ready", False)
    return tmp_path


def _history(job, durations):
    now = time.time()
    for index, duration in enumerate(durations):
        job_runs.record_run(job, "ok", now - (len(durations) - index) * 86400, duration)


def test_a_run_is_visible_while_it_runs_and_gone_afterwards(db):
    job_runs.mark_started("forecast_v1", time.time() - 120)
    running = job_runs.running_now()
    assert [r["job"] for r in running] == ["forecast_v1"] and 100 < running[0]["elapsed_s"] < 200
    job_runs.mark_finished("forecast_v1")
    assert job_runs.running_now() == []


def test_a_mark_left_by_a_killed_run_is_ignored_and_pruned(db):
    job_runs.mark_started("old", time.time() - 5 * 86400)
    assert job_runs.running_now() == []
    job_runs.prune_runs()
    conn = job_runs._connect()
    assert conn.execute("SELECT COUNT(*) FROM job_running").fetchone()[0] == 0


def test_duration_stats_ignore_instant_runs_and_need_history(db):
    _history("fire_outlook", [1.0] * 40 + [1500.0, 1600.0, 1700.0])   # mostly "nothing to do" exits
    stats = job_runs.duration_stats()["fire_outlook"]
    assert stats["runs"] == 3 and stats["median_s"] == 1600.0 and not stats["enough_history"]
    assert job_runs.overrun_threshold_s(stats) is None


def test_overrun_threshold_is_1_5x_p95_or_15_minutes_more(db):
    _history("forecast_v1", [21000, 21600, 22000, 21200, 21800, 22400])   # about 6 h
    stats = job_runs.duration_stats()["forecast_v1"]
    assert stats["enough_history"]
    assert job_runs.overrun_threshold_s(stats) == pytest.approx(stats["p95_s"] * 1.5)
    _history("short_job", [100, 110, 105, 95, 100, 102])
    short = job_runs.duration_stats()["short_job"]
    assert job_runs.overrun_threshold_s(short) == pytest.approx(short["p95_s"] + 900)   # +15 min wins for short jobs


def test_a_stuck_run_alerts_once(db):
    _history("forecast_v1", [21000, 21600, 22000, 21200, 21800, 22400])
    sent = []
    job_runs.mark_started("forecast_v1", time.time() - 12 * 3600)       # 12 h: well past 1.5 x 6 h
    assert job_runs.check_overruns(lambda *a, **k: sent.append(a)) == ["forecast_v1"]
    assert job_runs.check_overruns(lambda *a, **k: sent.append(a)) == []   # not again for the same run
    assert len(sent) == 1 and "forecast_v1" in sent[0][1]


def test_a_normal_long_run_does_not_alert(db):
    _history("forecast_v1", [21000, 21600, 22000, 21200, 21800, 22400])
    job_runs.mark_started("forecast_v1", time.time() - 6.5 * 3600)
    assert job_runs.check_overruns(lambda *a, **k: pytest.fail("should not alert")) == []


def test_a_failed_alert_is_retried_next_time(db):
    _history("forecast_v1", [21000, 21600, 22000, 21200, 21800, 22400])
    job_runs.mark_started("forecast_v1", time.time() - 12 * 3600)

    def boom(*a, **k):
        raise RuntimeError("discord down")

    assert job_runs.check_overruns(boom) == []
    assert job_runs.check_overruns(lambda *a, **k: None) == ["forecast_v1"]
