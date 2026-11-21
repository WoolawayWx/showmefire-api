import json
from datetime import datetime, timedelta, timezone

import pytest

from services import job_health
from tools import firedetections


def iso(now, minutes_ago):
    return (datetime.fromtimestamp(now, tz=timezone.utc) - timedelta(minutes=minutes_ago)).isoformat().replace("+00:00", "Z")


def run(now, minutes_ago, status="error", message=None):
    return {"started_at": iso(now, minutes_ago), "status": status, "message": message, "duration_s": 1, "warnings": 0, "errors": 0}


@pytest.mark.parametrize("message,expect", [
    ("FIRMS CSV request failed: 400 Client Error ... Invalid MAP_KEY.", "MODIS_KEY"),
    ("exit code 127", "PATH"),
    ("extended forecast exit code 143; see logs", "SIGTERM"),
    ("Missing data files.", "forecast file"),
    ("An error occurred (InternalError) when calling the GetObject operation (reached max retries: 4)", "server error"),
    ("urllib3 Read timed out. (read timeout=20)", "slow or unreachable"),
])
def test_diagnose_known_errors(message, expect):
    assert expect in job_health.diagnose(message)["cause"] + job_health.diagnose(message)["fix"]


def test_diagnose_unknown_returns_none():
    assert job_health.diagnose("something new") is None
    assert job_health.diagnose(None) is None


def test_failures_needed_scales_with_frequency():
    assert job_health.failures_needed(300) == 6
    assert job_health.failures_needed(3 * 3600) == 3
    assert job_health.failures_needed(86400) == 2
    assert job_health.failures_needed(None) == 2


def test_daily_job_alerts_after_two_failures_then_stays_quiet_then_recovers():
    now = 1_800_000_000.0
    runs = [run(now, 60, message="exit code 127"), run(now, 1500, message="exit code 127")]
    alerts, state = job_health.plan({}, {"empty_outlook_maps": runs}, {}, {}, now)
    assert [a[0] for a in alerts] == ["Blank outlook maps keeps failing"]
    assert "exit code 127" in alerts[0][1] and "What to do" in alerts[0][1]

    again, state2 = job_health.plan({}, {"empty_outlook_maps": runs}, {}, state, now + 600)
    assert again == []

    fixed = [run(now, 1, status="ok")] + runs
    recovered, state3 = job_health.plan({}, {"empty_outlook_maps": fixed}, {}, state2, now + 900)
    assert [a[0] for a in recovered] == ["Blank outlook maps recovered"] and state3 == {}


def test_frequent_job_ignores_short_streaks_and_reminds_later():
    now = 1_800_000_000.0
    five_failures = [run(now, 5 * i, message="x") for i in range(5)] + [run(now, 40, status="ok")]
    assert job_health.plan({}, {"fetch_synoptic": five_failures}, {}, {}, now)[0] == []
    six = [run(now, 5 * i, message="Invalid MAP_KEY") for i in range(6)]
    alerts, state = job_health.plan({}, {"fetch_synoptic": six}, {}, {}, now)
    assert len(alerts) == 1
    later, _ = job_health.plan({}, {"fetch_synoptic": six}, {}, state, now + 13 * 3600)
    assert later[0][0].endswith("(still)")


def test_overdue_cron_job_alerts_only_inside_window():
    now = 1_800_000_000.0
    cron = {"verification": {"label": "Nightly verification", "schedule": "Daily, 04:30 UTC", "every_s": 86400}}
    summary = lambda hours: {"verification": {"last": run(now, hours * 60, status="ok")}}
    assert job_health.plan(summary(30), {}, cron, {}, now)[0] == []
    assert job_health.plan(summary(60), {}, cron, {}, now)[0][0][0] == "Nightly verification has stopped running"
    assert job_health.plan(summary(24 * 30), {}, cron, {}, now)[0] == []  # long retired/disabled


def test_check_jobs_uses_state_file(tmp_path, monkeypatch):
    import time
    now = time.time()
    runs = [run(now, 60, message="exit code 127"), run(now, 1500, message="exit code 127")]
    monkeypatch.setattr("services.job_runs.job_summaries", lambda window_hours=24: {"empty_outlook_maps": {"last": runs[0]}})
    monkeypatch.setattr("services.job_runs.recent_runs", lambda job, limit=50: runs)
    sent = []
    state_path = tmp_path / "state.json"
    first = job_health.check_jobs(notify=lambda *a, **k: sent.append((a, k)), state_path=state_path, now=now)
    second = job_health.check_jobs(notify=lambda *a, **k: sent.append((a, k)), state_path=state_path, now=now + 900)
    assert first == ["Blank outlook maps keeps failing"] and second == [] and len(sent) == 1
    assert "empty_outlook_maps:failing" in json.loads(state_path.read_text())


def _save(tmp_path, monkeypatch, previous_meta, data):
    monkeypatch.setattr(firedetections, "GIS_DIR", tmp_path)
    (tmp_path / "satfiredetection.geojson").write_text(json.dumps(
        {"type": "FeatureCollection", "features": [{"id": 1}], "metadata": previous_meta}))
    firedetections.save_detections(data)
    return json.loads((tmp_path / "satfiredetection.geojson").read_text())


def test_empty_fetch_keeps_recent_previous_detections(tmp_path, monkeypatch):
    recent = (datetime.utcnow() - timedelta(hours=3)).isoformat() + "Z"
    out = _save(tmp_path, monkeypatch, {"features_as_of": recent}, {"type": "FeatureCollection", "features": [], "metadata": {}})
    assert out["features"] == [{"id": 1}] and out["metadata"]["features_as_of"] == recent


def test_empty_fetch_drops_day_old_previous_detections(tmp_path, monkeypatch):
    old = (datetime.utcnow() - timedelta(days=24)).isoformat() + "Z"
    out = _save(tmp_path, monkeypatch, {"features_as_of": old}, {"type": "FeatureCollection", "features": [], "metadata": {}})
    assert out["features"] == [] and out["metadata"]["dropped_stale_previous"] == 1


def test_all_sources_failing_raises(monkeypatch, tmp_path):
    import requests
    monkeypatch.setattr(firedetections, "GIS_DIR", tmp_path)
    monkeypatch.setenv("MODIS_KEY", "secret")
    def boom(*a, **k):
        raise requests.exceptions.HTTPError("400 Client Error: Invalid MAP_KEY")
    monkeypatch.setattr(firedetections.requests, "get", boom)
    with pytest.raises(RuntimeError, match="All 3 FIRMS requests failed"):
        firedetections.main()
