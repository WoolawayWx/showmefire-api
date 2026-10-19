"""Admin controls for the hourly HRRR product: which cycles are built, the pause switch, the build history."""
import json
from datetime import datetime, timezone

import pytest

from services.hrrr_hourly import controls, enabled_cycle_hours, job
# The fixtures and helpers of the hourly product's own tests: a tmp data dir with R2 blocked, a fake build, a router client.
from tests.test_hrrr_hourly import _admin_client, at, auto_run, root, write  # noqa: F401


def test_settings_default_save_and_validation(root):
    assert controls.load_settings(root) == {"cycle_hours": None, "paused": False, "updated_at": None, "updated_by": None}
    saved = controls.save_settings([12, 0, 6, 6, 18], True, "admin@example.org", root)
    assert saved["cycle_hours"] == [0, 6, 12, 18] and saved["paused"] is True and saved["updated_by"] == "admin@example.org"
    assert controls.load_settings(root)["cycle_hours"] == [0, 6, 12, 18]
    assert controls.save_settings(None, False, "admin@example.org", root)["cycle_hours"] is None      # clears it
    for bad in ([], [24], [-1], [3.5], ["x"], [True]):
        with pytest.raises(ValueError):
            controls.save_settings(bad, False, "admin@example.org", root)


def test_a_damaged_settings_file_means_defaults_not_a_crash(root):
    root.mkdir(parents=True)
    (root / "settings.json").write_text("{not json")
    assert controls.load_settings(root)["cycle_hours"] is None
    (root / "settings.json").write_text(json.dumps({"cycle_hours": [3, 99, "x"], "paused": "yes"}))
    loaded = controls.load_settings(root)
    assert loaded["cycle_hours"] == [3] and loaded["paused"] is True


def test_saved_hours_beat_the_environment_which_beats_the_default(root, monkeypatch):
    monkeypatch.delenv("SMF_HRRR_HOURLY_CYCLE_HOURS", raising=False)
    assert enabled_cycle_hours() == tuple(range(24)) and controls.cycle_hours_source(root) == "default"
    monkeypatch.setenv("SMF_HRRR_HOURLY_CYCLE_HOURS", "0,12")
    assert enabled_cycle_hours() == (0, 12) and controls.cycle_hours_source(root) == "env"
    controls.save_settings([6, 18], False, "a", root)
    assert enabled_cycle_hours() == (6, 18) and controls.cycle_hours_source(root) == "settings"
    controls.save_settings(None, False, "a", root)                               # clearing falls back to the environment
    assert enabled_cycle_hours() == (0, 12)


def test_saved_hours_change_which_cycles_are_built_without_a_restart(root, monkeypatch):
    monkeypatch.delenv("SMF_HRRR_HOURLY_CYCLE_HOURS", raising=False)
    now = datetime(2026, 10, 16, 16, 0, tzinfo=timezone.utc)
    everything = {at(h) for h in range(0, 16)}
    controls.save_settings([12], False, "a", root)
    result, built = auto_run(root, now, ready=everything)
    assert built == [at(12)]
    controls.save_settings(None, False, "a", root)                               # back to all hours, no restart
    result, built = auto_run(root, now, ready=everything)
    assert built == [at(14)]                                                     # the newest ready cycle is eligible again


def test_pause_stops_new_builds_and_resume_clears_the_paused_status(root):
    now = datetime(2026, 10, 16, 14, 30, tzinfo=timezone.utc)
    ready = {at(12), at(11)}
    write(root, cycle=at(10))                                                    # something is being served
    controls.save_settings(None, True, "admin@example.org", root)
    result, built = auto_run(root, now, ready)
    assert built == [] and result["reason"] == "paused" and result["action"] == "skip"
    status = job.read_status(root)
    assert status["state"] == "paused" and status["updating"] is False and status["pending"] is False
    first = status["updated_at"]
    auto_run(root, now, ready)
    assert job.read_status(root)["updated_at"] == first                          # not rewritten on every tick
    controls.save_settings(None, False, "admin@example.org", root)
    result, built = auto_run(root, now, ready)
    assert built == [at(12)] and job.read_status(root)["state"] == "complete"


def test_resuming_with_nothing_to_build_does_not_leave_the_status_paused(root):
    now = datetime(2026, 10, 16, 14, 30, tzinfo=timezone.utc)
    write(root, cycle=at(12))
    controls.save_settings(None, True, "a", root)
    auto_run(root, now, ready=set())
    assert job.read_status(root)["state"] == "paused"
    controls.save_settings(None, False, "a", root)
    auto_run(root, now, ready=set())
    assert job.read_status(root)["state"] != "paused"


def test_pause_does_not_stop_an_explicit_manual_build(root):
    controls.save_settings(None, True, "a", root)
    built = []

    def build(cycle_time, hours, root, on_progress=None, wait=None):
        built.append(cycle_time)
        return write(root, cycle=cycle_time, hours=hours, expected=hours)

    result = job.run_auto(now=datetime(2026, 10, 17, 14, tzinfo=timezone.utc), root=root, cycle_time=at(12),
                          hours=(0, 1, 2), force=True, build=build, ready=lambda c, h: True,
                          forecast_done=lambda c: True, lock_free=lambda: True)
    assert result["action"] == "built" and built == [at(12)]


def test_builds_are_recorded_in_the_history_newest_first(root):
    now = datetime(2026, 10, 16, 14, 30, tzinfo=timezone.utc)
    auto_run(root, now, ready={at(12), at(11)})
    auto_run(root, now, ready={at(12), at(11)})
    events = controls.recent_events(10, root)
    assert [e["cycle"] for e in events] == ["20261016_11z", "20261016_12z"]
    assert events[0]["outcome"] == "complete" and events[0]["hours"] == 3 and "time" in events[0]

    def boom(cycle_time, hours, root, on_progress=None, wait=None):
        raise RuntimeError("download failed badly")

    with pytest.raises(RuntimeError):
        job.run_auto(now=now, root=root, cycle_time=at(10), hours=(0, 1, 2), ready=lambda c, h: True,
                     forecast_done=lambda c: True, lock_free=lambda: True, build=boom)
    failed = controls.recent_events(1, root)[0]
    assert failed["cycle"] == "20261016_10z" and failed["outcome"] == "failed" and "download failed" in failed["error"]


def test_history_is_trimmed_and_survives_garbage_lines(root):
    for index in range(controls.HISTORY_TRIM_AT + 5):
        controls.record_event({"cycle": f"c{index}", "outcome": "complete"}, root)
    lines = (root / "history.jsonl").read_text().splitlines()
    assert len(lines) <= controls.HISTORY_TRIM_AT and json.loads(lines[-1])["cycle"] == f"c{controls.HISTORY_TRIM_AT + 4}"
    with (root / "history.jsonl").open("a") as handle:
        handle.write("not json\n")
    assert len(controls.recent_events(5, root)) >= 1                              # the bad line is skipped


def test_admin_settings_routes(root, monkeypatch):
    monkeypatch.delenv("SMF_HRRR_HOURLY_CYCLE_HOURS", raising=False)
    client = _admin_client()
    body = client.get("/admin/hrrr-hourly/settings").json()
    assert body["settings"]["paused"] is False and body["source"] == "default"
    assert body["effective_cycle_hours"] == list(range(24)) and "status" in body and body["keep_days"] == 7
    saved = client.put("/admin/hrrr-hourly/settings", json={"cycle_hours": [12, 0], "paused": True}).json()
    assert saved["settings"]["cycle_hours"] == [0, 12] and saved["settings"]["paused"] is True
    assert saved["settings"]["updated_by"] == "admin@example.org"
    assert saved["effective_cycle_hours"] == [0, 12] and saved["source"] == "settings"
    assert client.put("/admin/hrrr-hourly/settings", json={"cycle_hours": [], "paused": False}).status_code == 400
    assert client.put("/admin/hrrr-hourly/settings", json={"cycle_hours": [25], "paused": False}).status_code == 400
    assert client.put("/admin/hrrr-hourly/settings", json={"cycle_hours": "x"}).status_code == 422
    cleared = client.put("/admin/hrrr-hourly/settings", json={"cycle_hours": None, "paused": False}).json()
    assert cleared["settings"]["cycle_hours"] is None and cleared["source"] == "default"


def test_admin_history_and_coverage_routes(root, monkeypatch):
    monkeypatch.delenv("SMF_HRRR_HOURLY_CYCLE_HOURS", raising=False)
    controls.record_event({"cycle": "20261016_11z", "outcome": "complete"}, root)
    controls.save_settings([11, 12], False, "a", root)
    client = _admin_client()
    assert client.get("/admin/hrrr-hourly/history?limit=5").json()["events"][0]["cycle"] == "20261016_11z"
    coverage = client.get("/admin/hrrr-hourly/coverage?hours=24").json()
    assert len(coverage["cycles"]) == 24 and coverage["enabled_cycle_hours"] == [11, 12]
    assert all(c["enabled"] == (c["hour_utc"] in (11, 12)) for c in coverage["cycles"])
    assert all(c["built"] is False for c in coverage["cycles"])                   # nothing built in this tmp data dir
    assert client.get("/admin/hrrr-hourly/coverage?hours=2").status_code == 422


def test_coverage_shows_built_and_partial_cycles(root, monkeypatch):
    monkeypatch.delenv("SMF_HRRR_HOURLY_CYCLE_HOURS", raising=False)
    client = _admin_client()
    newest = client.get("/admin/hrrr-hourly/coverage?hours=6").json()["cycles"][0]
    when = datetime.strptime(newest["cycle"], "%Y%m%d_%Hz").replace(tzinfo=timezone.utc)
    write(root, cycle=when, hours=(0, 1, 2))                                      # expected 0-18, so it is partial
    row = client.get("/admin/hrrr-hourly/coverage?hours=6").json()["cycles"][0]
    assert row["cycle"] == newest["cycle"] and row["built"] is True and row["partial"] is True


def test_settings_routes_require_admin(root):
    anonymous = _admin_client(admin=False)
    for call in (lambda: anonymous.get("/admin/hrrr-hourly/settings"),
                 lambda: anonymous.put("/admin/hrrr-hourly/settings", json={"cycle_hours": None, "paused": False}),
                 lambda: anonymous.get("/admin/hrrr-hourly/history"),
                 lambda: anonymous.get("/admin/hrrr-hourly/coverage")):
        assert call().status_code == 401
