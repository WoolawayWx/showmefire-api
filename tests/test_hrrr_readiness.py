from datetime import datetime, timezone

import pandas as pd

from services.hrrr_readiness import REQUIRED_FIELDS, check_hrrr_ready, cycle_run_time


class FakeHerbie:
    def __init__(self, fxx, missing_search=None):
        self.fxx = fxx
        self.missing_search = missing_search

    def inventory(self, search):
        return pd.DataFrame() if search == self.missing_search else pd.DataFrame([{"search": search}])


class FakeFastHerbie:
    def __init__(self, *, fxx, missing_hour=None, missing_search=None, **kwargs):
        self.file_exists = [
            FakeHerbie(hour, missing_search if hour == 6 else None)
            for hour in fxx
            if hour != missing_hour
        ]


def test_ready_only_when_every_frame_and_field_exists():
    result = check_hrrr_ready(
        datetime(2026, 9, 4, 9),
        range(5, 19),
        fast_herbie_factory=lambda **kwargs: FakeFastHerbie(**kwargs),
    )
    assert result.ready


def test_missing_frame_blocks_forecast():
    result = check_hrrr_ready(
        datetime(2026, 9, 4, 9),
        range(5, 19),
        fast_herbie_factory=lambda **kwargs: FakeFastHerbie(missing_hour=18, **kwargs),
    )
    assert not result.ready
    assert result.missing_files == (18,)


def test_missing_required_field_blocks_forecast():
    missing_search = REQUIRED_FIELDS["precipitation"]
    result = check_hrrr_ready(
        datetime(2026, 9, 4, 9),
        range(5, 19),
        fast_herbie_factory=lambda **kwargs: FakeFastHerbie(missing_search=missing_search, **kwargs),
    )
    assert not result.ready
    assert result.missing_fields == ("f06 precipitation",)


def test_after_12z_selects_same_day_09z_cycle():
    result = cycle_run_time(datetime(2026, 9, 4, 12, 5, tzinfo=timezone.utc), 9)
    assert result == datetime(2026, 9, 4, 9)


# --- 12z early start: the availability lag and the not-before floor ------------------------------

def test_default_lag_is_two_hours_so_other_flows_are_unchanged(monkeypatch):
    monkeypatch.delenv("FORECAST_AVAILABILITY_LAG_HOURS", raising=False)
    # 13:00Z is before 12z + 2 h, so it is still "yesterday's" run.
    assert cycle_run_time(datetime(2026, 10, 19, 13, 0, tzinfo=timezone.utc), 12) == datetime(2026, 10, 18, 12)
    assert cycle_run_time(datetime(2026, 10, 19, 14, 0, tzinfo=timezone.utc), 12) == datetime(2026, 10, 19, 12)


def test_lag_of_one_makes_todays_12z_the_run_from_13z(monkeypatch):
    monkeypatch.setenv("FORECAST_AVAILABILITY_LAG_HOURS", "1")
    assert cycle_run_time(datetime(2026, 10, 19, 12, 59, tzinfo=timezone.utc), 12) == datetime(2026, 10, 18, 12)
    assert cycle_run_time(datetime(2026, 10, 19, 13, 0, tzinfo=timezone.utc), 12) == datetime(2026, 10, 19, 12)
    # An explicit argument still wins over the environment.
    assert cycle_run_time(datetime(2026, 10, 19, 13, 0, tzinfo=timezone.utc), 12, availability_lag_hours=2) \
        == datetime(2026, 10, 18, 12)


def test_not_before_is_730_central_in_utc_for_both_seasons():
    from services.hrrr_readiness import not_before_utc

    summer = not_before_utc(datetime(2026, 7, 15, 13, 0, tzinfo=timezone.utc), "07:30")   # CDT, UTC-5
    winter = not_before_utc(datetime(2026, 12, 15, 13, 0, tzinfo=timezone.utc), "07:30")  # CST, UTC-6
    assert summer == datetime(2026, 7, 15, 12, 30, tzinfo=timezone.utc)
    assert winter == datetime(2026, 12, 15, 13, 30, tzinfo=timezone.utc)
    assert not_before_utc(datetime(2026, 7, 15, 13, 0, tzinfo=timezone.utc), None) is None
    assert not_before_utc(datetime(2026, 7, 15, 13, 0, tzinfo=timezone.utc), "") is None


def _load_wait_script():
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "scripts" / "wait_for_hrrr.py"
    spec = importlib.util.spec_from_file_location("wait_for_hrrr_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Clock:
    """A fake clock: datetime.now() and time.sleep() both read and advance it."""

    def __init__(self, start):
        self.now = start
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        from datetime import timedelta
        self.now += timedelta(seconds=seconds)

    def datetime_class(self):
        clock = self

        class FakeDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return clock.now
        return FakeDatetime


def _run_wait(monkeypatch, start, ready_from, env):
    from services.hrrr_readiness import HRRRReadiness

    wait = _load_wait_script()
    clock = _Clock(start)
    monkeypatch.setattr(wait, "datetime", clock.datetime_class())
    monkeypatch.setattr(wait.time, "sleep", clock.sleep)
    monkeypatch.setattr(wait.time, "monotonic", lambda: (clock.now - start).total_seconds())
    monkeypatch.setattr(wait, "check_hrrr_ready", lambda run_time, hours: HRRRReadiness(ready=clock.now >= ready_from))
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return wait.main(), clock


def test_wait_holds_until_the_local_floor_even_when_hrrr_is_ready(monkeypatch):
    # December (CST): the 07:30 Central floor is 13:30Z. HRRR is ready at 13:10Z but the wait must hold to the floor.
    start = datetime(2026, 12, 15, 13, 0, tzinfo=timezone.utc)
    code, clock = _run_wait(monkeypatch, start, ready_from=datetime(2026, 12, 15, 13, 10, tzinfo=timezone.utc), env={
        "FORECAST_CYCLE_HOUR": "12", "FORECAST_START_HOUR": "4", "FORECAST_END_HOUR": "15",
        "FORECAST_AVAILABILITY_LAG_HOURS": "1", "FORECAST_NOT_BEFORE_LOCAL": "07:30",
        "HRRR_READINESS_POLL_SECONDS": "300", "HRRR_READINESS_TIMEOUT_SECONDS": "5400"})
    assert code == 0
    assert clock.now == datetime(2026, 12, 15, 13, 30, tzinfo=timezone.utc)


def test_wait_returns_when_hrrr_is_ready_in_daylight_time(monkeypatch):
    # July (CDT): the floor is 12:30Z, long before HRRR f15 (13:10Z), so readiness decides.
    start = datetime(2026, 7, 15, 13, 0, tzinfo=timezone.utc)
    code, clock = _run_wait(monkeypatch, start, ready_from=datetime(2026, 7, 15, 13, 10, tzinfo=timezone.utc), env={
        "FORECAST_CYCLE_HOUR": "12", "FORECAST_START_HOUR": "4", "FORECAST_END_HOUR": "15",
        "FORECAST_AVAILABILITY_LAG_HOURS": "1", "FORECAST_NOT_BEFORE_LOCAL": "07:30",
        "HRRR_READINESS_POLL_SECONDS": "300", "HRRR_READINESS_TIMEOUT_SECONDS": "5400"})
    assert code == 0
    assert datetime(2026, 7, 15, 13, 10, tzinfo=timezone.utc) <= clock.now < datetime(2026, 7, 15, 13, 20, tzinfo=timezone.utc)


def test_wait_gives_up_after_the_timeout(monkeypatch):
    start = datetime(2026, 7, 15, 13, 0, tzinfo=timezone.utc)
    code, clock = _run_wait(monkeypatch, start, ready_from=datetime(2099, 1, 1, tzinfo=timezone.utc), env={
        "FORECAST_CYCLE_HOUR": "12", "FORECAST_START_HOUR": "4", "FORECAST_END_HOUR": "15",
        "FORECAST_AVAILABILITY_LAG_HOURS": "1", "HRRR_READINESS_POLL_SECONDS": "300",
        "HRRR_READINESS_TIMEOUT_SECONDS": "900"})
    assert code == 1
