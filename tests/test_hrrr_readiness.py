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
