from datetime import datetime, timezone

from routers.verification import _report_covers_closed_window


def test_report_generated_before_local_window_close_is_hidden():
    entry = {"date": "2026-09-04", "generated_at": "2026-09-04T22:30:00Z"}
    now = datetime(2026, 9, 5, 12, tzinfo=timezone.utc)
    assert not _report_covers_closed_window(entry, now=now)


def test_report_generated_after_local_window_close_is_visible():
    entry = {"date": "2026-09-04", "generated_at": "2026-09-05T04:30:00Z"}
    now = datetime(2026, 9, 5, 12, tzinfo=timezone.utc)
    assert _report_covers_closed_window(entry, now=now)


def test_report_is_hidden_while_observation_window_is_open():
    entry = {"date": "2026-09-04", "generated_at": "2026-09-05T04:30:00Z"}
    now = datetime(2026, 9, 4, 20, tzinfo=timezone.utc)
    assert not _report_covers_closed_window(entry, now=now)
