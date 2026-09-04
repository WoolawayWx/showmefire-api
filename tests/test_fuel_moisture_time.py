from datetime import datetime, timezone

from services.synoptic import resolve_fuel_moisture_target_time


def test_explicit_09z_observation_time_is_utc():
    result = resolve_fuel_moisture_target_time("2026-09-04", 9)
    assert result == datetime(2026, 9, 4, 9, tzinfo=timezone.utc)


def test_existing_morning_date_request_remains_central_time():
    result = resolve_fuel_moisture_target_time("2026-09-04")
    assert result == datetime(2026, 9, 4, 12, tzinfo=timezone.utc)
