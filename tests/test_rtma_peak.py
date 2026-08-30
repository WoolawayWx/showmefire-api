from datetime import date

from services.rtma_peak import _hours_for_local_date, PEAK_WINDOW_HOURS


def test_rtma_peak_hours_match_verification_window():
    hours = _hours_for_local_date(date(2026, 8, 29))
    assert len(hours) == PEAK_WINDOW_HOURS
    chicago = hours[0].astimezone(__import__("zoneinfo").ZoneInfo("America/Chicago"))
    assert chicago.hour == 10
    last = hours[-1].astimezone(__import__("zoneinfo").ZoneInfo("America/Chicago"))
    assert last.hour == 21
    assert last.date() == date(2026, 8, 29)
