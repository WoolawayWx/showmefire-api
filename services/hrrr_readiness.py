"""Readiness checks for a complete HRRR forecast input window."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable
from zoneinfo import ZoneInfo


REQUIRED_FIELDS = {
    "2 m temperature": ":TMP:2 m",
    "2 m relative humidity": ":RH:2 m",
    "10 m u wind": ":UGRD:10 m",
    "10 m v wind": ":VGRD:10 m",
    "precipitation": ":APCP:",
    "surface wind gust": ":GUST:surface:",
}


@dataclass(frozen=True)
class HRRRReadiness:
    ready: bool
    missing_files: tuple[int, ...] = ()
    missing_fields: tuple[str, ...] = ()
    error: str | None = None


def default_availability_lag_hours() -> int:
    """Hours after a cycle's start before it counts as "today's run".

    HRRR's sfc f15 is on AWS about 1.2 h after the cycle, so the 12z forecast sets
    FORECAST_AVAILABILITY_LAG_HOURS=1 and the readiness wait decides when it really starts. 2 is the old rule and
    stays the default, so every other flow (09z, ad hoc runs) is unchanged."""
    return int(os.getenv("FORECAST_AVAILABILITY_LAG_HOURS", "2"))


def cycle_run_time(now: datetime, cycle_hour: int, availability_lag_hours: int | None = None) -> datetime:
    """Return the latest cycle expected to exist, matching DailyForecast logic."""
    if availability_lag_hours is None:
        availability_lag_hours = default_availability_lag_hours()
    current = now.astimezone(timezone.utc)
    run = current.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
    if current.hour < cycle_hour + availability_lag_hours:
        run -= timedelta(days=1)
    return run.replace(tzinfo=None)


def not_before_utc(now: datetime, local_time: str | None, tz: str = "America/Chicago") -> datetime | None:
    """Today's `HH:MM` wall-clock time in `tz` as an aware UTC datetime, or None when no floor is set.

    The 12z forecast starts from RAWS fuel moisture observed around 7 AM Central (+-1 h), so it must not start
    before about 07:30 Central however early the HRRR files appear. In daylight time that is 12:30Z, before the
    files exist; in standard time it is 13:30Z, which is later than they are and so is the real limit."""
    if not local_time:
        return None
    hour, minute = (int(part) for part in local_time.split(":"))
    zone = ZoneInfo(tz)
    local_now = now.astimezone(zone)
    return local_now.replace(hour=hour, minute=minute, second=0, microsecond=0).astimezone(timezone.utc)


def check_hrrr_ready(
    run_time: datetime,
    forecast_hours: Iterable[int],
    fast_herbie_factory: Callable | None = None,
) -> HRRRReadiness:
    """Confirm every HRRR file and required field is published for the window."""
    hours = tuple(forecast_hours)
    try:
        if fast_herbie_factory is None:
            from herbie import FastHerbie

            fast_herbie_factory = FastHerbie

        collection = fast_herbie_factory(
            DATES=[run_time],
            fxx=list(hours),
            model="hrrr",
            product="sfc",
        )
        available = {int(item.fxx): item for item in collection.file_exists}
        missing_files = tuple(hour for hour in hours if hour not in available)
        missing_fields: list[str] = []

        for hour in hours:
            item = available.get(hour)
            if item is None:
                continue
            for label, search in REQUIRED_FIELDS.items():
                inventory = item.inventory(search)
                if inventory is None or inventory.empty:
                    missing_fields.append(f"f{hour:02d} {label}")

        return HRRRReadiness(
            ready=not missing_files and not missing_fields,
            missing_files=missing_files,
            missing_fields=tuple(missing_fields),
        )
    except Exception as error:
        return HRRRReadiness(ready=False, error=str(error))
