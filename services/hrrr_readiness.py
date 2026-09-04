"""Readiness checks for a complete HRRR forecast input window."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable


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


def cycle_run_time(now: datetime, cycle_hour: int, availability_lag_hours: int = 2) -> datetime:
    """Return the latest cycle expected to exist, matching DailyForecast logic."""
    current = now.astimezone(timezone.utc)
    run = current.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
    if current.hour < cycle_hour + availability_lag_hours:
        run -= timedelta(days=1)
    return run.replace(tzinfo=None)


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
