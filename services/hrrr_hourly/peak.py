"""Peak fire danger product: for each local day an HRRR run covers, the worst of the day's 10 AM - 9 PM Central hours.

The same peak logic as the daily forecast (forecast/DailyForecast.py: the peak is the maximum over the afternoon window of
the hourly fire danger, with the minimum RH and fuel moisture and the maximum wind and temperature alongside), built from
the hourly product's own fields. So it is an HRRR-only ESTIMATE: it uses the hourly product's estimated fuel moisture (see
danger.py) rather than the RAWS-based one the official forecast starts from, and can differ from it.

A run is one product with one step per local day (`day` instead of `lead_hour`). It is filled in as the hourly product's hours
arrive: a day shows the peak so far until all its window hours are in (the manifest lists those days as `volatile_steps`).
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from services.hrrr_hourly import Element, cycle_id

PRODUCT = "hrrr_fire_danger_peak"
DATASET = "hrrr-fire-danger-peak"
SCHEMA_VERSION = 1
LOCAL_TZ = ZoneInfo("America/Chicago")
WINDOW_START_HOUR = 10            # local hours, both ends included: the daily forecast's "10:00-21:00 CT"
WINDOW_END_HOUR = 21
MIN_WINDOW_HOURS = 3              # a day the run reaches fewer window hours of than this is left out: its "peak" would mean little

# name, long name, units, scale, field of the hourly product it is taken from, how it is reduced over the window
_REDUCTIONS = (
    ("peak_fire_danger", "Peak fire danger (HRRR-only estimate, 0 Low .. 4 Extreme)", "category", 1, "fire_danger_est", "max"),
    ("min_rh_pct", "Lowest 2 m relative humidity", "%", 0.5, "rh_pct", "min"),
    ("min_fuel_moisture_pct", "Lowest estimated 1-hr fuel moisture", "%", 0.1, "fuel_moisture_pct", "min"),
    ("max_wind_mph", "Highest 10 m wind speed", "mph", 0.1, "wind_speed_mph", "max"),
    ("max_gust_mph", "Highest wind gust", "mph", 0.1, "wind_gust_mph", "max"),
    ("max_temp_f", "Highest 2 m temperature", "degF", 0.1, "temp_f", "max"),
    ("max_hdw", "Highest Hot-Dry-Windy index", "hPa m/s", 0.5, "hdw", "max"),
)
# The hourly fields a day's peak is computed from.
SOURCE_FIELDS = tuple(sorted({source for *_head, source, _how in _REDUCTIONS}))
PEAK_ELEMENTS: tuple[Element, ...] = (
    Element("peak_fire_danger", _REDUCTIONS[0][1], "category", 1, "maximum over the window of fire_danger_est", True,
            categorical=True),
    Element("peak_hour_local", "Hour of peak fire danger (Central time, 24 h)", "h CT", 1,
            "first hour of the window with the peak fire danger", True, cmap="viridis", vmin=10, vmax=21,
            ticks=tuple(range(10, 22, 2))),
    *(Element(name, long, units, scale, f"{how} over the window of {source}", True, image=False)
      for name, long, units, scale, source, how in _REDUCTIONS[1:]),
)
BY_NAME = {element.name: element for element in PEAK_ELEMENTS}


def root_dir() -> Path:
    return Path(os.environ.get("SMF_HRRR_PEAK_DIR", "data/hrrr_fire_danger_peak"))


@dataclass(frozen=True)
class Day:
    index: int
    date: str                       # local calendar date, YYYY-MM-DD
    window_leads: tuple[int, ...]   # forecast hours of the run that fall in this day's window
    window_local_hours: tuple[int, ...]

    @property
    def label(self) -> str:
        day = datetime.strptime(self.date, "%Y-%m-%d")
        return f"{day:%a %b} {day.day}"

    @property
    def window_end_utc(self) -> datetime:
        end = datetime.strptime(self.date, "%Y-%m-%d").replace(hour=WINDOW_END_HOUR, tzinfo=LOCAL_TZ)
        return end.astimezone(timezone.utc)


def plan(cycle_time: datetime, hours) -> list[Day]:
    """The local days the run's hours reach into, each with the hours that fall in its 10 AM - 9 PM window. A day the run
    reaches fewer than MIN_WINDOW_HOURS of the window hours of is left out."""
    by_date: dict[str, list[tuple[int, int]]] = {}
    for lead in hours:
        local = (cycle_time + timedelta(hours=lead)).astimezone(LOCAL_TZ)
        if WINDOW_START_HOUR <= local.hour <= WINDOW_END_HOUR:
            by_date.setdefault(local.strftime("%Y-%m-%d"), []).append((lead, local.hour))
    return [Day(index, date, tuple(lead for lead, _hour in rows), tuple(hour for _lead, hour in rows))
            for index, (date, rows) in enumerate(sorted((d, r) for d, r in by_date.items() if len(r) >= MIN_WINDOW_HOURS))]


def aggregate(day: Day, fields_by_lead: dict[int, dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """The day's peak elements from the hours of its window that are available (`fields_by_lead`). Empty if none are."""
    leads = [lead for lead in day.window_leads if lead in fields_by_lead]
    local = dict(zip(day.window_leads, day.window_local_hours))
    out: dict[str, np.ndarray] = {}
    if not leads:
        return out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)       # all-NaN cells at the grid's edge are expected
        for name, _long, _units, _scale, source, how in _REDUCTIONS:
            cube = [fields_by_lead[lead][source] for lead in leads if source in fields_by_lead[lead]]
            if cube:
                out[name] = (np.nanmax if how == "max" else np.nanmin)(np.stack(cube).astype("float64"), axis=0)
        danger = [(lead, fields_by_lead[lead]["fire_danger_est"]) for lead in leads
                  if "fire_danger_est" in fields_by_lead[lead]]
        if danger:
            stack = np.stack([values for _lead, values in danger]).astype("float64")
            first_max = np.where(np.isfinite(stack), stack, -1.0).argmax(axis=0)
            hours = np.array([local[lead] for lead, _values in danger], dtype="float64")
            out["peak_hour_local"] = np.where(np.isfinite(stack).any(axis=0), hours[first_max], np.nan)
    return {name: values.astype("float32") for name, values in out.items()}


def day_complete(day: Day, present_leads, finished: bool) -> bool:
    """Its data can no longer change: every window hour is in, or the run is over."""
    present = set(present_leads)
    return finished or all(lead in present for lead in day.window_leads)


def build_manifest(cycle_time: datetime, days: list[Day], present_leads, day_elements: dict[int, list[str]], grid: dict,
                   issued_at: datetime, updated_at: datetime, finished: bool, revision: int) -> dict:
    """`day_elements`: day index -> names of the elements written for it. Only days with data are listed as available."""
    cycle = cycle_id(cycle_time)
    present = set(present_leads)
    have = [day.index for day in days if day.index in day_elements]
    return {
        "product": PRODUCT, "schema_version": SCHEMA_VERSION, "model": "HRRR", "kind": "daily-peak", "estimate": True,
        "label": f"Peak fire danger (estimate) from the {cycle_time:%H}z HRRR run",
        "note": ("HRRR-only estimate of the afternoon (10 AM - 9 PM Central) peak: the Show Me Fire rule applied to "
                 "estimated fuel moisture and HRRR humidity and wind. It can differ from the official forecast."),
        "cycle": cycle, "cycle_time_utc": cycle_time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "issued_at_utc": issued_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "updated_at_utc": updated_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "forecast_hours": have,
        "missing_hours": [d.index for d in days if d.index not in have] if finished else [],
        "pending_hours": [] if finished else [d.index for d in days if d.index not in have],
        "hours_expected": len(days), "zarr_leads": [d.index for d in days], "complete": finished, "revision": revision,
        "valid_times_utc": [days[i].window_end_utc.strftime("%Y-%m-%dT%H:%M:%SZ") for i in have],
        "step_labels": [days[i].label for i in have],
        "days": [{"index": d.index, "date": d.date, "label": d.label, "window_total": len(d.window_leads),
                  "window_present": sum(1 for lead in d.window_leads if lead in present),
                  "complete": d.index in have and day_complete(d, present, finished)} for d in days],
        "volatile_steps": [d.index for d in days if d.index in have and not day_complete(d, present, finished)],
        "variables": {e.name: {"long_name": e.long_name, "units": e.units, "source": e.source, "derived": True,
                               "scale": e.scale,
                               "hours_present": [i for i in have if e.name in day_elements.get(i, [])]}
                      for e in PEAK_ELEMENTS if any(e.name in names for names in day_elements.values())},
        "grid": grid, "series": {}, "files": {"zarr": f"{cycle}/forecast.zarr"},
    }
