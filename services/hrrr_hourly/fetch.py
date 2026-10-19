"""Download one HRRR forecast hour (profiles + extra surface fields) for the hourly product."""
from __future__ import annotations

import logging
import shutil
import tempfile
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import requests

from core.domain import crop
from services.fire_weather import hrrr_profiles as hp

logger = logging.getLogger(__name__)

# One search per field so a field missing from a run drops only that element.
EXTRA_SEARCHES = {
    "cape": ":CAPE:surface:",
    "cin": ":CIN:surface:",
    "pwat": ":PWAT:entire atmosphere",
    "cloud": ":TCDC:entire atmosphere:",
    "gust": ":GUST:surface:",
    "rh2": ":RH:2 m above ground:",
    "smoke_surface": ":MASSDEN:8 m above ground:",
    "smoke_column": ":COLMD:entire atmosphere",
}


def _idx_url(run: datetime, product: str, fxx: int) -> str:
    return f"{hp.AWS}/hrrr.{run:%Y%m%d}/conus/hrrr.t{run:%H}z.wrf{product}f{fxx:02d}.grib2.idx"


def cycle_ready(run: datetime, last_hour: int, head: Callable[[str], int] | None = None) -> bool:
    """True when both the prs and sfc files for the last forecast hour are published."""
    head = head or (lambda url: requests.head(url, headers=hp.HEADERS, timeout=20).status_code)
    return all(head(_idx_url(run, product, last_hour)) == 200 for product in ("prs", "sfc"))


# HRRR's last forecast hour we use (f18) is published about 1h14m after the cycle (measured on AWS); a cycle is not worth
# checking before this.
EARLIEST_READY = timedelta(hours=1, minutes=5)


def cycle_candidates(now: Optional[datetime] = None, lookback: int = 3,
                     hours: Optional[Iterable[int]] = None) -> list[datetime]:
    """Cycles that could be fully published by `now`, newest first.

    The newest enabled cycle that is old enough to be ready, then `lookback` earlier enabled ones: if the builder falls
    behind, older cycles are skipped rather than built long after they stopped being current. `hours` (UTC) restricts
    which cycles count; by default every hour, or SMF_HRRR_HOURLY_CYCLE_HOURS (see enabled_cycle_hours)."""
    from services.hrrr_hourly import enabled_cycle_hours

    allowed = set(hours) if hours is not None else set(enabled_cycle_hours())
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    cycle = (now - EARLIEST_READY).replace(minute=0, second=0, microsecond=0)
    found: list[datetime] = []
    for _ in range(24 * 3):                       # never look back more than three days
        if cycle.hour in allowed:
            found.append(cycle)
            if len(found) > lookback:
                break
        cycle -= timedelta(hours=1)
    return found


def latest_cycle(now: Optional[datetime] = None, cycle_hour: int = 12) -> datetime:
    """Most recent run initialised at `cycle_hour` UTC that has started."""
    now = now or datetime.now(timezone.utc)
    cycle = now.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
    return cycle if now >= cycle else cycle - timedelta(days=1)


def _first_variable(datasets):
    for dataset in datasets:
        for name in dataset.data_vars:
            return np.asarray(dataset[name].values, dtype=float)
    raise KeyError("empty HRRR subset")


# Timings of the most recent fetch_hour() call, read (and cleared) by the build through take_timings().
_LAST_TIMINGS: dict = {}


def take_timings() -> dict:
    """Timings recorded by the last fetch_hour(): profiles_sec, extras_sec and per-field seconds. Empty if none."""
    timings = dict(_LAST_TIMINGS)
    _LAST_TIMINGS.clear()
    return timings


def extra_searches(fxx: int) -> dict[str, str]:
    searches = dict(EXTRA_SEARCHES)
    if fxx >= 1:
        searches["precip"] = f":APCP:surface:{fxx - 1}-{fxx} hour acc fcst"
    return searches


def fetch_extras(run: datetime, fxx: int, workdir: Path, herbie_factory=None,
                 field_seconds: Optional[dict] = None) -> dict[str, np.ndarray]:
    """Optional surface fields on the cropped grid. Fields that fail or are absent are left out.

    `field_seconds`, if given, receives the seconds spent on each field (including failed ones)."""
    if herbie_factory is None:
        from herbie import Herbie as herbie_factory
    naive = run.replace(tzinfo=None)
    searches = extra_searches(fxx)
    out = {}
    for key, search in searches.items():
        started = time.perf_counter()
        try:
            herbie = herbie_factory(naive, model="hrrr", product="sfc", fxx=fxx, save_dir=str(workdir),
                                    verbose=False, priority=["aws"])
            result = herbie.xarray(search, remove_grib=True)
            datasets = [crop(item) for item in (result if isinstance(result, list) else [result])]
            out[key] = _first_variable(datasets)
        except Exception as exc:
            logger.info("HRRR %s f%02d: %s unavailable (%s)", run.strftime("%Y%m%d%Hz"), fxx, key, exc)
        if field_seconds is not None:
            field_seconds[key] = round(time.perf_counter() - started, 2)
    return out


def fetch_hour(run: datetime, fxx: int, herbie_factory=None, workdir: Optional[Path] = None):
    """(Profiles, extras) for forecast hour `fxx` of `run`.

    With `workdir` (a Prefetcher's directory for this hour) the GRIB subsets already downloaded there are reused
    and the caller removes the directory; otherwise a temp directory is used and always removed."""
    valid = run + timedelta(hours=fxx)
    owned = workdir is None
    workdir = Path(workdir or tempfile.mkdtemp(prefix="smf-hrrr-hourly-"))
    try:
        started = time.perf_counter()
        profiles = hp.fetch_profiles(valid, workdir=workdir, herbie_factory=herbie_factory, run=run, fxx=fxx)
        profiles_sec = time.perf_counter() - started
        field_seconds: dict = {}
        started = time.perf_counter()
        extras = fetch_extras(run, fxx, workdir, herbie_factory=herbie_factory, field_seconds=field_seconds)
        _LAST_TIMINGS.update({"profiles_sec": round(profiles_sec, 2),
                              "extras_sec": round(time.perf_counter() - started, 2),
                              "extras_fields": field_seconds, "extras_missing": sorted(set(field_seconds) - set(extras))})
        return profiles, extras
    finally:
        if owned:
            shutil.rmtree(workdir, ignore_errors=True)


def download_hour(run: datetime, fxx: int, workdir: Path, herbie_factory=None, stop: Optional[threading.Event] = None) -> dict:
    """Download every GRIB subset one forecast hour needs into `workdir`, without decoding anything.

    Herbie names a subset file after its search, so fetch_hour() later finds these files and skips the
    download. A subset that fails here is not an error: fetch_hour() tries it again and reports it."""
    if herbie_factory is None:
        from herbie import Herbie as herbie_factory
    naive = run.replace(tzinfo=None)
    wanted = [("prs", hp.PRS_SEARCH)] + [("sfc", search) for search in (*hp.SFC_SEARCHES.values(),
                                                                      *extra_searches(fxx).values())]
    seconds, failed = {}, []
    for product, search in wanted:
        if stop is not None and stop.is_set():
            break
        started = time.perf_counter()
        try:
            herbie = herbie_factory(naive, model="hrrr", product=product, fxx=fxx, save_dir=str(workdir),
                                    verbose=False, priority=["aws"])
            herbie.download(search, verbose=False)
        except Exception as exc:
            failed.append(search)
            logger.info("HRRR %s f%02d prefetch of %s failed (%s)", run.strftime("%Y%m%d%Hz"), fxx, search, exc)
        seconds[product] = seconds.get(product, 0.0) + time.perf_counter() - started
    return {"download_sec": round(sum(seconds.values()), 2), "prs_sec": round(seconds.get("prs", 0.0), 2),
            "failed": failed}


class Prefetcher:
    """Downloads each hour's GRIB files on a background thread while the build decodes the previous hour.

    The link is the bottleneck (about 5 MB/s, ~60 MB per hour), and decoding is CPU work, so doing them one after
    the other leaves the link idle half the time. At most `lookahead` hours are downloaded ahead of the hour
    being processed, which bounds disk use (~60 MB each) and keeps the download from running on while the
    build is paused for a higher-priority job.

        prefetcher = Prefetcher(run, leads, work_dir)
        for lead in leads:
            workdir = prefetcher.wait_for(lead)      # blocks until that hour is on disk
            fetch_hour(run, lead, workdir=workdir)
            prefetcher.release(lead)                 # delete it and let the next hour start downloading
        prefetcher.close()
    """

    def __init__(self, run: datetime, leads, root: Path, lookahead: int = 1, herbie_factory=None):
        self.run, self.leads, self.root = run, list(leads), Path(root)
        self.herbie_factory = herbie_factory
        self.results: dict[int, dict] = {}
        self._ready = {lead: threading.Event() for lead in self.leads}
        self._slots = threading.Semaphore(lookahead + 1)  # hours on disk: the one in use plus the lookahead
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="hrrr-hourly-prefetch", daemon=True)
        self._thread.start()

    def _dir(self, lead: int) -> Path:
        return self.root / f"f{lead:02d}"

    def _run(self) -> None:
        for lead in self.leads:
            while not self._slots.acquire(timeout=1):
                if self._stop.is_set():
                    return
            if self._stop.is_set():
                return
            workdir = self._dir(lead)
            workdir.mkdir(parents=True, exist_ok=True)
            try:
                self.results[lead] = download_hour(self.run, lead, workdir, self.herbie_factory, self._stop)
            except Exception:
                logger.exception("hrrr_hourly prefetch of f%02d failed", lead)
                self.results[lead] = {"download_sec": 0.0, "prs_sec": 0.0, "failed": ["all"]}
            finally:
                self._ready[lead].set()

    def wait_for(self, lead: int) -> Path:
        """Block until `lead` has been downloaded (or failed), then return its directory."""
        self._ready[lead].wait()
        return self._dir(lead)

    def release(self, lead: int) -> None:
        shutil.rmtree(self._dir(lead), ignore_errors=True)
        self._slots.release()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=30)
        shutil.rmtree(self.root, ignore_errors=True)
