"""Download one HRRR forecast hour (profiles + extra surface fields) for the hourly product."""
from __future__ import annotations

import logging
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

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


def fetch_extras(run: datetime, fxx: int, workdir: Path, herbie_factory=None) -> dict[str, np.ndarray]:
    """Optional surface fields on the cropped grid. Fields that fail or are absent are left out."""
    if herbie_factory is None:
        from herbie import Herbie as herbie_factory
    naive = run.replace(tzinfo=None)
    searches = dict(EXTRA_SEARCHES)
    if fxx >= 1:
        searches["precip"] = f":APCP:surface:{fxx - 1}-{fxx} hour acc fcst"
    out = {}
    for key, search in searches.items():
        try:
            herbie = herbie_factory(naive, model="hrrr", product="sfc", fxx=fxx, save_dir=str(workdir),
                                    verbose=False, priority=["aws"])
            result = herbie.xarray(search, remove_grib=True)
            datasets = [crop(item) for item in (result if isinstance(result, list) else [result])]
            out[key] = _first_variable(datasets)
        except Exception as exc:
            logger.info("HRRR %s f%02d: %s unavailable (%s)", run.strftime("%Y%m%d%Hz"), fxx, key, exc)
    return out


def fetch_hour(run: datetime, fxx: int, herbie_factory=None):
    """(Profiles, extras) for forecast hour `fxx` of `run`. Temp files are always removed."""
    valid = run + timedelta(hours=fxx)
    workdir = Path(tempfile.mkdtemp(prefix="smf-hrrr-hourly-"))
    try:
        profiles = hp.fetch_profiles(valid, workdir=workdir, herbie_factory=herbie_factory, run=run, fxx=fxx)
        extras = fetch_extras(run, fxx, workdir, herbie_factory=herbie_factory)
        return profiles, extras
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
