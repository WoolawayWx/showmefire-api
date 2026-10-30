"""
GEFS acquisition for the seven-day outlook: 31 members x leads 0..168 h (3-hourly), 0.25 deg `pgrb2sp25`.

Cost model (measured 2026-10-29 from the AWS .idx files and real decodes, see _plans/CLAUDE_PLAN.md):
  * Needed fields are ~3.6 MB per member-lead at 0.25 deg -> ~6.4 GB per cycle (57 leads x 31 members) fetched
    with HTTP byte ranges; the whole 0.25 deg surface file is ~20 MB, so never download whole files.
  * Decoding is cheap (~15 ms per message, ~2 min per cycle). The dominant cost is REQUEST LATENCY
    (~0.4-0.6 s per request x ~10.6k requests), so leads within a member are fetched concurrently
    (I/O bound; this does not need extra processing threads) while members are processed one at a time.
  * Each finished member is cropped to the domain and cached as one small .npz (~4 MB), so a killed
    run resumes at the first missing member instead of starting over.

Reuses the ensemble product's byte-range fetcher (services/ensemble_fire_danger/grib_idx.py), which merges
contiguous ranges and serializes eccodes safely. GEFS APCP is a 6-hour-reset bucket (f003 0-3, f006 0-6, f009 6-9,
f012 6-12 ...); 3-hour amounts are the difference within a bucket. Precipitation is allocated evenly to the three
hours of its interval (the interval is kept in the manifest: interpolation does not imply hourly skill).
Anything missing raises MemberIncomplete - never a zero-filled field.
"""
from __future__ import annotations

import logging
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import requests
import xarray as xr

from services.fire_outlook import moisture

logger = logging.getLogger(__name__)

BASE_URL = os.getenv("SMF_FIRE_OUTLOOK_GEFS_BASE", "https://noaa-gefs-pds.s3.amazonaws.com")
STEP_HOURS = 3
MAX_LEAD = 168
LEADS = tuple(range(0, MAX_LEAD + 1, STEP_HOURS))
MEMBERS = ("c00",) + tuple(f"p{i:02d}" for i in range(1, 31))
DOMAIN_MARGIN_DEG = 0.75
DEFAULT_WORKERS = 8  # measured: 6 -> 19.6 s/member, 12 -> 16.0 s, 24 -> 16.1 s (decode is serialized; no gain past ~8)

SEARCHES = {
    "t2m": r"^:TMP:2 m above ground:(?:anl|\d+ hour fcst)",
    "r2": r"^:RH:2 m above ground:(?:anl|\d+ hour fcst)",
    "u10": r"^:UGRD:10 m above ground:(?:anl|\d+ hour fcst)",
    "v10": r"^:VGRD:10 m above ground:(?:anl|\d+ hour fcst)",
    "swe": r"^:WEASD:surface:(?:anl|\d+ hour fcst)",
    "apcp": r"^:APCP:surface:\d+-\d+ hour acc fcst",
}
INSTANT = ("t2m", "r2", "u10", "v10", "swe")


class MemberIncomplete(RuntimeError):
    """A member could not be fully acquired (missing lead/field); the run must not publish."""


def grib_url(cycle: datetime, member: str, lead: int) -> str:
    c = cycle.astimezone(timezone.utc)
    return (f"{BASE_URL}/gefs.{c:%Y%m%d}/{c:%H}/atmos/pgrb2sp25/"
            f"ge{member}.t{c:%H}z.pgrb2s.0p25.f{lead:03d}")


@dataclass(frozen=True)
class Domain:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float

    @classmethod
    def from_grid(cls, lat: np.ndarray, lon: np.ndarray, margin: float = DOMAIN_MARGIN_DEG) -> "Domain":
        return cls(float(lat.min()) - margin, float(lat.max()) + margin, float(lon.min()) - margin, float(lon.max()) + margin)

    def crop_indices(self, lat2d: np.ndarray, lon2d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lat1, lon1 = lat2d[:, 0], lon2d[0, :]
        rows = np.where((lat1 >= self.lat_min) & (lat1 <= self.lat_max))[0]
        cols = np.where((lon1 >= self.lon_min) & (lon1 <= self.lon_max))[0]
        if rows.size < 2 or cols.size < 2:
            raise MemberIncomplete("domain does not overlap the GEFS grid")
        return rows, cols


def _session(workers: int) -> requests.Session:
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=workers, pool_maxsize=workers)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch_lead(session, cycle: datetime, member: str, lead: int, domain: Domain, tmp_dir: Path) -> dict:
    """One member-lead file -> cropped float32 fields. Raises MemberIncomplete when a needed field is absent."""
    from services.ensemble_fire_danger import grib_idx
    url = grib_url(cycle, member, lead)
    try:
        found, lat, lon = grib_idx.fetch_fields(session, url, SEARCHES, tmp_dir)
    except grib_idx.SourceUnavailable as error:
        raise MemberIncomplete(f"{member} f{lead:03d} not published: {error}") from error
    if lat.size == 0:
        raise MemberIncomplete(f"{member} f{lead:03d}: no matching fields")
    rows, cols = domain.crop_indices(lat, lon)
    out: dict = {"rows_lat": lat[rows, 0].astype("float64"), "cols_lon": lon[0, cols].astype("float64")}
    for key in INSTANT:
        if len(found[key]) != 1:
            raise MemberIncomplete(f"{member} f{lead:03d}: expected 1 {key} field, found {len(found[key])}")
        out[key] = found[key][0][1][np.ix_(rows, cols)].astype("float32")
    windows = grib_idx.apcp_windows(found["apcp"])
    if lead == 0:
        out["apcp"] = None
    else:
        ending = {w: a for w, a in windows.items() if w[1] == lead}
        if len(ending) != 1:
            raise MemberIncomplete(f"{member} f{lead:03d}: expected one APCP window ending at the lead, found {sorted(windows)}")
        (start, _), arr = next(iter(ending.items()))
        out["apcp"] = (start, arr[np.ix_(rows, cols)].astype("float32"))
    return out


def three_hour_precip(per_lead: Dict[int, Tuple[int, np.ndarray]]) -> np.ndarray:
    """{lead: (bucket_start, accumulated mm since bucket start)} -> (56, ny, nx) 3-hour amounts for leads 3..168.
    Within a 6 h bucket the later lead minus the earlier gives the second 3 h; a lead whose bucket starts at
    lead-3 is already a 3 h amount."""
    amounts = []
    for lead in LEADS[1:]:
        start, acc = per_lead[lead]
        if start == lead - STEP_HOURS:
            amount = acc
        else:
            prev_start, prev_acc = per_lead[lead - STEP_HOURS]
            if prev_start != start:
                raise MemberIncomplete(f"APCP buckets do not line up at f{lead:03d}: {start} vs {prev_start}")
            amount = acc - prev_acc
        amounts.append(np.clip(amount, 0.0, None))
    return np.stack(amounts)


def fetch_member(cycle: datetime, member: str, cache_dir: Path, domain: Domain, *,
                 workers: int = DEFAULT_WORKERS, fetch=fetch_lead) -> Path:
    """Ensure one member is cached (cropped to the domain); returns the .npz path. Atomic, resumable."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{member}.npz"
    if target.exists():
        try:
            with np.load(target) as z:
                if int(z["complete"]) == 1:
                    return target
        except Exception:  # noqa: BLE001 - corrupt/partial cache entry: refetch
            target.unlink(missing_ok=True)
    session = _session(workers)
    tmp_dir = Path(tempfile.gettempdir())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda lead: fetch(session, cycle, member, lead, domain, tmp_dir), LEADS))
    first = results[0]
    for r in results[1:]:
        if not (np.array_equal(r["rows_lat"], first["rows_lat"]) and np.array_equal(r["cols_lon"], first["cols_lon"])):
            raise MemberIncomplete(f"{member}: grid changed between leads")
    arrays = {key: np.stack([r[key] for r in results]) for key in INSTANT}
    arrays["apcp3h"] = three_hour_precip({lead: r["apcp"] for lead, r in zip(LEADS, results) if r["apcp"] is not None})
    tmp = target.with_name(target.name + ".tmp.npz")
    np.savez_compressed(tmp, lat=first["rows_lat"], lon=first["cols_lon"], complete=np.int8(1), **arrays)
    os.replace(tmp, target)
    return target


# --- regrid + hourly dataset --------------------------------------------------------------------------

def _fractional_index(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Fractional index of `values` along a monotonic 1-D axis (either direction)."""
    idx = np.arange(axis.size, dtype="float64")
    return np.interp(values, axis, idx) if axis[0] < axis[-1] else np.interp(values, axis[::-1], idx[::-1])


class Regridder:
    """Bilinear regrid of the cropped regular lat/lon GEFS grid onto the 3 km public grid (weights computed once)."""

    def __init__(self, lat1d: np.ndarray, lon1d: np.ndarray, target_lat: np.ndarray, target_lon: np.ndarray):
        if (target_lat.min() < min(lat1d.min(), lat1d.max()) or target_lat.max() > max(lat1d.max(), lat1d.min()) or
                target_lon.min() < lon1d.min() or target_lon.max() > lon1d.max()):
            raise MemberIncomplete("cropped GEFS domain does not cover the target grid")
        self.coords = np.stack([_fractional_index(lat1d, target_lat), _fractional_index(lon1d, target_lon)])
        self.shape = target_lat.shape

    def __call__(self, field: np.ndarray) -> np.ndarray:
        from scipy.ndimage import map_coordinates
        return map_coordinates(field.astype("float64"), self.coords, order=1, mode="nearest").astype("float32")

    def stack(self, fields: np.ndarray) -> np.ndarray:
        return np.stack([self(f) for f in fields])


def member_dataset(cycle: datetime, npz_path: Path, regridder_for: "callable") -> xr.Dataset:
    """Cached member -> normalized hourly dataset (cycle+0..168 h) on the public grid, the driver's input contract."""
    with np.load(npz_path) as z:
        lat1d, lon1d = z["lat"], z["lon"]
        rg: Regridder = regridder_for(lat1d, lon1d)
        t = rg.stack(z["t2m"]) - 273.15
        rh = np.clip(rg.stack(z["r2"]), 0.0, 100.0)
        u, v = rg.stack(z["u10"]), rg.stack(z["v10"])
        swe = np.clip(rg.stack(z["swe"]), 0.0, None)
        apcp3 = rg.stack(z["apcp3h"])
    hours = MAX_LEAD + 1
    shape = (hours,) + rg.shape

    def hourly(a):
        return moisture.interpolate_hourly(a, STEP_HOURS).astype("float32")
    precip = np.zeros(shape, dtype="float32")
    for i, lead in enumerate(LEADS[1:]):
        precip[lead - STEP_HOURS + 1: lead + 1] = (apcp3[i] / STEP_HOURS)[None]
    times = (np.datetime64(cycle.astimezone(timezone.utc).replace(tzinfo=None), "h")
             + np.arange(hours).astype("timedelta64[h]"))
    dims = ("time", "y", "x")
    return xr.Dataset({
        "temperature_2m": (dims, hourly(t)), "relative_humidity_2m": (dims, hourly(rh)),
        "wind_u_10m": (dims, hourly(u)), "wind_v_10m": (dims, hourly(v)),
        "precipitation_increment": (dims, precip), "snow_water_equivalent": (dims, hourly(swe)),
    }, coords={"time": times},
        attrs={"source": "GEFS 0.25deg pgrb2sp25", "precip_source_interval_hours": STEP_HOURS,
               "instantaneous_source_step_hours": STEP_HOURS,
               "note": "hourly values are linear interpolation / even allocation of 3-hourly source data"})


# --- availability ---------------------------------------------------------------------------------------

def inputs_ready(cycle: datetime, *, session=None, members=MEMBERS, workers: int = DEFAULT_WORKERS) -> Tuple[bool, dict]:
    """All members' final-lead index published? One HEAD per member (cheap). Coverage names which are missing."""
    session = session or _session(workers)
    last = MAX_LEAD

    def head(member):
        try:
            r = session.head(grib_url(cycle, member, last) + ".idx", timeout=20, allow_redirects=True)
            ok = r.status_code == 200
            stamp = parsedate_to_datetime(r.headers["Last-Modified"]) if ok and r.headers.get("Last-Modified") else None
            return member, ok, stamp
        except requests.exceptions.RequestException:
            return member, False, None
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(head, members))
    ready = [m for m, ok, _ in rows if ok]
    stamps = [s for _, ok, s in rows if ok and s]
    coverage = {"members_expected": len(members), "members_ready": len(ready), "last_lead_hours": last,
                "missing_members": [m for m, ok, _ in rows if not ok],
                "available_utc": max(stamps).astimezone(timezone.utc) if stamps and len(ready) == len(members) else None,
                "resolution": "0.25deg", "product": "pgrb2sp25", "variables": list(SEARCHES)}
    return len(ready) == len(members), coverage


# --- worker wiring -------------------------------------------------------------------------------------

def build_adapters(config, *, raws_provider=None, fallback_ok: bool = False):
    """Real Adapters for services.fire_outlook.worker. raws_provider(cycle) -> RAWS payload dict (or None: fuel
    moisture is then conditioned but not RAWS-anchored). fallback_ok opts in to an EMC seed when RTMA history is short."""
    from services.fire_outlook import conditioning, point, rules, verification, worker

    grid = point.public_grid_dict()
    lat, lon = conditioning.public_lat_lon()
    domain = Domain.from_grid(lat, lon)
    regridders: dict = {}

    def regridder_for(lat1d, lon1d):
        key = (lat1d.shape, float(lat1d[0]), float(lon1d[0]))
        if key not in regridders:
            regridders[key] = Regridder(lat1d, lon1d, lat, lon)
        return regridders[key]

    def cycle_dir(cycle):
        return config.cache_dir / "gefs" / cycle.strftime("%Y%m%dT%HZ")

    def members(cycle):
        for member in MEMBERS:
            path = fetch_member(cycle, member, cycle_dir(cycle), domain)
            yield member_dataset(cycle, path, regridder_for)

    def default_raws(cycle):
        """RAWS 10-h fuel moisture at or before the cycle, from the observations table (causal; None if unreadable)."""
        from core.database import get_db_path
        try:
            return verification.SqliteObservations(get_db_path()).fuel_moisture_payload(cycle)
        except Exception as error:  # noqa: BLE001 - no anchor is a quality level, not a failure
            logger.warning("RAWS anchor unavailable for %s: %s", cycle, error)
            return None

    provider = raws_provider if raws_provider is not None else default_raws

    def condition(cycle):
        raws = provider(cycle)
        fallback = None
        if fallback_ok:
            first = fetch_member(cycle, MEMBERS[0], cycle_dir(cycle), domain)
            ds = member_dataset(cycle, first, regridder_for)
            fallback = (ds["temperature_2m"].values[0], ds["relative_humidity_2m"].values[0])
        return conditioning.condition_initial_state(cycle, lat, lon, raws, fallback_weather=fallback, raws_states=None)

    def stations():
        """RAWS stations that reported 10-h fuel moisture in the last 30 days (the only ones that can verify the rule)."""
        from datetime import timedelta
        from core.database import get_db_path
        return verification.SqliteObservations(get_db_path()).stations(datetime.now(timezone.utc) - timedelta(days=30))

    def reference(cycle, stations_):
        from services.fire_outlook import nbm
        return nbm.reference_for(cycle, stations_, config.cache_dir / "nbm")

    return worker.Adapters(
        inputs_ready=lambda cycle: inputs_ready(cycle), stations=stations, reference=reference, members=members, n_members=len(MEMBERS),
        conditioning=condition, rules=rules.rule_functions, grid=grid,
        source_cycles=lambda cycle: {"gefs": cycle.strftime("%Y-%m-%dT%HZ"), "members": len(MEMBERS),
                                     "resolution": "0.25deg", "product": "pgrb2sp25", "leads": f"0-{MAX_LEAD}h/{STEP_HOURS}h"})
