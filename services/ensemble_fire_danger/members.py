"""
Member registry, cycle resolution and cached per-member fetch for the
ensemble fire danger product.

Contract-mirror discipline (api/core/contract_mirrors.json, pair
"ensemble_fire_danger_members"): byte-identical to
model-training/ensemble_fire_danger/members.py. The same resolution rules
run live (api) and during historical capture/panel building (training),
so the training panel sees exactly the member mix the live product would
have seen for that date.

Resolution rule (per member, in config order): target cycle = anchor
cycle (12z of the forecast day) + cycle_offset_hours, snapped DOWN to the
nearest cycle that source actually runs. If that cycle is not published
yet (live: the .idx of the LAST lead the window needs is not there;
historical: the archive doesn't have it), step back one source cycle at a
time, up to max_cycle_age_hours before the anchor. A cycle already
claimed by another member of the same source is skipped, so a late 12z
NAM that falls back to 06z does not duplicate nam_m6 - nam_m6 then
resolves to 00z. Leads must fit inside the source's max_lead for that
cycle hour. Every decision is recorded (resolved cycle, reason) in the
returned plan, which ends up in the run's evidence file.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import requests
import xarray as xr

from . import grib_idx

CONFIG_PATH = Path(__file__).with_name("member_config.json")

# Verbatim port of api/forecast/DailyForecast.py's second, tighter crop
# (the `mo_bounds` block right after the HRRR download) - the grid every
# public map and county_cells.json is on. Kept identical to
# model-training/scripts/build_county_cells.py::API_MO_BOUNDS.
API_MO_BOUNDS = {"lat": (35.8, 40.8), "lon": (-95.8, -89.1)}


def load_config(path: Optional[Path] = None) -> dict:
    return json.loads(Path(path or CONFIG_PATH).read_text(encoding="utf-8"))


def api_bounds_slices(lat: np.ndarray, lon: np.ndarray) -> Tuple[slice, slice]:
    """Row/col slices of DailyForecast's mo_bounds bounding-box-of-mask crop."""
    lon = np.where(lon > 180, lon - 360, lon)
    mask = ((lat >= API_MO_BOUNDS["lat"][0]) & (lat <= API_MO_BOUNDS["lat"][1])
            & (lon >= API_MO_BOUNDS["lon"][0]) & (lon <= API_MO_BOUNDS["lon"][1]))
    rows = np.where(np.any(mask, axis=1))[0]
    cols = np.where(np.any(mask, axis=0))[0]
    if not len(rows) or not len(cols):
        raise ValueError("grid does not intersect API_MO_BOUNDS")
    return slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)


@dataclass
class ResolvedMember:
    member_id: str
    source_key: str
    kind: str                      # "member" or "ensprod"
    weight: float
    cycle: Optional[datetime]
    leads: List[int] = field(default_factory=list)
    status: str = "pending"        # resolved | unavailable | fetched | failed
    reason: str = ""
    control: bool = False
    optional: bool = False

    def as_dict(self) -> dict:
        return {
            "member_id": self.member_id, "source": self.source_key, "kind": self.kind,
            "weight": self.weight, "cycle_utc": self.cycle.strftime("%Y-%m-%dT%H:%MZ") if self.cycle else None,
            "leads": self.leads, "status": self.status, "reason": self.reason,
            "control": self.control, "optional": self.optional,
        }


def _max_lead(source: dict, hour: int) -> int:
    table = source.get("max_lead") or {}
    return int(table.get(str(hour), table.get("default", 18)))


def _cycles_back(source: dict, target: datetime, oldest: datetime):
    hours = sorted(int(h) for h in source["cycles"])
    t = target.replace(minute=0, second=0, microsecond=0)
    while t >= oldest:
        if t.hour in hours:
            yield t
        t -= timedelta(hours=1)


def source_url(source: dict, kind: str = "member", use_archive: bool = False, product: str = "mean") -> str:
    """Live URL template, or the archive one (when the source has a separate
    archive, e.g. NAM nest: NOMADS live vs the ~12h-late AWS copy)."""
    entry = source["products"][product] if kind == "ensprod" else source
    return entry.get("archive_url", entry["url"]) if use_archive else entry["url"]


def resolve_plan(config: dict, anchor: datetime, is_available: Callable[[str], bool],
                 now: Optional[datetime] = None, kinds: Tuple[str, ...] = ("member", "ensprod"),
                 emulate_live_at: Optional[datetime] = None, use_archive: bool = False) -> List[ResolvedMember]:
    """Decide which cycle of each configured member/ensprod source to use.

    is_available(url) -> bool is injected (live: HEAD the .idx; tests:
    a stub). `now` bounds "future" cycles for live use; None = no bound.

    emulate_live_at (historical capture only): also treat a cycle as
    unpublished unless cycle + the source's typical `latency_hours` <=
    emulate_live_at. An archive has every cycle, but at the live run time
    (~14:45Z) a 12z HiResW, say, does not exist yet - without this the
    training panel would be built from a fresher member mix than the live
    product ever gets. Live runs leave it None: the real .idx check is
    authoritative there."""
    window = config["window"]
    start, end = int(window["valid_lead_start"]), int(window["valid_lead_end"])
    oldest = anchor - timedelta(hours=int(config.get("max_cycle_age_hours", 30)))
    plan: List[ResolvedMember] = []
    claimed: Dict[str, set] = {}
    groups = []
    if "member" in kinds:
        groups.append(("member", config.get("members", []), config.get("sources", {})))
    if "ensprod" in kinds:
        groups.append(("ensprod", config.get("ensprod_members", []), config.get("ensprod_sources", {})))
    for kind, members, sources in groups:
        for member in members:
            source = sources[member["source"]]
            resolved = ResolvedMember(member["id"], member["source"], kind, float(member.get("weight", 1.0)), None,
                                      control=bool(member.get("control")), optional=bool(member.get("optional")))
            target = anchor + timedelta(hours=int(member.get("cycle_offset_hours", 0)))
            archive_start = source.get("archive_start")
            reasons = []
            for cycle in _cycles_back(source, target, oldest):
                if now is not None and cycle > now:
                    continue
                latency = timedelta(hours=float(source.get("latency_hours", 0)))
                if emulate_live_at is not None and cycle + latency > emulate_live_at:
                    reasons.append(f"{cycle:%d/%HZ} not yet published at emulated run time")
                    continue
                if archive_start and cycle < datetime.fromisoformat(archive_start):
                    reasons.append(f"{cycle:%d/%HZ} predates archive")
                    break
                key = (kind, member["source"])
                if cycle in claimed.setdefault(str(key), set()):
                    continue
                leads = grib_idx.cycle_leads(anchor, cycle, start, end)
                if leads[0] < 1 or leads[-1] > _max_lead(source, cycle.hour):
                    reasons.append(f"{cycle:%d/%HZ} leads f{leads[0]}-f{leads[-1]} out of range")
                    continue
                url = grib_idx.render_url(source_url(source, kind, use_archive), cycle, leads[-1])
                if not is_available(url):
                    reasons.append(f"{cycle:%d/%HZ} not published")
                    continue
                claimed[str(key)].add(cycle)
                resolved.cycle, resolved.leads, resolved.status = cycle, leads, "resolved"
                resolved.reason = "; ".join(reasons) or "target cycle"
                break
            if resolved.cycle is None:
                resolved.status = "unavailable"
                resolved.reason = "; ".join(reasons) or "no cycle within max_cycle_age_hours"
            plan.append(resolved)
    return plan


def live_availability(session: requests.Session) -> Callable[[str], bool]:
    cache: Dict[str, bool] = {}

    def check(url: str) -> bool:
        if url not in cache:
            cache[url] = grib_idx.idx_exists(session, url)
        return cache[url]

    return check


def cache_path(cache_dir: Path, item: ResolvedMember) -> Path:
    return Path(cache_dir) / (f"{item.kind}_{item.member_id}_{item.source_key}_{item.cycle:%Y%m%d_%H}z"
                              f"_f{item.leads[0]:02d}-{item.leads[-1]:02d}.nc")


def fetch_resolved(config: dict, item: ResolvedMember, cache_dir: Path, crop: Callable[[xr.Dataset], xr.Dataset],
                   tmp_dir: Path, session: Optional[requests.Session] = None, force: bool = False,
                   use_archive: bool = False) -> Path:
    """Fetch (or reuse a cached copy of) one resolved member/ensprod run."""
    target = cache_path(cache_dir, item)
    if target.exists() and not force:
        try:
            with xr.open_dataset(target) as ds:
                if ds.sizes.get("step") == len(item.leads):
                    return target
        except Exception:
            pass
        target.unlink(missing_ok=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    if item.kind == "member":
        source = dict(config["sources"][item.source_key])
        source["url"] = source_url(source, "member", use_archive)
        ds = grib_idx.fetch_member_run(source, item.cycle, item.leads, crop, tmp_dir, session=session)
    else:
        source = config["ensprod_sources"][item.source_key]
        products = {kind: {"url": source_url(source, "ensprod", use_archive, kind)} for kind in source["products"]}
        ds = grib_idx.fetch_ensprod_run(products, item.cycle, item.leads, crop, tmp_dir, session=session)
    ds.attrs.update({"member_id": item.member_id, "source_key": item.source_key, "kind": item.kind})
    temporary = target.with_suffix(".nc.tmp")
    try:
        ds.to_netcdf(temporary, engine="netcdf4")
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target
