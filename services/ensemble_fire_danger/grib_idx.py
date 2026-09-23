"""
Source-agnostic GRIB2 byte-range fetcher for the ensemble fire danger
product (HRRR, NAM 3km nest, RRFS, HiResW ARW/FV3/ARW-mem2, and the
HREF/REFS "ensprod" mean/spread/probability files).

Contract-mirror discipline (see api/core/contract_mirrors.json, pair
"ensemble_fire_danger_grib_idx"): this module is byte-identical to
model-training/ensemble_fire_danger/grib_idx.py and imports only
stdlib/numpy/requests/xarray/eccodes - never anything repo-specific. The
caller passes its own crop function (api: core.domain.crop, training:
spatial.domain.crop - themselves a mirror pair) so the domain definition
still lives in exactly one place per repo.

Same byte-subsetting technique as model-training/spatial/rrfs_capture.py
(read the .idx sidecar, HTTP Range GET only the messages needed), with
three generalizations that real files forced (all confirmed live
2026-09-22):

- NAM nest and HiResW ARW-mem2 pack UGRD+VGRD into ONE GRIB message with
  two fields; their .idx lines are numbered "624.1"/"624.2" and share the
  same byte offset. rrfs_capture's int(parts[0]) would crash on those, and
  a naive per-line range would fetch the same bytes twice. Entries are
  therefore grouped by offset, fetched once, and the fields inside are
  read back in sub-message order with eccodes' multi-field support.
- APCP comes in different accumulation layouts per model: HRRR/RRFS/
  HiResW publish a 1-hour bucket ("11-12 hour acc") plus a run total,
  NAM nest uses 3-hour resetting buckets ("9-10", "9-11", "9-12"), and
  ensprod mean publishes 1/3/6-hour windows. hourly_precip() turns
  whatever windows exist into a per-hour interval.
- HiResW 00z 2.5km ARW/FV3 do NOT carry an instantaneous 2 m TMP at every
  lead (only RH plus the hourly TMAX/TMIN), the same limitation
  spatial/fv3hires_capture.py already documents for FV3. When TMP is
  absent, t2m is (TMAX+TMIN)/2 and the run is tagged t2m_is_tmax_tmin_proxy.

Decoding uses eccodes directly (in memory for ordinary messages; a unique
tempfile.mkstemp file only for multi-field ones) rather than cfgrib,
because cfgrib cannot build one hypercube out of several APCP windows or
mixed 2 m/10 m levels.
"""
from __future__ import annotations

import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import requests
import xarray as xr

HTTP_TIMEOUT_SECONDS = 60
HTTP_RETRIES = 3

# Default idx search patterns for deterministic members. Matched with
# re.search against ":VAR:LEVEL:FORECAST[:EXTRA...]" (idx fields 4 onward).
# The "\d+ hour fcst" anchor matters: it excludes "anl" records and every
# "N-M hour max/min/acc" statistical record from the instantaneous fields.
DEFAULT_SEARCHES: Dict[str, str] = {
    "t2m": r"^:TMP:2 m above ground:\d+ hour fcst:?$",
    "d2m": r"^:DPT:2 m above ground:\d+ hour fcst:?$",
    "r2": r"^:RH:2 m above ground:\d+ hour fcst:?$",
    "u10": r"^:UGRD:10 m above ground:\d+ hour fcst:?$",
    "v10": r"^:VGRD:10 m above ground:\d+ hour fcst:?$",
    "tmax": r"^:TMAX:2 m above ground:\d+-\d+ hour max fcst:?$",
    "tmin": r"^:TMIN:2 m above ground:\d+-\d+ hour min fcst:?$",
    "apcp": r"^:APCP:surface:\d+-\d+ hour acc fcst:?$",
}

# ensprod (HREF/REFS) searches, keyed by product kind. The idx descriptor
# carries the ensemble-statistic suffix ("wt ens mean" / "ens spread").
ENSPROD_SEARCHES: Dict[str, Dict[str, str]] = {
    "mean": {
        "t2m": r"^:TMP:2 m above ground:\d+ hour fcst:wt ens mean",
        "d2m": r"^:DPT:2 m above ground:\d+ hour fcst:wt ens mean",
        "wind10": r"^:WIND:10 m above ground:\d+ hour fcst:wt ens mean",
        "apcp": r"^:APCP:surface:\d+-\d+ hour acc fcst:wt ens mean",
    },
    "sprd": {
        "t2m": r"^:TMP:2 m above ground:\d+ hour fcst:ens spread",
        "d2m": r"^:DPT:2 m above ground:\d+ hour fcst:ens spread",
        "wind10": r"^:WIND:10 m above ground:\d+ hour fcst:ens spread",
    },
    "prob": {
        # Joint fire-weather probability - NCEP's own "RH and wind" joint
        # exceedance from the same members (">=9 <20" per the idx text).
        "jfwprb": r"^:JFWPRB:10 m above ground:\d+ hour fcst:prob",
        "pwind_10p3": r"^:WIND:10 m above ground:\d+ hour fcst:prob >10\.3:",
    },
}

_APCP_WINDOW = re.compile(r":APCP:surface:(\d+)-(\d+) hour acc fcst")


@dataclass(frozen=True)
class IdxEntry:
    number: str       # "624" or "624.1" - kept as text, sub-message numbers are real
    offset: int
    descriptor: str   # ":VAR:LEVEL:FORECAST[:EXTRA...]"


class SourceUnavailable(RuntimeError):
    """The requested cycle/lead is not (or no longer) published - not a bug."""


def parse_idx(idx_text: str) -> List[IdxEntry]:
    entries = []
    for line in idx_text.strip().splitlines():
        parts = line.strip().split(":")
        if len(parts) < 5:
            continue
        try:
            offset = int(parts[1])
        except ValueError:
            continue
        entries.append(IdxEntry(parts[0], offset, ":" + ":".join(parts[3:])))
    return entries


def byte_ranges(entries: Sequence[IdxEntry]) -> Dict[int, Tuple[int, Optional[int]]]:
    """Maps each DISTINCT message offset to (start, end-inclusive); end is None
    for the final message. Sub-messages sharing an offset share one range."""
    offsets = sorted({entry.offset for entry in entries})
    ranges = {}
    for i, offset in enumerate(offsets):
        ranges[offset] = (offset, offsets[i + 1] - 1 if i + 1 < len(offsets) else None)
    return ranges


def select(entries: Sequence[IdxEntry], searches: Dict[str, str]) -> Dict[str, List[IdxEntry]]:
    """key -> matching entries (a key may match several, e.g. APCP windows)."""
    compiled = {key: re.compile(pattern) for key, pattern in searches.items()}
    matches: Dict[str, List[IdxEntry]] = {key: [] for key in searches}
    for entry in entries:
        for key, pattern in compiled.items():
            if pattern.search(entry.descriptor):
                matches[key].append(entry)
    return matches


def _sub_index(entry: IdxEntry) -> int:
    return int(entry.number.split(".")[1]) - 1 if "." in entry.number else 0


def _get(session: requests.Session, url: str, headers: Optional[dict] = None, timeout: int = HTTP_TIMEOUT_SECONDS):
    last_error: Optional[Exception] = None
    for attempt in range(HTTP_RETRIES):
        try:
            response = session.get(url, headers=headers or {}, timeout=timeout)
            if response.status_code == 404:
                raise SourceUnavailable(f"404 Not Found: {url}")
            response.raise_for_status()
            return response
        except SourceUnavailable:
            raise
        except requests.exceptions.HTTPError as error:
            status = error.response.status_code if error.response is not None else None
            if status is not None and 400 <= status < 500 and status != 429:
                raise
            last_error = error
        except requests.exceptions.RequestException as error:
            last_error = error
        time.sleep(1.5 * (attempt + 1))
    raise last_error  # type: ignore[misc]


def fetch_idx(session: requests.Session, grib_url: str) -> List[IdxEntry]:
    entries = parse_idx(_get(session, grib_url + ".idx", timeout=30).text)
    if not entries:
        raise SourceUnavailable(f"empty or unparsable index: {grib_url}.idx")
    return entries


def idx_exists(session: requests.Session, grib_url: str) -> bool:
    try:
        response = session.head(grib_url + ".idx", timeout=20, allow_redirects=True)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


_PENDING_TEMP_FILES: List[Path] = []
# Every eccodes call is serialized: concurrent first-use from worker threads
# segfaulted ("grib_handle_create: Cannot create handle, no definitions
# found" - confirmed live 2026-09-22), and codes_grib_multi_support_on/off is
# process-global state anyway. Downloads stay parallel; decoding is cheap.
_ECCODES_LOCK = threading.Lock()


def _cleanup_temp_files() -> None:
    """eccodes' Python binding keeps a file it read open (Windows then
    refuses the delete with WinError 32) until the process exits -
    confirmed live 2026-09-22 even after codes_release + close + gc. So
    temp files are only used for multi-field messages, and deletion is
    retried opportunistically instead of failing the fetch."""
    for path in list(_PENDING_TEMP_FILES):
        try:
            path.unlink(missing_ok=True)
            _PENDING_TEMP_FILES.remove(path)
        except OSError:
            pass


def _field_from_gid(gid) -> dict:
    import eccodes

    nx = _first_int(gid, ("Nx", "Ni"))
    ny = _first_int(gid, ("Ny", "Nj"))
    values = np.asarray(eccodes.codes_get_values(gid), dtype="float64")
    if eccodes.codes_get(gid, "bitmapPresent"):
        missing = eccodes.codes_get(gid, "missingValue")
        values = np.where(values == missing, np.nan, values)
    lat = np.asarray(eccodes.codes_get_array(gid, "latitudes"), dtype="float64")
    lon = np.asarray(eccodes.codes_get_array(gid, "longitudes"), dtype="float64")
    return {
        "values": values.reshape(ny, nx),
        "lat": lat.reshape(ny, nx),
        "lon": np.where(lon > 180.0, lon - 360.0, lon).reshape(ny, nx),
    }


def _decode_block(data: bytes, tmp_dir: Path, multi_field: bool = False) -> List[dict]:
    """Every GRIB field in one message, in order, as {values, lat, lon}.

    Single-field messages decode straight from memory. A multi-field
    message (packed U/V - idx sub-numbers "N.1"/"N.2") needs eccodes'
    multi-field file iterator, so only those go through a temp file; field
    order is what lets IdxEntry sub-numbers map back."""
    import eccodes

    with _ECCODES_LOCK:
        if not multi_field:
            gid = eccodes.codes_new_from_message(data)
            try:
                return [_field_from_gid(gid)]
            finally:
                eccodes.codes_release(gid)
        return _decode_multi_field(data, tmp_dir)


def _sweep_stale_temp_files(tmp_dir: Path, max_age_seconds: float = 3600.0) -> None:
    """Files a PREVIOUS process could not delete (eccodes held them until exit)."""
    cutoff = time.time() - max_age_seconds
    for path in Path(tmp_dir).glob("ensfd.*.grib2"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            pass


def _decode_multi_field(data: bytes, tmp_dir: Path) -> List[dict]:
    import eccodes

    _cleanup_temp_files()
    _sweep_stale_temp_files(tmp_dir)
    fd, name = tempfile.mkstemp(prefix="ensfd.", suffix=".grib2", dir=tmp_dir)
    path = Path(name)
    fields: List[dict] = []
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
        eccodes.codes_grib_multi_support_on()
        try:
            with open(path, "rb") as stream:
                while True:
                    gid = eccodes.codes_grib_new_from_file(stream)
                    if gid is None:
                        break
                    try:
                        fields.append(_field_from_gid(gid))
                    finally:
                        eccodes.codes_release(gid)
        finally:
            eccodes.codes_grib_multi_support_off()
    finally:
        _PENDING_TEMP_FILES.append(path)
        _cleanup_temp_files()
    return fields


def _first_int(gid, keys: Iterable[str]) -> int:
    import eccodes

    for key in keys:
        try:
            value = eccodes.codes_get(gid, key)
            if value is not None and int(value) > 0:
                return int(value)
        except Exception:
            continue
    raise KeyError(f"none of {tuple(keys)} present in GRIB message")


def fetch_fields(session: requests.Session, grib_url: str, searches: Dict[str, str],
                 tmp_dir: Path) -> Tuple[Dict[str, List[Tuple[IdxEntry, np.ndarray]]], np.ndarray, np.ndarray]:
    """Fetch every message matching `searches` from one GRIB file.

    Returns ({key: [(entry, 2D values), ...]}, lat, lon). Contiguous
    ranges are merged into one GET; each distinct offset is decoded once."""
    entries = fetch_idx(session, grib_url)
    ranges = byte_ranges(entries)
    matches = select(entries, searches)
    wanted_offsets = sorted({entry.offset for found in matches.values() for entry in found})
    multi_offsets = {entry.offset for entry in entries if "." in entry.number}
    if not wanted_offsets:
        return {key: [] for key in searches}, np.empty((0, 0)), np.empty((0, 0))

    # Merge adjacent message ranges into contiguous blocks (fewer GETs).
    blocks: List[List[int]] = []
    for offset in wanted_offsets:
        if blocks:
            previous = blocks[-1][-1]
            prev_end = ranges[previous][1]
            if prev_end is not None and prev_end + 1 == offset:
                blocks[-1].append(offset)
                continue
        blocks.append([offset])

    fields_by_offset: Dict[int, List[dict]] = {}
    for block in blocks:
        start = ranges[block[0]][0]
        end = ranges[block[-1]][1]
        header = {"Range": f"bytes={start}-{end}" if end is not None else f"bytes={start}-"}
        data = _get(session, grib_url, headers=header).content
        # Split the block back into per-offset chunks so field order maps
        # to IdxEntry sub-numbers unambiguously.
        for offset in block:
            o_start, o_end = ranges[offset]
            chunk = data[o_start - start:(o_end - start + 1) if o_end is not None else None]
            fields_by_offset[offset] = _decode_block(chunk, tmp_dir, multi_field=offset in multi_offsets)

    result: Dict[str, List[Tuple[IdxEntry, np.ndarray]]] = {key: [] for key in searches}
    lat = lon = None
    for key, found in matches.items():
        for entry in found:
            fields = fields_by_offset.get(entry.offset) or []
            index = _sub_index(entry)
            if index >= len(fields):
                continue
            result[key].append((entry, fields[index]["values"]))
            if lat is None:
                lat, lon = fields[index]["lat"], fields[index]["lon"]
    return result, lat, lon


def apcp_windows(found: Sequence[Tuple[IdxEntry, np.ndarray]]) -> Dict[Tuple[int, int], np.ndarray]:
    windows = {}
    for entry, values in found:
        match = _APCP_WINDOW.search(entry.descriptor)
        if match:
            windows[(int(match.group(1)), int(match.group(2)))] = np.clip(values, 0.0, None)
    return windows


def hourly_precip(windows_by_lead: Dict[int, Dict[Tuple[int, int], np.ndarray]], leads: Sequence[int],
                  shape: Tuple[int, int]) -> Dict[int, np.ndarray]:
    """Per-hour precip (mm, interval ending at each lead) from whatever
    accumulation windows the model published. Preference order per lead f:
    (f-1, f) bucket -> difference of two windows sharing a start (a, f) -
    (a, f-1) -> (a, f) spread evenly over its hours -> zeros (logged by
    the caller through the missing-precip flag)."""
    out: Dict[int, np.ndarray] = {}
    for f in leads:
        windows = windows_by_lead.get(f, {})
        if (f - 1, f) in windows:
            out[f] = windows[(f - 1, f)]
            continue
        previous = windows_by_lead.get(f - 1, {})
        shared = sorted(a for (a, b) in windows if b == f and (a, f - 1) in previous)
        if shared:
            a = shared[-1]
            out[f] = np.clip(windows[(a, f)] - previous[(a, f - 1)], 0.0, None)
            continue
        candidates = sorted((a for (a, b) in windows if b == f), reverse=True)
        if candidates:
            a = candidates[0]
            out[f] = windows[(a, f)] / max(1, f - a)
            continue
        out[f] = np.full(shape, np.nan)
    return out


def magnus_rh(t_k: np.ndarray, td_k: np.ndarray) -> np.ndarray:
    """Same Magnus constants as model-training/spatial/rtma_capture.py::relative_humidity."""
    t_c, td_c = np.asarray(t_k) - 273.15, np.asarray(td_k) - 273.15
    return np.clip(100.0 * np.exp((17.625 * td_c) / (243.04 + td_c) - (17.625 * t_c) / (243.04 + t_c)), 0.0, 100.0)


def render_url(template: str, cycle: datetime, lead: int) -> str:
    return template.format(date=f"{cycle:%Y%m%d}", hh=f"{cycle:%H}", fxx=lead)


def fetch_member_run(source: dict, cycle: datetime, leads: Sequence[int], crop: Callable[[xr.Dataset], xr.Dataset],
                     tmp_dir: Path, session: Optional[requests.Session] = None) -> xr.Dataset:
    """One deterministic member run -> Dataset(step, y, x) with t2m [K],
    r2 [%], u10/v10 [m/s], tp1h [mm per hour], cropped by `crop`.

    Raises SourceUnavailable if any required lead is unpublished (a
    partially-published run is not a usable member)."""
    own_session = session is None
    session = session or requests.Session()
    searches = dict(DEFAULT_SEARCHES, **(source.get("searches") or {}))
    per_lead = {}
    apcp_by_lead: Dict[int, Dict[Tuple[int, int], np.ndarray]] = {}
    lat = lon = None
    proxy_used = False
    try:
        for lead in leads:
            url = render_url(source["url"], cycle, lead)
            found, lead_lat, lead_lon = fetch_fields(session, url, searches, tmp_dir)
            first = {key: (values[0][1] if values else None) for key, values in found.items()}
            t2m = first.get("t2m")
            if t2m is None and first.get("tmax") is not None and first.get("tmin") is not None:
                t2m = 0.5 * (first["tmax"] + first["tmin"])
                proxy_used = True
            r2 = first.get("r2")
            if r2 is None and t2m is not None and first.get("d2m") is not None:
                r2 = magnus_rh(t2m, first["d2m"])
            if t2m is None or r2 is None or first.get("u10") is None or first.get("v10") is None:
                missing = [k for k in ("t2m", "r2", "u10", "v10") if {"t2m": t2m, "r2": r2}.get(k, first.get(k)) is None]
                raise KeyError(f"{url}: required fields missing {missing}")
            per_lead[lead] = {"t2m": t2m, "r2": r2, "u10": first["u10"], "v10": first["v10"]}
            apcp_by_lead[lead] = apcp_windows(found.get("apcp", []))
            if lat is None:
                lat, lon = lead_lat, lead_lon
    finally:
        if own_session:
            session.close()

    shape = lat.shape
    precip = hourly_precip(apcp_by_lead, list(leads), shape)
    data_vars = {}
    for name in ("t2m", "r2", "u10", "v10"):
        data_vars[name] = (("step", "y", "x"), np.stack([per_lead[f][name] for f in leads]).astype("float32"))
    data_vars["tp1h"] = (("step", "y", "x"), np.stack([precip[f] for f in leads]).astype("float32"))
    ds = xr.Dataset(
        data_vars,
        coords={
            "step": ("step", np.asarray(leads, dtype="int32"), {"units": "forecast lead hours"}),
            "latitude": (("y", "x"), lat),
            "longitude": (("y", "x"), lon),
        },
    )
    ds = crop(ds)
    ds.attrs.update({
        "source": source.get("name", ""),
        "cycle_utc": cycle.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "leads": ",".join(str(int(f)) for f in leads),
        "t2m_is_tmax_tmin_proxy": int(proxy_used),
        "precip_missing": int(bool(np.isnan(ds["tp1h"].values).all())),
    })
    return ds


def fetch_ensprod_run(sources: Dict[str, dict], cycle: datetime, leads: Sequence[int],
                      crop: Callable[[xr.Dataset], xr.Dataset], tmp_dir: Path,
                      session: Optional[requests.Session] = None) -> xr.Dataset:
    """HREF/REFS ensemble-product run -> Dataset(step, y, x) with
    t2m_mean/t2m_sprd/d2m_mean/d2m_sprd [K], wind10_mean/wind10_sprd [m/s],
    tp1h_mean [mm/h] and, when published, jfwprb/pwind_10p3 [%].

    `sources` maps product kind ("mean", "sprd", optional "prob") to a
    source dict with a `url` template. mean+sprd are required per lead;
    prob is best-effort."""
    own_session = session is None
    session = session or requests.Session()
    arrays: Dict[str, Dict[int, np.ndarray]] = {}
    apcp_by_lead: Dict[int, Dict[Tuple[int, int], np.ndarray]] = {}
    lat = lon = None
    try:
        for lead in leads:
            for kind in ("mean", "sprd", "prob"):
                source = sources.get(kind)
                if source is None:
                    continue
                url = render_url(source["url"], cycle, lead)
                try:
                    found, lead_lat, lead_lon = fetch_fields(session, url, ENSPROD_SEARCHES[kind], tmp_dir)
                except SourceUnavailable:
                    if kind == "prob":
                        continue
                    raise
                for key, values in found.items():
                    if key == "apcp":
                        apcp_by_lead[lead] = apcp_windows(values)
                        continue
                    if values:
                        arrays.setdefault(f"{key}_{kind}" if kind != "prob" else key, {})[lead] = values[0][1]
                if lat is None and lead_lat is not None and lead_lat.size:
                    lat, lon = lead_lat, lead_lon
            for required in ("t2m_mean", "t2m_sprd", "d2m_mean", "d2m_sprd", "wind10_mean", "wind10_sprd"):
                if lead not in arrays.get(required, {}):
                    raise KeyError(f"ensprod lead f{lead:02d}: {required} not published")
    finally:
        if own_session:
            session.close()

    shape = lat.shape
    data_vars = {}
    for name, by_lead in arrays.items():
        if all(f in by_lead for f in leads):
            data_vars[name] = (("step", "y", "x"), np.stack([by_lead[f] for f in leads]).astype("float32"))
    precip = hourly_precip(apcp_by_lead, list(leads), shape)
    data_vars["tp1h_mean"] = (("step", "y", "x"), np.stack([precip[f] for f in leads]).astype("float32"))
    ds = xr.Dataset(
        data_vars,
        coords={
            "step": ("step", np.asarray(leads, dtype="int32"), {"units": "forecast lead hours"}),
            "latitude": (("y", "x"), lat),
            "longitude": (("y", "x"), lon),
        },
    )
    ds = crop(ds)
    ds.attrs.update({
        "cycle_utc": cycle.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "leads": ",".join(str(int(f)) for f in leads),
        "precip_missing": int(bool(np.isnan(ds["tp1h_mean"].values).all())),
    })
    return ds


def cycle_leads(anchor_cycle: datetime, cycle: datetime, valid_lead_start: int, valid_lead_end: int) -> List[int]:
    """Leads of `cycle` that verify at anchor+valid_lead_start..end."""
    offset = int((anchor_cycle - cycle) / timedelta(hours=1))
    return [offset + lead for lead in range(valid_lead_start, valid_lead_end + 1)]
