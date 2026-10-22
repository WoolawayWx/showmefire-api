""""What happened": one day of hourly RTMA analysis, in the data viewer's dataset format.

The hourly HRRR product (services/hrrr_hourly) shows what a model run expects. This shows what the analysis says actually
happened: one run per local (Central) day, one step per hour, built from the RTMA hours rtma_capture.py keeps in
cache/rtma (a 7-day operational cache). Each finished day is kept here and on the CDN for keep_days(), so the viewer can
go back further than the cache does; `build --fetch` back-fills any older day straight from NOAA's archive on AWS.

Element names are the hourly HRRR product's (temp_f, rh_pct, wind_speed_mph, ...), so the viewer's colors, groups and
wind symbols apply unchanged, and fuel moisture and fire danger are the same estimate it uses (3 + 0.25 x RH, then the
Show Me Fire rule), so the two can be compared hour for hour. RTMA has no gust, mixing height or other upper-air field.

    <root>/<YYYYMMDD>/forecast.zarr   (step, y, x) arrays, one chunk per step: same layout as live_store.py writes
    <root>/<YYYYMMDD>/manifest.json   kind "analysis-day"
    <root>/latest.json                the newest day with any hour in it (today, while it fills in)

Today is rebuilt every hour as RTMA arrives (complete: false, and served by the API only); a day is published to the CDN
once it is over. The API side is routers/rtma_analysis.py.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import shutil
from datetime import date as Date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np

from services.fire_weather import indices as ix
from services.hrrr_hourly import BY_NAME as HOURLY, live_store
from services.hrrr_hourly import danger
from services.hrrr_hourly.derive import ESTIMATED_FM_INTERCEPT, ESTIMATED_FM_RH_SLOPE
from services.raws_calibration import calibrate, fade_km

logger = logging.getLogger(__name__)

PRODUCT = "rtma_analysis"
DATASET = "rtma-analysis"
KIND = "analysis-day"
TITLE = "What happened (RTMA analysis)"
SCHEMA_VERSION = 1
LOCAL_TZ = ZoneInfo("America/Chicago")
RUN_FORMAT = "%Y%m%d"
STEP_DIM = "step"

_SOURCES = {
    "temp_f": "RTMA TMP 2 m", "dewpoint_f": "RTMA DPT 2 m", "rh_pct": "RTMA RH 2 m",
    "vpd_hpa": "Bolton from RTMA T and Td", "wind_speed_mph": "RTMA UGRD/VGRD 10 m",
    "wind_dir_deg": "RTMA UGRD/VGRD 10 m", "precip_mm_1h": "RTMA/URMA APCP, previous hour",
    "fuel_moisture_pct": "3.0 + 0.25 x RTMA RH (uncalibrated, as the hourly HRRR product)",
    "fire_danger_est": "Show Me Fire rule on estimated fuel moisture, RTMA RH and wind",
}
CALIBRATED_FM = "fuel_moisture_cal_pct"
# Stored at archive precision, coarser than the forecast's but still finer than RTMA's own accuracy: it halves the size
# of a day (19 -> ~9 MB measured on 2026-10-19), which matters when months of days are kept.
_SCALES = {"temp_f": 0.5, "dewpoint_f": 0.5, "vpd_hpa": 0.25, "wind_speed_mph": 0.5, "wind_dir_deg": 5,
           "fuel_moisture_pct": 0.25}
ELEMENTS = tuple(dataclasses.replace(HOURLY[name], source=source, scale=_SCALES.get(name, HOURLY[name].scale))
                 for name, source in _SOURCES.items()) + (
    dataclasses.replace(HOURLY["fuel_moisture_pct"], name=CALIBRATED_FM, scale=_SCALES["fuel_moisture_pct"],
                        long_name="Fuel moisture, calibrated to RAWS",
                        source="The estimate above, corrected toward the RAWS 10-hr fuel sticks reporting that hour, "
                               "fading out away from them (services/raws_calibration.py)"),
)
BY_NAME = {element.name: element for element in ELEMENTS}


def root_dir() -> Path:
    return Path(os.environ.get("SMF_RTMA_ANALYSIS_DIR", "data/rtma_analysis"))


def keep_days() -> int:
    """Days kept on this server and on the CDN, about 9 MB each (so ~1 GB by default). Stay under
    cdn_datasets.RUNS_LISTED (400), which also caps the CDN."""
    return int(os.environ.get("SMF_RTMA_ANALYSIS_KEEP_DAYS", "120"))


def run_id(day: Date) -> str:
    return day.strftime(RUN_FORMAT)


def local_today(now: Optional[datetime] = None) -> Date:
    return (now or datetime.now(timezone.utc)).astimezone(LOCAL_TZ).date()


def day_hours(day: Date) -> list[datetime]:
    """The UTC analysis hours that fall on `day` in Central time: 24, or 23/25 on the days the clocks change."""
    start = datetime(day.year, day.month, day.day, tzinfo=LOCAL_TZ).astimezone(timezone.utc)
    end = datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=LOCAL_TZ).astimezone(timezone.utc)
    return [start + timedelta(hours=i) for i in range(int((end - start).total_seconds() // 3600))]


def step_label(hour_utc: datetime, hours: list[datetime]) -> str:
    """'1 PM'; the repeated hour on the day the clocks go back gets its zone ('1 AM CDT', '1 AM CST')."""
    local = hour_utc.astimezone(LOCAL_TZ)
    text = f"{local.hour % 12 or 12} {'AM' if local.hour < 12 else 'PM'}"
    if sum(1 for other in hours if other.astimezone(LOCAL_TZ).hour == local.hour) > 1:
        text += f" {local.tzname()}"
    return text


# --- one hour ---------------------------------------------------------------------------------

def _f(temp_k):
    return (np.asarray(temp_k, dtype="float64") - 273.15) * 1.8 + 32.0


def _projection(dataset) -> tuple[float, float]:
    """Central longitude and standard parallel of the RTMA Lambert grid (its winds are grid-relative)."""
    attrs = dataset["u10"].attrs
    return float(attrs.get("GRIB_LoVInDegrees", 265.0)), float(attrs.get("GRIB_Latin1InDegrees", 25.0))


def derive_hour(dataset) -> dict[str, np.ndarray]:
    """Element name -> 2-D float32 array from one cached RTMA hour. Elements whose inputs are missing are left out."""
    t_c = np.asarray(dataset["t2m"].values, dtype="float64") - 273.15
    out: dict[str, np.ndarray] = {"temp_f": _f(dataset["t2m"].values)}
    td_c = None
    if "d2m" in dataset:
        td_c = np.asarray(dataset["d2m"].values, dtype="float64") - 273.15
        out["dewpoint_f"] = td_c * 1.8 + 32.0
        out["vpd_hpa"] = ix.vapor_pressure_deficit_hpa(t_c, td_c)
    if "r2" in dataset:
        rh = np.asarray(dataset["r2"].values, dtype="float64")
    elif td_c is not None:
        rh = 100.0 * ix.saturation_vapor_pressure_hpa(td_c) / ix.saturation_vapor_pressure_hpa(t_c)
    else:
        rh = None
    if rh is not None:
        out["rh_pct"] = np.clip(rh, 0.0, 100.0)
        out["fuel_moisture_pct"] = ESTIMATED_FM_INTERCEPT + ESTIMATED_FM_RH_SLOPE * out["rh_pct"]
    if "u10" in dataset and "v10" in dataset:
        central_longitude, standard_parallel = _projection(dataset)
        east, north = ix.earth_relative_winds(
            np.asarray(dataset["u10"].values, dtype="float64"), np.asarray(dataset["v10"].values, dtype="float64"),
            np.asarray(dataset["longitude"].values, dtype="float64"), central_longitude, standard_parallel)
        out["wind_speed_mph"] = ix.wind_speed(east, north) * ix.MS_TO_MPH
        out["wind_dir_deg"] = ix.wind_direction_from(east, north)
    if "apcp" in dataset and not dataset["apcp"].attrs.get("missing", False):
        out["precip_mm_1h"] = np.maximum(np.asarray(dataset["apcp"].values, dtype="float64"), 0.0)
    if "fuel_moisture_pct" in out and "wind_speed_mph" in out:
        out["fire_danger_est"] = danger.fire_danger_category(
            out["fuel_moisture_pct"], out["rh_pct"], out["wind_speed_mph"] * danger.MPH_TO_KT)
    return {name: np.asarray(values, dtype="float32") for name, values in out.items()}


def _open_hour(hour_utc: datetime, cache_dir: Optional[Path], fetch: bool):
    """The cached RTMA hour as an xarray Dataset, fetching it from NOAA first if asked. None if it is not available."""
    import xarray as xr

    from services import rtma_capture

    path = rtma_capture.cache_path_for_hour(hour_utc, cache_dir=cache_dir)
    if not path.is_file() and fetch:
        rtma_capture.ensure_analysis_hour_cached(hour_utc.replace(tzinfo=None), cache_dir=cache_dir)
    if not path.is_file():
        return None
    try:
        return xr.open_dataset(path).load()
    except Exception as error:  # noqa: BLE001 - an unreadable cache file is just a missing hour
        logger.warning("Unreadable RTMA cache file %s: %s", path, error)
        return None


# --- RAWS fuel moisture for calibration ----------------------------------------------------------

def _read_json(path: Path) -> Optional[dict]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _history_observations(hour_utc: datetime, coordinates: dict) -> list[dict]:
    """RAWS fuel moisture recorded by the 5-minute station-history job (kept 36 h) closest to the hour."""
    from maps.station_history import get_values_n_hours_ago_batch

    values = get_values_n_hours_ago_batch("fuel_moisture_pct", 0, now_utc=hour_utc)
    return [{"station": stid, "longitude": coordinates[stid][0], "latitude": coordinates[stid][1], "fuel_moisture": value}
            for stid, value in values.items() if stid in coordinates and value is not None]


def _station_coordinates() -> dict:
    from services.synoptic import get_station_data

    coordinates = {}
    for station in get_station_data().get("stations") or []:
        if station.get("stid") and station.get("longitude") is not None and station.get("latitude") is not None:
            coordinates[station["stid"]] = (float(station["longitude"]), float(station["latitude"]))
    return coordinates


def raws_observations(day: Date, hours: list[datetime], now: datetime) -> dict[int, list[dict]]:
    """Step index -> RAWS fuel-moisture readings for that hour (only hours with enough stations to calibrate).

    The whole-day observation archive (services/full_obs_archive.py, written at 01:00 for the day before) is used when
    it exists, else the 10:00-21:00 end-of-day RAWS archive; hours neither covers (today, so far) come from the
    station history.
    """
    from core.config import ARCHIVE_RAW_DATA_DIR
    from services import full_obs_archive
    from services.raws_calibration import MIN_STATIONS, synoptic_fuel_observations

    payload = (_read_json(full_obs_archive.output_path(day.isoformat()))
               or _read_json(Path(ARCHIVE_RAW_DATA_DIR) / f"raw_data_{day:%Y%m%d}.json"))
    coordinates = None
    found = {}
    for index, hour in enumerate(hours):
        if hour > now:
            continue
        readings = synoptic_fuel_observations(payload, hour) if payload else []
        if len(readings) < MIN_STATIONS and now - hour < timedelta(hours=36):
            try:
                if coordinates is None:
                    coordinates = _station_coordinates()
                readings = _history_observations(hour, coordinates)
            except Exception as error:  # noqa: BLE001 - calibration is optional; the estimate still gets written
                logger.warning("No station-history RAWS fuel moisture for %s: %s", hour, error)
                coordinates = {}
        if len(readings) >= MIN_STATIONS:
            found[index] = readings
    return found


# --- one day ----------------------------------------------------------------------------------

def build_manifest(day: Date, hours: list[datetime], present: dict[int, list[str]], grid: dict, complete: bool,
                   issued_at: datetime) -> dict:
    """`present`: step index -> names of the elements written for it."""
    have = sorted(present)
    later = [i for i in range(len(hours)) if i not in present and hours[i] > issued_at - timedelta(hours=2)]
    label_day = datetime(day.year, day.month, day.day)
    return {
        "product": PRODUCT, "schema_version": SCHEMA_VERSION, "model": "RTMA", "kind": KIND,
        "label": f"What happened on {label_day:%a %b} {day.day}",
        "note": ("RTMA analysis: what was observed, gridded, hour by hour. Fuel moisture and fire danger are the same "
                 "estimate the hourly HRRR view uses, so the two compare directly; the calibrated fuel moisture is "
                 "that estimate corrected toward the RAWS fuel sticks near them."),
        "cycle": run_id(day), "date": day.isoformat(),
        "cycle_time_utc": hours[0].strftime("%Y-%m-%dT%H:%M:%SZ"),
        "issued_at_utc": issued_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "forecast_hours": have,
        "missing_hours": [i for i in range(len(hours)) if i not in present] if complete else [],
        "pending_hours": [] if complete else later,
        "hours_expected": len(hours), "zarr_leads": list(range(len(hours))), "complete": complete,
        "valid_times_utc": [hours[i].strftime("%Y-%m-%dT%H:%M:%SZ") for i in have],
        "step_labels": [step_label(hours[i], hours) for i in have],
        "step_names": [step_label(hour, hours) for hour in hours],   # every step, for the not-yet-in hours' chips
        "variables": {e.name: {"long_name": e.long_name, "units": e.units, "source": e.source, "derived": e.derived,
                               "scale": e.scale, "hours_present": [i for i in have if e.name in present[i]]}
                      for e in ELEMENTS if any(e.name in names for names in present.values())},
        "grid": grid, "series": {}, "files": {"zarr": f"{run_id(day)}/forecast.zarr"},
    }


def build_day(day: Date, *, root: Optional[Path] = None, cache_dir: Optional[Path] = None, fetch: bool = False,
              now: Optional[datetime] = None) -> Optional[dict]:
    """Write the day's store and manifest from whatever RTMA hours exist and swap them in. Returns the manifest, or
    None when not one hour of the day is available. A day is complete three hours after it ends (RTMA for the last hour
    posts about an hour after it, the 15-minute capture job then picks it up); its missing hours then stay missing."""
    root = Path(root or root_dir())
    now = now or datetime.now(timezone.utc)
    hours = day_hours(day)
    run = run_id(day)
    staging = root / f".{run}.tmp"
    shutil.rmtree(staging, ignore_errors=True)
    store_dir = staging / "forecast.zarr"
    present: dict[int, list[str]] = {}
    calibration: dict[int, dict] = {}
    raws = raws_observations(day, hours, now)
    lat = None
    # Each hour is written as soon as it is derived, so only one hour's fields are in memory at a time.
    for index, hour in enumerate(hours):
        if hour > now:
            continue
        dataset = _open_hour(hour, cache_dir, fetch)
        if dataset is None:
            continue
        try:
            if lat is None:
                lat = np.asarray(dataset["latitude"].values, dtype="float32")
                lon = np.asarray(dataset["longitude"].values, dtype="float32")
                lon = np.where(lon > 180, lon - 360, lon).astype("float32")
                valid = np.array([int(h.timestamp() // 3600) for h in hours], dtype="int64")
                live_store.create(store_dir, ELEMENTS, len(hours), lat, lon, np.arange(len(hours)), valid,
                                  step_dim=STEP_DIM)
            elif dataset["t2m"].shape != lat.shape:
                logger.warning("RTMA %s has a different grid (%s); skipped", hour, dataset["t2m"].shape)
                continue
            values = derive_hour(dataset)
        finally:
            dataset.close()
        if "fuel_moisture_pct" in values and index in raws:
            calibrated, meta = calibrate(values["fuel_moisture_pct"], lon, lat, raws[index])
            if meta["applied"]:
                values[CALIBRATED_FM] = calibrated.astype("float32")
                calibration[index] = {"stations": meta["station_count"], "mean_residual": meta["mean_residual"]}
        for name, array in values.items():
            live_store.write_step(store_dir, name, index, array, BY_NAME[name].scale)
        present[index] = sorted(values)
    if not present:
        shutil.rmtree(staging, ignore_errors=True)
        return None

    complete = now >= hours[-1] + timedelta(hours=3)
    grid = {"ny": int(lat.shape[0]), "nx": int(lat.shape[1]), "source": "RTMA 2.5 km, Missouri and a buffer around it"}
    manifest = build_manifest(day, hours, present, grid, complete, now)
    if calibration:
        manifest["calibration"] = {"method": "RAWS 10-hr fuel moisture residuals, fading out away from stations",
                                   "fade_km": fade_km(), "hours": {str(i): calibration[i] for i in sorted(calibration)}}
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))

    final, retired = root / run, root / f".{run}.old"
    shutil.rmtree(retired, ignore_errors=True)
    if final.exists():
        os.replace(final, retired)
    os.replace(staging, final)
    shutil.rmtree(retired, ignore_errors=True)
    _point_latest(root)
    return manifest


def _point_latest(root: Path) -> None:
    from services.hrrr_hourly import store

    runs = store.list_cycles(root)
    if runs:
        newest = store.read_manifest(runs[0], root) or {}
        temporary = root / "latest.json.tmp"
        temporary.write_text(json.dumps({"cycle": runs[0], "cycle_time_utc": newest.get("cycle_time_utc"),
                                         "issued_at_utc": newest.get("issued_at_utc")}))
        os.replace(temporary, root / "latest.json")


def prune(root: Optional[Path] = None, now: Optional[datetime] = None, days: Optional[int] = None) -> list[str]:
    root = Path(root or root_dir())
    cutoff = local_today(now) - timedelta(days=days if days is not None else keep_days())
    removed = []
    for folder in root.iterdir() if root.is_dir() else ():
        try:
            when = datetime.strptime(folder.name, RUN_FORMAT).date()
        except ValueError:
            continue
        if folder.is_dir() and when < cutoff:
            shutil.rmtree(folder, ignore_errors=True)
            removed.append(folder.name)
    return removed


# --- CDN ----------------------------------------------------------------------------------------

def run_summary(manifest: dict) -> dict:
    return {"id": manifest["cycle"], "cycle": manifest["cycle"], "label": manifest.get("label"),
            "cycle_time_utc": manifest.get("cycle_time_utc"), "issued_at_utc": manifest.get("issued_at_utc"),
            "forecast_hours": manifest.get("forecast_hours"), "missing_hours": manifest.get("missing_hours"),
            "variables": len(manifest.get("variables", {})), "time_utc": manifest["cycle_time_utc"],
            "complete": manifest.get("complete", True), "pending_hours": manifest.get("pending_hours", []),
            "hours_expected": manifest.get("hours_expected")}


def try_publish(run: str, root: Optional[Path] = None, client=None) -> Optional[dict]:
    """Upload a finished day to the CDN (never raises). A day still filling in stays API-only: the CDN caches a run's
    files for a year, which only suits files that will not change."""
    from services import cdn_datasets
    from services.hrrr_hourly import store

    root = Path(root or root_dir())
    manifest = store.read_manifest(run, root)
    if not manifest or not manifest.get("complete"):
        return None
    result = cdn_datasets.try_publish_run(
        DATASET, run, root / run, run_summary(manifest), title=TITLE, kind=KIND, schema_version=SCHEMA_VERSION,
        include=["forecast.zarr", "manifest.json"], client=client, keep_days=keep_days())
    if result is not None:
        (root / run / ".published").write_text(manifest["issued_at_utc"])
    return result


def refresh(now: Optional[datetime] = None, root: Optional[Path] = None, cache_dir: Optional[Path] = None) -> dict:
    """The scheduler's hourly job: rebuild today from the hours that are in, finish and publish yesterday, prune."""
    root = Path(root or root_dir())
    now = now or datetime.now(timezone.utc)
    today = local_today(now)
    built = {}
    for day in (today - timedelta(days=1), today):
        folder = root / run_id(day)
        existing = (folder / "manifest.json").is_file() and json.loads((folder / "manifest.json").read_text())
        if existing and existing.get("complete"):
            continue   # a finished day only changes if someone rebuilds it on purpose
        manifest = build_day(day, root=root, cache_dir=cache_dir, now=now)
        if manifest:
            built[manifest["cycle"]] = f"{len(manifest['forecast_hours'])}/{manifest['hours_expected']} hours" + (
                "" if manifest["complete"] else " so far")
    published = []
    for day in (today - timedelta(days=2), today - timedelta(days=1)):
        folder = root / run_id(day)
        if (folder / "manifest.json").is_file() and not (folder / ".published").is_file():
            if try_publish(run_id(day), root):
                published.append(run_id(day))
    return {"built": built, "published": published, "pruned": prune(root, now)}


def _cli() -> None:
    from core.logging_setup import configure_logging

    configure_logging()
    parser = argparse.ArgumentParser(description="Build 'what happened' days of RTMA analysis for the data viewer.")
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("build", help="build one or more local (Central) days")
    one.add_argument("dates", nargs="+", help="YYYY-MM-DD")
    one.add_argument("--fetch", action="store_true", help="download hours missing from cache/rtma from NOAA first")
    one.add_argument("--publish", action="store_true", help="upload finished days to the CDN")
    sub.add_parser("refresh", help="what the scheduler runs every hour")
    args = parser.parse_args()
    if args.command == "refresh":
        print(refresh())
        return
    for text in args.dates:
        manifest = build_day(datetime.strptime(text, "%Y-%m-%d").date(), fetch=args.fetch)
        if manifest is None:
            print(f"{text}: no RTMA hours available")
            continue
        print(f"{text}: {len(manifest['forecast_hours'])}/{manifest['hours_expected']} hours, complete={manifest['complete']}")
        if args.publish:
            print("  published" if try_publish(manifest["cycle"]) else "  not published")


if __name__ == "__main__":
    _cli()
