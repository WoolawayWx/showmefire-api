"""
Live runner for the Day-1 ensemble fire danger product (BETA).

Called by scripts/run_ensemble_fire_danger.py (from scripts/forecasts.sh,
right after DailyForecast, as a NON-blocking step). One run:

  1. resolve which cycle of every configured member / ensprod source to
     use for today's 12z-anchored peak window (members.resolve_plan);
  2. fetch them (cached NetCDF per member run, byte-range GRIB subsets);
  3. build the target grid + operational FM anchor from DailyForecast's
     forecast-state hand-off (forecast_state.read), or from HRRR if absent;
  4. score both tracks (tracks.track_members / tracks.track_synthetic);
  5. turn each track's member peaks into calibrated neighborhood/point
     probabilities + categorical (core.probability_products);
  6. render 5 graphics per track - the PRIMARY track (SMF_ENSEMBLE_FD_PRIMARY,
     default "members") to images/ for the public CDN, both tracks to the
     beta Testbed for side-by-side comparison;
  7. write immutable evidence (JSON summary + compact .npz of raw member
     peaks/probabilities, which is what future live-data recalibration
     reads), status.json key ForecastEnsembleFireDanger, and the Testbed
     manifest entry; upload the primary images to R2.

Failure isolation: this module never touches any production array, file
or status key other than its own. Its runner is wired into forecasts.sh
with `|| echo WARNING`, so even an uncaught crash cannot block or alter
the public forecast.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import xarray as xr

from . import bundle as bundle_mod
from . import core, forecast_state, members, tracks
from .regrid import Regridder, cell_size_km

logger = logging.getLogger(__name__)

API_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
CACHE_DIR = Path(os.getenv("SMF_ENSEMBLE_FD_CACHE") or
                 ("/app/cache/ensemble" if Path("/app/cache").exists() else str(DATA_DIR / "cache" / "ensemble")))
EVIDENCE_ROOT = Path(os.getenv("SMF_ENSEMBLE_FD_EVIDENCE_ROOT") or (DATA_DIR / "model-shadow" / "ensemble-fire-danger"))
STATE_PATH = EVIDENCE_ROOT / "runner-state.json"
PUBLIC_IMAGES_DIR = Path(os.getenv("SMF_ENSEMBLE_FD_IMAGES_DIR") or (API_ROOT / "images"))
STATUS_KEY = "ForecastEnsembleFireDanger"
TRACKS = ("members", "ensprod")
TRACK_LABELS = {
    "members": "HRRR, NAM 3km, RRFS, HiResW (+ time-lagged runs)",
    "ensprod": "REFS/HREF ensemble mean + spread",
}
PRODUCT_FILES = {
    "categorical": "mo-forecast-ens-firedanger",
    1: "mo-forecast-ens-prob-moderate",
    2: "mo-forecast-ens-prob-elevated",
    3: "mo-forecast-ens-prob-critical",
    4: "mo-forecast-ens-prob-extreme",
}


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "y", "on"}


def default_anchor(now: Optional[datetime] = None, cycle_hour: int = 12) -> datetime:
    """Same rule as DailyForecast: use yesterday's cycle before cycle+2h."""
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).replace(tzinfo=None)
    anchor = now.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)
    if now < anchor + timedelta(hours=2):
        anchor -= timedelta(days=1)
    return anchor


# --- rule plumbing -----------------------------------------------------------

def rule_functions():
    """(categorize, dampen, info) from THIS repo's canonical rule definitions."""
    from core.fire_danger import (MAX_DEMOTION_FRACTION, RULE_SPEC, _green_factor,
                                  calculate_fire_danger)
    from services.rule_uncertainty import category_vectorized, check_parity

    thresholds = RULE_SPEC["thresholds"]
    check_parity(calculate_fire_danger, thresholds)  # refuses to run on vectorized/scalar drift
    try:
        from services.seasonal_fuel_state import current_gdd_accum
        gdd = current_gdd_accum()
    except Exception:
        gdd = None
    green = _green_factor(gdd)

    def categorize(fm, rh, wind):
        return category_vectorized(fm, rh, wind, thresholds)

    def dampen(category, fm, rh, wind):
        return core.seasonal_dampening_vectorized(category, fm, rh, wind, green, thresholds, MAX_DEMOTION_FRACTION)

    return categorize, dampen, {"gdd_accum_since_mar1": gdd, "green_factor": green,
                                "rule_spec_version": RULE_SPEC["version"]}


def load_fm_booster():
    import xgboost as xgb

    override = os.getenv("SMF_ENSEMBLE_FM_MODEL", "").strip()
    if override:
        path = Path(override)
    else:
        from models.versioning import load_active_model_path
        path = load_active_model_path("fuel_moisture", auto_rollback=False)
    booster = xgb.Booster()
    booster.load_model(str(path))
    if not booster.feature_names:
        from models.features import LEGACY_FEATURES
        booster.feature_names = list(LEGACY_FEATURES)
    return booster, str(path)


def county_cell_lists(grid_shape) -> Optional[Dict[str, list]]:
    try:
        from core.risk_fusion_county_reference import county_cells
        cells = county_cells()
        if list(cells["grid_shape"]) != list(grid_shape):
            logger.warning("county_cells grid %s != ensemble grid %s - county summaries skipped",
                           cells["grid_shape"], list(grid_shape))
            return None
        out: Dict[str, list] = {}
        for key, fips in cells["cell_to_fips"].items():
            row, col = (int(v) for v in key.split(","))
            out.setdefault(fips, []).append((row, col))
        return out
    except Exception as error:
        logger.warning("county_cells unavailable: %s", error)
        return None


# --- the run -----------------------------------------------------------------

def _fetch_all(config: dict, plan: List[members.ResolvedMember], workers: int) -> Dict[str, Path]:
    from core.domain import crop

    tmp_dir = CACHE_DIR / "tmp"
    paths: Dict[str, Path] = {}

    def one(item: members.ResolvedMember):
        started = time.time()
        try:
            # One Session per worker thread - requests.Session is not thread-safe.
            with requests.Session() as worker_session:
                path = members.fetch_resolved(config, item, CACHE_DIR / "runs", crop, tmp_dir,
                                              session=worker_session)
            item.status = "fetched"
            item.reason += f"; fetched in {time.time() - started:.0f}s"
            return item.member_id, path
        except Exception as error:
            item.status = "failed"
            item.reason += f"; fetch failed: {type(error).__name__}: {error}"
            logger.warning("ensemble member %s (%s %s) failed: %s", item.member_id, item.source_key,
                           item.cycle, error)
            return item.member_id, None

    todo = [item for item in plan if item.status == "resolved"]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for member_id, path in pool.map(one, todo):
            if path is not None:
                paths[member_id] = path
    return paths


def _target_grid(anchor: datetime, fetched: Dict[str, Path], plan: List[members.ResolvedMember]):
    state = forecast_state.read(anchor)
    if state is not None:
        return state["lat"], state["lon"], state
    for item in plan:
        if item.kind == "member" and item.source_key == "hrrr" and item.member_id in fetched:
            with xr.open_dataset(fetched[item.member_id]) as ds:
                lat, lon = ds["latitude"].values, ds["longitude"].values
            rows, cols = members.api_bounds_slices(lat, lon)
            return lat[rows, cols], lon[rows, cols], None
    raise RuntimeError("no target grid: neither a forecast-state hand-off nor any HRRR member is available")


def _regridder_factory(dst_lat, dst_lon):
    cache: Dict[tuple, Regridder] = {}

    def make(src_lat, src_lon) -> Regridder:
        key = (src_lat.shape, float(src_lat[0, 0]), float(src_lon[0, 0]), float(src_lat[-1, -1]))
        if key not in cache:
            cache[key] = Regridder(src_lat, src_lon, dst_lat, dst_lon, cache_dir=CACHE_DIR / "regrid")
        return cache[key]

    return make


def run(anchor: Optional[datetime] = None, *, now: Optional[datetime] = None, render: bool = True,
        upload: Optional[bool] = None, workers: int = 4) -> dict:
    started = time.time()
    anchor = anchor or default_anchor(now)
    bundle = bundle_mod.load_bundle()
    config = bundle.get("member_config") or members.load_config()
    window = config["window"]
    valid_times = [pd.Timestamp(anchor + timedelta(hours=lead), tz="UTC")
                   for lead in range(int(window["valid_lead_start"]), int(window["valid_lead_end"]) + 1)]
    primary = os.getenv("SMF_ENSEMBLE_FD_PRIMARY", "members").strip().lower()
    if primary not in TRACKS:
        primary = "members"
    run_id = f"{anchor:%Y%m%d_%H}z"
    logger.info("ensemble fire danger run %s (primary track %s, bundle %s)", run_id, primary, bundle["source"])

    with requests.Session() as session:
        live_now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).replace(tzinfo=None)
        plan = members.resolve_plan(config, anchor, members.live_availability(session), now=live_now)
    fetched = _fetch_all(config, plan, workers)

    dst_lat, dst_lon, state = _target_grid(anchor, fetched, plan)
    make_regridder = _regridder_factory(dst_lat, dst_lon)
    production_fm = None
    swe_grid = None
    snow_mask = None
    if state is not None:
        if state["hourly_fm"].shape[0] == len(valid_times):
            production_fm = state["hourly_fm"]
        swe_grid = state["swe_grid"]
        if swe_grid is not None and swe_grid.shape == dst_lat.shape:
            snow_mask = swe_grid * 0.03937 > state["snow_threshold_in"]
        else:
            swe_grid = None
    categorize, dampen, rule_info = rule_functions()
    booster, fm_model_path = load_fm_booster()
    feature_names = list(booster.feature_names)
    fm_clip = tuple(config.get("fm_clip", [1.0, 40.0]))
    cell_km = cell_size_km(dst_lat, dst_lon)
    radius_cells = float(config.get("neighborhood_radius_km", 20.0)) / max(cell_km, 0.1)
    sigma = float(config.get("smooth_sigma_cells", 1.5))

    results: Dict[str, dict] = {}
    # --- Track M
    member_fields, weights, wind_factors, control_id = {}, {}, {}, None
    for item in plan:
        if item.kind != "member" or item.member_id not in fetched:
            continue
        try:
            with xr.open_dataset(fetched[item.member_id]) as ds:
                ds = ds.load()
            regridder = make_regridder(ds["latitude"].values, ds["longitude"].values)
            member_fields[item.member_id] = tracks.regridded_member_fields(ds, regridder)
            weights[item.member_id] = item.weight
            wind_factors[item.member_id] = float(config["sources"][item.source_key].get("wind_factor", 0.8))
            if item.control:
                control_id = item.member_id
        except Exception as error:
            item.status, item.reason = "failed", item.reason + f"; regrid failed: {error}"
    n_members = len(member_fields)
    if n_members >= int(config.get("abort_below_members", 3)):
        track = tracks.track_members(member_fields, weights, wind_factors, control_id, booster, feature_names,
                                     valid_times, categorize, dampen, production_fm, snow_mask, swe_grid, fm_clip)
        track["degraded"] = n_members < int(config.get("min_members", 5))
        results["members"] = track
    else:
        logger.warning("Track M skipped: only %d usable members", n_members)

    # --- Track E
    ens_fields, ens_weights = {}, {}
    for item in plan:
        if item.kind != "ensprod" or item.member_id not in fetched:
            continue
        try:
            with xr.open_dataset(fetched[item.member_id]) as ds:
                ds = ds.load()
            regridder = make_regridder(ds["latitude"].values, ds["longitude"].values)
            ens_fields[item.member_id] = tracks.regridded_ensprod_fields(ds, regridder)
            ens_weights[item.member_id] = item.weight
        except Exception as error:
            item.status, item.reason = "failed", item.reason + f"; regrid failed: {error}"
    if ens_fields:
        track = tracks.track_synthetic(ens_fields, ens_weights, config.get("synthetic", {}), booster,
                                       feature_names, valid_times, categorize, dampen, production_fm, snow_mask,
                                       swe_grid, fm_clip)
        track["degraded"] = "refs" not in ens_fields
        results["ensprod"] = track
    else:
        logger.warning("Track E skipped: no ensprod source available")

    if not results:
        raise RuntimeError("no ensemble track could be computed (see plan in evidence)")
    if primary not in results:
        primary = next(iter(results))

    products = {}
    for name, track in results.items():
        products[name] = core.probability_products(
            track["peaks"], track["weights"], radius_cells=radius_cells, smooth_sigma=sigma,
            calibration=bundle_mod.track_calibration(bundle, name),
            categorical_thresholds=bundle_mod.track_thresholds(bundle, name),
            default_threshold=float(config.get("default_categorical_threshold", 0.5)))

    images = {"public": [], "testbed": []}
    if render:
        images = _render_all(anchor, results, products, primary, bundle, config, dst_lat, dst_lon, plan)

    summary = _write_evidence(run_id, anchor, plan, results, products, primary, bundle, rule_info, fm_model_path,
                              production_fm is not None, cell_km, radius_cells, dst_lat.shape, images,
                              time.time() - started)
    _update_status(anchor, images, summary, primary)
    _update_testbed_manifest(images, summary)
    upload = env_bool("uploadForecast", True) and env_bool("SMF_ENSEMBLE_FD_UPLOAD", True) if upload is None else upload
    if upload and images["public"]:
        _upload(images["public"])
    _prune_cache()
    _persist_state({"last_run": run_id, "last_success": datetime.now(timezone.utc).isoformat(),
                    "primary": primary, "tracks": list(results), "consecutive_failures": 0,
                    "last_error": None, "runtime_sec": round(time.time() - started, 1)})
    return summary


# --- outputs -----------------------------------------------------------------

SOURCE_SHORT = {"hrrr": "HRRR", "nam_nest": "NAM 3km", "rrfs": "RRFS", "hiresw_arw": "HiResW ARW",
                "hiresw_fv3": "HiResW FV3", "hiresw_arw2": "HiResW ARW2", "refs": "REFS", "href": "HREF"}


def _members_text(plan: List[members.ResolvedMember], track: str, result: dict) -> str:
    """Compact, wrapped member list, grouped by model: "HRRR 00/06/12Z, NAM 3km 06/12Z, ..."."""
    if track == "members":
        used = [i for i in plan if i.kind == "member" and i.member_id in result["member_ids"]]
        cycles: Dict[str, List[str]] = {}
        for item in used:
            cycles.setdefault(SOURCE_SHORT.get(item.source_key, item.source_key), []).append(f"{item.cycle:%H}")
        groups = [f"{name} {'/'.join(sorted(set(hours)))}Z" for name, hours in cycles.items()]
        text = f"{len(used)} members: " + ", ".join(groups)
    else:
        text = (f"{len(result['member_ids'])} synthetic members from "
                f"{' + '.join(SOURCE_SHORT.get(s, s) for s in result.get('sources', []))} mean/spread")
    wrapped, line = [], ""
    for word in text.split(" "):
        if line and len(line) + len(word) > 52:
            wrapped.append(line.rstrip())
            line = ""
        line += word + " "
    wrapped.append(line.rstrip())
    if result.get("degraded"):
        wrapped.append("DEGRADED: fewer members than configured minimum")
    return "\n".join(wrapped)


def _render_all(anchor, results, products, primary, bundle, config, lat, lon, plan) -> dict:
    from services.beta_products import BETA_IMAGES_DIR
    from .render import Renderer, descriptions

    renderer = Renderer(lat, lon)
    suffix = os.getenv("FORECAST_IMAGE_SUFFIX", "")
    valid_date = (anchor + timedelta(hours=4)).strftime("%Y-%m-%d")
    subtitle = f"Ensemble Run: {anchor:%Y-%m-%d %HZ} | Valid: {valid_date}"
    calibrated = (f"calibrated, model v{bundle['version']}" if bundle.get("calibrated")
                  else "uncalibrated raw member fraction")
    images = {"public": [], "testbed": []}
    for track, product in products.items():
        text = descriptions(window_label=config["window"]["label"],
                            members_text=_members_text(plan, track, results[track]),
                            calibrated_text=calibrated, track_label=TRACK_LABELS[track])
        rendered = []
        testbed_cat = BETA_IMAGES_DIR / f"{PRODUCT_FILES['categorical']}-{track}.png"
        rendered.append(renderer.categorical(product["categorical"], testbed_cat, subtitle=subtitle,
                                             description=text["categorical"], run_date=anchor))
        for k in core.CATEGORY_IDS:
            out = BETA_IMAGES_DIR / f"{PRODUCT_FILES[k]}-{track}.png"
            rendered.append(renderer.probability(k, product["neighborhood"][k - 1], out, subtitle=subtitle,
                                                 description=text["probability"], run_date=anchor))
        images["testbed"].extend(str(p) for p in rendered)
        if track == primary:
            for path in rendered:
                public_name = path.name.replace(f"-{track}.png", f"{suffix}.png")
                target = PUBLIC_IMAGES_DIR / public_name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                webp = path.with_suffix(".webp")
                if webp.exists():
                    shutil.copyfile(webp, target.with_suffix(".webp"))
                images["public"].append(str(target))
    return images


def _write_evidence(run_id, anchor, plan, results, products, primary, bundle, rule_info, fm_model_path,
                    fm_anchored, cell_km, radius_cells, grid_shape, images, runtime) -> dict:
    counties = county_cell_lists(grid_shape)
    track_summaries = {}
    arrays = {}
    for name, product in products.items():
        track = results[name]
        summary = {
            "member_ids": track["member_ids"],
            "member_count": len(track["member_ids"]),
            "degraded": bool(track.get("degraded")),
            "statewide_max_neighborhood_probability": {
                core.CATEGORY_KEYS[k]: float(np.nanmax(product["neighborhood"][k - 1]))
                for k in core.CATEGORY_IDS},
            "categorical_cell_counts": {str(c): int(np.sum(product["categorical"] == c)) for c in range(5)},
        }
        if counties:
            summary["county_max_neighborhood_probability"] = {
                core.CATEGORY_KEYS[k]: core.county_summary(product["neighborhood"][k - 1], counties)
                for k in core.CATEGORY_IDS}
            summary["county_max_categorical"] = core.county_summary(product["categorical"], counties)
        track_summaries[name] = summary
        arrays[f"{name}__peaks"] = track["peaks"]
        arrays[f"{name}__weights"] = np.asarray(track["weights"], dtype="float32")
        arrays[f"{name}__raw_point"] = product["raw_point"].astype("float32")
        arrays[f"{name}__raw_neighborhood"] = product["raw_neighborhood"].astype("float32")
        arrays[f"{name}__categorical"] = product["categorical"].astype("float32")
        for extra in ("jfwprb", "pwind_10p3", "spread_rh_proxy"):
            if extra in track:
                arrays[f"{name}__{extra}"] = np.asarray(track[extra], dtype="float32")

    record = {
        "run_id": run_id,
        "anchor_cycle_utc": anchor.strftime("%Y-%m-%dT%H:%MZ"),
        "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "model_type": bundle_mod.MODEL_TYPE,
        "advisory_only": True,
        "primary_track": primary,
        "bundle": {k: bundle.get(k) for k in ("source", "version", "calibrated", "checksum")},
        "fm_anchor": "operational_forecast_state" if fm_anchored else "none",
        "fm_model_path": fm_model_path,
        "rule": rule_info,
        "grid_shape": list(grid_shape),
        "cell_km": round(cell_km, 3),
        "neighborhood_radius_cells": round(radius_cells, 2),
        "plan": [item.as_dict() for item in plan],
        "tracks": track_summaries,
        "images": images,
        "runtime_sec": round(runtime, 1),
    }
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    json_path = EVIDENCE_ROOT / f"{run_id}.ensemble_fire_danger.json"
    json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    np.savez_compressed(EVIDENCE_ROOT / f"{run_id}.ensemble_fire_danger.npz", **arrays)
    return record


def _update_status(anchor, images, summary, primary) -> None:
    try:
        status_file = Path(os.getenv("SMF_ENSEMBLE_FD_STATUS_FILE") or (API_ROOT / "status.json"))
        status = json.loads(status_file.read_text()) if status_file.exists() else {}
        now_ct = pd.Timestamp.now(tz="America/Chicago").strftime("%Y-%m-%d %H:%M CT")
        status[STATUS_KEY] = {
            "last_update": now_ct,
            "model_run": anchor.strftime("%Y-%m-%d %HZ"),
            "status": "updated",
            "beta": True,
            "primary_track": primary,
            "runtime_sec": summary["runtime_sec"],
            "maps_generated": [Path(p).name for p in images["public"]],
            "log": [f"Ensemble fire danger (beta) updated at {now_ct}"],
        }
        tmp = status_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(status, indent=4))
        os.replace(tmp, status_file)
    except Exception as error:
        logger.warning("status.json update failed (non-fatal): %s", error)


def _update_testbed_manifest(images, summary) -> None:
    try:
        from services.beta_products import BETA_ROOT, load_manifest, save_manifest
        manifest = load_manifest()
        manifest.setdefault("products", {})["ensemble_fire_danger"] = {
            "kind": "raster-preview",
            "beta": True,
            "not_for_operations": True,
            "model_type": bundle_mod.MODEL_TYPE,
            "model_version": summary["bundle"].get("version"),
            "calibrated": summary["bundle"].get("calibrated"),
            "primary_track": summary["primary_track"],
            "generated_at": summary["recorded_at"],
            "anchor_cycle_utc": summary["anchor_cycle_utc"],
            "previews": [str(Path(p).relative_to(BETA_ROOT)).replace("\\", "/") for p in images["testbed"]
                         if Path(p).resolve().is_relative_to(BETA_ROOT.resolve())],
            "public_files": [Path(p).name for p in images["public"]],
            "tracks": {name: {"member_count": t["member_count"], "degraded": t["degraded"],
                              "statewide_max_neighborhood_probability": t["statewide_max_neighborhood_probability"]}
                       for name, t in summary["tracks"].items()},
        }
        save_manifest(manifest)
    except Exception as error:
        logger.warning("testbed manifest update failed (non-fatal): %s", error)


def _upload(paths: List[str]) -> None:
    try:
        import sys
        scripts_dir = API_ROOT / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        import upload_cdn
        # PNG only: upload_forecast_files labels every non-geojson file image/png.
        upload_cdn.upload_forecast_files(files_to_upload=[Path(p) for p in paths],
                                         path_prefix=os.getenv("CDN_TEST_PREFIX") or None)
    except Exception as error:
        logger.error("ensemble CDN upload failed (non-fatal): %s", error)


def _prune_cache() -> None:
    """Member-run NetCDFs are ~15-30 MB each, ~180 MB per forecast day. Keep
    SMF_ENSEMBLE_FD_CACHE_DAYS (default 10) days so a failed run can be
    re-rendered or the files pulled for training, then delete. The regrid
    weight cache is small and permanent. Never raises."""
    try:
        keep_days = float(os.getenv("SMF_ENSEMBLE_FD_CACHE_DAYS", "10"))
        cutoff = time.time() - keep_days * 86400
        for folder, pattern in ((CACHE_DIR / "runs", "*.nc"), (CACHE_DIR / "tmp", "*")):
            if not folder.exists():
                continue
            for path in folder.glob(pattern):
                try:
                    if path.is_file() and path.stat().st_mtime < cutoff:
                        path.unlink()
                except OSError:
                    pass
    except Exception as error:
        logger.warning("ensemble cache prune failed (non-fatal): %s", error)


def _persist_state(update: dict) -> None:
    try:
        EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
        state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
        state.update(update)
        state["runs"] = int(state.get("runs", 0)) + 1
        STATE_PATH.write_text(json.dumps(state, indent=2))
    except Exception as error:
        logger.warning("runner state write failed: %s", error)


def record_failure(error: Exception) -> None:
    try:
        EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
        state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
        state.update({"last_error": f"{type(error).__name__}: {error}",
                      "last_failure": datetime.now(timezone.utc).isoformat()})
        state["consecutive_failures"] = int(state.get("consecutive_failures", 0)) + 1
        STATE_PATH.write_text(json.dumps(state, indent=2))
    except Exception:
        pass


def diagnostics() -> dict:
    try:
        state = json.loads(STATE_PATH.read_text()) if STATE_PATH.exists() else {}
    except Exception:
        state = {}
    latest = sorted(EVIDENCE_ROOT.glob("*.ensemble_fire_danger.json"))[-1:] if EVIDENCE_ROOT.exists() else []
    summary = None
    if latest:
        try:
            record = json.loads(latest[0].read_text())
            summary = {k: record.get(k) for k in ("run_id", "primary_track", "bundle", "fm_anchor", "runtime_sec")}
            summary["tracks"] = {n: {"member_count": t["member_count"], "degraded": t["degraded"],
                                     "statewide_max_neighborhood_probability":
                                         t["statewide_max_neighborhood_probability"]}
                                 for n, t in (record.get("tracks") or {}).items()}
            summary["members"] = [{k: p[k] for k in ("member_id", "cycle_utc", "status")} for p in record["plan"]]
        except Exception:
            summary = None
    return {"model_type": bundle_mod.MODEL_TYPE, "advisory_only": True, "state": state, "latest": summary}
