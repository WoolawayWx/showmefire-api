"""Opt-in, isolated staged-input worker. No public forecast hooks."""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import time
import numpy as np
import xarray as xr
from services.fire_weather_hybrid.contracts import GRID, FUELS, LOCAL_TZ, day_hours, utc
from services.fire_weather_hybrid.runtime import Predictor, PhysicsPredictor
from services import fire_weather_hybrid_store as store

logger = logging.getLogger(__name__)


def run(input_path, bundle_path=None, *, root=None, days=1):
    """bundle_path=None (default) runs the physics path; a bundle runs the legacy ML path for comparison."""
    root = Path(root or store.root_dir())
    root.mkdir(parents=True, exist_ok=True)
    lock = root/"worker.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    started = time.monotonic()
    try:
        os.write(descriptor, str(os.getpid()).encode())
        if bundle_path is None:
            predictor = PhysicsPredictor()
            model_version, evidence, scale_version = PhysicsPredictor.version, "physics", predictor.scale["version"]
        else:
            from services.fire_weather_hybrid.trees import Bundle
            bundle = Bundle(bundle_path)
            predictor = Predictor(bundle)
            model_version, evidence = bundle.manifest["version"], bundle.manifest.get("evidence_status", "experimental")
            scale_version = bundle.scale["version"]
        with xr.open_dataset(input_path) as ds:
            if ds.attrs.get("grid_fingerprint") != GRID.fingerprint:
                raise ValueError("staged input grid fingerprint mismatch")
            if not np.array_equal(ds.x.values, GRID.x) or not np.array_equal(ds.y.values, GRID.y):
                raise ValueError("staged input axes do not match the canonical grid")
            issued = utc(ds.attrs["issued_at"])
            model_reference=utc(ds.attrs.get("model_reference_at",ds.attrs["issued_at"]))
            if model_reference>issued:raise ValueError("model time reference cannot follow issuance")
            if "window_start" in ds.attrs:  # rolling window written by staging
                start = utc(ds.attrs["window_start"])
                times = [start+timedelta(hours=i) for i in range(int(ds.attrs["window_hours"]))]
            else:
                times = day_hours(issued.astimezone(LOCAL_TZ).date(), days)
            time_values = [np.datetime64(value.replace(tzinfo=None), "ns") for value in times]
            if not set(time_values) <= set(ds.time.values.astype("datetime64[ns]")):
                raise ValueError("staged input does not cover the complete forecast window")
            required = {"temp_c", "rh_pct", "wind_speed_ms", "burnable_fraction", "missouri_mask", *[f"fm{fuel}_pct" for fuel in FUELS]}
            if not required <= set(ds):
                raise ValueError(f"missing staged fields: {sorted(required-set(ds))}")
            provenance = json.loads(ds.attrs.get("provenance", "{}"))
            if not provenance.get("sources"):
                raise ValueError("staged input requires source provenance")
            for source in provenance["sources"]:
                from services.fire_weather_hybrid.sources import validate_availability
                validate_availability(source["name"], source["cycle_time"], source["available_at"], issued)
            def hours():
                for index, (valid, coordinate) in enumerate(zip(times, time_values)):
                    store.atomic_json(root/"status.json", {"state": "running", "updating": True, "step": index,
                        "steps": len(times), "valid_time": valid.isoformat(), "elapsed_seconds": round(time.monotonic()-started, 1)})
                    hour = ds.sel(time=coordinate)
                    fields = {name: np.asarray(value.values, "float32") for name,value in hour.data_vars.items()
                              if value.dims == ("y", "x") and name != "office"}
                    core_valid = np.isfinite(fields["temp_c"]) & np.isfinite(fields["rh_pct"]) & np.isfinite(fields["wind_speed_ms"])
                    domain = fields["missouri_mask"] > 0
                    if not domain.any() or not np.any(core_valid & domain):
                        raise ValueError("staged hour has no Missouri weather coverage")
                    result = predictor.predict_hour(fields, valid, model_reference, office=hour.office.values if "office" in hour else None)
                    for name in result:
                        result[name] = np.where(fields["missouri_mask"] > 0, result[name], np.nan)
                    logger.info("hybrid hour complete valid=%s step=%d", valid.isoformat(), index)
                    yield result
            manifest = store.write_run(root, issued, times, hours(), cycle_time=utc(ds.attrs["anchor_cycle"]) if "anchor_cycle" in ds.attrs else None, metadata={"model_version": model_version,
                "quality": {"evidence_status": evidence,
                            "fuel_physics": ds.attrs.get("fuel_physics", "unspecified"), "provenance": provenance,
                            "model_reference_at":model_reference.isoformat()},
                "risk_scale_version": scale_version,
                "step_kinds":json.loads(ds.attrs["step_kinds"]) if "step_kinds" in ds.attrs else None}, grid=GRID,
                render_maps=os.getenv("SMF_FIRE_WEATHER_HYBRID_MAPS", "true").lower() == "true")
            store.atomic_json(root/"status.json", {"state": "complete", "updating": False, "cycle": manifest["cycle"],
                "serving_cycle": manifest["cycle"], "elapsed_seconds": round(time.monotonic()-started, 1),
                "updated_at": datetime.now(timezone.utc).isoformat()})
            return manifest
    except Exception as error:
        # No credentials or upstream request URLs are written to browser status.
        logger.exception("hybrid worker failed")
        store.atomic_json(root/"status.json", {"state": "failed", "updating": False,
                                              "error_type": type(error).__name__, "updated_at": datetime.now(timezone.utc).isoformat()})
        raise
    finally:
        os.close(descriptor)
        lock.unlink(missing_ok=True)

