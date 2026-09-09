"""
Failure-isolated, shadow-only scoring of the fire_weather_ml XGBoost
candidate (model-training/fire_weather_ml/) alongside the real Rothermel
calculation this repo already runs (services/spread_rate.py) - mirrors
services/risk_fusion_glm_shadow.py's structure (kill switch, persisted
state, immutable evidence) for the same reason: this scores a real, trained
model in shadow only, and must never affect the public spread-rate output.

fire_weather_ml's own model inputs
(model-training/fire_weather_ml/features.py::MODEL_FEATURE_COLUMNS -
temp_c, rh_pct, wind_ms, precip_mm, fm1_pct, fm10_pct, fm100_pct, slope_deg,
aspect_deg, canopy_cover_pct, canopy_height_m) are EXACTLY the same
weather/fuel-moisture/terrain quantities services/spread_rate.py already
computes every run (via fire_behavior_static.load_static_fields() and
spread_rate_moisture.condition_moisture()) - so this shadow scores the SAME
grid the real Rothermel calculation just ran on, rather than needing any
separate inputs, and can directly compare its predictions against the real
Rothermel-computed rate of spread as an ongoing, LIVE accuracy check (the
offline held-out R^2=0.9622 recorded in
model-training/docs/fire_weather_ml_plan.md, now checked against live
production data too, not just the historical panel it was fit/evaluated on).

Since the model's feature set is a fixed list of already-computed grid
quantities (no derivation pipeline like risk_fusion's KBDI/GDD features to
byte-for-byte mirror), the safety contract here is simpler than
risk_fusion_glm_shadow's core/risk_fusion_features.py mirror: this module
just hardcodes EXPECTED_FEATURE_COLUMNS and refuses to score a bundle whose
own recorded feature_columns don't match it exactly (order matters - it's
positional input to the booster).

Loaded from a raw bundle directory via SMF_FIRE_WEATHER_ML_BUNDLE - the
same shadow-only pattern risk_fusion_glm_shadow.py/V4/V5 use, NOT through
the model registry (models/versioning.py). The registered training-side
beta candidate (model-training/fire_weather_ml/register_beta.py) is
explicitly "not production eligible... nothing here touches the api/ repo"
until a real promotion pipeline exists for this model family - this module
is the first thing that does, and it does so in shadow only, exactly like
risk_fusion's GLM did before any such pipeline existed for it either.

Every public function catches its own exceptions and never raises into the
caller - a bug here must never affect spread-rate generation.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

import numpy as np

BUNDLE_ENV = "SMF_FIRE_WEATHER_ML_BUNDLE"
ENABLED_ENV = "FIRE_WEATHER_ML_SHADOW_ENABLED"
EVIDENCE_ROOT = Path(os.getenv("SMF_FIRE_WEATHER_ML_EVIDENCE_ROOT") or
                     (Path(os.getenv("DATA_DIR", "data")) / "model-shadow" / "fire-weather-ml"))
MAX_FAILURES = int(os.getenv("FIRE_WEATHER_ML_SHADOW_MAX_FAILURES", "3"))
STATE_PATH = EVIDENCE_ROOT / "shadow-state.json"

MODEL_TYPE = "fire_weather_ml"
NOT_FOR_OPERATIONS_LABEL = "SHADOW ONLY — NOT FOR OPERATIONS"
IMAGE_FILENAME = "fire_weather_ml_shadow_latest.png"

# Same values as services/spread_rate.py's own ROS_CLASS_*/NODATA_CLASS -
# duplicated rather than imported, same reason as _aspect_degrees above:
# importing services.spread_rate at all pulls in its full production
# import chain (rtma_capture -> herbie, etc.), which this module has no
# other reason to depend on. Keep these in sync by hand if the real
# product's class scheme ever changes.
ROS_CLASS_BOUNDS = (0.0, 2.0, 5.0, 20.0, 50.0, 150.0)
ROS_CLASS_LABELS = ("Very Low", "Low", "Moderate", "High", "Very High", "Extreme")
ROS_CLASS_COLORS = ("#90EE90", "#ADFF2F", "#FFFF00", "#FFA500", "#FF4500", "#8B0000")
NODATA_CLASS = 255


def _classify_ros_ch_per_h(rate: np.ndarray) -> np.ndarray:
    """Same logic as services/spread_rate.py::classify_ros_ch_per_h - see the module-level note above."""
    classes = np.full(rate.shape, NODATA_CLASS, dtype=np.uint8)
    valid = np.isfinite(rate) & (rate >= 0.0)
    if not np.any(valid):
        return classes
    bucket = np.digitize(rate[valid], ROS_CLASS_BOUNDS, right=False) - 1
    classes[valid] = np.clip(bucket, 0, len(ROS_CLASS_LABELS) - 1).astype(np.uint8)
    return classes


def _image_path() -> Path:
    """
    Rendered alongside the real Testbed spread-rate PNG (same BETA_ROOT the
    stable/beta products already publish into - services/beta_products.py,
    imported lazily here to avoid this module's own light import footprint
    pulling in beta_products' heavier chain at module load time) so this is
    visible the same way, but under its own filename/manifest key and never
    routed through services/gis_publisher.py's public-map pipeline - it must
    look and be findable as a distinct, clearly-labeled shadow product, not
    something that could be mistaken for a served layer.
    """
    from services.beta_products import BETA_ROOT
    return BETA_ROOT / "images" / IMAGE_FILENAME

# Must match model-training/fire_weather_ml/features.py::MODEL_FEATURE_COLUMNS
# exactly, in order - positional input to the XGBoost booster.
EXPECTED_FEATURE_COLUMNS = (
    "temp_c", "rh_pct", "wind_ms", "precip_mm",
    "fm1_pct", "fm10_pct", "fm100_pct",
    "slope_deg", "aspect_deg", "canopy_cover_pct", "canopy_height_m",
)

BUNDLE_ASSET_FILENAMES = {
    "contract": "contract.json",
    "metadata": "fire_weather_ml_metadata.json",
    "model": "fire_weather_ml_model.json",
}


def _configured() -> bool:
    return bool(os.getenv(BUNDLE_ENV, "").strip())


def _requested() -> bool:
    return os.getenv(ENABLED_ENV, "false").strip().lower() in {"1", "true", "yes", "on"}


def _initial_state() -> dict:
    state = {
        "configured": _configured(),
        "enabled": _configured() and _requested(),
        "healthy": True,
        "auto_disabled": False,
        "consecutive_failures": 0,
        "last_error": None,
        "last_success": None,
        "runs": 0,
        "successful_runs": 0,
        "cells_scored": 0,
        "bundle_checksum": None,
        "model_version": None,
        "last_comparison": None,
        "last_image": None,
        "public_path_unchanged": True,
    }
    try:
        if STATE_PATH.exists():
            stored = json.loads(STATE_PATH.read_text())
            for key in state:
                state[key] = stored.get(key, state[key])
    except Exception:
        pass
    state["configured"] = _configured()
    state["enabled"] = bool(state["configured"] and _requested() and not state.get("auto_disabled", False))
    return state


_state = _initial_state()


def _persist_state() -> None:
    try:
        EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
        temporary = STATE_PATH.with_suffix(".tmp")
        temporary.write_text(json.dumps(_state, indent=2))
        temporary.replace(STATE_PATH)
    except Exception:
        pass


def diagnostics() -> dict:
    """Mirrors risk_fusion_glm_shadow.diagnostics()'s cron-process reread."""
    try:
        if STATE_PATH.exists():
            stored = json.loads(STATE_PATH.read_text())
            for key in _state:
                if key in stored:
                    _state[key] = stored[key]
    except Exception as error:
        _state["healthy"] = False
        _state["last_error"] = f"unable to read shadow state: {error}"
    _state["configured"] = _configured()
    _state["enabled"] = bool(_state["configured"] and _requested() and not _state.get("auto_disabled", False))
    return dict(_state)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_bundle(directory: Optional[Path] = None) -> Dict:
    """
    Loads and validates a fire_weather_ml candidate bundle directory.
    Raises on any contract mismatch - a caller must never score with a
    bundle that isn't advisory_only, isn't the model_family this module
    knows how to score, or whose feature_columns don't match
    EXPECTED_FEATURE_COLUMNS exactly.
    """
    import xgboost as xgb

    directory = Path(directory or os.getenv(BUNDLE_ENV, ""))
    if not str(directory) or not directory.is_dir():
        raise FileNotFoundError(f"{BUNDLE_ENV} is not a bundle directory")
    for filename in BUNDLE_ASSET_FILENAMES.values():
        if not (directory / filename).is_file():
            raise FileNotFoundError(f"fire_weather_ml bundle missing asset: {filename}")

    contract = json.loads((directory / BUNDLE_ASSET_FILENAMES["contract"]).read_text(encoding="utf-8"))
    if contract.get("advisory_only") is not True:
        raise ValueError("fire_weather_ml bundle is not advisory_only")
    if contract.get("model_family") != "xgboost_regressor":
        raise ValueError(
            f"fire_weather_ml shadow only knows how to score model_family='xgboost_regressor', "
            f"got {contract.get('model_family')!r}")
    feature_columns = tuple(contract.get("feature_columns") or ())
    if feature_columns != EXPECTED_FEATURE_COLUMNS:
        raise ValueError(
            f"fire_weather_ml bundle feature_columns {feature_columns!r} does not match "
            f"this module's EXPECTED_FEATURE_COLUMNS {EXPECTED_FEATURE_COLUMNS!r}")

    booster = xgb.Booster()
    booster.load_model(str(directory / BUNDLE_ASSET_FILENAMES["model"]))

    bundle_checksum = hashlib.sha256(
        "".join(_sha256_file(directory / filename) for filename in BUNDLE_ASSET_FILENAMES.values()).encode()
    ).hexdigest()

    # Written by model-training/fire_weather_ml/register_beta.py after a
    # successful registration - optional (an older/manually-copied bundle
    # directory won't have it), never fatal. Without it, graphics/evidence
    # just report "unknown" rather than refusing to score.
    version_path = directory / "registered_version.json"
    version = None
    if version_path.is_file():
        try:
            version = json.loads(version_path.read_text(encoding="utf-8")).get("version")
        except Exception:
            version = None

    return {"booster": booster, "contract": contract, "bundle_checksum": bundle_checksum, "version": version}


def _aspect_degrees(aspect_sin: np.ndarray, aspect_cos: np.ndarray) -> np.ndarray:
    """Same trivial trig as services/spread_rate.py::aspect_degrees, defined here directly rather than importing
    that module (which would pull in its full heavy import chain - xarray, rtma_capture, etc. - just for this)."""
    return (np.degrees(np.arctan2(aspect_sin, aspect_cos)) + 360.0) % 360.0


def score_grid(bundle: Dict, static: dict, moisture: dict) -> np.ndarray:
    """
    Predicts ros_ch_per_h over the same valid-cell mask
    services/spread_rate.py::compute_spread_rate_grid uses, from the SAME
    static/moisture inputs that function already received this run. Returns
    a full-shape array with NaN outside the valid mask.
    """
    import pandas as pd

    shape = static["lat"].shape
    predictions = np.full(shape, np.nan, dtype=np.float32)

    wind_ms = np.asarray(moisture["wind_ms"], dtype=float)
    valid = static["valid_mask"] & np.isfinite(wind_ms)
    if not np.any(valid):
        return predictions

    aspect = _aspect_degrees(static["aspect_sin"], static["aspect_cos"])
    features = pd.DataFrame({
        "temp_c": np.asarray(moisture["temp_c"], dtype=float)[valid],
        "rh_pct": np.asarray(moisture["rh"], dtype=float)[valid],
        "wind_ms": wind_ms[valid],
        "precip_mm": np.asarray(moisture["precip_mm"], dtype=float)[valid],
        "fm1_pct": np.asarray(moisture["fm1_pct"], dtype=float)[valid],
        "fm10_pct": np.asarray(moisture["fm10_pct"], dtype=float)[valid],
        "fm100_pct": np.asarray(moisture["fm100_pct"], dtype=float)[valid],
        "slope_deg": np.asarray(static["slope_deg"], dtype=float)[valid],
        "aspect_deg": aspect[valid],
        "canopy_cover_pct": np.asarray(static["canopy_cover_pct"], dtype=float)[valid],
        "canopy_height_m": np.asarray(static["canopy_height_m"], dtype=float)[valid],
    }, columns=list(EXPECTED_FEATURE_COLUMNS))

    import xgboost as xgb
    dmatrix = xgb.DMatrix(features, feature_names=list(EXPECTED_FEATURE_COLUMNS))
    raw = bundle["booster"].predict(dmatrix)
    # The regressor has no built-in floor - a real rate of spread can never
    # be negative, but nothing stops the model from predicting one for a
    # near-zero-spread cell (observed on real data: -0.00 to -0.03 ch/h).
    # Clipped here, once, so every consumer (evidence, comparison stats,
    # the rendered graphic) sees the same physically-sane value.
    predictions[valid] = np.clip(raw, 0.0, None)
    return predictions


def _correlation_or_none(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    """np.corrcoef divides by each array's stddev - zero variance (e.g. a near-constant sample) makes that a NaN,
    which json.dumps would otherwise emit as the invalid-JSON literal `NaN`. None is the honest "not computable" value."""
    if len(a) < 2 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return None
    value = float(np.corrcoef(a, b)[0, 1])
    return value if np.isfinite(value) else None


def _compare(predicted: np.ndarray, real: np.ndarray) -> Dict:
    """Diagnostic comparison stats over cells where both are finite - never a promotion gate, informational only."""
    both_valid = np.isfinite(predicted) & np.isfinite(real)
    if not np.any(both_valid):
        return {"compared_cells": 0}
    diff = predicted[both_valid] - real[both_valid]
    return {
        "compared_cells": int(both_valid.sum()),
        "mean_absolute_error_ch_per_h": float(np.mean(np.abs(diff))),
        "mean_predicted_ch_per_h": float(np.mean(predicted[both_valid])),
        "mean_real_ch_per_h": float(np.mean(real[both_valid])),
        "correlation": _correlation_or_none(predicted[both_valid], real[both_valid]),
    }


def _render_png(predicted: np.ndarray, static: dict, bundle: Dict, comparison: Dict, out_path: Path) -> None:
    """
    Same map style/classification as services/spread_rate.py::_render_png
    (reused via lazy import - safe here since this only runs from inside
    that same live pipeline, which has already paid for matplotlib/
    cartopy/geopandas), but explicitly re-labeled: title says "Shadow", a
    prominent not-for-operations banner, and the scored model's name and
    registered version - so this can never be mistaken for the real,
    served Testbed product it's rendered alongside.
    """
    import cartopy.crs as ccrs
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    out_path.parent.mkdir(parents=True, exist_ok=True)
    classes = _classify_ros_ch_per_h(predicted).astype(np.float32)
    classes[classes == NODATA_CLASS] = np.nan
    cmap = ListedColormap(ROS_CLASS_COLORS)
    norm = BoundaryNorm(np.arange(-0.5, len(ROS_CLASS_LABELS) + 0.5, 1.0), len(ROS_CLASS_LABELS))

    pixel_width, pixel_height, dpi = 2048, 1152, 144
    data_crs = ccrs.PlateCarree()
    map_crs = ccrs.LambertConformal(central_longitude=-92.45, central_latitude=38.3)
    fig = plt.figure(figsize=(pixel_width / dpi, pixel_height / dpi), dpi=dpi, facecolor="#E8E8E8")
    ax = fig.add_axes([0.04, 0.04, 0.70, 0.92], projection=map_crs)
    ax.set_extent((-95.8, -89.1, 35.8, 40.8), crs=data_crs)
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    mesh = ax.pcolormesh(
        static["lon"], static["lat"], classes, transform=data_crs,
        cmap=cmap, norm=norm, shading="auto", alpha=0.78, zorder=2,
    )

    maps_dir = Path(__file__).resolve().parent.parent / "maps"
    county_path = maps_dir / "shapefiles" / "MO_County_Boundaries" / "MO_County_Boundaries.shp"
    state_path = maps_dir / "shapefiles" / "MO_State_Boundary" / "MO_State_Boundary.shp"
    if county_path.is_file():
        counties = gpd.read_file(county_path).to_crs("EPSG:4326")
        ax.add_geometries(counties.geometry, crs=data_crs, edgecolor="#B6B6B6", facecolor="none",
                          linewidth=0.7, zorder=4)
    if state_path.is_file():
        state = gpd.read_file(state_path).to_crs("EPSG:4326")
        ax.add_geometries(state.geometry, crs=data_crs, edgecolor="#111111", facecolor="none",
                          linewidth=1.5, zorder=5)

    cax = fig.add_axes([0.02, 0.12, 0.018, 0.58])
    cbar = fig.colorbar(mesh, cax=cax, ticks=np.arange(len(ROS_CLASS_LABELS)))
    cbar.ax.set_yticklabels(ROS_CLASS_LABELS)
    cbar.set_label("Predicted Head-Fire Spread Rate")

    # Large, impossible-to-miss banner - this is the actual point of this
    # graphic existing at all, per the request that motivated it.
    fig.text(0.5, 0.965, NOT_FOR_OPERATIONS_LABEL, fontsize=22, fontweight="bold",
             ha="center", va="top", color="#B00000")
    fig.text(0.98, 0.92, "fire_weather_ml Shadow Comparison", fontsize=22, fontweight="bold", ha="right", va="top")
    fig.text(0.98, 0.865, f"Model: {MODEL_TYPE}  Version: {bundle.get('version') or 'unknown'}",
             fontsize=14, ha="right", va="top")
    mae = comparison.get("mean_absolute_error_ch_per_h")
    correlation = comparison.get("correlation")
    summary_lines = [f"Compared cells: {comparison.get('compared_cells', 0)}"]
    if mae is not None:
        summary_lines.append(f"Mean absolute error vs. real Rothermel: {mae:.3f} ch/h")
    if correlation is not None:
        summary_lines.append(f"Correlation with real output: {correlation:.3f}")
    fig.text(0.77, 0.66, "\n\n".join(summary_lines), fontsize=12, ha="left", va="top", linespacing=1.35)
    fig.text(
        0.77, 0.30,
        "This is a machine-learning approximation of the real Rothermel\n"
        "calculation, scored for internal comparison only. It is NOT\n"
        "reviewed, validated, or approved for any operational decision.\n"
        "See the real Observed Rothermel Spread Rate product for the\n"
        "actual physics-based Testbed output.",
        fontsize=11, ha="left", va="top", linespacing=1.4, color="#444444",
    )

    fd, temporary = tempfile.mkstemp(prefix=".fire_weather_ml_shadow.", suffix=".png", dir=out_path.parent)
    os.close(fd)
    temp_path = Path(temporary)
    try:
        fig.savefig(temp_path, facecolor=fig.get_facecolor())
        plt.close(fig)
        temp_path.replace(out_path)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)
        plt.close("all")


def _update_manifest(image_path: Path, bundle: Dict, comparison: Dict, generated_at: str) -> None:
    """Mirrors services/spread_rate.py::_update_manifest's shape, under a distinct product key so it never
    overwrites or is confused with the real spread_rate manifest entry."""
    from services.beta_products import BETA_ROOT, load_manifest, save_manifest

    manifest = load_manifest()
    manifest.setdefault("products", {})["fire_weather_ml_shadow"] = {
        "kind": "raster-preview",
        "preview": str(image_path.relative_to(BETA_ROOT)).replace("\\", "/"),
        "not_for_operations": True,
        "model_type": MODEL_TYPE,
        "model_version": bundle.get("version"),
        "generated_at": generated_at,
        "comparison": comparison,
    }
    save_manifest(manifest)


def score_for_spread_rate(
    static: dict, moisture: dict, real_grids: dict, *,
    bundle_dir: Optional[Path] = None, evidence_root: Optional[Path] = None,
) -> bool:
    """
    Scores fire_weather_ml against the same static/moisture inputs
    services/spread_rate.py just used, compares against the real Rothermel
    output (real_grids["ros_ch_per_h"]), and writes an immutable evidence
    file. Never raises; returns False on any failure (including shadow
    being disabled).
    """
    if not diagnostics()["enabled"]:
        return False
    try:
        bundle = load_bundle(bundle_dir)
        predicted = score_grid(bundle, static, moisture)
        comparison = _compare(predicted, np.asarray(real_grids["ros_ch_per_h"], dtype=float))
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        record = {
            "recorded_at": generated_at,
            "bundle_checksum": bundle["bundle_checksum"],
            "model_version": bundle.get("version"),
            "comparison": comparison,
        }
        root = Path(evidence_root or EVIDENCE_ROOT)
        root.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = root / f"{run_id}.fire_weather_ml_score.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)

        # Best-effort: a plotting/manifest failure must never count against
        # the shadow's own health/auto-disable tracking - the scoring and
        # evidence write above are the part that actually matters.
        image_path = None
        try:
            image_path = _image_path()
            _render_png(predicted, static, bundle, comparison, image_path)
            _update_manifest(image_path, bundle, comparison, generated_at)
        except Exception as render_error:
            import logging
            logging.getLogger(__name__).warning(
                "fire_weather_ml shadow graphic/manifest failed (non-fatal): %s", render_error)

        _state.update(
            consecutive_failures=0, last_error=None, healthy=True,
            last_success=record["recorded_at"],
            runs=_state["runs"] + 1, successful_runs=_state.get("successful_runs", 0) + 1,
            cells_scored=comparison.get("compared_cells", 0),
            bundle_checksum=bundle["bundle_checksum"],
            model_version=bundle.get("version"),
            last_comparison=comparison,
            last_image=str(image_path) if image_path else None,
        )
        _persist_state()
        return True
    except Exception as error:
        _state["runs"] += 1
        _state["consecutive_failures"] += 1
        _state["last_error"] = str(error)
        _state["healthy"] = False
        if _state["consecutive_failures"] >= MAX_FAILURES:
            _state["enabled"] = False
            _state["auto_disabled"] = True
        _persist_state()
        return False


def record_skipped_run(reason: str) -> bool:
    """Makes an enabled-but-empty shadow attempt observable, mirroring risk_fusion_glm_shadow.record_skipped_run."""
    if not diagnostics()["enabled"]:
        return False
    _state["runs"] += 1
    _state["consecutive_failures"] += 1
    _state["last_error"] = str(reason)
    _state["healthy"] = False
    if _state["consecutive_failures"] >= MAX_FAILURES:
        _state["enabled"] = False
        _state["auto_disabled"] = True
    _persist_state()
    return True
