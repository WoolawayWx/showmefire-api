"""
Failure-isolated, immutable fire_weather_index shadow evidence - mirrors
services/risk_fusion_glm_shadow.py's structure (kill switch, persisted
state, immutable evidence files) for the same reason: this scores a real
model in shadow only, and must never affect the public forecast path.

fire_weather_index (model-training/fire_weather_index/) is a fourth,
independent model family - a continuous, weighted fire-WEATHER-conditions
score (not an ignition-risk count model like fire_risk_fusion, and not a
distillation of the live if/then rule like the standalone
api/fire-danger-model/ experiment). See that package's __init__.py for the
full rationale. There is no GLM here, no fitted coefficients at all - the
"model" is a fixed set of ramp functions and weights (calibrated only in
the sense that model-training/fire_weather_index/calibrate.py sets the
category cutpoints from the score's own historical percentile
distribution), reimplemented here in pure numpy/stdlib from the bundle's
two small JSON files, same repo-independence reason as
core/risk_fusion_features.py.

Live scoring uses HRRR-derived grids only (the same hourly_rh/hourly_ws_kts/
hourly_temp_c/hourly_precip_mm arrays risk_fusion_hook.py already builds
from the live forecast run) - RRFS/FV3-HIRES blending only happens in the
OFFLINE training panel (model-training/fire_weather_index/build_county_days.py),
for accumulating history toward a future retrain, not for live scoring.
KBDI and the seasonal cure/green-up factor need multi-day accumulated
state this hook does not track, so they are treated as unavailable at
live-scoring time - factors.compute_score's existing renormalize-over-
available-factors behavior handles that the same way it handles any other
missing factor, not a special case.

Loaded from a raw bundle directory via SMF_FIRE_WEATHER_INDEX_BUNDLE - same
shadow-only pattern as risk_fusion_glm_shadow.py/fire_weather_ml_shadow.py,
not through the model registry.

Every public function catches its own exceptions and never raises into the
caller - a bug here must never affect forecast generation.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

BUNDLE_ENV = "SMF_FIRE_WEATHER_INDEX_BUNDLE"
ENABLED_ENV = "FIRE_WEATHER_INDEX_SHADOW_ENABLED"
EVIDENCE_ROOT = Path(os.getenv("SMF_FIRE_WEATHER_INDEX_EVIDENCE_ROOT") or
                     (Path(os.getenv("DATA_DIR", "data")) / "model-shadow" / "fire-weather-index"))
MAX_FAILURES = int(os.getenv("FIRE_WEATHER_INDEX_SHADOW_MAX_FAILURES", "3"))
STATE_PATH = EVIDENCE_ROOT / "shadow-state.json"

MODEL_TYPE = "fire_weather_index"
NOT_FOR_OPERATIONS_LABEL = "SHADOW ONLY — NOT FOR OPERATIONS"
IMAGE_FILENAME = "fire_weather_index_shadow_latest.png"
CATEGORY_LABELS = ("Low", "Moderate", "Elevated", "Critical", "Extreme")
# Same 5-class palette convention as fire_weather_ml_shadow.py's ROS_CLASS_COLORS
# (calm green -> extreme dark red) - kept visually consistent across shadow
# products even though this is a county choropleth, not a grid.
CATEGORY_COLORS = ("#90EE90", "#ADFF2F", "#FFFF00", "#FFA500", "#8B0000")
NODATA_COLOR = "#CCCCCC"

BUNDLE_ASSET_FILENAMES = {
    "factor_weights": "factor_weights.json",
    "category_thresholds": "category_thresholds.json",
}

# Must match model-training/fire_weather_index/factors.py's ramp anchors
# exactly - reimplemented here rather than imported, same independence
# reason as EXPECTED_FEATURE_COLUMNS in fire_weather_ml_shadow.py. These
# are read from the bundle's factor_weights.json (see build_factor_weights_asset
# in model_bundle.py), not hardcoded here, so a future recalibration doesn't
# need a code change - this module only hardcodes the FUNCTIONAL FORM
# (linear ramp, weighted average), not the numeric anchors.


def _configured() -> bool:
    if os.getenv(BUNDLE_ENV, "").strip():
        return True
    try:
        from models.versioning import get_model_entry
        return bool(get_model_entry("fire_weather_index").get("stable"))
    except Exception:
        return False


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
        "counties_scored": 0,
        "bundle_checksum": None,
        "model_version": None,
        "last_score_summary": None,
        "last_category_counts": None,
        "last_image": None,
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


def _resolve_bundle_files(directory: Optional[Path]):
    """Resolve role -> file path for a bundle, preferring an explicit
    directory / BUNDLE_ENV override, and falling back to the unified
    registry's `stable` channel once this family has been migrated and
    nothing has overridden it. Returns (files, version_hint)."""
    directory = Path(directory or os.getenv(BUNDLE_ENV, "")) if (directory or os.getenv(BUNDLE_ENV, "").strip()) else None
    if directory is not None:
        if not directory.is_dir():
            raise FileNotFoundError(f"{BUNDLE_ENV} is not a bundle directory")
        files = {role: directory / filename for role, filename in BUNDLE_ASSET_FILENAMES.items()}
        version = None
        version_path = directory / "registered_version.json"
        if version_path.is_file():
            try:
                version = json.loads(version_path.read_text(encoding="utf-8")).get("version")
            except Exception:
                version = None
        return files, version

    from models.versioning import get_model_entry, load_active_assets
    resolved = load_active_assets("fire_weather_index", channel="stable")
    files = {role: asset["path"] for role, asset in resolved.items()}
    stable = get_model_entry("fire_weather_index").get("stable") or {}
    version = (stable.get("metadata") or {}).get("shadow_bundle_version") or stable.get("version")
    return files, version


def load_bundle(directory: Optional[Path] = None) -> Dict:
    """Loads and validates a fire_weather_index candidate bundle, either
    from an explicit directory / BUNDLE_ENV (fixed filenames), or - once
    this family is migrated and nothing overrides it - from the unified
    registry's active stable version."""
    files, version = _resolve_bundle_files(directory)
    missing = [role for role in BUNDLE_ASSET_FILENAMES if role not in files]
    if missing:
        raise FileNotFoundError(f"fire_weather_index bundle missing asset(s): {missing}")
    assets = {role: json.loads(Path(files[role]).read_text(encoding="utf-8")) for role in BUNDLE_ASSET_FILENAMES}

    weights_asset = assets["factor_weights"]
    if weights_asset.get("schema") != "fire-weather-index-factor-weights-v1":
        raise ValueError(f"unexpected factor_weights schema: {weights_asset.get('schema')!r}")
    thresholds_asset = assets["category_thresholds"]
    if thresholds_asset.get("schema") != "fire-weather-index-category-thresholds-v1":
        raise ValueError(f"unexpected category_thresholds schema: {thresholds_asset.get('schema')!r}")

    bundle_checksum = hashlib.sha256(
        "".join(_sha256_file(Path(files[role])) for role in BUNDLE_ASSET_FILENAMES).encode()
    ).hexdigest()

    return {**assets, "bundle_checksum": bundle_checksum, "version": version}


def _ramp(value: Optional[float], benign: float, extreme: float) -> Optional[float]:
    if value is None or not np.isfinite(value):
        return None
    return float(np.clip((value - benign) / (extreme - benign), 0.0, 1.0))


def compute_factors(anchors: Dict, weather_row: Dict) -> Dict[str, Optional[float]]:
    """rh/wind/vpd/precip_relief only - kbdi/cure need multi-day state this
    live hook doesn't track (see module docstring)."""
    return {
        "rh": _ramp(weather_row.get("rh_min_afternoon"), anchors["rh"]["benign"], anchors["rh"]["extreme"]),
        "wind": _ramp(weather_row.get("wind_kts_max"), anchors["wind"]["benign"], anchors["wind"]["extreme"]),
        "vpd": _ramp(weather_row.get("vpd_kpa_max"), anchors["vpd"]["benign"], anchors["vpd"]["extreme"]),
        "precip_relief": _ramp(weather_row.get("precip_24h_mm"), anchors["precip_relief"]["benign"],
                                anchors["precip_relief"]["extreme"]),
    }


def compute_score(weights: Dict[str, float], factor_values: Dict[str, Optional[float]],
                   raw_score_ceiling: float = 1.0) -> Optional[float]:
    """raw_score_ceiling: divides the raw weighted average before the final
    clip - mirrors model-training's factors.compute_score's rescale against
    factors.RAW_SCORE_CEILING (see that constant's docstring for what it's
    anchored to). Defaults to 1.0 (no-op) only so this function still works
    if ever called without a bundle-provided value; score_county_day below
    always passes the bundle's own value."""
    numerator, denominator = 0.0, 0.0
    for name, weight in weights.items():
        value = factor_values.get(name)
        if value is None:
            continue
        signed_weight = -weight if name == "precip_relief" else weight
        numerator += signed_weight * value
        denominator += weight
    if denominator == 0.0:
        return None
    return float(np.clip((numerator / denominator) / raw_score_ceiling, 0.0, 1.0))


def score_to_category(score: float, thresholds: List[float]) -> int:
    category = 0
    for cutpoint in thresholds:
        if score >= cutpoint:
            category += 1
        else:
            break
    return min(category, 4)


def score_county_day(bundle: Dict, weather_row: Dict) -> Dict:
    weights = bundle["factor_weights"]["weights"]
    anchors = bundle["factor_weights"]["ramp_anchors"]
    raw_score_ceiling = bundle["factor_weights"].get("raw_score_ceiling", {}).get("value") or 1.0
    factor_values = compute_factors(anchors, weather_row)
    score = compute_score(weights, factor_values, raw_score_ceiling)
    category = score_to_category(score, bundle["category_thresholds"]["thresholds"]) if score is not None else None
    return {"score": score, "category": category, "factors": factor_values}


def _image_path() -> Path:
    """Rendered alongside the other beta/shadow products (services/beta_products.py's
    BETA_ROOT), under its own filename/manifest key - see fire_weather_ml_shadow.py's
    _image_path docstring for why (never mistakable for a served layer)."""
    from services.beta_products import BETA_ROOT
    return BETA_ROOT / "images" / IMAGE_FILENAME


def _county_geometries():
    """Missouri county boundaries in EPSG:4326, keyed by 5-digit FIPS - same
    join technique as services/burn_ban_map.py::_read_counties (COUNTYFIPS is
    the shapefile's 3-digit county-only code, zero-padded and prefixed with
    Missouri's state FIPS "29")."""
    import geopandas as gpd

    shp_path = Path(__file__).resolve().parent.parent / "maps" / "shapefiles" / "MO_County_Boundaries" / "MO_County_Boundaries.shp"
    counties = gpd.read_file(shp_path).to_crs("EPSG:4326")
    counties["fips"] = counties["COUNTYFIPS"].astype(str).str.zfill(3).radd("29")
    return counties


def _render_png(county_fips: List[str], scored: Dict[str, Dict], bundle: Dict, out_path: Path) -> None:
    """
    County choropleth, one fill color per category (0=Low .. 4=Extreme) -
    same categorical add_geometries-per-subset technique
    services/burn_ban_map.py already uses for Missouri county maps (no
    existing continuous-colormap choropleth in this repo to mirror instead).
    Explicitly labeled shadow/experimental, same convention as
    fire_weather_ml_shadow.py's _render_png.
    """
    import cartopy.crs as ccrs
    import matplotlib
    matplotlib.use("Agg")  # headless/non-interactive - see fire_weather_ml_shadow.py's _render_png for why
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    counties = _county_geometries()
    category_by_fips = {fips: scored[fips]["category"] for fips in county_fips}
    counties["category"] = counties["fips"].map(category_by_fips)

    pixel_width, pixel_height, dpi = 2048, 1152, 144
    data_crs = ccrs.PlateCarree()
    map_crs = ccrs.LambertConformal(central_longitude=-92.45, central_latitude=38.3)
    fig = plt.figure(figsize=(pixel_width / dpi, pixel_height / dpi), dpi=dpi, facecolor="#E8E8E8")
    ax = fig.add_axes([0.04, 0.04, 0.70, 0.92], projection=map_crs)
    ax.set_extent((-95.8, -89.1, 35.8, 40.8), crs=data_crs)
    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])

    for category_id, color in enumerate(CATEGORY_COLORS):
        subset = counties[counties["category"] == category_id]
        if not subset.empty:
            ax.add_geometries(subset.geometry, crs=data_crs, facecolor=color, edgecolor="none", zorder=6)
    no_data = counties[counties["category"].isna()]
    if not no_data.empty:
        ax.add_geometries(no_data.geometry, crs=data_crs, facecolor=NODATA_COLOR, edgecolor="none", zorder=6)
    ax.add_geometries(counties.geometry, crs=data_crs, edgecolor="#444444", facecolor="none", linewidth=0.7, zorder=9)

    state_shp = Path(__file__).resolve().parent.parent / "maps" / "shapefiles" / "MO_State_Boundary" / "MO_State_Boundary.shp"
    if state_shp.exists():
        import geopandas as gpd
        state = gpd.read_file(state_shp).to_crs("EPSG:4326")
        ax.add_geometries(state.geometry, crs=data_crs, edgecolor="#111111", facecolor="none", linewidth=1.5, zorder=10)

    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=color, label=label) for color, label in zip(CATEGORY_COLORS, CATEGORY_LABELS)]
    legend_handles.append(Patch(facecolor=NODATA_COLOR, label="No data"))
    ax.legend(handles=legend_handles, loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=10, frameon=False)

    fig.text(0.5, 0.965, NOT_FOR_OPERATIONS_LABEL, fontsize=22, fontweight="bold",
             ha="center", va="top", color="#B00000")
    fig.text(0.98, 0.92, "Fire Weather Severity Index (Shadow)", fontsize=22, fontweight="bold", ha="right", va="top")
    fig.text(0.98, 0.865, f"Model: {MODEL_TYPE}  Version: {bundle.get('version') or 'unknown'}",
             fontsize=14, ha="right", va="top")
    fig.text(
        0.77, 0.66,
        "A continuous, numeric\n"
        "fire-weather danger score,\n"
        "mapped to these 5 categories\n"
        "via calibrated score\n"
        "percentiles - NOT the same\n"
        "computation as the public\n"
        "if/then rule-based Fire\n"
        "Danger category shown\n"
        "elsewhere on this site.\n\n"
        "Experimental output, scored\n"
        "for internal comparison only.\n"
        "It is NOT reviewed, validated,\n"
        "or approved for any\n"
        "operational decision.",
        fontsize=11, ha="left", va="top", linespacing=1.5, color="#444444",
    )

    fd, temporary = tempfile.mkstemp(prefix=".fire_weather_index_shadow.", suffix=".png", dir=out_path.parent)
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


def _update_manifest(image_path: Path, bundle: Dict, category_counts: Dict, score_summary: Dict, generated_at: str) -> None:
    """Mirrors fire_weather_ml_shadow.py::_update_manifest's shape, under a distinct product key."""
    from services.beta_products import BETA_ROOT, load_manifest, save_manifest

    manifest = load_manifest()
    manifest.setdefault("products", {})["fire_weather_index_shadow"] = {
        "kind": "raster-preview",
        "preview": str(image_path.relative_to(BETA_ROOT)).replace("\\", "/"),
        "not_for_operations": True,
        "model_type": MODEL_TYPE,
        "model_version": bundle.get("version"),
        "generated_at": generated_at,
        "category_counts": category_counts,
        "score_summary": score_summary,
    }
    save_manifest(manifest)


def score_for_forecast(
    run_id: str,
    valid_local_date: str,
    county_fips: List[str],
    weather_rows: Dict[str, Dict],
    bundle_dir: Optional[Path] = None,
    evidence_root: Optional[Path] = None,
) -> bool:
    """Scores one forecast run's county-day weather rows and writes an
    immutable evidence file. weather_rows keyed by county_fips, same shape
    risk_fusion_hook.py already builds (rh_min_afternoon, wind_kts_max,
    vpd_kpa_max, precip_24h_mm). Never raises; returns False on any failure
    (including shadow being disabled)."""
    if not diagnostics()["enabled"]:
        return False
    try:
        bundle = load_bundle(bundle_dir)
        scored = {fips: score_county_day(bundle, weather_rows[fips]) for fips in county_fips}
        scores = [scored[fips]["score"] for fips in county_fips if scored[fips]["score"] is not None]
        categories = [scored[fips]["category"] for fips in county_fips if scored[fips]["category"] is not None]
        category_counts = {label: categories.count(i) for i, label in enumerate(CATEGORY_LABELS)}
        score_summary = ({"mean": float(np.mean(scores)), "min": float(np.min(scores)), "max": float(np.max(scores))}
                        if scores else {})

        record = {
            "run_id": str(run_id),
            "recorded_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "valid_local_date": valid_local_date,
            "bundle_checksum": bundle["bundle_checksum"],
            "county_fips": list(county_fips),
            "score": [scored[fips]["score"] for fips in county_fips],
            "category": [scored[fips]["category"] for fips in county_fips],
        }
        root = Path(evidence_root or EVIDENCE_ROOT)
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{run_id}.fire_weather_index_score.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(record, stream, indent=2)

        # Best-effort: a plotting/manifest failure must never count against
        # the shadow's own health/auto-disable tracking - the scoring and
        # evidence write above are the part that actually matters.
        image_path = None
        try:
            image_path = _image_path()
            _render_png(county_fips, scored, bundle, image_path)
            _update_manifest(image_path, bundle, category_counts, score_summary, record["recorded_at"])
            print(f"fire_weather_index shadow map written to: {image_path.resolve()}")
        except Exception as render_error:
            import logging
            logging.getLogger(__name__).warning(
                "fire_weather_index shadow graphic/manifest failed (non-fatal): %s", render_error)

        _state.update(
            consecutive_failures=0, last_error=None, healthy=True,
            last_success=record["recorded_at"],
            runs=_state["runs"] + 1, successful_runs=_state.get("successful_runs", 0) + 1,
            counties_scored=len(county_fips),
            bundle_checksum=bundle["bundle_checksum"],
            model_version=bundle.get("version"),
            last_score_summary=score_summary,
            last_category_counts=category_counts,
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
