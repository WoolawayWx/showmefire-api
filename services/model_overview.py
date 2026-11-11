"""One sheet of every county-scoring fire danger model for a day, side by side.

SHADOW ONLY - NOT FOR OPERATIONS. The companion of the fire escape overview (services/fire_escape/render.py):
that sheet shows the fire escape models, this one shows the others - the public rule, the Fire Weather Severity
Index (HRRR and the HRRR+RRFS blend), the two ensemble tracks and the risk fusion GLM.

It draws what the models already recorded, nothing new: the rows fire danger tracking keeps
(services/fire_danger_tracking.py, table fire_danger_model_scores), after asking it to ingest any evidence it
has not seen yet. For each model the run produced last for that valid date is shown, and a model with no run
for the day gets an empty panel that says so rather than yesterday's map.

Not on the sheet: models that only publish a grid (the ML risk score, the spread rate, the hybrid). They have
no county rows to draw.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from textwrap import fill
from typing import Optional

import numpy as np

from core import database
from core.fire_danger import CATEGORY_HEX_COLORS, CATEGORY_LABELS
from services import fire_danger_tracking as tracking
from services.fire_escape import render as style

logger = logging.getLogger(__name__)
IMAGE_FILENAME = "model_overview_latest.png"
MANIFEST_KEY = "model_overview"
# model tag, column drawn, scale, panel title. Scales: "level" = 0 Low .. 4 Extreme; the others are percent.
PANELS = (
    ("public_rule", "category", "level", "Public fire danger (rule)"),
    ("fire_weather_index", "category", "level", "Severity index, HRRR"),
    ("fire_weather_index:blend", "category", "level", "Severity index, HRRR + RRFS"),
    ("ensemble:members", "category", "level", "Ensemble, members track"),
    ("ensemble:ensprod", "category", "level", "Ensemble, REFS track"),
    ("ensemble:members", "probability", "elevated", "Ensemble: chance of Elevated or higher"),
    ("risk_fusion_glm", "probability", "any_fire", "Risk fusion: chance of any fire"),
)
SCALES = {"elevated": (5, 10, 20, 30, 40, 50, 60, 75, 90, 100), "any_fire": (1, 2, 5, 10, 15, 20, 30, 40, 60, 100)}
NOTE = ("The same day, as each model sees it.\n\n"
        "Levels are not one scale: the public rule and the\nensemble use weather thresholds; the severity\n"
        "index ranks the day against its own history\n(its top level is its top 5 % of days).\n\n"
        "The two right-hand maps are chances, in percent:\nthat fire danger reaches Elevated, and that any\n"
        "wildfire is reported in the county.\n\n"
        "Gray counties have no forecast data.\nA gray panel has no data for the day.")


def collect(day: date, db_path: Optional[Path] = None, ingest: bool = True) -> dict[str, dict]:
    """{model: {"run_id", "produced_at", "category": {fips: value}, "probability": {...}, "score": {...}}} for the
    run each model produced last for `day`. Models without a row that day are absent."""
    if ingest:
        try:
            tracking.ingest_scores(db_path)
        except Exception as exc:                                   # the sheet can still show what is already in the table
            logger.warning("model overview: ingest failed (non-fatal): %s", exc)
    conn = sqlite3.connect(db_path or database.get_db_path(), timeout=30)
    try:
        rows = conn.execute('''SELECT model, run_id, county_fips, score, category, probability, produced_at
                               FROM fire_danger_model_scores WHERE valid_date = ?''', (day.isoformat(),)).fetchall()
    except sqlite3.OperationalError:
        rows = []
    finally:
        conn.close()
    latest: dict[str, tuple] = {}
    for model, run_id, _fips, _score, _category, _probability, produced_at in rows:
        key = (str(produced_at or ""), str(run_id))
        if model not in latest or key > latest[model]:
            latest[model] = key
    out: dict[str, dict] = {}
    for model, run_id, fips, score, category, probability, produced_at in rows:
        if (str(produced_at or ""), str(run_id)) != latest[model]:
            continue
        entry = out.setdefault(model, {"run_id": run_id, "produced_at": produced_at, "category": {}, "probability": {}, "score": {}})
        for name, value in (("category", category), ("probability", probability), ("score", score)):
            if value is not None:
                entry[name][str(fips)] = float(value)
    return out


def panel_summary(values: dict, scale: str) -> str:
    """One line under a panel: how many counties at each level, or the highest and median chance."""
    numbers = np.array(list(values.values()), dtype=float)
    if not numbers.size:
        return "No data for this day"
    if scale == "level":
        counts = [(label, int((np.rint(numbers) == level).sum())) for level, label in enumerate(CATEGORY_LABELS)]
        return " | ".join(f"{label} {count}" for label, count in counts if count)
    return f"Highest {style.percent(float(numbers.max()))} ({style.odds(float(numbers.max()))}) | median {style.percent(float(np.median(numbers)))}"


def _draw_panel(ax, data_crs, values: dict, scale: str, counties):
    """Fill the counties and return (mappable, ticks, labels) for the colorbar."""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import BoundaryNorm, ListedColormap

    if scale == "level":
        cmap, norm = ListedColormap(CATEGORY_HEX_COLORS), BoundaryNorm(style.LEVEL_BINS, len(CATEGORY_HEX_COLORS))
        ticks, labels, low = [0, 1, 2, 3, 4], list(CATEGORY_LABELS), -np.inf
        to_plot = lambda value: float(np.clip(np.rint(value), 0, 4))
    else:
        levels = SCALES[scale]
        cmap = ListedColormap(style.PROBABILITY_COLORS)
        cmap.set_under(style.BELOW_SCALE_COLOR)
        norm = BoundaryNorm(levels, cmap.N)
        ticks, labels, low = list(levels[:-1]), [f"{v:g}%" for v in levels[:-1]], levels[0]
        to_plot = lambda value: 100.0 * value
    for row in counties.itertuples():
        value = values.get(row.fips)
        if value is None or not np.isfinite(value):
            color = "#d9d9d9"
        else:
            shown = to_plot(value)
            color = style.BELOW_SCALE_COLOR if shown < low else cmap(norm(shown))
        ax.add_geometries([row.geometry], crs=data_crs, facecolor=color, edgecolor="#b6b6b6", linewidth=0.3, alpha=0.9, zorder=7)
    return ScalarMappable(norm=norm, cmap=cmap), ticks, labels


def render(models: dict[str, dict], day: date, out_path: Path) -> Path:
    import cartopy.crs as ccrs
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data_crs = ccrs.PlateCarree()
    projection = ccrs.LambertConformal(central_longitude=-92.45, central_latitude=38.3)
    counties = style._county_shapes()
    state = gpd.read_file(style.API_ROOT / "maps" / "shapefiles" / "MO_State_Boundary" / "MO_State_Boundary.shp").to_crs("EPSG:4326")
    fig = plt.figure(figsize=(23, 11.5), dpi=110, facecolor="white")
    try:
        note = fig.add_subplot(2, 4, 1)
        note.axis("off")
        note.text(0.0, 0.95, f"Fire danger models\n{day:%A}\n{day:%B %d, %Y}", fontsize=18, fontweight="bold", va="top")
        note.text(0.0, 0.70, style.NOT_FOR_OPERATIONS, fontsize=12, fontweight="bold", color="#B00000", va="top")
        note.text(0.0, 0.60, NOTE, fontsize=10.5, va="top", linespacing=1.35)
        for position, (model, column, scale, title) in enumerate(PANELS, start=2):
            ax = fig.add_subplot(2, 4, position, projection=projection)
            ax.set_extent(style.EXTENT, crs=data_crs)
            ax.set_frame_on(False)
            values = (models.get(model) or {}).get(column) or {}
            mappable, ticks, labels = _draw_panel(ax, data_crs, values, scale, counties)
            ax.add_geometries(state.geometry, crs=data_crs, edgecolor="black", facecolor="none", linewidth=1.0, zorder=9)
            ax.set_title(fill(title, width=37), fontsize=12, fontweight="bold")
            bar = fig.colorbar(mappable, ax=ax, orientation="horizontal", pad=0.02, fraction=0.045, shrink=0.9,
                               **({} if scale == "level" else {"extend": "min", "spacing": "uniform"}))
            bar.set_ticks(ticks)
            bar.set_ticklabels(labels, fontsize=8)
            ax.text(0.5, -0.24, fill(panel_summary(values, scale), width=48), transform=ax.transAxes, ha="center", va="top", fontsize=8.5)
        fig.subplots_adjust(left=0.02, right=0.99, top=0.95, bottom=0.09, wspace=0.06, hspace=0.32)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=110)
    finally:
        plt.close(fig)
    return out_path


def publish(image: Path, models: dict[str, dict], day: date) -> None:
    """Copy the sheet into the Testbed images and list it in the Testbed manifest."""
    import shutil

    from services.beta_products import BETA_ROOT, load_manifest, save_manifest

    target = BETA_ROOT / "images" / IMAGE_FILENAME
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(image, target)
    shown = {model for model, _column, _scale, _title in PANELS}
    manifest = load_manifest()
    manifest.setdefault("products", {})[MANIFEST_KEY] = {
        "kind": "raster-preview", "not_for_operations": True, "preview": f"images/{IMAGE_FILENAME}",
        "valid_local_date": day.isoformat(), "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "models": {model: {"label": tracking.model_label(model), "run_id": models[model]["run_id"]} for model in sorted(shown & set(models))},
        "missing": sorted(shown - set(models)),
    }
    save_manifest(manifest)


def run(day: date, out_dir: Path, db_path: Optional[Path] = None, publish_to_testbed: bool = False) -> dict:
    """Draw the day's sheet. Never raises: returns {"status": "ok", ...} or {"status": "skipped", "reason": ...}."""
    try:
        models = collect(day, db_path)
        shown = {model for model, _column, _scale, _title in PANELS}
        if not shown & set(models):
            return {"status": "skipped", "reason": f"no model has county scores for {day}"}
        image = render(models, day, Path(out_dir) / f"{day:%Y%m%d}_model_overview.png")
        if publish_to_testbed:
            publish(image, models, day)
        return {"status": "ok", "image": str(image), "models": sorted(shown & set(models)), "missing": sorted(shown - set(models))}
    except Exception as exc:
        logger.warning("model overview for %s failed (non-fatal): %s", day, exc, exc_info=True)
        return {"status": "skipped", "reason": f"error: {exc}"}
