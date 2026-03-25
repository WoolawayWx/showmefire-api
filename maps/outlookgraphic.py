import json
import logging
import argparse
from datetime import datetime
from io import BytesIO
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
import textwrap
from zoneinfo import ZoneInfo

import cairosvg
import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.font_manager as font_manager
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR))

from core.database import get_latest_forecast

STATUS_FILE = PROJECT_DIR / "status.json"

OUTLOOK_COLOR = "#FFA500"
OUTLINE_COLOR = "#B35A00"
SUPPORTED_DAYS = {2, 3}
CENTRAL_TZ = ZoneInfo("America/Chicago")
NO_OUTLOOK_BANNER_TEXT = "No outlook created for this day, or not high enough confidence/probability."


def central_now() -> datetime:
    return datetime.now(CENTRAL_TZ)


def format_central_timestamp(value: datetime | None = None) -> str:
    base = value.astimezone(CENTRAL_TZ) if value else central_now()
    return base.strftime("%Y-%m-%d %H:%M CT")


def parse_issue_time(issue_time: str | None) -> datetime | None:
    if issue_time is None:
        return None

    value = str(issue_time).strip()
    if not value:
        return None

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("issue_time must be a valid ISO datetime") from exc

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=CENTRAL_TZ)

    return parsed


def extract_payload_issue_time(payload: dict) -> datetime | None:
    features = payload.get("features") or []
    if not features:
        return None

    first_props = (features[0] or {}).get("properties") or {}
    raw_issue_time = first_props.get("issue_time")
    if not raw_issue_time:
        return None

    try:
        return parse_issue_time(str(raw_issue_time))
    except ValueError:
        return None


def normalize_day(day: int) -> int:
    try:
        normalized_day = int(day)
    except (TypeError, ValueError) as exc:
        raise ValueError("day must be an integer") from exc

    if normalized_day not in SUPPORTED_DAYS:
        raise ValueError("day must be 2 or 3")

    return normalized_day


def normalize_valid_date(valid_date: str | None) -> str:
    if valid_date is None or not str(valid_date).strip():
        return central_now().strftime("%Y-%m-%d")
    try:
        parsed = datetime.strptime(str(valid_date).strip(), "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("valid_date must be in YYYY-MM-DD format") from exc
    return parsed.strftime("%Y-%m-%d")


def published_outlook_file(day: int) -> Path:
    return PROJECT_DIR / "gis" / f"outlook_day{day}_published.geojson"


def output_image_file(day: int) -> Path:
    return PROJECT_DIR / "images" / f"mo-outlook-day{day}.png"


def output_webp_file(day: int) -> Path:
    return PROJECT_DIR / "images" / f"mo-outlook-day{day}.webp"


def log_file(day: int) -> Path:
    return PROJECT_DIR / "logs" / f"outlookgraphic_day{day}.log"


def setup_logger(day: int) -> logging.Logger:
    logger = logging.getLogger(f"outlookgraphic_day{day}")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        active_log_file = log_file(day)
        active_log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(active_log_file, maxBytes=5 * 1024 * 1024, backupCount=5)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)

    return logger


def generate_basemap():
    pixelw = 2048
    pixelh = 1152
    mapdpi = 144

    extent = (-95.8, -89.1, 35.8, 40.8)
    data_crs = ccrs.PlateCarree()
    map_crs = ccrs.LambertConformal(central_longitude=-92.45, central_latitude=38.3)

    figsize_width = pixelw / mapdpi
    figsize_height = pixelh / mapdpi

    fig = plt.figure(figsize=(figsize_width, figsize_height), dpi=mapdpi, facecolor="#E8E8E8")
    ax = plt.axes([0, 0, 1, 1], projection=map_crs)

    ax.set_frame_on(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_extent(extent, crs=data_crs)

    counties = gpd.read_file(SCRIPT_DIR / "shapefiles/MO_County_Boundaries/MO_County_Boundaries.shp")
    if counties.crs != data_crs.proj4_init:
        counties = counties.to_crs(data_crs.proj4_init)
    ax.add_geometries(counties.geometry, crs=data_crs, edgecolor="#B6B6B6", facecolor="none", linewidth=1, zorder=5)

    missouri_border = gpd.read_file(SCRIPT_DIR / "shapefiles/MO_State_Boundary/MO_State_Boundary.shp")
    if missouri_border.crs != data_crs.proj4_init:
        missouri_border = missouri_border.to_crs(data_crs.proj4_init)
    ax.add_geometries(missouri_border.geometry, crs=data_crs, edgecolor="#000000", facecolor="none", linewidth=1.5, zorder=8)

    # Keep map placement aligned with realtimefiredanger layout.
    ax.set_anchor("W")
    plt.subplots_adjust(left=0.05)

    return fig, ax, data_crs, mapdpi


def add_branding(fig, ax, day: int, valid_date: str, updated_time: datetime):
    font_paths = [
        str(PROJECT_DIR / "assets/Montserrat/static/Montserrat-Regular.ttf"),
        str(PROJECT_DIR / "assets/Plus_Jakarta_Sans/static/PlusJakartaSans-Regular.ttf"),
        str(PROJECT_DIR / "assets/Plus_Jakarta_Sans/static/PlusJakartaSans-Bold.ttf"),
    ]

    for font_path in font_paths:
        if Path(font_path).exists():
            font_manager.fontManager.addfont(font_path)
    plt.rcParams["font.family"] = "Montserrat"

    fig.text(
        0.99,
        0.97,
        f"Missouri Fire Weather Outlook - Day {day}",
        fontsize=26,
        fontweight="bold",
        ha="right",
        va="top",
        fontname="Plus Jakarta Sans",
    )
    fig.text(
        0.99,
        0.90,
        f"Valid For: {valid_date} | Issued: {format_central_timestamp(updated_time)}",
        fontsize=16,
        ha="right",
        va="top",
        fontname="Montserrat",
    )
    fig.text(0.02, 0.01, "ShowMeFire.org", fontsize=20, fontweight="bold", ha="left", va="bottom", fontname="Montserrat")

    svg_path = str(PROJECT_DIR / "assets/LightBackGroundLogo.svg")
    try:
        png_bytes = cairosvg.svg2png(url=svg_path)
        image = mpimg.imread(BytesIO(png_bytes), format="png")
        image_box = OffsetImage(image, zoom=0.03)
        logo = AnnotationBbox(image_box, (0.99, 0.01), frameon=False, xycoords="figure fraction", box_alignment=(1, 0))
        ax.add_artist(logo)
    except Exception:
        # Logo rendering is optional and should not block map generation.
        pass


def load_outlook_geojson(day: int) -> dict:
    published_file = published_outlook_file(day)
    if not published_file.exists():
        return {"type": "FeatureCollection", "features": [], "outlook_text": ""}

    with published_file.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if payload.get("type") != "FeatureCollection":
        return {"type": "FeatureCollection", "features": [], "outlook_text": ""}

    payload.setdefault("features", [])
    payload.setdefault("outlook_text", "")
    return payload

def wrap_paragraphs(text: str, width: int = 58) -> str:
    parts = []
    for p in str(text).splitlines():
        if not p.strip():
            parts.append("")
        else:
            parts.append(
                textwrap.fill(
                    p,
                    width=width,
                    break_long_words=False,
                    break_on_hyphens=False,
                )
            )
    return "\n".join(parts)

def get_forecast_discussion_text() -> str:
    try:
        latest_forecast = get_latest_forecast()
        if not latest_forecast:
            return ""
        return str(latest_forecast.get("discussion") or "").strip()
    except Exception:
        return ""


def render_outlook(fig, ax, data_crs, payload: dict, day: int):
    features = payload.get("features", [])

    if features:
        gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
        if not gdf.empty:
            missouri_border = gpd.read_file(SCRIPT_DIR / "shapefiles/MO_State_Boundary/MO_State_Boundary.shp")
            if missouri_border.crs != "EPSG:4326":
                missouri_border = missouri_border.to_crs("EPSG:4326")

            # Clip outlook polygons to Missouri so out-of-state geometry is masked.
            clipped = gpd.overlay(gdf, missouri_border[["geometry"]], how="intersection")
            if not clipped.empty:
                ax.add_geometries(
                    clipped.geometry,
                    crs=data_crs,
                    facecolor=OUTLOOK_COLOR,
                    edgecolor=OUTLINE_COLOR,
                    linewidth=1.3,
                    alpha=0.42,
                    zorder=7,
                )

    outlook_text = str(payload.get("outlook_text") or "").strip()
    forecast_discussion = get_forecast_discussion_text()
    feature_count = len(features)
    no_risk = feature_count == 0

    if no_risk:
        fig.text(
            0.5,
            0.50,
            wrap_paragraphs(NO_OUTLOOK_BANNER_TEXT, width=52),
            fontsize=23,
            fontweight="bold",
            ha="center",
            va="center",
            linespacing=1.3,
            fontname="Plus Jakarta Sans",
            bbox={
                "boxstyle": "round,pad=0.65",
                "facecolor": "#E6E6E6",
                "edgecolor": "#E65100",
                "linewidth": 3,
            },
            color="#434343",
        )
    elif forecast_discussion:
        synopsis = forecast_discussion
    elif outlook_text:
        synopsis = outlook_text
    else:
        synopsis = "15% Risk (Elevated/High Fire Weather Conditions)"

    if not no_risk:
        wrapped_discussion = wrap_paragraphs(f"Discussion: {synopsis}", width=58)

        fig.text(
            0.99,
            0.62,
            wrapped_discussion + "\n\n",
            fontsize=14,
            ha="right",
            va="top",
            linespacing=1.6,
            fontname="Montserrat",
        )

    fig.text(
        0.99,
        0.2,
        "ORANGE: 15% PROBABILITY OF ELEVATED FIRE WEATHER\n"
        "ISSUED BY: SHOW ME FIRE STAFF\n"
        "FOR MORE INFO: SHOWMEFIRE.ORG",
        fontsize=10,
        ha="right",
        va="top",
        linespacing=1.8,
        fontname="Montserrat",
        fontweight="bold", # Makes small text pop
        color="#333333"    # Dark grey is often cleaner than pure black
    )


def update_status(runtime_sec: float, feature_count: int, day: int, output_file: Path, output_webp: Path):
    if STATUS_FILE.exists():
        try:
            with STATUS_FILE.open("r", encoding="utf-8") as f:
                status = json.load(f)
        except json.JSONDecodeError:
            status = {}
    else:
        status = {}

    status[f"OutlookGraphicDay{day}"] = {
        "last_update": format_central_timestamp(),
        "status": "updated",
        "runtime_sec": round(runtime_sec, 2),
        "feature_count": feature_count,
        "day": day,
        "output": str(output_file),
        "output_webp": str(output_webp),
    }

    with STATUS_FILE.open("w", encoding="utf-8") as f:
        json.dump(status, f, indent=4)


def main():
    parser = argparse.ArgumentParser(description="Generate Missouri outlook graphics for day 2 or day 3")
    parser.add_argument("--day", type=int, default=2, help="Outlook day to render (2 or 3)")
    parser.add_argument("--valid-date", type=str, default=None, help="Outlook valid date in YYYY-MM-DD")
    parser.add_argument("--issue-time", type=str, default=None, help="Issue datetime in ISO format")
    args = parser.parse_args()
    day = normalize_day(args.day)
    valid_date = normalize_valid_date(args.valid_date)
    cli_issue_time = parse_issue_time(args.issue_time)

    active_output_file = output_image_file(day)
    active_output_webp_file = output_webp_file(day)
    start = datetime.now()
    logger = setup_logger(day)

    payload = load_outlook_geojson(day)
    payload_issue_time = extract_payload_issue_time(payload)
    updated_time = cli_issue_time or payload_issue_time or central_now()

    fig, ax, data_crs, mapdpi = generate_basemap()
    add_branding(fig, ax, day, valid_date, updated_time)
    render_outlook(fig, ax, data_crs, payload, day)

    active_output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(active_output_file, dpi=mapdpi, bbox_inches=None, pad_inches=0)
    fig.savefig(
        active_output_webp_file,
        dpi=mapdpi,
        bbox_inches=None,
        pad_inches=0,
        format="webp",
        pil_kwargs={"quality": 82},
    )
    plt.close(fig)

    runtime_sec = (datetime.now() - start).total_seconds()
    feature_count = len(payload.get("features", []))

    logger.info("Outlook day %s graphic updated at %s", day, format_central_timestamp())
    logger.info("Feature count: %s", feature_count)
    logger.info("Script runtime: %.2f seconds", runtime_sec)
    logger.info("Saved PNG: %s", active_output_file)
    logger.info("Saved WebP: %s", active_output_webp_file)

    update_status(runtime_sec, feature_count, day, active_output_file, active_output_webp_file)
    print(f"Outlook day {day} graphic updated: {active_output_file} | {active_output_webp_file}")


if __name__ == "__main__":
    main()
