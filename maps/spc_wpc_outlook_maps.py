"""Oklahoma-Mesonet-style national outlook maps for Missouri: SPC severe weather (categorical, tornado, wind and hail
probabilities, each with its conditional-intensity areas hatched on the same map, as on SPC's own graphics), SPC fire
weather (and dry thunderstorms) and the WPC excessive rainfall outlook, all for Day 1.

Same look as maps/drought_monitor.py and maps/advisory_category_maps.py: the Missouri basemap, polygons clipped to the
state, legend at the right, title / valid time / "Data Source" block, one PNG per product in images/ named
mo-noaa-outlook-<key>.png (maps.sh uploads new images to the CDN; the website and the app show them from there).

maps.sh runs every script in this folder every 15 minutes, but an outlook changes only a few times a day. So each
product is fetched, compared with a hash of what was last drawn, and only redrawn (and pushed to the CDN) when the
outlook actually changed. A source that is down is reported and skipped; the other products still update.

Colors: SPC supplies its own fill color on every polygon, and NOAA's fire weather service defines its classes, so those
are used as published. The WPC rainfall outlook carries no colors, so the standard green / yellow / red / magenta
scale is used.
"""
import argparse
import hashlib
import json
import sys
import textwrap
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import geopandas as gpd
import requests
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_DIR))

from maps import station_plot as sp

# Not "mo-outlook-day*": that is the Missouri fire weather outlook (maps/outlookgraphic.py), a different product that the API
# also serves with its own cache headers. These are NOAA's national outlooks, drawn for Missouri.
FILE_PREFIX = "mo-noaa-outlook"
STATUS_DIR = PROJECT_DIR / "data" / "current_conditions" / "status"
IMAGES_DIR = PROJECT_DIR / "images"
HEADERS = {"User-Agent": "ShowMeFire/1.0 graphics@showmefire.org"}
CENTRAL = ZoneInfo("America/Chicago")
# Generous box around Missouri so the national polygons are clipped before drawing (as drought_monitor.py does).
CLIP_BOUNDS = (-98.0, 34.5, -88.0, 42.0)

SPC = "https://www.spc.noaa.gov/products/outlook"
FIREWX = "https://mapservices.weather.noaa.gov/vector/rest/services/fire_weather/SPC_firewx/MapServer"
QUERY = "query?where=1%3D1&outFields=*&returnGeometry=true&outSR=4326&f=geojson"

SPC_SOURCE = "NOAA Storm Prediction Center"
PROBABILITY_NOTE = ("Chance of {hazard} within 25 miles of a point. "
                    "Hatching: how strong it could be IF it occurs (level 3 is the highest).")

# Fallbacks for when a polygon carries no label or color of its own.
SPC_CATEGORIES = {
    2: ("General Thunderstorms", "#C1E9C1"), 3: ("Marginal Risk (1 of 5)", "#66C57D"), 4: ("Slight Risk (2 of 5)", "#F6F67B"),
    5: ("Enhanced Risk (3 of 5)", "#E6B366"), 6: ("Moderate Risk (4 of 5)", "#E67E7E"), 8: ("High Risk (5 of 5)", "#FF7CFF"),
}
# Straight from NOAA's own SPC fire weather map service (the layer's renderer: value, label, fill).
FIRE_CATEGORIES = {5: ("Elevated", "#E69800"), 8: ("Critical", "#FF0000"), 10: ("Extremely Critical", "#E600A9")}
DRY_THUNDERSTORM = {5: ("Isolated Dry Thunderstorms", "#732600"), 8: ("Scattered Dry Thunderstorms", "#FF0000")}
ERO_CATEGORIES = {1: ("Marginal Risk (at least 5%)", "#2E9E4F"), 2: ("Slight Risk (at least 15%)", "#FFE600"),
                  3: ("Moderate Risk (at least 40%)", "#FF2D1A"), 4: ("High Risk (at least 70%)", "#FF00FF")}
CIG_HATCH = {1: "///", 2: "xxx", 3: "+++"}


@dataclass(frozen=True)
class Product:
    key: str                 # file is mo-noaa-outlook-<key>.png
    title: str               # map title
    url: str
    kind: str                # categorical | probability | cig | fire | ero
    source: str              # "Data Source" line
    note: str                # what the map shows, in the description block
    categories: dict | None = None
    overlay_url: str | None = None   # the hazard's conditional-intensity (CIG) areas, hatched over the probabilities
    day: str = "Day 1"               # how the website and the app group it: Day 1 | Day 2 | Day 3 | Days 4-8
    hazard: str = "severe"           # severe | fire | flood


FIRE_EXPER = "https://www.spc.noaa.gov/products/exper/fire_wx"
DAY4_8 = "https://www.spc.noaa.gov/products/exper/day4-8"
ERO = "https://www.wpc.ncep.noaa.gov/exper/eromap/geojson"
FLOOD_SOURCE = "NOAA Weather Prediction Center"
CAT_NOTE = "Highest severe thunderstorm risk category for the day."
FIRE_NOTE = "Where dry, windy conditions favor the spread of wildfires."
DRYT_NOTE = "Where thunderstorms with little rain could start fires."
FIRE_LATER_NOTE = "Where strong wind with low humidity favors the spread of wildfires."
DRYT_LATER_NOTE = "Where dry thunderstorms could start fires."
ERO_NOTE = "Chance of rainfall causing flash flooding within 25 miles of a point."
DAY_LABEL = {1: "Day 1", 2: "Day 2", 3: "Day 3"}


def _severe_day(day: int) -> list[Product]:
    """Day 1 and Day 2: categorical plus tornado, wind and hail probabilities, each with its intensity (CIG) overlay."""
    label, base = DAY_LABEL[day], f"{SPC}/day{day}otlk"
    products = [Product(f"day{day}-severe", f"Missouri SPC {label} Severe Weather Outlook", f"{base}_cat.nolyr.geojson", "categorical",
                        SPC_SOURCE, CAT_NOTE, SPC_CATEGORIES, day=label)]
    for key, name, hazard, cig in (("tornado", "Tornado", "a tornado", "cigtorn"),
                                   ("wind", "Damaging Wind", "damaging wind (58 mph or more)", "cigwind"),
                                   ("hail", "Hail", "1 inch or larger hail", "cighail")):
        products.append(Product(f"day{day}-{key}", f"Missouri SPC {label} {name} Outlook", f"{base}_{ 'torn' if key == 'tornado' else key }.nolyr.geojson",
                                "probability", SPC_SOURCE, PROBABILITY_NOTE.format(hazard=hazard),
                                overlay_url=f"{base}_{cig}.nolyr.geojson", day=label))
    return products


def _build_products() -> tuple:
    products: list[Product] = []
    # Severe weather. Days 1-2 have a category and a probability map per hazard; Day 3 has a category and one "any severe"
    # probability; Days 4-8 have one probability outlook per day (SPC outlines areas of 15% or more).
    products += _severe_day(1) + _severe_day(2)
    products.append(Product("day3-severe", "Missouri SPC Day 3 Severe Weather Outlook", f"{SPC}/day3otlk_cat.nolyr.geojson", "categorical",
                            SPC_SOURCE, CAT_NOTE, SPC_CATEGORIES, day="Day 3"))
    products.append(Product("day3-severe-prob", "Missouri SPC Day 3 Severe Weather Probability", f"{SPC}/day3otlk_prob.nolyr.geojson",
                            "probability", SPC_SOURCE,
                            PROBABILITY_NOTE.format(hazard="any severe weather (tornado, damaging wind or large hail)"),
                            overlay_url=f"{SPC}/day3otlk_cigprob.nolyr.geojson", day="Day 3"))
    for day in range(4, 9):
        products.append(Product(f"day{day}-severe", f"Missouri SPC Day {day} Severe Weather Outlook", f"{DAY4_8}/day{day}prob.nolyr.geojson",
                                "probability", SPC_SOURCE,
                                "Chance of severe weather within 25 miles of a point. SPC outlines areas of 15% or more.",
                                day="Days 4-8"))
    # Fire weather. Days 1-2 come from NOAA's map service (Elevated / Critical / Extreme and dry thunderstorms); Days 3-8 from
    # SPC's own GeoJSON files (Days 4-8: wind and low humidity only).
    products.append(Product("day1-fire", "Missouri SPC Day 1 Fire Weather Outlook", f"{FIREWX}/1/{QUERY}", "fire", SPC_SOURCE,
                            FIRE_NOTE, FIRE_CATEGORIES, day="Day 1", hazard="fire"))
    products.append(Product("day1-drythunder", "Missouri SPC Day 1 Dry Thunderstorm Outlook", f"{FIREWX}/2/{QUERY}", "fire", SPC_SOURCE,
                            DRYT_NOTE, DRY_THUNDERSTORM, day="Day 1", hazard="fire"))
    products.append(Product("day2-fire", "Missouri SPC Day 2 Fire Weather Outlook", f"{FIREWX}/4/{QUERY}", "fire", SPC_SOURCE,
                            FIRE_NOTE, FIRE_CATEGORIES, day="Day 2", hazard="fire"))
    products.append(Product("day2-drythunder", "Missouri SPC Day 2 Dry Thunderstorm Outlook", f"{FIREWX}/5/{QUERY}", "fire", SPC_SOURCE,
                            DRYT_NOTE, DRY_THUNDERSTORM, day="Day 2", hazard="fire"))
    products.append(Product("day3-fire", "Missouri SPC Day 3 Fire Weather Outlook", f"{FIRE_EXPER}/day3fw_windrhcat.nolyr.geojson",
                            "categorical", SPC_SOURCE, FIRE_LATER_NOTE, day="Day 3", hazard="fire"))
    products.append(Product("day3-drythunder", "Missouri SPC Day 3 Dry Thunderstorm Outlook", f"{FIRE_EXPER}/day3fw_drytcat.nolyr.geojson",
                            "categorical", SPC_SOURCE, DRYT_LATER_NOTE, day="Day 3", hazard="fire"))
    for day in range(4, 9):
        products.append(Product(f"day{day}-fire", f"Missouri SPC Day {day} Fire Weather Outlook",
                                f"{FIRE_EXPER}/day{day}fw_windrhcat.nolyr.geojson", "categorical", SPC_SOURCE, FIRE_LATER_NOTE,
                                day="Days 4-8", hazard="fire"))
    # Flooding: the WPC excessive rainfall outlook, Days 1-5.
    for day in range(1, 6):
        products.append(Product(f"day{day}-rainfall", f"Missouri WPC Day {day} Excessive Rainfall Outlook", f"{ERO}/Day{day}_Latest.geojson",
                                "ero", FLOOD_SOURCE, ERO_NOTE, ERO_CATEGORIES,
                                day=DAY_LABEL.get(day, "Days 4-8"), hazard="flood"))
    return tuple(products)


PRODUCTS = _build_products()
BY_KEY = {product.key: product for product in PRODUCTS}
CIG = Product("cig", "", "", "cig", "", "")      # only used to read the hatched intensity areas (see classify)


def png_path(product: Product) -> Path:
    return IMAGES_DIR / f"{FILE_PREFIX}-{product.key}.png"


# --- fetching and change detection ----------------------------------------------------------------------------------
def fetch_url(url: str) -> bytes:
    response = requests.get(url, headers=HEADERS, timeout=45)
    response.raise_for_status()
    return response.content


def signature(payload: bytes) -> str:
    """Hash of the outlook as published; changes only when a new outlook is issued."""
    return hashlib.sha1(payload).hexdigest()


# --- reading the polygons ---------------------------------------------------------------------------------------------
def _prop(properties: dict, *names, default=None):
    """A property by any of several spellings (SPC writes DN/LABEL, NOAA's service writes dn/valid)."""
    lowered = {str(key).lower(): value for key, value in properties.items()}
    for name in names:
        value = lowered.get(name.lower())
        if value not in (None, ""):
            return value
    return default


def _int(value, default=0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_time(value) -> datetime | None:
    """UTC times as the three sources write them: ISO with offset, 'YYYYMMDDHHMM', or 'YYYY-MM-DD HH:MM:SS'."""
    if value in (None, ""):
        return None
    text = str(value).strip()
    for pattern in ("%Y%m%d%H%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text, pattern).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _clock(moment: datetime) -> str:
    return f"{moment:%I:%M %p}".lstrip("0")


def _stamp(moment: datetime) -> str:
    return f"{moment:%b} {moment.day}, {_clock(moment)}"


def valid_text(properties: dict) -> tuple[str, str | None]:
    """('Valid: Nov 20, 7:00 PM - 6:00 AM CT', 'Issued: Nov 20, 6:41 PM CT'), local time. Either part may be unknown."""
    start = parse_time(_prop(properties, "VALID_ISO", "START_TIME", "valid"))
    end = parse_time(_prop(properties, "EXPIRE_ISO", "END_TIME", "expire"))
    issued = parse_time(_prop(properties, "ISSUE_ISO", "ISSUE_TIME", "issue"))
    valid = "Valid: current outlook"
    if start and end:
        start, end = start.astimezone(CENTRAL), end.astimezone(CENTRAL)
        valid = f"Valid: {_stamp(start)} – {_clock(end) if start.date() == end.date() else _stamp(end)} CT"
    elif end:
        valid = f"Valid until {_stamp(end.astimezone(CENTRAL))} CT"
    return valid, (f"Issued: {_stamp(issued.astimezone(CENTRAL))} CT" if issued else None)


def read_polygons(payload: bytes) -> gpd.GeoDataFrame:
    """Every polygon with its properties, as a frame with the fields lower-cased. No geometry gives an empty frame."""
    collection = json.loads(payload)
    features = [f for f in collection.get("features", []) if f.get("geometry")]
    if not features:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs="EPSG:4326")
    frame = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")
    frame.columns = [column if column == "geometry" else str(column).lower() for column in frame.columns]
    frame["geometry"] = frame.geometry.buffer(0)             # SPC polygons can be self-touching; clipping needs valid ones
    return frame


def first_properties(payload: bytes) -> dict:
    """Properties of the first feature, geometry or not: a quiet day still says when the outlook is valid."""
    features = json.loads(payload).get("features", [])
    return dict(features[0].get("properties") or {}) if features else {}


def clip_to_missouri(frame: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if frame.empty:
        return frame
    frame = frame.clip(CLIP_BOUNDS)
    state = sp._missouri_geometry()
    if state is not None and not frame.empty:
        frame = frame.clip(state)
    return frame[~frame.geometry.is_empty]


# --- what each polygon means ----------------------------------------------------------------------------------------------
def classify(product: Product, row) -> tuple[int, str, str]:
    """(level, legend label, color) for one polygon. level 0 means 'no risk': the polygon is not drawn."""
    level = _int(_prop(row, "dn"))
    if product.kind == "ero":
        level = _int(_prop(row, "dn")) or next((n for n, (name, _) in ERO_CATEGORIES.items()
                                               if name.split(" Risk")[0].lower() in str(_prop(row, "outlook", default="")).lower()), 0)
        name, color = ERO_CATEGORIES.get(level, (str(_prop(row, "outlook", default="Risk area")), "#9CA3AF"))
        return level, name, color
    if product.kind in ("fire", "categorical"):
        name, color = (product.categories or {}).get(level, (None, None))
        label = _prop(row, "label2") or _prop(row, "label") or name or f"Outlook area ({level})"   # Days 3-8 fire: only LABEL
        return level, str(label), _prop(row, "fill", default=color) or "#9CA3AF"
    if product.kind == "cig":
        return level, f"Level {level}" + (" (highest)" if level >= 3 else ""), "#000000"
    # Probability: DN is the percent. LABEL is not safe to show (Day 3 writes it as the fraction "0.05").
    return level, f"{level}%", _prop(row, "fill", default="#9CA3AF")


# --- drawing --------------------------------------------------------------------------------------------------------------
def draw_product(product: Product, payload: bytes, out_png: Path, status_path: Path, overlay_payload: bytes | None = None) -> dict:
    started = time.time()
    properties = first_properties(payload)
    frame = clip_to_missouri(read_polygons(payload))
    fig, ax, data_crs, _map_crs, _dpi = sp.generate_basemap()

    entries: dict[int, tuple[str, str]] = {}
    rows = sorted((classify(product, row) + (row["geometry"],) for _, row in frame.iterrows()), key=lambda item: item[0])
    for level, label, color, geometry in rows:
        if level <= 0:
            continue
        # Opaque: outlook areas are nested (the Slight area lies inside the Marginal one), and a see-through fill would
        # blend each level with the one beneath it so the map no longer matches the legend's colors.
        ax.add_geometries([geometry], crs=data_crs, facecolor=color, edgecolor="#555555", alpha=1.0, linewidth=1.2, zorder=7)
        entries.setdefault(level, (label, color))

    # The conditional-intensity areas go on the same map, hatched over the probabilities, as on SPC's own graphics.
    hatched: dict[int, str] = {}
    if overlay_payload is not None:
        overlay = clip_to_missouri(read_polygons(overlay_payload))
        for level, label, _color, geometry in sorted((classify(CIG, row) + (row["geometry"],) for _, row in overlay.iterrows()),
                                                     key=lambda item: item[0]):
            if level <= 0:
                continue
            ax.add_geometries([geometry], crs=data_crs, facecolor="none", edgecolor="black", hatch=CIG_HATCH.get(level, "///"),
                              linewidth=1.4, alpha=0.8, zorder=8)
            hatched.setdefault(level, label)

    legend_kwargs = dict(loc="upper right", bbox_transform=fig.transFigure, fontsize=12, title_fontsize=13, frameon=False,
                         labelspacing=0.9, handlelength=1.6, handleheight=1.3)
    if entries:
        handles = [Patch(facecolor=color, edgecolor="#555555", linewidth=0.6, label=textwrap.fill(label, 38))
                   for _level, (label, color) in sorted(entries.items())]
        # Side by side with the intensity legend when there is one, so the two never run into the description above.
        anchor_x = 0.865 if hatched else 0.99
        legend = fig.legend(handles=handles, title="Risk Level", bbox_to_anchor=(anchor_x, 0.38), **legend_kwargs)
        legend.set_zorder(20)
    if hatched:
        handles = [Patch(facecolor="white", edgecolor="black", hatch=CIG_HATCH.get(level), linewidth=0.6, label=label)
                   for level, label in sorted(hatched.items())]
        legend = fig.legend(handles=handles, title="Intensity", bbox_to_anchor=(0.99, 0.38), **legend_kwargs)
        legend.set_zorder(20)
    status_line = None if (entries or hatched) else "No areas in this outlook\ncover Missouri right now."

    valid, issued = valid_text(properties)
    # The legend sits under this block, so it stays short: source, what the map shows, when it was issued.
    lines = [f"Data Source: {product.source}", "", textwrap.fill(product.note, 40), "", issued, "" if status_line else None, status_line]
    description = "\n".join(line for line in lines if line is not None)       # None drops a line; "" keeps a blank spacer
    sp.add_branding(fig, ax, title=product.title, subtitle=valid, description=description,
                    update_schedule="Redrawn whenever a new outlook is issued")
    status = {
        "product": f"outlook_{product.key}", "status": "updated", "valid": valid, "issued": issued,
        "levels_over_missouri": [entries[level][0] for level in sorted(entries)],
        "intensity_levels_over_missouri": [hatched[level] for level in sorted(hatched)],
        "last_update": datetime.now(CENTRAL).strftime("%Y-%m-%d %H:%M CT"),
        "runtime_sec": round(time.time() - started, 2),
    }
    sp.save_and_record_status(fig, out_png, status_path, status)
    return status


def refresh(products=PRODUCTS, force: bool = False, fetch_fn=fetch_url, images_dir: Path | None = None,
            status_dir: Path | None = None) -> tuple[list[Path], list[str]]:
    """Redraw each product whose outlook changed. Returns (PNGs written, keys that failed)."""
    images_dir, status_dir = images_dir or IMAGES_DIR, status_dir or STATUS_DIR
    status_dir.mkdir(parents=True, exist_ok=True)
    changed, failed = [], []
    for product in products:
        out_png = images_dir / f"{FILE_PREFIX}-{product.key}.png"
        sig_path = status_dir / f"outlook_{product.key}.signature"
        try:
            payload = fetch_fn(product.url)
            overlay_payload = fetch_fn(product.overlay_url) if product.overlay_url else None
            digest = signature(payload + (overlay_payload or b""))
            try:
                if not force and out_png.is_file() and sig_path.read_text() == digest:
                    print(f"{product.key}: outlook unchanged; skipping.")
                    continue
            except OSError:
                pass
            status = draw_product(product, payload, out_png, status_dir / f"outlook_{product.key}.json", overlay_payload)
            sig_path.write_text(digest)
            changed.append(out_png)
            print(f"{product.key}: updated ({status['valid']}; {len(status['levels_over_missouri'])} level(s) over Missouri, "
                  f"{status['runtime_sec']:.1f}s)")
        except Exception as error:  # noqa: BLE001 - one source being down must not stop the others
            failed.append(product.key)
            print(f"{product.key}: FAILED ({type(error).__name__}: {str(error)[:160]})")
    return changed, failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--force", action="store_true", help="redraw even if the outlook has not changed")
    parser.add_argument("--only", action="append", choices=sorted(BY_KEY), help="just this product (repeatable)")
    args = parser.parse_args()
    products = tuple(BY_KEY[key] for key in args.only) if args.only else PRODUCTS
    changed, failed = refresh(products, force=args.force)
    if changed:
        try:
            from scripts.upload_cdn import upload_to_cdn
            upload_to_cdn(changed, [f"latest/{path.name}" for path in changed], ["image/png"] * len(changed),
                          ["max-age=300"] * len(changed))
        except Exception as exc:  # noqa: BLE001 - maps.sh's own upload pass will also try
            print(f"Outlook map CDN upload failed: {exc}")
    if failed and len(failed) == len(products):
        sys.exit(1)             # nothing could be fetched at all: let maps.sh log the failure


if __name__ == "__main__":
    main()
