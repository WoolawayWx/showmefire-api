"""Oklahoma-Mesonet-style U.S. Drought Monitor map for Missouri.

The Monitor is published once a week (Thursday mornings, valid the prior
Tuesday), but maps.sh runs every script in this folder every 15 minutes. So
this script first makes one cheap HEAD request and compares the file's
Last-Modified/ETag with the last one it rendered; it only downloads the ~18 MB
national file, renders, and uploads when a new weekly map is actually out.
"""
import io
import json
import sys
import time
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path

import geopandas as gpd
import requests
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.append(str(PROJECT_DIR))

from maps import station_plot as sp
from maps import station_products as spx

USDM_URL = "https://droughtmonitor.unl.edu/data/json/usdm_current.json"
STATUS_DIR = PROJECT_DIR / "data" / "current_conditions" / "status"
IMAGES_DIR = PROJECT_DIR / "images"
OUT_PNG = IMAGES_DIR / "mo-drought.png"
STATUS_PATH = STATUS_DIR / "drought_monitor.json"
SIGNATURE_PATH = STATUS_DIR / "drought_monitor.signature"
HEADERS = {"User-Agent": "ShowMeFire/1.0 graphics@showmefire.org"}
# Generous box around Missouri so the national polygons are clipped before drawing.
CLIP_BOUNDS = (-98.0, 34.5, -88.0, 42.0)

# Official U.S. Drought Monitor category colors.
CATEGORIES = {
    0: ("D0", "Abnormally Dry", "#ffff00"),
    1: ("D1", "Moderate Drought", "#fcd37f"),
    2: ("D2", "Severe Drought", "#ffaa00"),
    3: ("D3", "Extreme Drought", "#e60000"),
    4: ("D4", "Exceptional Drought", "#730000"),
}


def _signature() -> tuple[str, str]:
    """(signature, last_modified) from a HEAD request; changes only on a new release."""
    response = requests.head(USDM_URL, headers=HEADERS, timeout=30, allow_redirects=True)
    response.raise_for_status()
    last_modified = response.headers.get("Last-Modified", "")
    return f"{response.headers.get('ETag', '')}|{last_modified}", last_modified


def _valid_date(last_modified: str) -> str:
    """The Monitor is valid the Tuesday before its Thursday release."""
    try:
        released = parsedate_to_datetime(last_modified).astimezone(spx.now_central().tzinfo)
    except (TypeError, ValueError):
        return "Latest weekly release"
    days_back = (released.weekday() - 1) % 7 or 7
    return (released - timedelta(days=days_back)).strftime("%b %d, %Y")


def render(last_modified: str) -> None:
    started = time.time()
    response = requests.get(USDM_URL, headers=HEADERS, timeout=120)
    response.raise_for_status()
    frame = gpd.read_file(io.BytesIO(response.content)).set_crs("EPSG:4326", allow_override=True)
    frame = frame.clip(CLIP_BOUNDS)
    # Mask to the state outline so nothing is drawn outside Missouri.
    state = sp._missouri_geometry()
    if state is not None:
        frame = frame.clip(state)
    frame = frame[~frame.geometry.is_empty]

    fig, ax, data_crs, _map_crs, _dpi = sp.generate_basemap()
    present = set()
    for dm, row_geometry in sorted(zip(frame["DM"].astype(int), frame.geometry), key=lambda item: item[0]):
        color = CATEGORIES.get(dm, ("D?", "Drought", "#d1d5db"))[2]
        ax.add_geometries([row_geometry], crs=data_crs, facecolor=color, edgecolor="none", zorder=7)
        present.add(dm)

    handles = [Patch(facecolor=color, edgecolor="#555555", linewidth=0.4, label=f"{code}  {name}")
               for code, name, color in CATEGORIES.values()]
    legend = fig.legend(
        handles=handles, title="Drought Intensity", loc="upper right", bbox_to_anchor=(0.99, 0.38),
        bbox_transform=fig.transFigure, fontsize=12, title_fontsize=13, frameon=False,
        labelspacing=0.9, handlelength=1.6, handleheight=1.3,
    )
    legend.set_zorder(20)

    valid = _valid_date(last_modified)
    sp.add_branding(
        fig, ax, title="Missouri Drought Monitor", subtitle=f"Valid: {valid}",
        description=(
            "Data Source: U.S. Drought Monitor\n(NDMC, USDA, NOAA)\n\n"
            "Updated weekly on Thursdays.\nDrought categories reflect\nlong-term dryness and impacts."
        ),
    )
    status = {
        "product": "drought_monitor", "status": "updated", "valid_date": valid,
        "categories_present": [CATEGORIES[dm][0] for dm in sorted(present) if dm in CATEGORIES],
        "last_update": spx.now_central().strftime("%Y-%m-%d %H:%M CT"),
        "runtime_sec": round(time.time() - started, 2),
    }
    sp.save_and_record_status(fig, OUT_PNG, STATUS_PATH, status)
    print(f"Drought Monitor map updated, valid {valid} ({status['runtime_sec']:.2f}s)")


def main() -> None:
    signature, last_modified = _signature()
    try:
        if OUT_PNG.is_file() and SIGNATURE_PATH.read_text() == signature:
            print("Drought Monitor unchanged since last render; skipping.")
            return
    except OSError:
        pass
    render(last_modified)
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    SIGNATURE_PATH.write_text(signature)
    try:
        from scripts.upload_cdn import upload_to_cdn
        upload_to_cdn([OUT_PNG], [f"latest/{OUT_PNG.name}"], ["image/png"], ["max-age=300"])
    except Exception as exc:  # noqa: BLE001 - maps.sh's own upload pass will also try
        print(f"Drought Monitor CDN upload failed: {exc}")


if __name__ == "__main__":
    main()
