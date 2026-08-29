"""Generate the public Missouri fire-weather RSS feed."""

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import pytz
from feedgen.feed import FeedGenerator

from ai.ai_summary import generate_summary
from core.config import PUBLIC_DIR

logger = logging.getLogger(__name__)

MAPS = [
    {
        "title": "Fire Danger Assessment",
        "url": "https://api.showmefire.org/images/mo-realtimefiredanger.png",
        "id": "danger",
    },
    {
        "title": "Relative Humidity",
        "url": "https://api.showmefire.org/images/mo-rh.png",
        "id": "rh",
    },
    {
        "title": "Fuel Moisture",
        "url": "https://api.showmefire.org/images/mo-fuelmoisture.png",
        "id": "fuel",
    },
    {
        "title": "Sustained Winds",
        "url": "https://api.showmefire.org/images/mo-windfilmap.png",
        "id": "wind",
    },
]


def generate_rss_feed(add_summary: bool = False) -> str:
    fg = FeedGenerator()
    fg.title("Show Me Fire | Missouri Weather & Danger Maps")
    fg.description("Real-time fire weather analysis for Missouri")
    fg.link(href="https://api.showmefire.org/rss.xml", rel="self")

    now_utc = datetime.now(timezone.utc)
    now_central = now_utc.astimezone(pytz.timezone("US/Central"))
    valid_time = now_central.strftime("%H:%M CT")
    fg.lastBuildDate(now_utc)

    for map_info in MAPS:
        image_url = f"{map_info['url']}?t={int(now_utc.timestamp())}"
        entry = fg.add_entry()
        entry.title(f"{map_info['title']} - {valid_time}")
        entry.description(f'<img src="{image_url}" alt="{map_info["title"]}">')
        entry.link(href=map_info["url"])
        entry.guid(f"mo-map-{map_info['id']}", permalink=False)
        entry.pubDate(now_utc)
        entry.enclosure(image_url, 0, "image/png")

    if add_summary:
        try:
            summary_text = generate_summary()
        except Exception:
            logger.exception("RSS summary generation failed")
            summary_text = None
        if summary_text:
            entry = fg.add_entry()
            entry.title(f"Current Fire Weather Summary - {valid_time}")
            entry.description(summary_text)
            entry.link(href="https://api.showmefire.org/rss.xml")
            entry.guid("mo-summary", permalink=False)
            entry.pubDate(now_utc)

    return fg.rss_str(pretty=True).decode("utf-8")


def write_rss_feed(add_summary: bool = False) -> Path:
    """Write the feed to the path served by the API."""
    output_file = Path(PUBLIC_DIR) / "rss.xml"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        generate_rss_feed(add_summary=add_summary),
        encoding="utf-8",
    )
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate the public RSS feed.")
    parser.add_argument(
        "--add-summary",
        action="store_true",
        help="Include the numeric/AI-generated fire-weather summary.",
    )
    args = parser.parse_args()
    output_file = write_rss_feed(add_summary=args.add_summary)
    print(f"Successfully created {output_file} at {datetime.now().strftime('%H:%M:%S')}")
