#!/usr/bin/env python3
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))

from core.database import get_latest_forecast
from services.discord_notifier import notify_forecast_ready

MAP_FILES = [
    "mo-forecastfiredanger.png",
    "mo-forecastfuelmoisture.png",
    "mo-forecastmaxwind.png",
    "mo-forecastminrh.png",
]


def _build_image_urls() -> list[str]:
    base_url = os.getenv("PUBLIC_API_BASE_URL", "https://api.showmefire.org").rstrip("/")
    cache_key = str(int(datetime.now(timezone.utc).timestamp()))

    urls: list[str] = []
    for image_name in MAP_FILES:
        image_path = BASE_DIR / "images" / image_name
        if image_path.exists():
            urls.append(f"{base_url}/images/{image_name}?v={cache_key}")
    return urls


def main() -> int:
    latest = get_latest_forecast()
    if not latest:
        print("No forecast available; skipping Discord completion post")
        return 0

    image_urls = _build_image_urls()
    if not image_urls:
        print("No forecast map images found; sending forecast event without image")

    sent = notify_forecast_ready(
        title=str(latest.get("title") or "Daily Forecast Update"),
        discussion=str(latest.get("discussion") or ""),
        valid_time=str(latest.get("valid_time") or ""),
        updated_at=str(latest.get("updated_at") or datetime.now(timezone.utc).isoformat()),
        url=os.getenv("FORECAST_PUBLIC_URL", "https://showmefire.org/forecasts"),
        image_urls=image_urls,
    )

    if sent:
        print(f"Forecast completion Discord event sent with {len(image_urls)} map image(s)")
    else:
        print("Forecast completion Discord event was not sent")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
