"""Render and publish the NOAA precipitation graphics to R2."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))
load_dotenv(PROJECT_DIR / ".env")

from maps.precipitation_graphics import generate_graphics  # noqa: E402

LOG = logging.getLogger("publish_precipitation_graphics")


def _get_r2_client_and_bucket():
    import upload_cdn

    credentials = (upload_cdn.R2_ACCESS_KEY_ID, upload_cdn.R2_SECRET_ACCESS_KEY, upload_cdn.R2_ACCOUNT_ID)
    if not all(credentials):
        raise RuntimeError("R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, and R2_ACCOUNT_ID are required")
    return upload_cdn.get_r2_client(), os.getenv("R2_BUCKET", upload_cdn.BUCKET_NAME)


def render(stems=None) -> list[Path]:
    """Render the graphics to images/ without touching R2 (CPU-heavy: the scheduler runs this in its process pool)."""
    return generate_graphics(stems=stems) if stems else generate_graphics()


def publish(stems=None) -> list[str]:
    # Render all products before connecting to R2 so source/render errors never
    # replace a previously published CDN image.
    return upload(render(stems))


def upload(outputs) -> list[str]:
    """Upload rendered graphics (and their WebP thumbnails) to latest/ on R2."""
    from services.cdn_thumbnails import thumbnails_enabled

    client, bucket = _get_r2_client_and_bucket()
    # The small WebP next to each PNG (only when CDN_THUMBNAILS is on), so both change together.
    upload_thumbnail = None
    if thumbnails_enabled():
        import upload_cdn
        upload_thumbnail = upload_cdn._upload_thumbnail
    published = []
    for path in outputs:
        key = f"latest/{path.name}"
        client.upload_file(
            str(path),
            bucket,
            key,
            ExtraArgs={"ContentType": "image/png", "CacheControl": "public,max-age=300,stale-while-revalidate=300"},
        )
        if upload_thumbnail:
            upload_thumbnail(client, path)
        published.append(key)
        LOG.info("Published %s to %s/%s", path.name, bucket, key)
    return published


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        publish()
    except Exception:
        LOG.exception("Precipitation graphics publish failed; next scheduled run will retry")
        raise
