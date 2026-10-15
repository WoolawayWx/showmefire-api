"""Small WebP thumbnails for the map images pushed to the CDN.

The maps are 2048x1152 PNGs of 170-750 KB, but the website shows them in a slot a few hundred pixels
wide, so every phone downloads ~10-20x more than it can display. For each PNG uploaded to latest/
(scripts/upload_cdn.py) this makes an 800px-wide WebP next to it, named `<name>-sm.webp`
(`mo-rh.png` -> `mo-rh-sm.webp`, typically 15-35 KB). The website uses it for the in-page picture and
loads the full PNG only when someone opens the zoom view.

Purely additive and OFF by default: nothing about the existing PNGs changes, and the thumbnail is
not created, uploaded or archived unless CDN_THUMBNAILS is switched on. Pillow only: no boto3 here,
so it can be tested without the CDN client.

Environment:
  CDN_THUMBNAILS           1/true/yes/on to enable (default off)
  CDN_THUMBNAIL_WIDTH      pixels wide (default 800, 200-2048)
  CDN_THUMBNAIL_QUALITY    WebP quality (default 75, 30-95)
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from PIL import Image

logger = logging.getLogger(__name__)

THUMB_SUFFIX = "-sm.webp"
DEFAULT_WIDTH = 800
DEFAULT_QUALITY = 75


def thumbnails_enabled() -> bool:
    return os.getenv("CDN_THUMBNAILS", "").strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(high, int(os.getenv(name, str(default)))))
    except ValueError:
        return default


def thumbnail_name(filename: str) -> Optional[str]:
    """`mo-rh.png` -> `mo-rh-sm.webp`; None for anything that is not a PNG."""
    name = Path(filename).name
    if not name.lower().endswith(".png"):
        return None
    return name[:-4] + THUMB_SUFFIX


def make_thumbnail(source: Path, out_dir: Path, width: Optional[int] = None,
                   quality: Optional[int] = None) -> Optional[Path]:
    """Write the thumbnail for a PNG into ``out_dir`` and return its path.

    Returns None (and logs) if the source is not a readable PNG: a thumbnail is optional and must
    never get in the way of the real upload. Images already narrower than ``width`` are not
    enlarged. Transparency is kept when the PNG has any.
    """
    source = Path(source)
    out_name = thumbnail_name(source.name)
    if out_name is None or not source.is_file():
        return None
    width = width or _int_env("CDN_THUMBNAIL_WIDTH", DEFAULT_WIDTH, 200, 2048)
    quality = quality or _int_env("CDN_THUMBNAIL_QUALITY", DEFAULT_QUALITY, 30, 95)
    try:
        with Image.open(source) as image:
            has_alpha = image.mode in ("RGBA", "LA") or "transparency" in image.info
            image = image.convert("RGBA" if has_alpha else "RGB")
            if image.width > width:
                image = image.resize((width, max(1, round(image.height * width / image.width))), Image.LANCZOS)
            destination = Path(out_dir) / out_name
            image.save(destination, "WEBP", quality=quality, method=4)
        return destination
    except Exception as error:  # corrupt/truncated PNG, disk full, ...
        logger.warning("Could not make a thumbnail for %s: %s", source.name, error)
        return None
