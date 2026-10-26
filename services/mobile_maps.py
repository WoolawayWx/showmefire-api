"""The weather-map catalog for the mobile app's "Weather maps" screens (GET /api/mobile/maps).

The same ~137 map images the website's /mesonet page lists, grouped into a handful of groups so the app can show a landing page of
buttons and one page of graphics per group. The list lives in core/map_catalog.json (one place to add a map for the app); this
module validates it once and adds each map's CDN URLs. Every map has a small WebP thumbnail next to its PNG (see
services/cdn_thumbnails.py) which the app uses for its grids, so a page of maps costs tens of KB per image, not hundreds.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from services.cdn_thumbnails import thumbnail_name, thumbnails_enabled
from services.mobile_content import PUBLIC_CDN_BASE_URL

CATALOG_PATH = Path(__file__).resolve().parent.parent / "core" / "map_catalog.json"
# The maps are rebuilt every 15 minutes; one cache-busting value per slot keeps phones and the CDN from serving a stale one.
VERSION_MINUTES = 15


@lru_cache(maxsize=1)
def load_catalog() -> dict:
    """The catalog file, validated: groups and sections are non-empty, map keys are unique and every file is a PNG."""
    catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    keys: set[str] = set()
    group_ids: set[str] = set()
    for group in catalog["groups"]:
        if group["id"] in group_ids:
            raise ValueError(f"Duplicate map group id {group['id']!r}")
        group_ids.add(group["id"])
        if not group["title"].strip() or not group["sections"]:
            raise ValueError(f"Map group {group['id']!r} needs a title and sections")
        for section in group["sections"]:
            if not section["maps"]:
                raise ValueError(f"Map section {section['id']!r} has no maps")
            for item in section["maps"]:
                if item["key"] in keys:
                    raise ValueError(f"Duplicate map key {item['key']!r}")
                keys.add(item["key"])
                if not item["title"].strip() or thumbnail_name(item["file"]) is None:
                    raise ValueError(f"Map {item['key']!r} needs a title and a .png file")
    return catalog


def version_token(now: datetime) -> str:
    minutes = now.hour * 60 + now.minute
    return f"{now:%Y%m%d}-{minutes // VERSION_MINUTES}"


def build_catalog(now: Optional[datetime] = None, *, cdn_base_url: str = PUBLIC_CDN_BASE_URL,
                  use_thumbnails: Optional[bool] = None) -> dict[str, Any]:
    """The response: each map with its full-size `url` and the `thumbUrl` the grids show (the same as `url` when the CDN has no thumbnails)."""
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    use_thumbnails = thumbnails_enabled() if use_thumbnails is None else use_thumbnails
    version = f"?v={version_token(now)}"
    base = cdn_base_url.rstrip("/")
    groups = []
    for group in load_catalog()["groups"]:
        sections = []
        for section in group["sections"]:
            maps = []
            for item in section["maps"]:
                thumb = thumbnail_name(item["file"]) if use_thumbnails else item["file"]
                maps.append({
                    "key": item["key"],
                    "title": item["title"],
                    "description": item.get("description", ""),
                    "url": f"{base}/{item['file']}{version}",
                    "thumbUrl": f"{base}/{thumb}{version}",
                })
            sections.append({"id": section["id"], "title": section["title"], "maps": maps})
        groups.append({
            "id": group["id"],
            "title": group["title"],
            "description": group.get("description", ""),
            "mapCount": sum(len(section["maps"]) for section in sections),
            "sections": sections,
        })
    return {"generatedAt": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "groups": groups}
