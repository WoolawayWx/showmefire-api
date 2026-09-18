"""Browser-based renderer: screenshots the live-preview Vue page with headless Chromium.

This exists alongside graphic_renderer.py (matplotlib) rather than replacing it, so the
saved image is produced by the exact same GraphicsLivePreview.vue component a department
edits against - one styling source of truth instead of two renderers that can drift
apart. Switch it on with GRAPHICS_RENDERER=browser; the default keeps the matplotlib path.

Playwright is imported lazily so departments that never enable this path don't need the
package (or its Chromium download) installed.
"""
from __future__ import annotations

import asyncio
import json
import os
import secrets
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from core.database import get_db_path
from services.graphic_renderer import HEIGHT, WIDTH, _render_fingerprint, _render_sources

BROWSER_RENDERER_VERSION = "graphics-browser-v1"
RENDER_PAGE_URL = os.getenv("SMF_GRAPHICS_RENDER_URL", "https://showmefire.org/graphics/render")
RENDER_TIMEOUT_MS = int(os.getenv("SMF_GRAPHICS_RENDER_TIMEOUT_MS", "20000"))
RENDER_TOKEN_TTL_SECONDS = int(os.getenv("SMF_GRAPHICS_RENDER_TOKEN_TTL", "300"))
DEFAULT_CENTER = [-92.45, 38.343121]


@contextmanager
def _db():
    db = sqlite3.connect(get_db_path(), timeout=30)
    try:
        with db:
            yield db
    finally:
        db.close()


def _store_render_config(config: dict) -> str:
    """Save the resolved render config under a short-lived token instead of
    embedding it in the URL - Cloudflare Pages serves the render page from a
    different origin than the API, so the page fetches its config back by
    token rather than carrying a large base64 blob through the querystring."""
    payload = {
        "config": config,
        "center": config.get("center") or DEFAULT_CENTER,
        "zoom": config.get("zoom") or (7.5 if config["product_id"] == "mo_alerts" else 4.0),
        "departmentName": config.get("department_name", ""),
        "logoUrl": config.get("department_logo_url") or "",
    }
    token = secrets.token_urlsafe(24)
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=RENDER_TOKEN_TTL_SECONDS)
    with _db() as db:
        db.execute("DELETE FROM graphic_render_tokens WHERE expires_at <= CURRENT_TIMESTAMP")
        db.execute(
            "INSERT INTO graphic_render_tokens(token,payload_json,expires_at) VALUES (?,?,?)",
            (token, json.dumps(payload), expires_at.isoformat()),
        )
    return token


def render_graphic_browser(config: dict) -> dict:
    """Render one configured product by screenshotting the live-preview page.

    Safe to call from a worker thread (a fresh Chromium is launched and closed per
    call, matching the ProcessPoolExecutor-per-call model the matplotlib renderer
    uses); pool this if per-render browser launch latency (~100-200ms) matters.
    """
    from playwright.sync_api import sync_playwright

    product_ids, payloads, urls, frames = _render_sources(config)
    source_hash = _render_fingerprint(config, payloads)
    token = _store_render_config(config)
    url = f"{RENDER_PAGE_URL}?token={token}"

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            args=[
                "--use-gl=swiftshader", "--enable-webgl", "--ignore-gpu-blocklist",
                # Docker has no sandbox namespaces and a tiny default /dev/shm;
                # both crash Chromium unless disabled/redirected to /tmp.
                "--no-sandbox", "--disable-dev-shm-usage",
            ],
        )
        try:
            page = browser.new_page(viewport={"width": WIDTH, "height": HEIGHT})
            page.goto(url, wait_until="load", timeout=RENDER_TIMEOUT_MS)
            page.wait_for_function("window.__SMF_RENDER_READY__ === true", timeout=RENDER_TIMEOUT_MS)
            data = page.screenshot(type="png")
        finally:
            browser.close()

    return {
        "bytes": data,
        "source_fingerprint": source_hash,
        "source_urls": urls,
        "basemap_tiles": 0,
        "renderer_version": BROWSER_RENDERER_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


async def render_graphic_browser_async(config: dict) -> dict:
    return await asyncio.to_thread(render_graphic_browser, config)
