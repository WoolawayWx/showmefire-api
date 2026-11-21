"""The building blocks of a feature article, as validated data.

An article is an ordered list of blocks, stored as JSON and drawn by the website (and, later, the app). There is no raw HTML anywhere:
text is plain, with a tiny inline markup the reader draws itself (`**bold**`, `*italic*`, `[text](https://link)`), so nothing an
author types can run as a script. URLs are checked here and again where they are drawn.

Block types: heading, paragraph, image, quote, callout, list, table, video, weather_map, button, divider.
"""
from __future__ import annotations

import re
from typing import Annotated, Any, Dict, List, Literal, Optional, Union
from urllib.parse import urlparse

from pydantic import BaseModel, Field, TypeAdapter, field_validator

MAX_BLOCKS = 200
TEXT_MAX = 8000
SHORT_MAX = 300
YOUTUBE_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
# Image files are uploaded to our CDN (services/post_media.py); other https hosts are allowed so a figure can be linked.
ALLOWED_TONES = ("info", "tip", "warning", "danger")


def check_url(value: str, *, allow_relative: bool = False) -> str:
    """An https URL (or, for links, a path on this site). Anything else (javascript:, data:, http:) is refused."""
    value = str(value or "").strip()
    if allow_relative and re.fullmatch(r"/(?!/)[^\s]*", value):
        return value
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc or any(ch.isspace() for ch in value):
        raise ValueError("must be an https:// link" + (" or a path on this site starting with /" if allow_relative else ""))
    return value


class _Block(BaseModel):
    model_config = {"extra": "forbid"}


class Heading(_Block):
    type: Literal["heading"]
    level: Literal[2, 3] = 2
    text: str = Field(min_length=1, max_length=SHORT_MAX)


class Paragraph(_Block):
    type: Literal["paragraph"]
    text: str = Field(min_length=1, max_length=TEXT_MAX)


class Image(_Block):
    type: Literal["image"]
    url: str
    alt: str = Field(min_length=1, max_length=SHORT_MAX)  # required: a picture without alt text is unreadable to screen readers
    caption: str = Field(default="", max_length=SHORT_MAX)
    credit: str = Field(default="", max_length=SHORT_MAX)

    _url = field_validator("url")(lambda cls, value: check_url(value))


class Quote(_Block):
    type: Literal["quote"]
    text: str = Field(min_length=1, max_length=2000)
    attribution: str = Field(default="", max_length=SHORT_MAX)


class Callout(_Block):
    type: Literal["callout"]
    tone: Literal["info", "tip", "warning", "danger"] = "info"
    title: str = Field(default="", max_length=SHORT_MAX)
    text: str = Field(min_length=1, max_length=2000)


class ListBlock(_Block):
    type: Literal["list"]
    style: Literal["bullet", "number"] = "bullet"
    items: List[str] = Field(min_length=1, max_length=60)

    @field_validator("items")
    @classmethod
    def clean_items(cls, items: List[str]) -> List[str]:
        cleaned = [str(item).strip() for item in items]
        if any(not item or len(item) > 1000 for item in cleaned):
            raise ValueError("list items must be 1-1000 characters")
        return cleaned


class Table(_Block):
    type: Literal["table"]
    header: List[str] = Field(default_factory=list, max_length=12)
    rows: List[List[str]] = Field(min_length=1, max_length=100)
    caption: str = Field(default="", max_length=SHORT_MAX)

    @field_validator("rows")
    @classmethod
    def rows_fit(cls, rows: List[List[str]]) -> List[List[str]]:
        for row in rows:
            if not 1 <= len(row) <= 12 or any(len(str(cell)) > 500 for cell in row):
                raise ValueError("table rows need 1-12 cells of at most 500 characters")
        return [[str(cell) for cell in row] for row in rows]


class Video(_Block):
    type: Literal["video"]
    provider: Literal["youtube"] = "youtube"
    id: str
    title: str = Field(min_length=1, max_length=SHORT_MAX)
    caption: str = Field(default="", max_length=SHORT_MAX)

    @field_validator("id")
    @classmethod
    def youtube_id(cls, value: str) -> str:
        value = str(value).strip()
        if not YOUTUBE_ID.fullmatch(value):
            raise ValueError("must be the 11-character YouTube video ID, not a link")
        return value


class WeatherMap(_Block):
    """One of the weather maps in core/map_catalog.json, shown at its live address so it stays current."""
    type: Literal["weather_map"]
    key: str = Field(min_length=1, max_length=80)
    caption: str = Field(default="", max_length=SHORT_MAX)


class Button(_Block):
    type: Literal["button"]
    label: str = Field(min_length=1, max_length=80)
    url: str

    _url = field_validator("url")(lambda cls, value: check_url(value, allow_relative=True))


class Divider(_Block):
    type: Literal["divider"]


Block = Annotated[
    Union[Heading, Paragraph, Image, Quote, Callout, ListBlock, Table, Video, WeatherMap, Button, Divider],
    Field(discriminator="type"),
]
_BLOCKS = TypeAdapter(List[Block])

BLOCK_TYPES = ("heading", "paragraph", "image", "quote", "callout", "list", "table", "video", "weather_map", "button", "divider")


def validate_blocks(raw: Any) -> List[Dict[str, Any]]:
    """The blocks as clean dicts, or ValueError (with a readable message naming the block) when one is invalid."""
    if not isinstance(raw, list):
        raise ValueError("blocks must be a list")
    if len(raw) > MAX_BLOCKS:
        raise ValueError(f"an article can have at most {MAX_BLOCKS} blocks")
    from pydantic import ValidationError

    try:
        return [block.model_dump() for block in _BLOCKS.validate_python(raw)]
    except ValidationError as error:
        first = error.errors()[0]
        location = first.get("loc") or ()
        position = next((part for part in location if isinstance(part, int)), None)
        where = f"block {position + 1}" if position is not None else "blocks"
        field = next((str(part) for part in reversed(location) if isinstance(part, str) and part not in BLOCK_TYPES), "")
        message = str(first.get("msg", "invalid")).replace("Value error, ", "")
        raise ValueError(f"{where}{' (' + field + ')' if field else ''}: {message}") from error


def _strip_inline(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return re.sub(r"[*_]{1,2}", "", text)


def plain_text(blocks: List[Dict[str, Any]]) -> str:
    """The article's words with markup removed (for the summary, search and reading time)."""
    parts: List[str] = []
    for block in blocks:
        kind = block.get("type")
        if kind in ("heading", "paragraph"):
            parts.append(_strip_inline(block["text"]))
        elif kind == "quote":
            parts.append(_strip_inline(block["text"]))
        elif kind == "callout":
            parts += [block.get("title", ""), _strip_inline(block["text"])]
        elif kind == "list":
            parts += [_strip_inline(item) for item in block["items"]]
        elif kind == "table":
            parts += [cell for row in block["rows"] for cell in row]
    return " ".join(part for part in parts if part).strip()


def reading_minutes(blocks: List[Dict[str, Any]]) -> int:
    return max(1, round(len(plain_text(blocks).split()) / 220))


def make_summary(blocks: List[Dict[str, Any]], limit: int = 200) -> str:
    text = plain_text(blocks)
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",;:.") + "…"
