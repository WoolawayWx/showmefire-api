"""Feature articles: long-form pieces built from blocks (services/article_blocks.py), separate from the staff discussion `posts`.

Who may do what (core/admin_auth/permissions.py):
  * `articles.write` (owner, admin, staff): create articles and edit drafts.
  * `articles.publish` (owner, admin): publish, unpublish, delete a published article, and edit one that is live.
Staff therefore cannot change anything the public can see. The router enforces that; this module just does the work and writes the
editorial entries in the moderation log (subject types "article" / "newsletter").

Public read model (camelCase, what the website and later the app consume):
  {id, slug, title, subtitle, summary, author, tags, hero, blocks, readingMinutes, publishedAt, updatedAt, seoTitle, seoDescription}
`weather_map` blocks gain a `map` {title, url, thumbUrl} at read time so they always show the current image; a map that no longer
exists in the catalog is dropped from the page rather than breaking it.
"""
from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from core.database import get_db_path
from services import audit
from services.article_blocks import check_url, make_summary, plain_text, reading_minutes, validate_blocks

TITLE_MAX = 160
TAG_MAX = 30
MAX_TAGS = 8


class ArticleError(ValueError):
    """A problem the author can fix (bad field, cannot publish yet). The router turns it into a 400."""


class NotFound(LookupError):
    pass


# ----------------------------------------------------------------------------------------------------------- helpers

def slugify(text: str) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:80].strip("-") or "article"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path(), timeout=15)
    conn.row_factory = sqlite3.Row
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _iso(value: Optional[str]) -> Optional[str]:
    """SQLite's UTC 'YYYY-MM-DD HH:MM:SS' as an ISO 8601 UTC time the browser reads the same everywhere."""
    if not value:
        return None
    return str(value).replace(" ", "T") + ("Z" if not str(value).endswith("Z") else "")


def _unique_slug(conn: sqlite3.Connection, wanted: str, own_id: Optional[int] = None) -> str:
    base = slugify(wanted)
    candidate, n = base, 1
    while True:
        row = conn.execute("SELECT id FROM articles WHERE slug = ?", (candidate,)).fetchone()
        if row is None or row["id"] == own_id:
            return candidate
        n += 1
        candidate = f"{base}-{n}"


def map_lookup() -> Dict[str, Dict[str, str]]:
    """Every weather map in the catalog by key: {title, url, thumbUrl}."""
    from services.mobile_maps import build_catalog

    found: Dict[str, Dict[str, str]] = {}
    for group in build_catalog()["groups"]:
        for section in group["sections"]:
            for item in section["maps"]:
                found[item["key"]] = {"title": item["title"], "url": item["url"], "thumbUrl": item["thumbUrl"]}
    return found


def _clean_hero(raw: Any) -> Optional[Dict[str, str]]:
    if not raw:
        return None
    if not isinstance(raw, dict):
        raise ArticleError("hero must be an object with url and alt")
    url = str(raw.get("url") or "").strip()
    if not url:
        return None
    try:
        url = check_url(url)
    except ValueError as error:
        raise ArticleError(f"hero image: {error}") from error
    alt = str(raw.get("alt") or "").strip()
    if not alt:
        raise ArticleError("hero image needs alt text describing the picture")
    return {"url": url, "alt": alt[:300], "caption": str(raw.get("caption") or "").strip()[:300],
            "credit": str(raw.get("credit") or "").strip()[:300]}


def _clean_tags(raw: Any) -> List[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ArticleError("tags must be a list")
    tags: List[str] = []
    for tag in raw:
        value = re.sub(r"\s+", " ", str(tag).strip().lower())[:TAG_MAX]
        if value and value not in tags:
            tags.append(value)
    if len(tags) > MAX_TAGS:
        raise ArticleError(f"use at most {MAX_TAGS} tags")
    return tags


def clean_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate what an author sent. Raises ArticleError with a message fit to show next to the form."""
    title = re.sub(r"\s+", " ", str(data.get("title") or "")).strip()
    if not title:
        raise ArticleError("title is required")
    if len(title) > TITLE_MAX:
        raise ArticleError(f"title must be at most {TITLE_MAX} characters")
    try:
        blocks = validate_blocks(data.get("blocks") or [])
    except ValueError as error:
        raise ArticleError(str(error)) from error
    known = None
    for block in blocks:
        if block["type"] == "weather_map":
            known = known if known is not None else map_lookup()
            if block["key"] not in known:
                raise ArticleError(f"weather map {block['key']!r} is not in the map catalog")
    summary = re.sub(r"\s+", " ", str(data.get("summary") or "")).strip()[:400]
    return {
        "title": title,
        "subtitle": re.sub(r"\s+", " ", str(data.get("subtitle") or "")).strip()[:300],
        "summary": summary,
        "author_name": re.sub(r"\s+", " ", str(data.get("author_name") or data.get("author") or "")).strip()[:120],
        "hero": _clean_hero(data.get("hero")),
        "tags": _clean_tags(data.get("tags")),
        "blocks": blocks,
        "seo_title": str(data.get("seo_title") or data.get("seoTitle") or "").strip()[:160],
        "seo_description": str(data.get("seo_description") or data.get("seoDescription") or "").strip()[:300],
    }


def _decode(row: sqlite3.Row) -> Dict[str, Any]:
    def load(column: str, default: Any) -> Any:
        try:
            return json.loads(row[column]) if row[column] else default
        except json.JSONDecodeError:
            return default

    return {
        "id": row["id"], "slug": row["slug"], "title": row["title"], "subtitle": row["subtitle"], "summary": row["summary"],
        "author_name": row["author_name"], "hero": load("hero_json", None), "tags": load("tags_json", []),
        "blocks": load("blocks_json", []), "seo_title": row["seo_title"], "seo_description": row["seo_description"],
        "status": row["status"], "published_at": row["published_at"], "created_by": row["created_by"],
        "updated_by": row["updated_by"], "created_at": row["created_at"], "updated_at": row["updated_at"],
    }


def admin_view(article: Dict[str, Any]) -> Dict[str, Any]:
    """Everything, for the editor."""
    return {**article, "published_at": _iso(article["published_at"]), "created_at": _iso(article["created_at"]),
            "updated_at": _iso(article["updated_at"]), "reading_minutes": reading_minutes(article["blocks"]),
            "effective_summary": article["summary"] or make_summary(article["blocks"])}


def _resolve_maps(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not any(block["type"] == "weather_map" for block in blocks):
        return blocks
    maps = map_lookup()
    resolved = []
    for block in blocks:
        if block["type"] == "weather_map":
            found = maps.get(block["key"])
            if not found:
                continue
            block = {**block, "map": found}
        resolved.append(block)
    return resolved


def public_view(article: Dict[str, Any], *, with_body: bool = True) -> Dict[str, Any]:
    view = {
        "id": article["id"], "slug": article["slug"], "title": article["title"], "subtitle": article["subtitle"],
        "summary": article["summary"] or make_summary(article["blocks"]), "author": article["author_name"],
        "tags": article["tags"], "hero": article["hero"], "readingMinutes": reading_minutes(article["blocks"]),
        "publishedAt": _iso(article["published_at"]), "updatedAt": _iso(article["updated_at"]),
    }
    if with_body:
        view.update({"blocks": _resolve_maps(article["blocks"]), "seoTitle": article["seo_title"],
                     "seoDescription": article["seo_description"] or view["summary"]})
    return view


# ----------------------------------------------------------------------------------------------------------- reading

def get_admin(article_id: int) -> Dict[str, Any]:
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM articles WHERE id = ?", (article_id,)).fetchone()
    finally:
        conn.close()
    if row is None:
        raise NotFound("article not found")
    return _decode(row)


def list_admin(status: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = _connect()
    try:
        if status in ("draft", "published"):
            rows = conn.execute("SELECT * FROM articles WHERE status = ? ORDER BY updated_at DESC, id DESC", (status,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM articles ORDER BY updated_at DESC, id DESC").fetchall()
    finally:
        conn.close()
    return [_decode(row) for row in rows]


def list_public(limit: int = 12, offset: int = 0, tag: Optional[str] = None) -> Tuple[List[Dict[str, Any]], int]:
    """Published articles, newest first, and how many there are in all (for paging). `tag` filters to one tag."""
    limit = max(1, min(int(limit), 50))
    offset = max(0, int(offset))
    conn = _connect()
    try:
        rows = conn.execute("SELECT * FROM articles WHERE status = 'published' ORDER BY published_at DESC, id DESC").fetchall()
    finally:
        conn.close()
    articles = [_decode(row) for row in rows]
    if tag:
        wanted = tag.strip().lower()
        articles = [article for article in articles if wanted in article["tags"]]
    return articles[offset:offset + limit], len(articles)


def public_tags() -> List[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    for article in _all_published():
        for tag in article["tags"]:
            counts[tag] = counts.get(tag, 0) + 1
    return [{"tag": tag, "count": count} for tag, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))]


def _all_published() -> List[Dict[str, Any]]:
    conn = _connect()
    try:
        return [_decode(row) for row in conn.execute("SELECT * FROM articles WHERE status = 'published'").fetchall()]
    finally:
        conn.close()


def get_public(slug: str) -> Dict[str, Any]:
    conn = _connect()
    try:
        row = conn.execute("SELECT * FROM articles WHERE slug = ? AND status = 'published'", (slug,)).fetchone()
    finally:
        conn.close()
    if row is None:
        raise NotFound("article not found")
    return _decode(row)


def related_public(article: Dict[str, Any], limit: int = 3) -> List[Dict[str, Any]]:
    """Other published articles that share a tag, newest first (falls back to the newest others)."""
    others = [other for other in _all_published() if other["id"] != article["id"]]
    others.sort(key=lambda other: (other["published_at"] or "", other["id"]), reverse=True)
    shared = [other for other in others if set(other["tags"]) & set(article["tags"])]
    return (shared + [other for other in others if other not in shared])[:limit]


# ----------------------------------------------------------------------------------------------------------- writing

def _store(fields: Dict[str, Any]) -> Tuple:
    return (fields["title"], fields["subtitle"], fields["summary"], fields["author_name"],
            json.dumps(fields["hero"]) if fields["hero"] else None, json.dumps(fields["tags"]),
            json.dumps(fields["blocks"]), fields["seo_title"], fields["seo_description"])


def create(data: Dict[str, Any], actor: str) -> Dict[str, Any]:
    fields = clean_fields(data)
    conn = _connect()
    try:
        slug = _unique_slug(conn, str(data.get("slug") or "") or fields["title"])
        cursor = conn.execute(
            """INSERT INTO articles (slug, title, subtitle, summary, author_name, hero_json, tags_json, blocks_json,
                                     seo_title, seo_description, status, created_by, updated_by, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'draft', ?, ?, ?, ?)""",
            (slug, *_store(fields), actor, actor, _now(), _now()))
        article_id = cursor.lastrowid
        audit.record(conn, action="article_create", subject_type="article", subject_id=article_id, actor=actor,
                     after={"title": fields["title"], "slug": slug})
        conn.commit()
    finally:
        conn.close()
    return get_admin(article_id)


def update(article_id: int, data: Dict[str, Any], actor: str) -> Dict[str, Any]:
    current = get_admin(article_id)
    fields = clean_fields(data)
    conn = _connect()
    try:
        wanted_slug = str(data.get("slug") or "").strip()
        # A published article keeps its address unless someone deliberately renames it; a draft's address follows its title.
        slug = _unique_slug(conn, wanted_slug, article_id) if wanted_slug else current["slug"]
        conn.execute(
            """UPDATE articles SET slug = ?, title = ?, subtitle = ?, summary = ?, author_name = ?, hero_json = ?, tags_json = ?,
                                   blocks_json = ?, seo_title = ?, seo_description = ?, updated_by = ?, updated_at = ? WHERE id = ?""",
            (slug, *_store(fields), actor, _now(), article_id))
        audit.record(conn, action="article_edit", subject_type="article", subject_id=article_id, actor=actor,
                     before={"title": current["title"], "status": current["status"]}, after={"title": fields["title"], "slug": slug})
        conn.commit()
    finally:
        conn.close()
    return get_admin(article_id)


def publish(article_id: int, actor: str) -> Dict[str, Any]:
    article = get_admin(article_id)
    if not article["blocks"]:
        raise ArticleError("add some content before publishing")
    if not plain_text(article["blocks"]) and not article["hero"]:
        raise ArticleError("an article needs some text before it can be published")
    conn = _connect()
    try:
        conn.execute("UPDATE articles SET status = 'published', published_at = COALESCE(published_at, ?), updated_by = ?, updated_at = ? "
                     "WHERE id = ?", (_now(), actor, _now(), article_id))
        audit.record(conn, action="article_publish", subject_type="article", subject_id=article_id, actor=actor,
                     before={"status": article["status"]}, after={"status": "published", "slug": article["slug"]})
        conn.commit()
    finally:
        conn.close()
    return get_admin(article_id)


def unpublish(article_id: int, actor: str) -> Dict[str, Any]:
    article = get_admin(article_id)
    conn = _connect()
    try:
        conn.execute("UPDATE articles SET status = 'draft', updated_by = ?, updated_at = ? WHERE id = ?", (actor, _now(), article_id))
        audit.record(conn, action="article_unpublish", subject_type="article", subject_id=article_id, actor=actor,
                     before={"status": article["status"]}, after={"status": "draft"})
        conn.commit()
    finally:
        conn.close()
    return get_admin(article_id)


def delete(article_id: int, actor: str) -> None:
    article = get_admin(article_id)
    conn = _connect()
    try:
        conn.execute("DELETE FROM articles WHERE id = ?", (article_id,))
        audit.record(conn, action="article_delete", subject_type="article", subject_id=article_id, actor=actor,
                     before={"title": article["title"], "status": article["status"], "slug": article["slug"]})
        conn.commit()
    finally:
        conn.close()
