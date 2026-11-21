"""Feature articles and the newsletter web archive.

Public (no sign-in, cached a minute):
    GET /api/articles                       published articles, newest first (?limit, ?offset, ?tag), plus the tag list
    GET /api/articles/{slug}                one article with its blocks, and related articles
    GET /api/newsletters                    sent newsletters on the web archive, newest first
    GET /api/newsletters/{id}               one newsletter's cleaned HTML, with the previous and next one

Admin (see core/admin_auth/permissions.py): staff write drafts (`articles.write`); owner and admin publish, unpublish, edit
what is live and delete it (`articles.publish`), and hide or restore a newsletter on the web (`newsletter_archive.manage`).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field

from core.admin_auth import permissions
from services import articles, newsletter_web
from services.article_blocks import BLOCK_TYPES

router = APIRouter(tags=["articles"])
WRITE = permissions.ARTICLES_WRITE
PUBLISH = permissions.ARTICLES_PUBLISH
ARCHIVE = permissions.NEWSLETTER_ARCHIVE_MANAGE


class ArticleIn(BaseModel):
    """What the editor sends. Field-level checks (blocks, URLs, tags) live in services.articles so the messages are readable."""
    title: str = Field(default="", max_length=400)
    slug: str = Field(default="", max_length=120)
    subtitle: str = Field(default="", max_length=600)
    summary: str = Field(default="", max_length=1000)
    author_name: str = Field(default="", max_length=300)
    hero: Optional[Dict[str, Any]] = None
    tags: List[str] = Field(default_factory=list, max_length=40)
    blocks: List[Dict[str, Any]] = Field(default_factory=list, max_length=300)
    seo_title: str = Field(default="", max_length=400)
    seo_description: str = Field(default="", max_length=800)


class HiddenIn(BaseModel):
    hidden: bool


def _public_headers(response: Response) -> None:
    response.headers["Cache-Control"] = "public, max-age=60"


def _no_store(response: Response) -> None:
    response.headers["Cache-Control"] = "no-store"


def _fail(error: Exception) -> HTTPException:
    if isinstance(error, (articles.NotFound, LookupError)):
        return HTTPException(status_code=404, detail="Not found")
    return HTTPException(status_code=400, detail=str(error))


# ------------------------------------------------------------------------------------------------------------ public

@router.get("/api/articles")
def public_articles(response: Response, limit: int = 12, offset: int = 0, tag: Optional[str] = None):
    _public_headers(response)
    items, total = articles.list_public(limit=limit, offset=offset, tag=tag)
    return {"success": True, "articles": [articles.public_view(item, with_body=False) for item in items], "total": total,
            "tags": articles.public_tags()}


@router.get("/api/articles/{slug}")
def public_article(slug: str, response: Response):
    _public_headers(response)
    try:
        article = articles.get_public(slug)
    except articles.NotFound:
        raise HTTPException(status_code=404, detail="Article not found")
    return {"success": True, "article": articles.public_view(article),
            "related": [articles.public_view(item, with_body=False) for item in articles.related_public(article)]}


@router.get("/api/newsletters")
def public_newsletters(response: Response, limit: int = 20, offset: int = 0):
    _public_headers(response)
    items, total = newsletter_web.list_public(limit=limit, offset=offset)
    return {"success": True, "newsletters": items, "total": total}


@router.get("/api/newsletters/{newsletter_id}")
def public_newsletter(newsletter_id: int, response: Response):
    _public_headers(response)
    item = newsletter_web.get_public(newsletter_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Newsletter not found")
    return {"success": True, "newsletter": item}


# ------------------------------------------------------------------------------------------------------------- admin

@router.get("/api/admin/articles/editor-config", dependencies=[Depends(permissions.require_permission(WRITE))])
def admin_editor_config(response: Response):
    """What the editor needs to build its menus: block types and the weather maps an article can embed."""
    _no_store(response)
    from services.mobile_maps import build_catalog

    maps = [{"key": item["key"], "title": item["title"], "group": group["title"], "thumbUrl": item["thumbUrl"]}
            for group in build_catalog()["groups"] for section in group["sections"] for item in section["maps"]]
    return {"success": True, "blockTypes": list(BLOCK_TYPES), "maps": maps, "tones": ["info", "tip", "warning", "danger"]}


@router.get("/api/admin/articles", dependencies=[Depends(permissions.require_permission(WRITE))])
def admin_list_articles(response: Response, status: Optional[str] = None):
    _no_store(response)
    rows = []
    for article in articles.list_admin(status):
        view = articles.admin_view(article)
        view.pop("blocks", None)  # the list stays light; open an article for its blocks
        rows.append(view)
    return {"success": True, "articles": rows}


@router.post("/api/admin/articles", status_code=201)
def admin_create_article(payload: ArticleIn, email: str = Depends(permissions.require_permission(WRITE))):
    try:
        return {"success": True, "article": articles.admin_view(articles.create(payload.model_dump(), email))}
    except articles.ArticleError as error:
        raise _fail(error)


@router.get("/api/admin/articles/{article_id}", dependencies=[Depends(permissions.require_permission(WRITE))])
def admin_get_article(article_id: int, response: Response):
    _no_store(response)
    try:
        return {"success": True, "article": articles.admin_view(articles.get_admin(article_id))}
    except articles.NotFound as error:
        raise _fail(error)


def _can_publish() -> bool:
    user = permissions.authorize_user(None, WRITE)
    return permissions.has(user, PUBLISH)


@router.put("/api/admin/articles/{article_id}")
def admin_update_article(article_id: int, payload: ArticleIn, email: str = Depends(permissions.require_permission(WRITE))):
    try:
        current = articles.get_admin(article_id)
        if current["status"] == "published" and not _can_publish():
            # Staff can write drafts; what the public can read is changed only by someone who can publish.
            raise HTTPException(status_code=403, detail="This article is live. Ask an admin to edit it, or unpublish it first.")
        return {"success": True, "article": articles.admin_view(articles.update(article_id, payload.model_dump(), email))}
    except (articles.NotFound, articles.ArticleError) as error:
        raise _fail(error)


@router.post("/api/admin/articles/{article_id}/publish")
def admin_publish_article(article_id: int, email: str = Depends(permissions.require_permission(PUBLISH))):
    try:
        return {"success": True, "article": articles.admin_view(articles.publish(article_id, email))}
    except (articles.NotFound, articles.ArticleError) as error:
        raise _fail(error)


@router.post("/api/admin/articles/{article_id}/unpublish")
def admin_unpublish_article(article_id: int, email: str = Depends(permissions.require_permission(PUBLISH))):
    try:
        return {"success": True, "article": articles.admin_view(articles.unpublish(article_id, email))}
    except articles.NotFound as error:
        raise _fail(error)


@router.delete("/api/admin/articles/{article_id}")
def admin_delete_article(article_id: int, email: str = Depends(permissions.require_permission(WRITE))):
    try:
        if articles.get_admin(article_id)["status"] == "published" and not _can_publish():
            raise HTTPException(status_code=403, detail="This article is live. Ask an admin to unpublish or delete it.")
        articles.delete(article_id, email)
    except articles.NotFound as error:
        raise _fail(error)
    return {"success": True}


@router.get("/api/admin/newsletter-archive", dependencies=[Depends(permissions.require_permission(ARCHIVE))])
def admin_newsletter_archive(response: Response):
    _no_store(response)
    return {"success": True, "newsletters": newsletter_web.list_admin()}


@router.put("/api/admin/newsletter-archive/{newsletter_id}")
def admin_set_newsletter_hidden(newsletter_id: int, payload: HiddenIn, email: str = Depends(permissions.require_permission(ARCHIVE))):
    try:
        return {"success": True, "newsletter": newsletter_web.set_hidden(newsletter_id, payload.hidden, email)}
    except LookupError as error:
        raise _fail(error)
