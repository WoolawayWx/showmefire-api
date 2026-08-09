import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from core.database import (
    create_comment,
    create_post,
    create_post_category,
    delete_comment,
    delete_post,
    get_post,
    list_post_tags,
    list_post_categories,
    list_posts,
    update_post,
)
from core.security import verify_token

logger = logging.getLogger(__name__)

router = APIRouter(tags=["posts"])


def _require_admin(token: Optional[str] = None) -> str:
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return email


def _normalize_tags(tags: Optional[List[str]]) -> List[str]:
    normalized: List[str] = []
    for raw in tags or []:
        tag = str(raw or "").strip().lower().replace(" ", "-")
        tag = "".join(c for c in tag if c.isalnum() or c in "-_")
        if tag and tag not in normalized:
            normalized.append(tag)
    return normalized


class PostCreate(BaseModel):
    title: str
    body: str
    author_name: str
    tags: List[str] = []
    excerpt: str = ""
    status: str = "published"
    category: str = "Field Notes"
    cover_image: Optional[str] = None
    seo_title: Optional[str] = None
    seo_description: Optional[str] = None
    body_format: str = "plain"

    @field_validator("title", "body", "author_name")
    @classmethod
    def _require_non_empty(cls, value: str) -> str:
        stripped = str(value or "").strip()
        if not stripped:
            raise ValueError("must not be empty")
        return stripped

    @field_validator("tags", mode="before")
    @classmethod
    def _normalize_tags_field(cls, value: Optional[List[str]]) -> List[str]:
        return _normalize_tags(value)


class PostUpdate(BaseModel):
    title: Optional[str] = None
    body: Optional[str] = None
    author_name: Optional[str] = None
    tags: Optional[List[str]] = None
    excerpt: Optional[str] = None
    status: Optional[str] = None
    category: Optional[str] = None
    cover_image: Optional[str] = None
    seo_title: Optional[str] = None
    seo_description: Optional[str] = None
    body_format: Optional[str] = None
    slug: Optional[str] = None

    @field_validator("title", "body")
    @classmethod
    def _strip_optional(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        stripped = str(value).strip()
        if not stripped:
            raise ValueError("must not be empty")
        return stripped

    @field_validator("tags", mode="before")
    @classmethod
    def _normalize_tags_field(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is None:
            return None
        return _normalize_tags(value)


class CommentCreate(BaseModel):
    author_name: str
    body: str

    @field_validator("author_name", "body")
    @classmethod
    def _require_non_empty(cls, value: str) -> str:
        stripped = str(value or "").strip()
        if not stripped:
            raise ValueError("must not be empty")
        return stripped


@router.get("/api/posts")
def public_list_posts(tag: Optional[str] = None, category: Optional[str] = None, limit: int = 50, offset: int = 0):
    """Return published discussion posts for the public blog."""
    try:
        safe_limit = min(max(limit, 1), 50)
        safe_offset = max(offset, 0)
        return {
            "posts": list_posts(tag=tag, category=category, status="published", limit=safe_limit, offset=safe_offset),
            "available_tags": list_post_tags(),
            "available_categories": list_post_categories(),
        }
    except Exception as e:
        logger.error(f"Failed to list public posts: {e}")
        raise HTTPException(status_code=500, detail="Failed to load posts")


@router.get("/api/posts/{post_ref}")
def public_get_post(post_ref: str):
    """Return one public discussion post."""
    try:
        post = get_post(int(post_ref)) if post_ref.isdigit() else get_post(0, slug=post_ref)
    except Exception as e:
        logger.error(f"Failed to get public post {post_ref}: {e}")
        raise HTTPException(status_code=500, detail="Failed to load post")

    if not post or post.get("status") != "published":
        raise HTTPException(status_code=404, detail="Post not found")

    post.pop("comments", None)
    return {"post": post}


@router.get("/api/admin/posts")
def admin_list_posts(token: Optional[str] = None, tag: Optional[str] = None, category: Optional[str] = None,
                     limit: int = 50, offset: int = 0):
    _require_admin(token)
    try:
        posts = list_posts(tag=tag, category=category, status=None, limit=limit, offset=offset)
        available_tags = list_post_tags()
        available_categories = list_post_categories()
    except Exception as e:
        logger.error(f"Failed to list posts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {"success": True, "posts": posts, "available_tags": available_tags, "available_categories": available_categories}


@router.post("/api/admin/posts")
def admin_create_post(payload: PostCreate, token: Optional[str] = None):
    _require_admin(token)
    try:
        post = create_post(
            title=payload.title,
            body=payload.body,
            author_name=payload.author_name,
            tags=payload.tags,
            excerpt=payload.excerpt,
            status=payload.status,
            category=payload.category,
            cover_image=payload.cover_image,
            seo_title=payload.seo_title,
            seo_description=payload.seo_description,
            body_format=payload.body_format,
        )
    except Exception as e:
        logger.error(f"Failed to create post: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {"success": True, "post": post}


@router.get("/api/admin/posts/{post_id}")
def admin_get_post(post_id: int, token: Optional[str] = None):
    _require_admin(token)
    try:
        post = get_post(post_id)
    except Exception as e:
        logger.error(f"Failed to get post {post_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    return {"success": True, "post": post}


@router.put("/api/admin/posts/{post_id}")
def admin_update_post(post_id: int, payload: PostUpdate, token: Optional[str] = None):
    _require_admin(token)
    try:
        post = update_post(
            post_id,
            title=payload.title,
            body=payload.body,
            author_name=payload.author_name,
            tags=payload.tags,
            excerpt=payload.excerpt,
            status=payload.status,
            category=payload.category,
            cover_image=payload.cover_image,
            seo_title=payload.seo_title,
            seo_description=payload.seo_description,
            body_format=payload.body_format,
            slug=payload.slug,
        )
    except Exception as e:
        logger.error(f"Failed to update post {post_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    return {"success": True, "post": post}


@router.post("/api/admin/post-categories")
def admin_create_post_category(payload: dict, token: Optional[str] = None):
    _require_admin(token)
    try:
        return {"success": True, "category": create_post_category(payload.get("name", ""))}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/api/post-categories")
def public_post_categories():
    return {"categories": list_post_categories()}


@router.delete("/api/admin/posts/{post_id}")
def admin_delete_post(post_id: int, token: Optional[str] = None):
    _require_admin(token)
    try:
        deleted = delete_post(post_id)
    except Exception as e:
        logger.error(f"Failed to delete post {post_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if not deleted:
        raise HTTPException(status_code=404, detail="Post not found")

    return {"success": True}


@router.post("/api/admin/posts/{post_id}/comments")
def admin_create_comment(post_id: int, payload: CommentCreate, token: Optional[str] = None):
    _require_admin(token)
    try:
        comment = create_comment(post_id, author_name=payload.author_name, body=payload.body)
    except Exception as e:
        logger.error(f"Failed to add comment to post {post_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if not comment:
        raise HTTPException(status_code=404, detail="Post not found")

    return {"success": True, "comment": comment}


@router.delete("/api/admin/posts/{post_id}/comments/{comment_id}")
def admin_delete_comment(post_id: int, comment_id: int, token: Optional[str] = None):
    _require_admin(token)
    try:
        deleted = delete_comment(post_id, comment_id)
    except Exception as e:
        logger.error(f"Failed to delete comment {comment_id} on post {post_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if not deleted:
        raise HTTPException(status_code=404, detail="Comment not found")

    return {"success": True}
