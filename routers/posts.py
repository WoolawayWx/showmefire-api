import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, field_validator

from core.database import (
    create_comment,
    create_post,
    delete_comment,
    delete_post,
    get_post,
    list_post_tags,
    list_posts,
    update_post,
)
from core.security import verify_token

logger = logging.getLogger(__name__)

router = APIRouter(tags=["posts"])


def _require_admin(token: str) -> str:
    email = verify_token(token)
    if not email:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return email


def _normalize_tags(tags: Optional[List[str]]) -> List[str]:
    normalized: List[str] = []
    for raw in tags or []:
        tag = str(raw or "").strip()
        if tag and tag not in normalized:
            normalized.append(tag)
    return normalized


class PostCreate(BaseModel):
    title: str
    body: str
    author_name: str
    tags: List[str] = []

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
    tags: Optional[List[str]] = None

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
def public_list_posts(tag: Optional[str] = None, limit: int = 50, offset: int = 0):
    """Return published discussion posts for the public blog."""
    try:
        safe_limit = min(max(limit, 1), 50)
        safe_offset = max(offset, 0)
        return {
            "posts": list_posts(tag=tag, limit=safe_limit, offset=safe_offset),
            "available_tags": list_post_tags(),
        }
    except Exception as e:
        logger.error(f"Failed to list public posts: {e}")
        raise HTTPException(status_code=500, detail="Failed to load posts")


@router.get("/api/posts/{post_id}")
def public_get_post(post_id: int):
    """Return one public discussion post."""
    try:
        post = get_post(post_id)
    except Exception as e:
        logger.error(f"Failed to get public post {post_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to load post")

    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    post.pop("comments", None)
    return {"post": post}


@router.get("/api/admin/posts")
def admin_list_posts(token: str, tag: Optional[str] = None, limit: int = 50, offset: int = 0):
    _require_admin(token)
    try:
        posts = list_posts(tag=tag, limit=limit, offset=offset)
        available_tags = list_post_tags()
    except Exception as e:
        logger.error(f"Failed to list posts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {"success": True, "posts": posts, "available_tags": available_tags}


@router.post("/api/admin/posts")
def admin_create_post(payload: PostCreate, token: str):
    _require_admin(token)
    try:
        post = create_post(
            title=payload.title,
            body=payload.body,
            author_name=payload.author_name,
            tags=payload.tags,
        )
    except Exception as e:
        logger.error(f"Failed to create post: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {"success": True, "post": post}


@router.get("/api/admin/posts/{post_id}")
def admin_get_post(post_id: int, token: str):
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
def admin_update_post(post_id: int, payload: PostUpdate, token: str):
    _require_admin(token)
    try:
        post = update_post(
            post_id,
            title=payload.title,
            body=payload.body,
            tags=payload.tags,
        )
    except Exception as e:
        logger.error(f"Failed to update post {post_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if not post:
        raise HTTPException(status_code=404, detail="Post not found")

    return {"success": True, "post": post}


@router.delete("/api/admin/posts/{post_id}")
def admin_delete_post(post_id: int, token: str):
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
def admin_create_comment(post_id: int, payload: CommentCreate, token: str):
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
def admin_delete_comment(post_id: int, comment_id: int, token: str):
    _require_admin(token)
    try:
        deleted = delete_comment(post_id, comment_id)
    except Exception as e:
        logger.error(f"Failed to delete comment {comment_id} on post {post_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    if not deleted:
        raise HTTPException(status_code=404, detail="Comment not found")

    return {"success": True}
