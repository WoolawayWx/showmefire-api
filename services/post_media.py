import hashlib
import logging
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from core.database import create_post_media_record, list_post_media_records

logger = logging.getLogger(__name__)

POST_MEDIA_DIR = Path(os.getenv("POST_MEDIA_DIR", str(Path(os.getenv("DATA_DIR", ".")) / "files" / "post-media")))
CDN_BASE_URL = os.getenv("CDN_BASE_URL", "https://cdn.showmefire.org").rstrip("/")
API_PUBLIC_BASE_URL = os.getenv("API_PUBLIC_BASE_URL", "https://api.showmefire.org").rstrip("/")
CDN_PREFIX = "assets/posts"
MAX_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}


def ensure_post_media_dir() -> None:
    POST_MEDIA_DIR.mkdir(parents=True, exist_ok=True)


def _public_url(filename: str, cdn_url: Optional[str] = None) -> str:
    if cdn_url:
        return cdn_url
    return f"{API_PUBLIC_BASE_URL}/files/post-media/{filename}"


def _upload_to_cdn(local_path: Path, key: str, content_type: str) -> Optional[str]:
    access_key = os.getenv("R2_ACCESS_KEY_ID")
    secret_key = os.getenv("R2_SECRET_ACCESS_KEY")
    account_id = os.getenv("R2_ACCOUNT_ID")
    if not all([access_key, secret_key, account_id]):
        return None

    try:
        import boto3
        from botocore.config import Config

        client = boto3.client(
            "s3",
            endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version="s3v4"),
            region_name="auto",
        )
        client.upload_file(
            str(local_path),
            "cdn-showmefire",
            key,
            ExtraArgs={
                "ContentType": content_type,
                "CacheControl": "public, max-age=31536000, immutable",
            },
        )
        return f"{CDN_BASE_URL}/{key}"
    except Exception as exc:
        logger.warning("Post media CDN upload failed for %s: %s", key, exc)
        return None


def save_post_media(content: bytes, content_type: str, original_name: str, uploaded_by: Optional[str] = None) -> Dict:
    normalized_type = (content_type or "").lower().split(";")[0].strip()
    if normalized_type not in ALLOWED_TYPES:
        raise ValueError("Only JPEG, PNG, WebP, and GIF images are allowed")
    if len(content) > MAX_BYTES:
        raise ValueError("Image exceeds 10 MB")

    ensure_post_media_dir()
    extension = ALLOWED_TYPES[normalized_type]
    filename = f"{uuid.uuid4().hex}{extension}"
    target = POST_MEDIA_DIR / filename
    target.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()

    cdn_key = f"{CDN_PREFIX}/{filename}"
    cdn_url = _upload_to_cdn(target, cdn_key, normalized_type)
    public_url = _public_url(filename, cdn_url)

    record = create_post_media_record(
        filename=filename,
        original_name=Path(original_name or filename).name[:180],
        content_type=normalized_type,
        size_bytes=len(content),
        public_url=public_url,
        cdn_url=cdn_url,
        sha256=digest,
        uploaded_by=uploaded_by,
    )
    record["url"] = public_url
    return record


def list_post_media(limit: int = 100, offset: int = 0) -> List[Dict]:
    items = list_post_media_records(limit=limit, offset=offset)
    for item in items:
        item["url"] = item.get("cdn_url") or item.get("public_url")
    return items
