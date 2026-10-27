"""Publish the detection training export to Cloudflare R2 so the training machine can download it.

Layout (bucket resolution mirrors services/day_archive.py; override with R2_TRAINING_BUCKET / R2_TRAINING_PREFIX):

    <prefix>/latest.json                          pointer to the newest snapshot (written last)
    <prefix>/snapshots/<UTC stamp>/detection_training_export.zip
    <prefix>/snapshots/<UTC stamp>/manifest.json  the export manifest (row counts, per-source date ranges)

The zip holds only the columns scripts/export_detection_training_data.py selects (no contact details or IP hashes), but
it does include moderation outcomes, so keep this prefix off any public/CDN path. model-training/detection_dataset/r2.py
reads the same layout; change both together.

Publish order is zip, then latest.json, then pruning, so a reader following the pointer never meets a half-uploaded
snapshot and the newest snapshot is never the one deleted.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from services.cdn_datasets import configured, make_client

logger = logging.getLogger(__name__)

ZIP_NAME = "detection_training_export.zip"
MANIFEST_NAME = "manifest.json"
DEFAULT_KEEP = 14


def training_bucket() -> str:
    return os.getenv("R2_TRAINING_BUCKET") or os.getenv("R2_ARCHIVE_BUCKET") or os.getenv("R2_BUCKET", "cdn-showmefire")


def prefix() -> str:
    return os.getenv("R2_TRAINING_PREFIX", "training/detections").strip("/")


def pointer_key() -> str:
    return f"{prefix()}/latest.json"


def snapshot_key(stamp: str, name: str) -> str:
    return f"{prefix()}/snapshots/{stamp}/{name}"


def _export_module():
    """The export script is stdlib-only and shared with the training repo, so it is loaded by path."""
    path = Path(__file__).resolve().parent.parent / "scripts" / "export_detection_training_data.py"
    spec = importlib.util.spec_from_file_location("_smf_export_detection_training", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def list_snapshots(client) -> List[str]:
    """Snapshot stamps under the prefix, newest first."""
    stamps: List[str] = []
    token = None
    while True:
        kwargs: Dict[str, Any] = {"Bucket": training_bucket(), "Prefix": f"{prefix()}/snapshots/", "Delimiter": "/"}
        if token:
            kwargs["ContinuationToken"] = token
        page = client.list_objects_v2(**kwargs)
        stamps += [p["Prefix"].rstrip("/").rsplit("/", 1)[-1] for p in page.get("CommonPrefixes", [])]
        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken")
    return sorted(stamps, reverse=True)


def prune_snapshots(client, keep: int, protect: str) -> List[str]:
    """Delete all but the newest `keep` snapshots (never `protect`, the one latest.json points to)."""
    doomed = [s for s in list_snapshots(client)[max(keep, 1):] if s != protect]
    for stamp in doomed:
        token = None
        while True:
            kwargs: Dict[str, Any] = {"Bucket": training_bucket(), "Prefix": f"{prefix()}/snapshots/{stamp}/"}
            if token:
                kwargs["ContinuationToken"] = token
            page = client.list_objects_v2(**kwargs)
            keys = [{"Key": item["Key"]} for item in page.get("Contents", [])]
            if keys:
                client.delete_objects(Bucket=training_bucket(), Delete={"Objects": keys, "Quiet": True})
            if not page.get("IsTruncated"):
                break
            token = page.get("NextContinuationToken")
    return doomed


def publish(db_path: Optional[Path] = None, since: Optional[str] = None, keep: int = DEFAULT_KEEP,
            client=None, now: Optional[datetime] = None) -> Dict[str, Any]:
    """Export the DB (read-only), upload the zip + manifest, then repoint latest.json. Returns the pointer document."""
    client = client or make_client()
    now = now or datetime.now(timezone.utc)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    export = _export_module()
    with tempfile.TemporaryDirectory() as tmp:
        manifest_path = export.export(since, db_path, Path(tmp))
        zip_path = export.bundle(manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        zip_key = snapshot_key(stamp, ZIP_NAME)
        client.upload_file(str(zip_path), training_bucket(), zip_key, ExtraArgs={"ContentType": "application/zip"})
        client.put_object(Bucket=training_bucket(), Key=snapshot_key(stamp, MANIFEST_NAME), Body=manifest_path.read_bytes(),
                          ContentType="application/json")
        pointer = {
            "schema_version": manifest.get("schema_version"),
            "snapshot": stamp,
            "generated_at": manifest.get("generated_at"),
            "zip_key": zip_key,
            "zip_sha256": _sha256(zip_path),
            "zip_bytes": zip_path.stat().st_size,
            "rows": {name: info["rows"] for name, info in manifest["files"].items()},
            "detection_ranges": manifest.get("detection_ranges", {}),
        }
    client.put_object(Bucket=training_bucket(), Key=pointer_key(), Body=json.dumps(pointer, indent=1).encode(),
                      ContentType="application/json", CacheControl="no-store")
    pointer["pruned"] = prune_snapshots(client, keep, protect=stamp)
    logger.info("detection_r2: published %s (%s bytes, %s detections), pruned %d",
                stamp, pointer["zip_bytes"], pointer["rows"].get("detections"), len(pointer["pruned"]))
    return pointer


def publish_job() -> None:
    """Scheduler entry point (core/scheduler.py, off unless SMF_DETECTION_R2_PUBLISH=true). Never raises."""
    if not configured():
        logger.warning("detection_r2: R2 credentials are not configured; skipping the training export")
        return
    try:
        publish(keep=int(os.getenv("SMF_DETECTION_R2_KEEP", str(DEFAULT_KEEP))))
    except Exception as exc:  # noqa: BLE001 - a failed upload must never disturb the scheduler
        logger.error("detection_r2: publish failed: %s", exc, exc_info=True)
