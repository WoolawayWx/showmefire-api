"""Publish dataset runs to the CDN (Cloudflare R2, bucket cdn-showmefire) so browsers read them from the edge
instead of the API.

A dataset is anything that is produced in runs and read by a viewer: the hourly HRRR forecast today, more
model or plotted data later. Nothing here knows about HRRR; a dataset is just an id and a folder of files.

    datasets/index.json                         registry of every published dataset
    datasets/<dataset>/latest.json              pointer to the newest run (short cache)
    datasets/<dataset>/runs.json                recent runs, newest first (short cache)
    datasets/<dataset>/runs/<run>/...           the run's files; never rewritten, so cached for a year, except the files
                                                of a run still being filled in (see `publish_update`)

Publish order is files, then the manifest, then runs.json, latest.json and the registry, so a reader that
follows the pointer never meets a half-uploaded run. The first run to be pruned from the CDN is never the one
`latest.json` points to. Publishing is best effort (see `try_publish_run`): the API keeps serving every run.
"""
from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

BUCKET = "cdn-showmefire"
PREFIX = "datasets"
MANIFEST_NAME = "manifest.json"
IMMUTABLE_CACHE = "public, max-age=31536000, immutable"
POINTER_CACHE = "public, max-age=60"
# Files of a run that is still being filled in, and that a later update may rewrite (its manifest, a day that is not complete).
LIVE_CACHE = "public, max-age=60"
UPLOAD_WORKERS = 16
# A backstop on how many runs runs.json lists (and so keeps on the CDN), beyond the age limit. It has to be well above
# runs per day x keep_days (24 x 7 = 168 for the hourly HRRR product), or runs still inside the window get deleted.
RUNS_LISTED = 400


def keep_days() -> int:
    return int(os.environ.get("SMF_DATASETS_KEEP_DAYS", "7"))


def configured() -> bool:
    return all(os.getenv(name) for name in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"))


def make_client():
    """The same S3-compatible R2 client scripts/upload_cdn.py uses."""
    import boto3
    from botocore.config import Config

    return boto3.client(
        service_name="s3", endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4", max_pool_connections=UPLOAD_WORKERS), region_name="auto")


def dataset_key(dataset: str, *parts: str) -> str:
    return "/".join([PREFIX, dataset, *parts])


def run_key(dataset: str, run_id: str, relative: str) -> str:
    return dataset_key(dataset, "runs", run_id, relative)


def _content_type(path: Path) -> str:
    # Zarr's metadata files (.zmetadata, .zarray, .zattrs, .zgroup) are JSON; chunks are already compressed.
    if path.suffix == ".json" or path.name.startswith(".z"):
        return "application/json"
    return "application/octet-stream"


def _now_iso(now: datetime) -> str:
    return now.strftime("%Y-%m-%dT%H:%M:%SZ")


# --- reading and writing the small JSON documents ------------------------------------------

def _get_json(client, key: str) -> Optional[dict]:
    try:
        body = client.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    except Exception as error:  # noqa: BLE001 - S3 clients raise their own ClientError type
        if getattr(error, "response", {}).get("Error", {}).get("Code") in ("NoSuchKey", "404", "NotFound"):
            return None
        raise
    try:
        return json.loads(body)
    except ValueError:
        return None


def _put_json(client, key: str, payload: dict) -> None:
    client.put_object(Bucket=BUCKET, Key=key, Body=json.dumps(payload, indent=1, sort_keys=True).encode(),
                      ContentType="application/json", CacheControl=POINTER_CACHE)


def _delete_prefix(client, prefix: str) -> int:
    deleted = 0
    pager = client.get_paginator("list_objects_v2")
    for page in pager.paginate(Bucket=BUCKET, Prefix=prefix):
        keys = [{"Key": item["Key"]} for item in page.get("Contents", [])]
        if keys:
            client.delete_objects(Bucket=BUCKET, Delete={"Objects": keys, "Quiet": True})
            deleted += len(keys)
    return deleted


# --- publishing -------------------------------------------------------------------------------

def _upload_files(client, dataset: str, run_id: str, files: list[tuple[Path, str]], cache: str = IMMUTABLE_CACHE) -> int:
    """Upload (local path, path inside the run) pairs in parallel. Raises if any file fails."""
    def upload(item: tuple[Path, str]) -> None:
        path, relative = item
        client.upload_file(str(path), BUCKET, run_key(dataset, run_id, relative),
                           ExtraArgs={"ContentType": _content_type(path), "CacheControl": cache})

    with ThreadPoolExecutor(max_workers=UPLOAD_WORKERS) as pool:
        for _ in pool.map(upload, files):   # iterating re-raises the first failure
            pass
    return len(files)


def _update_indexes(client, dataset: str, run_id: str, summary: dict, *, title: str, kind: str, schema_version: int,
                    now: datetime) -> list[str]:
    """Add or replace the run's row in runs.json, repoint latest.json and the registry, then delete expired runs.

    A row with `"ready": false` is listed but never becomes the dataset's latest: a run still filling in is reachable by
    its id until enough of it is there (the caller sets `ready`). Returns the run ids that were deleted."""
    previous = _get_json(client, dataset_key(dataset, "runs.json")) or {}
    runs = [row for row in previous.get("runs", []) if row.get("id") != run_id] + [summary]
    runs.sort(key=lambda row: row["time_utc"], reverse=True)
    newest = next((row for row in runs if row.get("ready", True)), None)
    # With nothing ready yet the pointer stays where it was (and that run is kept), so a run still filling in is never "latest".
    keep = newest["id"] if newest else previous.get("latest")
    cutoff = _now_iso(now - timedelta(days=keep_days()))
    expired = list(dict.fromkeys(  # never the run latest.json points to
        [row["id"] for row in runs if row["id"] != keep and row["time_utc"] < cutoff]
        + [row["id"] for row in runs[RUNS_LISTED:] if row["id"] != keep]))
    runs = [row for row in runs if row["id"] not in expired]

    _put_json(client, dataset_key(dataset, "runs.json"), {"latest": keep, "runs": runs})
    if newest:
        _put_json(client, dataset_key(dataset, "latest.json"), {
            "run": newest["id"], "manifest": f"runs/{newest['id']}/{MANIFEST_NAME}", "time_utc": newest["time_utc"],
            "updated_at": _now_iso(now)})
        registry = _get_json(client, f"{PREFIX}/index.json") or {}
        datasets = registry.get("datasets", {})
        datasets[dataset] = {"title": title, "kind": kind, "schema_version": schema_version,
                             "latest": f"{dataset}/latest.json", "runs": f"{dataset}/runs.json",
                             "updated_at": _now_iso(now)}
        _put_json(client, f"{PREFIX}/index.json", {"datasets": datasets})

    for run in expired:   # only after the pointers no longer mention them
        _delete_prefix(client, dataset_key(dataset, "runs", run) + "/")
    return expired


def publish_run(dataset: str, run_id: str, run_dir: Path, summary: dict, *, title: str, kind: str,
                schema_version: int, include: Optional[list[str]] = None, client=None,
                now: Optional[datetime] = None) -> dict:
    """Upload a run and make it the dataset's newest.

    `run_dir` holds the run; everything under it is uploaded unless `include` names the entries to
    publish (files or folders, relative to it). It must contain manifest.json. `summary` is the run's row
    in runs.json and needs `id` (the run id) and `time_utc` (ISO, used to age runs out); put any other
    fields viewers list (label, forecast hours, ...) in it too. Runs older than `keep_days()` are removed
    from the CDN. Returns {"uploaded": file count, "pruned": [run ids]}."""
    run_dir = Path(run_dir)
    if not (run_dir / MANIFEST_NAME).is_file():
        raise FileNotFoundError(f"{run_dir / MANIFEST_NAME} is missing")
    if summary.get("id") != run_id or not summary.get("time_utc"):
        raise ValueError("summary needs id (the run id) and time_utc")
    client = client or make_client()
    now = now or datetime.now(timezone.utc)

    roots = [run_dir / name for name in include] if include else [run_dir]
    files = []
    for root in roots:
        for path in ([root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())):
            relative = path.relative_to(run_dir).as_posix()
            if relative != MANIFEST_NAME and not path.name.endswith(".tmp"):
                files.append((path, relative))
    uploaded = _upload_files(client, dataset, run_id, files)
    client.upload_file(str(run_dir / MANIFEST_NAME), BUCKET, run_key(dataset, run_id, MANIFEST_NAME),
                       ExtraArgs={"ContentType": "application/json", "CacheControl": IMMUTABLE_CACHE})

    expired = _update_indexes(client, dataset, run_id, summary, title=title, kind=kind, schema_version=schema_version,
                              now=now)
    return {"uploaded": uploaded + 1, "pruned": expired}


def publish_update(dataset: str, run_id: str, run_dir: Path, summary: dict, *, title: str, kind: str, schema_version: int,
                   stable: list[str], changing: list[str], complete: bool, client=None, now: Optional[datetime] = None) -> dict:
    """Upload part of a run that is still being built, and refresh the pointers. Call it again as more arrives.

    `stable` are files that never change once uploaded (finished step chunks, array metadata): cached for a year.
    `changing` are files a later update may rewrite (a day still filling in): cached for a minute. The manifest is
    uploaded last, with the short cache until `complete`, when it (and every file in `changing`) gets the year-long one:
    the caller passes the finished files in `changing` once more with `complete=True`. Paths are relative to `run_dir`."""
    run_dir = Path(run_dir)
    if not (run_dir / MANIFEST_NAME).is_file():
        raise FileNotFoundError(f"{run_dir / MANIFEST_NAME} is missing")
    if summary.get("id") != run_id or not summary.get("time_utc"):
        raise ValueError("summary needs id (the run id) and time_utc")
    client = client or make_client()
    now = now or datetime.now(timezone.utc)
    uploaded = _upload_files(client, dataset, run_id, [(run_dir / name, name) for name in stable])
    uploaded += _upload_files(client, dataset, run_id, [(run_dir / name, name) for name in changing],
                              cache=IMMUTABLE_CACHE if complete else LIVE_CACHE)
    client.upload_file(str(run_dir / MANIFEST_NAME), BUCKET, run_key(dataset, run_id, MANIFEST_NAME),
                       ExtraArgs={"ContentType": "application/json",
                                  "CacheControl": IMMUTABLE_CACHE if complete else LIVE_CACHE})
    expired = _update_indexes(client, dataset, run_id, summary, title=title, kind=kind, schema_version=schema_version,
                              now=now)
    return {"uploaded": uploaded + 1, "pruned": expired}


def try_publish_run(*args: Any, **kwargs: Any) -> Optional[dict]:
    """publish_run that never raises and does nothing without R2 credentials. A failed upload must not fail
    the build: the run is still served by the API, and the next run publishes normally."""
    if kwargs.get("client") is None and not configured():
        logger.info("R2 credentials are not set; not publishing to the CDN")
        return None
    try:
        result = publish_run(*args, **kwargs)
    except Exception as error:  # noqa: BLE001
        logger.warning("Publishing to the CDN failed: %s", error)
        return None
    logger.info("Published %s files to the CDN (pruned %s)", result["uploaded"], result["pruned"] or "none")
    return result


def try_publish_update(*args: Any, **kwargs: Any) -> Optional[dict]:
    """publish_update that never raises and does nothing without R2 credentials (see try_publish_run)."""
    if kwargs.get("client") is None and not configured():
        return None
    try:
        return publish_update(*args, **kwargs)
    except Exception as error:  # noqa: BLE001
        logger.warning("Publishing an update to the CDN failed: %s", error)
        return None
