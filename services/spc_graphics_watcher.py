"""Detect new SPC outlook GIS data and fan out department graphic renders."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
import uuid
from contextlib import contextmanager
from typing import Iterator

from core.database import get_db_path
from services.graphic_renderer import PRODUCT_URLS, fetch_spc_product

logger = logging.getLogger(__name__)


@contextmanager
def _db() -> Iterator[sqlite3.Connection]:
    db = sqlite3.connect(get_db_path(), timeout=30)
    db.row_factory = sqlite3.Row
    try:
        with db:
            yield db
    finally:
        db.close()


def _fingerprint(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _changed_products(observed: dict[str, str]) -> set[str]:
    with _db() as db:
        rows = db.execute(
            "SELECT product_id,source_fingerprint FROM graphic_source_state"
        ).fetchall()
    previous = {row["product_id"]: row["source_fingerprint"] for row in rows}
    return {product_id for product_id, digest in observed.items() if previous.get(product_id) != digest}


def _queue_affected_bundles(changed_products: set[str]) -> list[dict]:
    """Create jobs for active bundles affected by this SPC source update."""
    queued: list[dict] = []
    with _db() as db:
        rows = db.execute(
            "SELECT id,department_id,config_json FROM graphic_bundles WHERE active=1"
        ).fetchall()
        for row in rows:
            bundle = json.loads(row["config_json"])
            product_id = bundle.get("product_id")
            affected = product_id in changed_products or (
                product_id == "spc_four_panel" and bool(changed_products)
            )
            if not affected:
                continue
            bundle["id"] = row["id"]
            job_id = str(uuid.uuid4())
            db.execute(
                "INSERT INTO graphic_jobs(id,bundle_id,department_id,status,config_json) VALUES (?,?,?,?,?)",
                (job_id, row["id"], row["department_id"], "queued", json.dumps(bundle)),
            )
            queued.append({
                "job_id": job_id,
                "bundle": bundle,
                "department_id": row["department_id"],
            })
    return queued


def _jobs_succeeded(job_ids: list[str]) -> bool:
    if not job_ids:
        return True
    placeholders = ",".join("?" for _ in job_ids)
    with _db() as db:
        rows = db.execute(
            f"SELECT id,status FROM graphic_jobs WHERE id IN ({placeholders})",
            job_ids,
        ).fetchall()
    statuses = {row["id"]: row["status"] for row in rows}
    return all(statuses.get(job_id) in {"completed", "skipped"} for job_id in job_ids)


def _record_processed(observed: dict[str, str], changed_products: set[str]) -> None:
    with _db() as db:
        for product_id in changed_products:
            db.execute(
                """INSERT INTO graphic_source_state(product_id,source_url,source_fingerprint)
                   VALUES (?,?,?)
                   ON CONFLICT(product_id) DO UPDATE SET
                     source_url=excluded.source_url,
                     source_fingerprint=excluded.source_fingerprint,
                     observed_at=CURRENT_TIMESTAMP,
                     processed_at=CURRENT_TIMESTAMP""",
                (product_id, PRODUCT_URLS[product_id], observed[product_id]),
            )


async def refresh_spc_graphics() -> dict:
    """Poll SPC once and render every active bundle affected by an update.

    Active SPC bundles are the current subscription registry. Source state is
    advanced only after every queued render completes or is skipped, allowing
    a failed department render to be retried on the next scheduler tick.
    """
    payloads = await asyncio.gather(*(
        asyncio.to_thread(fetch_spc_product, product_id)
        for product_id in PRODUCT_URLS
    ))
    observed = {
        product_id: _fingerprint(payload)
        for product_id, payload in zip(PRODUCT_URLS, payloads)
    }
    changed_products = _changed_products(observed)
    if not changed_products:
        return {"changed_products": [], "queued": 0, "completed": True}

    jobs = _queue_affected_bundles(changed_products)
    logger.info(
        "SPC outlook update detected for %s; queued %s department bundle(s)",
        ", ".join(sorted(changed_products)),
        len(jobs),
    )

    # Imported lazily to avoid coupling API router initialization to the
    # scheduler module at application startup.
    from routers.graphics import _run_job

    for job in jobs:
        await _run_job(
            job["job_id"], job["bundle"], job["department_id"], None,
        )

    job_ids = [job["job_id"] for job in jobs]
    succeeded = _jobs_succeeded(job_ids)
    if succeeded:
        _record_processed(observed, changed_products)
    else:
        logger.warning(
            "SPC source state was not advanced because one or more department renders failed"
        )
    return {
        "changed_products": sorted(changed_products),
        "queued": len(jobs),
        "completed": succeeded,
    }


async def refresh_spc_graphics_job() -> None:
    """Scheduler wrapper that reports failures without stopping other jobs."""
    try:
        result = await refresh_spc_graphics()
        if result["changed_products"]:
            logger.info("SPC graphics refresh finished: %s", result)
    except Exception as error:
        logger.error("SPC graphics source poll failed: %s", error, exc_info=True)
