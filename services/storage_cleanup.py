"""Keeps the server's disks from filling up.

Hourly (core/scheduler.py ``storage_cleanup``), only things that are safe to lose:

- forecast_v1 raw downloads of finished runs (``forecast_v1.repository.prune_download_cache``;
  a run also clears its own downloads when it publishes, this catches crashes in between)
- staging folders left behind by an interrupted publish or archive build, once a day old
- a free-space check on every disk the app writes to: WARNING under SMF_DISK_WARN_FREE_GB (15),
  ERROR plus a staff Discord alert (at most every 6 h) under SMF_DISK_CRITICAL_FREE_GB (5)

Everything else that grows (HRRR/ensemble/RTMA caches, hrrr_hourly, rtma_analysis, logs) already
has its own retention; day_archive prunes source files verified in R2 when
SMF_DAY_ARCHIVE_PRUNE_LOCAL=true.

By hand:

    python -m services.storage_cleanup report             # free space and the big folders
    python -m services.storage_cleanup run                # the hourly pass, now
    python -m services.storage_cleanup archive-folder /legacy --r2-prefix legacy-archive/data_archive_day [--delete]

``archive-folder`` uploads every file in a folder to R2, reads each one back and compares its
sha256, writes a manifest, and with ``--delete`` removes a local file only after its copy verified.
Re-running it skips files already verified.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

GB = 1024 ** 3
STALE_SECONDS = 24 * 3600
ALERT_EVERY_SECONDS = 6 * 3600
# Folders reported by ``report`` (relative to the app root); the disks checked are the ones they live on.
WATCHED = ("data", "data/forecast-v1", "data/forecast-v1/download-cache", "data/hrrr_hourly", "data/rtma_analysis",
           "data/testbed", "cache", "cache/hrrr", "cache/ensemble", "cache/rtma", "gis", "archive", "logs",
           "reports", "images")


def root() -> Path:
    from services.day_archive import root as app_root
    return app_root()


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


# --- cleanup rules -------------------------------------------------------------------------------

def _staging_dirs(base: Path) -> list[Path]:
    """Work folders that only exist while a publish/build is running."""
    found = list((base / "gis").glob(".publish-*"))
    for parent in (base / "data" / "forecast-v1" / ".staging", base / "cache" / "day-archive"):
        if parent.is_dir():
            found += list(parent.iterdir())
    return found


def prune_stale_staging(base: Optional[Path] = None, now: Optional[float] = None) -> list[str]:
    base = base or root()
    now = now or time.time()
    removed = []
    for path in _staging_dirs(base):
        try:
            if now - path.stat().st_mtime < STALE_SECONDS:
                continue
            shutil.rmtree(path) if path.is_dir() else path.unlink()
            removed.append(str(path.relative_to(base)))
        except OSError as error:
            logger.warning("Could not remove stale %s: %s", path, error)
    return removed


def prune_forecast_v1_downloads(base: Optional[Path] = None) -> int:
    from forecast_v1.repository import prune_download_cache

    forecast_root = (base or root()) / "data" / "forecast-v1"
    return prune_download_cache(forecast_root) if forecast_root.is_dir() else 0


# --- disk check -----------------------------------------------------------------------------------

def disks(base: Optional[Path] = None) -> list[dict]:
    """Free space of each distinct filesystem the watched folders are on."""
    base = base or root()
    seen, rows = set(), []
    for rel in ("",) + WATCHED:
        path = base / rel
        if not path.exists():
            continue
        device = path.stat().st_dev
        if device in seen:
            continue
        seen.add(device)
        usage = shutil.disk_usage(path)
        rows.append({"path": str(path), "total_gb": round(usage.total / GB, 1), "used_gb": round(usage.used / GB, 1),
                     "free_gb": round(usage.free / GB, 1), "used_pct": round(100 * usage.used / usage.total, 1)})
    return rows


def _alert_state_path(base: Path) -> Path:
    return base / "data" / "storage_alert.json"


def check_disks(base: Optional[Path] = None, now: Optional[float] = None) -> list[dict]:
    base = base or root()
    now = now or time.time()
    warn = _float_env("SMF_DISK_WARN_FREE_GB", 15)
    critical = _float_env("SMF_DISK_CRITICAL_FREE_GB", 5)
    rows = disks(base)
    low = []
    for disk in rows:
        if disk["free_gb"] < critical:
            disk["level"] = "critical"
            logger.error("Disk almost full: %s has %.1f GB free (%.0f%% used)", disk["path"], disk["free_gb"], disk["used_pct"])
            low.append(disk)
        elif disk["free_gb"] < warn:
            disk["level"] = "low"
            logger.warning("Disk space low: %s has %.1f GB free (%.0f%% used)", disk["path"], disk["free_gb"], disk["used_pct"])
        else:
            disk["level"] = "ok"
    if low:
        _alert(base, low, now)
    return rows


def _alert(base: Path, low: list[dict], now: float) -> None:
    state_path = _alert_state_path(base)
    try:
        last = json.loads(state_path.read_text()).get("alerted_at")
    except (OSError, ValueError):
        last = None
    if last is not None and now - last < ALERT_EVERY_SECONDS:
        return
    from services.discord_notifier import notify_staff_alert

    fields = [{"name": disk["path"], "value": f"{disk['free_gb']} GB free of {disk['total_gb']} GB ({disk['used_pct']}% used)"}
              for disk in low]
    notify_staff_alert(alert_type="system_issue", title="Server disk almost full",
                       description="Run `python -m services.storage_cleanup report` on the server to see what is using it.",
                       fields=fields, admin_path="/admin/logs", background=False)
    try:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps({"alerted_at": now}))
    except OSError as error:
        logger.warning("Could not save disk alert state: %s", error)


# --- passes ---------------------------------------------------------------------------------------

def run(base: Optional[Path] = None) -> dict:
    base = base or root()
    result = {"forecast_v1_download_files": 0, "stale_staging": []}
    try:
        result["forecast_v1_download_files"] = prune_forecast_v1_downloads(base)
    except Exception as error:  # noqa: BLE001 - keep going with the other rules
        logger.error("forecast_v1 download cleanup failed: %s", error, exc_info=True)
    result["stale_staging"] = prune_stale_staging(base)
    result["disks"] = check_disks(base)
    return result


def folder_size(path: Path) -> int:
    total = 0
    for dirpath, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.lstat(os.path.join(dirpath, name)).st_size
            except OSError:
                pass
    return total


def report(base: Optional[Path] = None) -> dict:
    base = base or root()
    folders = [{"path": rel, "gb": round(folder_size(base / rel) / GB, 2)} for rel in WATCHED if (base / rel).is_dir()]
    data = base / "data"
    files = sorted(({"path": str(p.relative_to(base)), "gb": round(p.stat().st_size / GB, 2)}
                    for p in data.glob("*") if p.is_file() and p.stat().st_size > 50 * 1024 ** 2),
                   key=lambda row: -row["gb"]) if data.is_dir() else []
    return {"disks": disks(base), "folders": folders, "large_files_in_data": files}


# --- archive a folder to R2, then (optionally) delete it locally -------------------------------------

def archive_folder(folder: Path, r2_prefix: str, delete: bool = False, s3=None, log=print) -> dict:
    """Upload every file under ``folder`` to ``r2_prefix/<relative path>``, verify by reading it back."""
    from services import day_archive

    if s3 is None:
        from services.archive_bundler import _r2_client
        s3 = _r2_client()
    folder = Path(folder).resolve()
    prefix = r2_prefix.strip("/")
    manifest_key = f"{prefix}/manifest.json"
    manifest = day_archive._get_json(s3, manifest_key) or {"source": str(folder), "files": {}}
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    summary = {"files": len(files), "uploaded": 0, "already_verified": 0, "deleted": 0, "failed": []}
    for index, path in enumerate(files, 1):
        rel = path.relative_to(folder).as_posix()
        key = f"{prefix}/{rel}"
        size = path.stat().st_size
        try:
            sha = day_archive.sha256_file(path)
            known = manifest["files"].get(rel)
            if known and known["sha256"] == sha and known["size"] == size:
                summary["already_verified"] += 1
            else:
                s3.upload_file(str(path), day_archive.bucket(), key,
                               ExtraArgs={"Metadata": {"sha256": sha}, "ContentType": "application/octet-stream"})
                day_archive.verify_remote(s3, key, size, sha)
                manifest["files"][rel] = {"key": key, "size": size, "sha256": sha,
                                          "verified_at": datetime.now(timezone.utc).isoformat()}
                day_archive._put_json(s3, manifest_key, manifest)
                summary["uploaded"] += 1
            if delete:
                path.unlink()
                summary["deleted"] += 1
            log(f"[{index}/{len(files)}] {rel} ({size / GB:.2f} GB) verified{' and deleted' if delete else ''}")
        except Exception as error:  # noqa: BLE001 - keep the local file, carry on with the rest
            summary["failed"].append(f"{rel}: {error}")
            log(f"[{index}/{len(files)}] {rel} FAILED, kept locally: {error}")
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("report", help="free space and the size of the big folders")
    sub.add_parser("run", help="the hourly cleanup pass, now")
    archive = sub.add_parser("archive-folder", help="upload a folder to R2 (verified), optionally delete it locally")
    archive.add_argument("folder", type=Path)
    archive.add_argument("--r2-prefix", required=True, help="e.g. legacy-archive/data_archive_day")
    archive.add_argument("--delete", action="store_true", help="delete each local file once its R2 copy verified")
    args = parser.parse_args(argv)
    if args.command == "report":
        print(json.dumps(report(), indent=2))
    elif args.command == "run":
        print(json.dumps(run(), indent=2))
    else:
        summary = archive_folder(args.folder, args.r2_prefix, delete=args.delete)
        print(json.dumps(summary, indent=2))
        return 1 if summary["failed"] else 0
    return 0


async def run_scheduled() -> None:
    import asyncio

    result = await asyncio.to_thread(run)
    free = ", ".join(f"{d['path']} {d['free_gb']} GB free" for d in result["disks"])
    logger.info("Storage cleanup: forecast_v1 download files removed=%d, stale staging removed=%d; %s",
                result["forecast_v1_download_files"], len(result["stale_staging"]), free)


if __name__ == "__main__":
    raise SystemExit(main())
