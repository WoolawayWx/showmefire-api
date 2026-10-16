"""Daily zips of the 15-minute map copies on the CDN, so the CDN keeps only
recent days and the history can be downloaded and stored elsewhere.

scripts/maps.sh uploads every changed map to the CDN under YYYY/MM/DD/HHMM/
(the image timeline and recent-event lookups use these). Nightly, for each
completed day this job:
  1. zips that day's HHMM/ folders into map-archive/YYYYMMDD.zip in the same
     bucket (next to data-archive/ and ngfs-ogc-raw/), verified after upload;
  2. deletes the day's HHMM/ folders once the day is older than
     SMF_MAP_ARCHIVE_KEEP_DAYS (14) - only if its zip exists and checks out.
     YYYY/MM/DD/forecast/, latest/ and everything else are never touched;
  3. deletes zips SMF_MAP_ZIP_KEEP_DAYS (30) after they were created
     (0 keeps them forever), with a Discord reminder 7 days before;
  4. posts to the staff Discord channel when new zips are ready to download.
Large backlogs are worked through SMF_MAP_ARCHIVE_DAYS_PER_RUN (3) days a night.
scripts/pull_map_archives.py downloads the zips to a local folder.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[1]
LEDGER_PATH = PROJECT_DIR / "data" / "map_archive" / "ledger.json"
WORK_DIR = PROJECT_DIR / "cache" / "map-archive"
BUCKET = os.getenv("R2_BUCKET", "cdn-showmefire")
CDN_BASE_URL = os.getenv("CDN_BASE_URL", "https://cdn.showmefire.org").rstrip("/")
ZIP_PREFIX = "map-archive"
QUARTER_HOUR = re.compile(r"^\d{4}/\d{2}/\d{2}/(\d{4})/$")
STORED = {".png", ".webp", ".jpg", ".jpeg", ".gif", ".zip"}  # already compressed
REMIND_DAYS_BEFORE = 7


def _setting(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def keep_days() -> int:
    return _setting("SMF_MAP_ARCHIVE_KEEP_DAYS", 14)


def zip_keep_days() -> int:
    return _setting("SMF_MAP_ZIP_KEEP_DAYS", 30)


def days_per_run() -> int:
    return _setting("SMF_MAP_ARCHIVE_DAYS_PER_RUN", 3)


def zip_key(day: str) -> str:
    return f"{ZIP_PREFIX}/{day}.zip"


def download_url(day: str) -> str:
    return f"{CDN_BASE_URL}/{zip_key(day)}"


def load_ledger(path: Optional[Path] = None) -> dict:
    try:
        return json.loads((path or LEDGER_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_ledger(ledger: dict, path: Optional[Path] = None) -> None:
    path = path or LEDGER_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


# --- bucket inventory -----------------------------------------------------------

def _prefixes(s3, prefix: str) -> list[str]:
    found = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix, Delimiter="/"):
        found += [item["Prefix"] for item in page.get("CommonPrefixes", [])]
    return found


def dated_days(s3) -> list[str]:
    """Every YYYYMMDD with a dated folder on the CDN, oldest first."""
    days = []
    for year in _prefixes(s3, ""):
        if not re.fullmatch(r"\d{4}/", year):
            continue
        for month in _prefixes(s3, year):
            for day in _prefixes(s3, month):
                if re.fullmatch(r"\d{4}/\d{2}/\d{2}/", day):
                    days.append(day.replace("/", ""))
    return sorted(days)


def quarter_hour_objects(s3, day: str) -> list[dict]:
    """Objects in the day's HHMM/ folders (never forecast/ or anything else)."""
    root = f"{day[:4]}/{day[4:6]}/{day[6:]}/"
    objects = []
    for folder in _prefixes(s3, root):
        if not QUARTER_HOUR.match(folder):
            continue
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=folder):
            objects += page.get("Contents", [])
    return objects


# --- steps ----------------------------------------------------------------------

def build_zip(s3, day: str, objects: list[dict]) -> dict:
    """Stream the day's objects into a zip, upload it, verify the size, remove the local copy."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    needed = sum(item["Size"] for item in objects) * 1.1 + 512 * 1024 * 1024
    free = shutil.disk_usage(WORK_DIR).free
    if free < needed:
        raise RuntimeError(f"{day}: only {free / 1e9:.1f} GB free for a ~{needed / 1e9:.1f} GB zip")
    local = WORK_DIR / f"{day}.zip"
    root = f"{day[:4]}/{day[4:6]}/{day[6:]}/"
    try:
        with zipfile.ZipFile(local, "w", allowZip64=True) as archive:
            for item in objects:
                name = item["Key"][len(root):]  # HHMM/file.png
                method = zipfile.ZIP_STORED if Path(name).suffix.lower() in STORED else zipfile.ZIP_DEFLATED
                info = zipfile.ZipInfo(name, date_time=item["LastModified"].timetuple()[:6])
                info.compress_type = method
                with archive.open(info, "w", force_zip64=True) as entry:
                    s3.download_fileobj(BUCKET, item["Key"], entry)
        with zipfile.ZipFile(local) as archive:
            if len(archive.namelist()) != len(objects) or archive.testzip() is not None:
                raise RuntimeError(f"{day}: zip verification failed")
        size = local.stat().st_size
        s3.upload_file(str(local), BUCKET, zip_key(day), ExtraArgs={"ContentType": "application/zip"})
        remote = s3.head_object(Bucket=BUCKET, Key=zip_key(day))["ContentLength"]
        if remote != size:
            raise RuntimeError(f"{day}: uploaded zip is {remote} bytes, local {size}")
        return {"key": zip_key(day), "objects": len(objects), "image_bytes": sum(i["Size"] for i in objects),
                "zip_bytes": size, "created": datetime.now(timezone.utc).isoformat()}
    finally:
        local.unlink(missing_ok=True)


def zip_is_intact(s3, entry: dict) -> bool:
    try:
        return s3.head_object(Bucket=BUCKET, Key=entry["key"])["ContentLength"] == entry["zip_bytes"]
    except Exception:  # noqa: BLE001 - missing or unreachable: treat as not safe to prune
        return False


def delete_objects(s3, keys: list[str]) -> None:
    for start in range(0, len(keys), 1000):
        batch = [{"Key": key} for key in keys[start:start + 1000]]
        response = s3.delete_objects(Bucket=BUCKET, Delete={"Objects": batch, "Quiet": True})
        if response.get("Errors"):
            raise RuntimeError(f"delete failed for {len(response['Errors'])} objects: {response['Errors'][0]}")


# --- notification -----------------------------------------------------------------

def _gb(value: float) -> str:
    return f"{value / 1e9:.1f} GB"


def notify(title: str, description: str, days: list[tuple[str, dict]]) -> bool:
    from services.discord_notifier import notify_staff_alert

    fields = []
    for day, entry in days[:20]:
        expires = ""
        if entry.get("expires"):
            expires = f"\nAvailable until {entry['expires'][:10]}"
        fields.append({"name": f"{day[:4]}-{day[4:6]}-{day[6:]}",
                       "value": f"{entry['objects']:,} images, {_gb(entry['zip_bytes'])}\n{download_url(day)}{expires}",
                       "inline": False})
    if len(days) > 20:
        fields.append({"name": "More", "value": f"{len(days) - 20} more days; see /api/map-archives", "inline": False})
    return notify_staff_alert(alert_type="archive_ready", title=title, description=description, fields=fields,
                              admin_path="/admin", force=True, background=False)


# --- driver -------------------------------------------------------------------------

def run(now: Optional[datetime] = None, s3=None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    if s3 is None:
        from services.archive_bundler import _r2_client

        s3 = _r2_client()
    ledger = load_ledger()
    today = now.strftime("%Y%m%d")
    yesterday = (now - timedelta(days=1)).strftime("%Y%m%d")
    keep_cutoff = (now - timedelta(days=keep_days())).strftime("%Y%m%d")
    created, pruned, expired, reminders = [], [], [], []

    # 1. Zip completed days (folders are named in the worker's clock; a day is
    #    complete once the next day has started, so only days before yesterday's
    #    end are taken), oldest first, a few per run.
    for day in dated_days(s3):
        if len(created) >= days_per_run():
            break
        if day >= today or day > yesterday or day in ledger:
            continue
        objects = quarter_hour_objects(s3, day)
        if not objects:
            ledger[day] = {"key": None, "objects": 0, "note": "no 15-minute folders"}
            continue
        try:
            entry = build_zip(s3, day, objects)
        except Exception as exc:  # noqa: BLE001 - leave the day for the next run
            logger.error("map archive %s failed: %s", day, exc)
            continue
        if zip_keep_days() > 0:
            entry["expires"] = (now + timedelta(days=zip_keep_days())).isoformat()
        ledger[day] = entry
        created.append(day)
        save_ledger(ledger)
        logger.info("map archive %s: %s images -> %s", day, entry["objects"], _gb(entry["zip_bytes"]))

    # 2. Prune the 15-minute folders of days past the CDN retention, only behind a good zip.
    for day, entry in sorted(ledger.items()):
        if day >= keep_cutoff or entry.get("pruned") or not entry.get("key"):
            continue
        if not zip_is_intact(s3, entry):
            logger.error("map archive %s: zip missing or wrong size; not pruning its folders", day)
            continue
        objects = quarter_hour_objects(s3, day)
        delete_objects(s3, [item["Key"] for item in objects])
        entry["pruned"] = now.isoformat()
        pruned.append(day)
        save_ledger(ledger)

    # 3. Expire old zips (after a reminder), so R2 does not keep building up.
    for day, entry in sorted(ledger.items()):
        if not entry.get("expires") or entry.get("expired"):
            continue
        expires = datetime.fromisoformat(entry["expires"])
        if now >= expires and entry.get("reminded"):
            delete_objects(s3, [entry["key"]])
            entry["expired"] = now.isoformat()
            expired.append(day)
        elif now >= expires - timedelta(days=REMIND_DAYS_BEFORE) and not entry.get("reminded"):
            reminders.append(day)
            entry["reminded"] = now.isoformat()
    save_ledger(ledger)

    # 4. Tell staff.
    if created:
        notify("Daily map archives ready to download",
               f"{len(created)} day(s) of 15-minute maps were zipped. Download them to keep a local copy"
               + (f"; each zip is removed from the CDN {zip_keep_days()} days after it is created." if zip_keep_days() else "."),
               [(day, ledger[day]) for day in created])
    if reminders:
        notify("Map archives expiring soon",
               f"These zips will be removed from the CDN within {REMIND_DAYS_BEFORE} days. Download any you want to keep.",
               [(day, ledger[day]) for day in reminders])
    return {"created": created, "pruned": pruned, "expired": expired, "reminded": reminders}


def available(ledger: Optional[dict] = None) -> list[dict]:
    """Zips currently on the CDN, newest first (for the API and the local pull script)."""
    ledger = ledger if ledger is not None else load_ledger()
    rows = []
    for day, entry in sorted(ledger.items(), reverse=True):
        if entry.get("key") and not entry.get("expired"):
            rows.append({"day": day, "url": download_url(day), "images": entry["objects"],
                         "zip_bytes": entry["zip_bytes"], "created": entry["created"],
                         "expires": entry.get("expires"), "folders_pruned": bool(entry.get("pruned"))})
    return rows


async def run_scheduled() -> None:
    """APScheduler entry point (core/scheduler.py); blocking R2 IO runs off the event loop."""
    import asyncio

    try:
        result = await asyncio.to_thread(run)
        logger.info("map archive run: %s", result)
    except Exception as exc:  # noqa: BLE001
        logger.error("map archive run failed: %s", exc, exc_info=True)
