"""Day bundles: everything needed to reconstruct one Central day, kept in R2.

For each completed Central day this builds component zips plus a manifest under
``day-archive/YYYY-MM-DD/`` in R2, verifies each upload by reading it back, and
only then publishes ``manifest.json``. R2 is the long-term store; the server
keeps only a transient work directory and (optionally) the source files for
``SMF_DAY_ARCHIVE_LOCAL_DAYS`` days.

Ledger (``day-archive/ledger.json``, cached under ``data/day_archive/``) tracks the
30-day window. A bundle is removed from R2 only when it is past its window AND a
download receipt (``day-archive/receipts/<date>.json``, written by the CLI or the
admin endpoint after every checksum verified) exists, and only when
``SMF_DAY_ARCHIVE_DELETE=true``. Staff get Discord reminders 7/3/1 days before
the window ends, and a daily overdue notice afterwards; nothing is ever deleted
undownloaded.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import zipfile
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

CENTRAL = ZoneInfo("America/Chicago")
SCHEMA_VERSION = 1
DATE_IN_NAME = re.compile(r"(20\d{6})")
STORED = {".nc", ".zip", ".png", ".webp", ".jpg", ".jpeg", ".gif", ".grib2", ".grb2"}
SETTLE_SECONDS = 300
REMINDER_DAYS = (7, 3, 1)
# Component source files that may be removed from the server once their bundle is verified in R2.
# derived is excluded: the public /verification routes keep serving reports and peak rasters.
PRUNABLE = {"forecasts", "observations", "observations_full", "hrrr", "rrfs", "ensemble",
            "forecasts_beta", "hrrr_beta", "rtma"}


def _flag(name: str, default: str = "false") -> bool:
    return os.getenv(name, default).strip().lower() == "true"


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def prefix() -> str:
    return os.getenv("SMF_DAY_ARCHIVE_PREFIX", "day-archive").strip("/")


def bucket() -> str:
    return os.getenv("R2_ARCHIVE_BUCKET") or os.getenv("R2_BUCKET", "cdn-showmefire")


def keep_days() -> int:
    return _int("SMF_DAY_ARCHIVE_KEEP_DAYS", 30)


def local_days() -> int:
    return max(1, _int("SMF_ARCHIVE_LOCAL_DAYS", 7))


def root() -> Path:
    return Path(os.getenv("SMF_ARCHIVE_ROOT") or ("/app" if Path("/app").exists() else Path(__file__).resolve().parents[1]))


def work_dir() -> Path:
    return root() / "cache" / "day-archive"


def ledger_cache_path() -> Path:
    return root() / "data" / "day_archive" / "ledger.json"


def day_key(day: str, name: str) -> str:
    return f"{prefix()}/{day}/{name}"


def ledger_key() -> str:
    return f"{prefix()}/ledger.json"


def receipt_key(day: str) -> str:
    return f"{prefix()}/receipts/{day}.json"


def compact(day: str) -> str:
    return day.replace("-", "")


# --- Central day helpers -----------------------------------------------------------

def central_today(now: Optional[datetime] = None) -> date:
    return (now or datetime.now(timezone.utc)).astimezone(CENTRAL).date()


def central_day_hours_utc(day: str) -> list[datetime]:
    """The 24 UTC analysis hours that fall inside a Central calendar day (23 or 25 on DST days)."""
    local = date.fromisoformat(day)
    start = datetime.combine(local, time.min, tzinfo=CENTRAL).astimezone(timezone.utc)
    end = datetime.combine(local + timedelta(days=1), time.min, tzinfo=CENTRAL).astimezone(timezone.utc)
    hours, current = [], start
    while current < end:
        hours.append(current)
        current += timedelta(hours=1)
    return hours


# --- hashing and ledger ----------------------------------------------------------------

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _get_json(s3, key: str) -> Optional[dict]:
    try:
        body = s3.get_object(Bucket=bucket(), Key=key)["Body"].read()
        data = json.loads(body)
        return data if isinstance(data, dict) else None
    except Exception as exc:  # noqa: BLE001 - missing key or unreachable
        code = getattr(exc, "response", {}).get("Error", {}).get("Code") if hasattr(exc, "response") else None
        if code not in (None, "NoSuchKey", "404", "NotFound"):
            logger.warning("Could not read %s: %s", key, exc)
        return None


def _put_json(s3, key: str, payload: dict) -> None:
    s3.put_object(Bucket=bucket(), Key=key, ContentType="application/json",
                  Body=json.dumps(payload, indent=2, sort_keys=True).encode())


def load_ledger(s3) -> dict:
    """R2 is authoritative; fall back to the local cache only if R2 has none."""
    remote = _get_json(s3, ledger_key())
    if remote is not None:
        return remote
    try:
        return json.loads(ledger_cache_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_ledger(s3, ledger: dict) -> None:
    _put_json(s3, ledger_key(), ledger)
    path = ledger_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(ledger, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def read_manifest(s3, day: str) -> Optional[dict]:
    return _get_json(s3, day_key(day, "manifest.json"))


def merge_receipts(s3, ledger: dict) -> list[str]:
    """Copy download receipts (written by the CLI/endpoint) into ledger entries."""
    merged = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket(), Prefix=f"{prefix()}/receipts/"):
        for item in page.get("Contents", []):
            day = Path(item["Key"]).stem
            entry = ledger.get(day)
            if not entry or entry.get("downloaded_verified"):
                continue
            receipt = _get_json(s3, item["Key"]) or {}
            if receipt.get("verified") is True:
                entry.update(downloaded_at=receipt.get("downloaded_at"), downloaded_by=receipt.get("downloaded_by"),
                             downloaded_verified=True)
                merged.append(day)
    return merged


def write_receipt(s3, day: str, downloaded_by: str, files: Optional[list] = None, verified: bool = True) -> dict:
    receipt = {"day": day, "downloaded_at": datetime.now(timezone.utc).isoformat(),
               "downloaded_by": downloaded_by, "verified": bool(verified), "files": files or []}
    _put_json(s3, receipt_key(day), receipt)
    return receipt


# --- component collectors ---------------------------------------------------------------

def _source_files(rel_dir: str, day: str, now_ts: float) -> list[tuple[Path, str]]:
    """Settled files in a source dir whose filename date is the day (arcname keeps the source layout)."""
    base = root() / rel_dir
    if not base.is_dir():
        return []
    wanted = compact(day)
    found = []
    for path in sorted(base.glob("*")):
        match = DATE_IN_NAME.search(path.name)
        if path.is_file() and match and match[1] == wanted and now_ts - path.stat().st_mtime >= SETTLE_SECONDS:
            found.append((path, f"{rel_dir}/{path.name}"))
    return found


def collect_source(name: str, day: str, now_ts: float) -> list[tuple[Path, str]]:
    from services.archive_store import SOURCES
    return _source_files(SOURCES[name], day, now_ts)


def collect_rtma(day: str, now_ts: float) -> list[tuple[Path, str]]:
    from services.rtma_capture import cache_path_for_hour
    found = []
    for hour in central_day_hours_utc(day):
        path = cache_path_for_hour(hour, cache_dir=root() / "cache" / "rtma")
        if path.is_file():
            found.append((path, f"cache/rtma/{path.name}"))
    return found


def collect_derived(day: str, now_ts: float) -> list[tuple[Path, str]]:
    base = root()
    found: list[tuple[Path, str]] = []
    gis = base / "gis"
    if gis.is_dir():
        for archive in sorted(gis.glob("*/archive")):
            for path in sorted(archive.glob(f"{day}.*")):
                if path.is_file():
                    found.append((path, str(path.relative_to(base))))
    for report_dir in (base / "reports" / day, base / "reports" / compact(day)):
        if report_dir.is_dir():
            found += [(p, str(p.relative_to(base))) for p in sorted(report_dir.rglob("*")) if p.is_file()]
    for rel in ("archive/forecast_09z_rtma_metrics", "archive/forecast_09z_metrics"):
        path = base / rel / f"{day}.json"
        if path.is_file():
            found.append((path, f"{rel}/{day}.json"))
    return found


FILE_COMPONENTS: dict[str, Callable[[str, float], list[tuple[Path, str]]]] = {}
SOURCE_COMPONENTS = ("forecasts", "observations", "observations_full", "hrrr", "rrfs", "ensemble",
                     "forecasts_beta", "hrrr_beta", "hrrr_hourly")
for _name in SOURCE_COMPONENTS:
    FILE_COMPONENTS[_name] = (lambda day, ts, _n=_name: collect_source(_n, day, ts))
FILE_COMPONENTS["rtma"] = collect_rtma
FILE_COMPONENTS["derived"] = collect_derived


def map_zip_days(day: str) -> list[str]:
    """Map zips are named by the worker clock's (UTC) day; a Central day spans two of them."""
    start = datetime.combine(date.fromisoformat(day), time.min, tzinfo=CENTRAL).astimezone(timezone.utc)
    last = start + timedelta(hours=23, minutes=59)
    return sorted({start.strftime("%Y%m%d"), last.strftime("%Y%m%d")})


# --- build ---------------------------------------------------------------------------------

def _free_space_check(path: Path, needed: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path).free
    if free < needed * 1.2 + 256 * 1024 * 1024:
        raise RuntimeError(f"only {free / 1e9:.1f} GB free for ~{needed / 1e9:.2f} GB of archive work")


def _zip_files(files: list[tuple[Path, str]], target: Path) -> list[dict]:
    members = []
    with zipfile.ZipFile(target, "w", allowZip64=True) as archive:
        for path, arcname in files:
            digest = hashlib.sha256()
            size = 0
            info = zipfile.ZipInfo.from_file(path, arcname)
            info.compress_type = zipfile.ZIP_STORED if path.suffix.lower() in STORED else zipfile.ZIP_DEFLATED
            with path.open("rb") as source, archive.open(info, "w", force_zip64=True) as entry:
                for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
                    entry.write(chunk)
            members.append({"name": arcname, "size": size, "sha256": digest.hexdigest(),
                            "src": str(path.relative_to(root())) if path.is_relative_to(root()) else None})
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None or len(archive.namelist()) != len(files):
            raise RuntimeError(f"{target.name}: zip verification failed")
    return members


def verify_remote(s3, key: str, size: int, sha256: str) -> None:
    """Read the object back and compare length and sha256 (R2 egress is free)."""
    digest, total = hashlib.sha256(), 0
    body = s3.get_object(Bucket=bucket(), Key=key)["Body"]
    for chunk in iter(lambda: body.read(8 * 1024 * 1024), b""):
        digest.update(chunk)
        total += len(chunk)
    if total != size or digest.hexdigest() != sha256:
        raise RuntimeError(f"{key}: read-back mismatch ({total} bytes vs {size})")


def build_file_component(s3, day: str, name: str, files: list[tuple[Path, str]]) -> dict:
    needed = sum(path.stat().st_size for path, _ in files)
    work = work_dir()
    _free_space_check(work, needed)
    local = work / f"{day}_{name}.zip"
    try:
        members = _zip_files(files, local)
        size, sha = local.stat().st_size, sha256_file(local)
        key = day_key(day, f"{name}.zip")
        s3.upload_file(str(local), bucket(), key, ExtraArgs={"ContentType": "application/zip"})
        verify_remote(s3, key, size, sha)
        return {"status": "present", "key": key, "size": size, "sha256": sha, "members": members,
                "built_at": datetime.now(timezone.utc).isoformat()}
    finally:
        local.unlink(missing_ok=True)


def build_maps_component(s3, day: str) -> dict:
    """Server-side copy of the day's map-archive zips (size-checked; the CLI verifies on download)."""
    parts, missing = [], []
    for utc_day in map_zip_days(day):
        source = f"map-archive/{utc_day}.zip"
        try:
            size = s3.head_object(Bucket=bucket(), Key=source)["ContentLength"]
        except Exception:  # noqa: BLE001 - not built yet
            missing.append(utc_day)
            continue
        name = f"maps_utc_{utc_day}.zip"
        key = day_key(day, name)
        s3.copy({"Bucket": bucket(), "Key": source}, bucket(), key)
        if s3.head_object(Bucket=bucket(), Key=key)["ContentLength"] != size:
            raise RuntimeError(f"{key}: copied size mismatch")
        parts.append({"name": name, "key": key, "size": size, "source_key": source})
    if not parts:
        return {"status": "missing", "reason": "map zips not built yet", "missing": missing}
    return {"status": "partial" if missing else "present", "parts": parts, "missing": missing,
            "size": sum(p["size"] for p in parts), "verified_by": "size",
            "built_at": datetime.now(timezone.utc).isoformat()}


def build_day(s3, day: str, now: Optional[datetime] = None, previous: Optional[dict] = None) -> Optional[dict]:
    """Build/complete one day's bundle and publish its manifest. None when there is nothing to archive."""
    now = now or datetime.now(timezone.utc)
    now_ts = now.timestamp()
    components: dict[str, Any] = dict((previous or {}).get("components", {}))
    for name, collect in FILE_COMPONENTS.items():
        if components.get(name, {}).get("status") == "present":
            continue
        files = collect(day, now_ts)
        if not files:
            components[name] = {"status": "missing", "reason": "no local files for this day"}
            continue
        try:
            components[name] = build_file_component(s3, day, name, files)
        except Exception as exc:  # noqa: BLE001 - leave for the next run
            logger.error("day archive %s/%s failed: %s", day, name, exc)
            components[name] = {"status": "failed", "error": str(exc)}
    if components.get("maps", {}).get("status") != "present":
        try:
            components["maps"] = build_maps_component(s3, day)
        except Exception as exc:  # noqa: BLE001
            logger.error("day archive %s/maps failed: %s", day, exc)
            components["maps"] = {"status": "failed", "error": str(exc)}
    if not any(c.get("status") in ("present", "partial") for c in components.values()):
        return None
    manifest = {
        "schema_version": SCHEMA_VERSION, "day": day, "day_definition": "America/Chicago calendar day",
        "built_at": (previous or {}).get("built_at") or now.isoformat(), "updated_at": now.isoformat(),
        "components": components,
        "complete": is_complete(components),
        "notes": ["Source files are selected by the date in their filename (UTC run date); RTMA by Central-day "
                  "hours; map zips by UTC day, so a Central day spans two map zips.",
                  "A component marked missing had no data on the server when the bundle was built."],
    }
    _put_json(s3, day_key(day, "manifest.json"), manifest)
    return manifest


REQUIRED = ("forecasts", "observations", "derived", "maps")


def is_complete(components: dict) -> bool:
    """Required components present and nothing failed/partial; optional ones (hrrr, ensemble...) may be absent."""
    if any(c.get("status") in ("failed", "partial") for c in components.values()):
        return False
    return all(components.get(name, {}).get("status") == "present" for name in REQUIRED)


def bundle_bytes(manifest: dict) -> int:
    total = 0
    for component in manifest.get("components", {}).values():
        total += int(component.get("size") or 0)
    return total


# --- local retention -------------------------------------------------------------------------

def prune_local(manifest: dict, today: date, dry_run: bool) -> dict:
    """Delete this day's source files from the server once they are verified in R2.

    Only days older than the local window, only components with status present (zip read back
    from R2), and only files whose sha256 still equals the manifest's.
    """
    day = date.fromisoformat(manifest["day"])
    result = {"deleted": 0, "would_delete": 0, "kept": 0}
    if (today - day).days <= local_days():
        return result
    for name, component in manifest.get("components", {}).items():
        if name not in PRUNABLE or component.get("status") != "present":
            continue
        for member in component.get("members", []):
            src = member.get("src")
            path = root() / src if src else None
            if path is None or not path.is_file() or sha256_file(path) != member["sha256"]:
                result["kept"] += 1
                continue
            if dry_run:
                result["would_delete"] += 1
            else:
                path.unlink()
                result["deleted"] += 1
    return result


# --- window, reminders, deletion --------------------------------------------------------------

def eligible_for_delete(entry: dict, now: datetime) -> bool:
    """A bundle may leave R2 only when past its window, downloaded and verified, and not already gone."""
    if entry.get("deleted") or not entry.get("expires"):
        return False
    return bool(entry.get("downloaded_verified")) and now >= datetime.fromisoformat(entry["expires"])


def reminder_due(entry: dict, now: datetime) -> Optional[str]:
    """'7'/'3'/'1' once per threshold before expiry, 'overdue' once a day after; None if nothing to send."""
    if entry.get("deleted") or entry.get("downloaded_verified") or not entry.get("expires"):
        return None
    expires = datetime.fromisoformat(entry["expires"])
    if now >= expires:
        return "overdue" if entry.get("overdue_notified") != now.date().isoformat() else None
    sent = set(entry.get("reminders", []))
    crossed = [d for d in REMINDER_DAYS if expires - now <= timedelta(days=d)]
    if not crossed or str(min(crossed)) in sent:
        return None
    return str(min(crossed))


def delete_bundle(s3, day: str) -> int:
    keys = []
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket(), Prefix=f"{prefix()}/{day}/"):
        keys += [{"Key": item["Key"]} for item in page.get("Contents", [])]
    for start in range(0, len(keys), 1000):
        response = s3.delete_objects(Bucket=bucket(), Delete={"Objects": keys[start:start + 1000], "Quiet": True})
        if response.get("Errors"):
            raise RuntimeError(f"delete failed for {len(response['Errors'])} objects")
    return len(keys)


def _gb(value: float) -> str:
    return f"{value / 1e9:.2f} GB"


def notify(title: str, description: str, rows: list[tuple[str, dict]]) -> bool:
    from services.discord_notifier import notify_staff_alert

    fields = []
    for day, entry in rows[:10]:
        left = ""
        if entry.get("expires"):
            left = f"\nExpires {entry['expires'][:10]}"
        state = "complete" if entry.get("complete") else "incomplete" + (
            f" ({', '.join(entry['pending'])})" if entry.get("pending") else "")
        fields.append({"name": day, "value": f"{_gb(entry.get('bytes', 0))}, {state}{left}", "inline": False})
    if len(rows) > 10:
        fields.append({"name": "More", "value": f"{len(rows) - 10} more days; GET /admin/day-archive", "inline": False})
    return notify_staff_alert(alert_type="archive_ready", title=title, description=description, fields=fields,
                              admin_path="/admin", force=True, background=False)


def _pending(manifest: dict) -> list[str]:
    return sorted(n for n, c in manifest["components"].items() if c.get("status") in ("failed", "partial"))


# --- driver ------------------------------------------------------------------------------------

def target_days(now: datetime, ledger: dict) -> list[str]:
    """Completed Central days not yet bundled, or incomplete within the retry window, oldest first."""
    today = central_today(now)
    backfill = _int("SMF_DAY_ARCHIVE_BACKFILL_DAYS", local_days())
    retry = _int("SMF_DAY_ARCHIVE_RETRY_DAYS", 4)
    days = []
    for offset in range(backfill, 0, -1):
        day = (today - timedelta(days=offset)).isoformat()
        entry = ledger.get(day)
        if entry is None or (not entry.get("complete") and offset <= retry and not entry.get("deleted")):
            days.append(day)
    return days


def run(now: Optional[datetime] = None, s3=None) -> dict[str, Any]:
    now = now or datetime.now(timezone.utc)
    if s3 is None:
        from services.archive_bundler import _r2_client
        s3 = _r2_client()
    ledger = load_ledger(s3)
    result: dict[str, Any] = {"built": [], "updated": [], "pruned": {}, "reminded": [], "overdue": [], "deleted": [],
                              "received": merge_receipts(s3, ledger)}

    built_rows = []
    for day in target_days(now, ledger)[:_int("SMF_DAY_ARCHIVE_DAYS_PER_RUN", 3)]:
        previous = read_manifest(s3, day) if day in ledger else None
        try:
            manifest = build_day(s3, day, now=now, previous=previous)
        except Exception as exc:  # noqa: BLE001
            logger.error("day archive %s failed: %s", day, exc, exc_info=True)
            notify("Day archive failed", f"{day}: {exc}", [])
            continue
        if manifest is None:
            continue
        entry = ledger.setdefault(day, {"created": now.isoformat(),
                                        "expires": (now + timedelta(days=keep_days())).isoformat()})
        entry.update(complete=manifest["complete"], bytes=bundle_bytes(manifest), pending=_pending(manifest),
                     components=sorted(manifest["components"]), updated=now.isoformat())
        (result["updated"] if previous else result["built"]).append(day)
        if not previous:
            built_rows.append((day, entry))
        save_ledger(s3, ledger)

    dry_run = not _flag("SMF_DAY_ARCHIVE_PRUNE_LOCAL")
    today = central_today(now)
    for day, entry in sorted(ledger.items()):
        if entry.get("deleted") or not entry.get("components"):
            continue
        manifest = read_manifest(s3, day)
        if manifest:
            outcome = prune_local(manifest, today, dry_run)
            if outcome["deleted"] or outcome["would_delete"]:
                result["pruned"][day] = outcome

    reminders, overdue = [], []
    for day, entry in sorted(ledger.items()):
        due = reminder_due(entry, now)
        if due == "overdue":
            entry["overdue_notified"] = now.date().isoformat()
            overdue.append((day, entry))
        elif due:
            entry.setdefault("reminders", []).extend(str(d) for d in REMINDER_DAYS if d >= int(due))
            reminders.append((day, entry))
    for day, entry in sorted(ledger.items()):
        if eligible_for_delete(entry, now):
            if _flag("SMF_DAY_ARCHIVE_DELETE"):
                delete_bundle(s3, day)
                entry["deleted"] = now.isoformat()
                result["deleted"].append(day)
            else:
                logger.info("day archive %s eligible for R2 deletion (SMF_DAY_ARCHIVE_DELETE is off)", day)
    save_ledger(s3, ledger)

    if built_rows:
        notify("Day archives ready to download",
               f"{len(built_rows)} day bundle(s) are in R2. Download with scripts/pull_day_archives.py; "
               f"each is removed {keep_days()} days after creation once it has been downloaded.", built_rows)
    if reminders:
        notify("Day archives expiring soon",
               "Download these before their window ends (pull_day_archives.py --all-pending).", reminders)
    if overdue:
        notify("Day archives overdue - NOT deleted",
               "Past the window but not downloaded; they stay in R2 until downloaded.", overdue)
    result["reminded"] = [d for d, _ in reminders]
    result["overdue"] = [d for d, _ in overdue]
    return result


# --- endpoint helpers ---------------------------------------------------------------------------

def list_days(s3) -> list[dict]:
    ledger = load_ledger(s3)
    now = datetime.now(timezone.utc)
    rows = []
    for day, entry in sorted(ledger.items(), reverse=True):
        expires = entry.get("expires")
        days_left = None
        if expires and not entry.get("deleted"):
            days_left = (datetime.fromisoformat(expires) - now).days
        rows.append({"day": day, "bytes": entry.get("bytes"), "complete": entry.get("complete"),
                     "pending": entry.get("pending", []), "created": entry.get("created"), "expires": expires,
                     "days_left": days_left, "downloaded": bool(entry.get("downloaded_verified")),
                     "downloaded_at": entry.get("downloaded_at"), "downloaded_by": entry.get("downloaded_by"),
                     "deleted": bool(entry.get("deleted"))})
    return rows


def presigned_links(s3, day: str, expires_in: int = 3600) -> Optional[list[dict]]:
    manifest = read_manifest(s3, day)
    if manifest is None:
        return None
    links = [{"name": "manifest.json", "size": None, "url": s3.generate_presigned_url(
        "get_object", Params={"Bucket": bucket(), "Key": day_key(day, "manifest.json")}, ExpiresIn=expires_in)}]
    for name, component in sorted(manifest["components"].items()):
        parts = component.get("parts") or ([component] if component.get("key") else [])
        for part in parts:
            if part.get("key"):
                links.append({"name": part.get("name") or f"{name}.zip", "component": name, "size": part.get("size"),
                              "sha256": part.get("sha256"), "url": s3.generate_presigned_url(
                    "get_object", Params={"Bucket": bucket(), "Key": part["key"]}, ExpiresIn=expires_in)})
    return links


async def run_scheduled() -> None:
    """APScheduler entry point; blocking R2 and zip work runs off the event loop."""
    import asyncio

    try:
        result = await asyncio.to_thread(run)
        logger.info("day archive run: %s", result)
    except Exception as exc:  # noqa: BLE001
        logger.error("day archive run failed: %s", exc, exc_info=True)
