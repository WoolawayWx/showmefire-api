"""
End-of-day archiving: bundle each day's HRRR grids, station forecasts, and
raw station observations into one verified zip per date, then delete the
originals for that date once the zip is confirmed intact.

Groups files by the date embedded in their filenames, processes one date at
a time (zip, verify, delete), keeping peak extra disk usage to about one
day's worth of data - cache/hrrr alone runs into the tens of GB, so disk
space is tight.

Path resolution mirrors core/database.py's get_db_path(): prefer the
Docker-mounted /app root when present, falling back to the project root for
local dev. This means source/output locations move together with wherever
docker-compose (or its production override) actually mounts cache/archive/
data_archive_day on the host - no separate env var needed here.

Runs both as a standalone script (useful for working through the existing
backlog) and as a scheduled job via core/scheduler.py.
"""
import asyncio
import logging
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()


def _resolve_root():
    docker_root = Path("/app")
    if docker_root.exists():
        return docker_root
    return Path(__file__).resolve().parent.parent


ROOT_DIR = _resolve_root()
SOURCE_DIRS = [
    ROOT_DIR / "cache" / "hrrr",
    ROOT_DIR / "archive" / "forecasts",
    ROOT_DIR / "archive" / "raw_data",
]
OUTPUT_DIR = ROOT_DIR / "data_archive_day"

# Finished daily zips are pushed here and removed from local disk immediately
# after a verified upload - the server is meant to hold at most one day's
# in-progress zip at a time, not an ever-growing archive.
R2_ACCESS_KEY_ID = os.getenv("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.getenv("R2_SECRET_ACCESS_KEY")
R2_ACCOUNT_ID = os.getenv("R2_ACCOUNT_ID")
R2_BUCKET = os.getenv("R2_BUCKET", "cdn-showmefire")
R2_ARCHIVE_PREFIX = "data-archive"

DATE_RE = re.compile(r"(20\d{6})")

# .nc files are large binary blobs that barely compress; storing them
# uncompressed is much faster and avoids burning CPU/time for no space savings.
STORED_SUFFIXES = {".nc"}


def date_for_file(path: Path):
    match = DATE_RE.search(path.name)
    return match.group(1) if match else None


def collect_files_by_date():
    files_by_date = {}
    for source_dir in SOURCE_DIRS:
        if not source_dir.is_dir():
            logger.warning(f"source directory not found, skipping: {source_dir}")
            continue
        for path in source_dir.iterdir():
            if not path.is_file():
                continue
            date = date_for_file(path)
            if date is None:
                logger.warning(f"no date found in filename, skipping: {path}")
                continue
            files_by_date.setdefault(date, []).append(path)
    return files_by_date


def write_zip_for_date(date, paths):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = OUTPUT_DIR / f"{date}.zip.tmp"
    final_path = OUTPUT_DIR / f"{date}.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for path in sorted(paths):
            compression = (
                zipfile.ZIP_STORED
                if path.suffix in STORED_SUFFIXES
                else zipfile.ZIP_DEFLATED
            )
            arcname = f"{path.parent.name}/{path.name}"
            zf.write(path, arcname=arcname, compress_type=compression)
    zip_path.replace(final_path)
    return final_path


def verify_zip(zip_path, paths):
    expected = {f"{path.parent.name}/{path.name}": path.stat().st_size for path in paths}
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad_file = zf.testzip()
        if bad_file is not None:
            return False, f"corrupt member: {bad_file}"
        actual = {info.filename: info.file_size for info in zf.infolist()}
    if actual != expected:
        return False, f"member mismatch: expected {expected}, got {actual}"
    return True, ""


def add_files_to_existing_zip(date, paths):
    """Atomically add new daily members without mutating the only archive.

    Existing members win by archive name. This makes retries idempotent while
    still allowing a partially captured RTMA day to receive missing hours.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_path = OUTPUT_DIR / f"{date}.zip"
    temp_path = OUTPUT_DIR / f"{date}.zip.merge.tmp"
    shutil.copy2(final_path, temp_path)

    added = []
    try:
        with zipfile.ZipFile(temp_path, "a") as zf:
            existing = {info.filename: info.file_size for info in zf.infolist()}
            for path in sorted(paths):
                arcname = f"{path.parent.name}/{path.name}"
                if arcname in existing:
                    if existing[arcname] != path.stat().st_size:
                        raise RuntimeError(f"existing member differs from source: {arcname}")
                    continue
                compression = zipfile.ZIP_STORED if path.suffix in STORED_SUFFIXES else zipfile.ZIP_DEFLATED
                zf.write(path, arcname=arcname, compress_type=compression)
                existing[arcname] = path.stat().st_size
                added.append((path, arcname, path.stat().st_size))

        with zipfile.ZipFile(temp_path, "r") as zf:
            bad_file = zf.testzip()
            if bad_file is not None:
                raise RuntimeError(f"corrupt member after merge: {bad_file}")
            actual = {info.filename: info.file_size for info in zf.infolist()}
            for _, arcname, size in added:
                if actual.get(arcname) != size:
                    raise RuntimeError(f"merged member mismatch: {arcname}")

        temp_path.replace(final_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    for path, _, _ in added:
        path.unlink()
    # Sources already represented in the archive are also safe to remove.
    added_paths = {item[0] for item in added}
    for path in paths:
        if path not in added_paths:
            path.unlink()
    logger.info(f"{date}: atomically merged {len(added)} new file(s) into {final_path}")
    return final_path


def _r2_credentials_present():
    return all([R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ACCOUNT_ID])


def _r2_client():
    import boto3
    from botocore.config import Config

    return boto3.client(
        service_name="s3",
        endpoint_url=f"https://{R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=R2_ACCESS_KEY_ID,
        aws_secret_access_key=R2_SECRET_ACCESS_KEY,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def _r2_key_for_date(date):
    return f"{R2_ARCHIVE_PREFIX}/{date}.zip"


def restore_from_r2_if_needed(date, final_path):
    """Pull a previously-uploaded zip back down so a later run for the same
    date (e.g. backlog reprocessing) can still merge into it instead of
    silently starting over and clobbering the R2 object with a partial zip.
    """
    if final_path.exists() or not _r2_credentials_present():
        return
    key = _r2_key_for_date(date)
    try:
        s3 = _r2_client()
        s3.head_object(Bucket=R2_BUCKET, Key=key)
    except Exception:
        return
    try:
        final_path.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(R2_BUCKET, key, str(final_path))
        logger.info(f"{date}: restored existing archive from R2 ({R2_BUCKET}/{key}) for merge")
    except Exception as e:
        logger.warning(f"{date}: failed to restore existing archive from R2 ({R2_BUCKET}/{key}): {e}")


def upload_and_remove_local(date, zip_path):
    """Push the finished zip to R2 and delete the local copy. Leaves the zip
    on disk (rather than losing data) if credentials are missing or the
    upload fails, so the next run can retry."""
    if not _r2_credentials_present():
        logger.warning(f"{date}: R2 credentials not configured; leaving archive on local disk at {zip_path}")
        return
    key = _r2_key_for_date(date)
    try:
        s3 = _r2_client()
        s3.upload_file(str(zip_path), R2_BUCKET, key, ExtraArgs={"ContentType": "application/zip"})
    except Exception as e:
        logger.error(f"{date}: failed to upload archive to R2 ({R2_BUCKET}/{key}); leaving local copy at {zip_path}: {e}")
        return
    zip_path.unlink()
    logger.info(f"{date}: uploaded archive to R2 ({R2_BUCKET}/{key}) and removed local copy")


def process_date(date, paths):
    final_path = OUTPUT_DIR / f"{date}.zip"
    restore_from_r2_if_needed(date, final_path)

    if final_path.exists():
        final_path = add_files_to_existing_zip(date, paths)
    else:
        zip_path = write_zip_for_date(date, paths)
        ok, reason = verify_zip(zip_path, paths)
        if not ok:
            logger.error(f"{date}: verification FAILED ({reason}); leaving originals and zip in place for inspection")
            return

        freed = 0
        for path in paths:
            freed += path.stat().st_size
            path.unlink()

        logger.info(f"{date}: wrote {len(paths)} file(s) -> {zip_path} (verified, freed {freed / 1e6:.1f} MB)")
        final_path = zip_path

    upload_and_remove_local(date, final_path)


def run_archive_bundle():
    """Sync entry point - safe to run standalone (e.g. against the backlog)."""
    files_by_date = collect_files_by_date()
    if not files_by_date:
        logger.info("No matching files found.")
        return

    for date in sorted(files_by_date):
        process_date(date, files_by_date[date])


async def run_end_of_day_archive():
    """Async entry point for the scheduler - offloads the blocking zip/IO
    work to a thread so it doesn't stall the event loop while processing
    what can be tens of GB of data."""
    try:
        await asyncio.to_thread(run_archive_bundle)
    except Exception as e:
        logger.error(f"end-of-day archive run failed: {e}", exc_info=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    run_archive_bundle()
