"""Download day archive bundles from R2 (run over SSH or on your own machine).

Each day's bundle (forecasts, observations, HRRR/RRFS/ensemble, RTMA, derived products,
map zips) lives in R2 under day-archive/YYYY-MM-DD/ with a manifest of sha256 checksums.
This reads R2 directly (not the API server), verifies every file against the manifest,
and only then records a download receipt in R2, which is what allows a bundle to be
removed from R2 after its 30-day window.

    python pull_day_archives.py --list
    python pull_day_archives.py --dest ~/showmefire-archive --all-pending
    python pull_day_archives.py --dest ~/showmefire-archive --date 2026-10-16
    python pull_day_archives.py --dest ~/showmefire-archive --since 2026-10-01 --no-mark

Needs Python 3 and boto3, and R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY in the
environment (R2_BUCKET defaults to cdn-showmefire, SMF_DAY_ARCHIVE_PREFIX to day-archive).
"""
import argparse
import getpass
import hashlib
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

BUCKET = os.getenv("R2_ARCHIVE_BUCKET") or os.getenv("R2_BUCKET", "cdn-showmefire")
PREFIX = os.getenv("SMF_DAY_ARCHIVE_PREFIX", "day-archive").strip("/")


def client():
    import boto3
    from botocore.config import Config

    missing = [k for k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY") if not os.getenv(k)]
    if missing:
        sys.exit(f"Missing environment variables: {', '.join(missing)}")
    return boto3.client(
        "s3", endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        region_name="auto", config=Config(retries={"max_attempts": 5, "mode": "standard"}),
    )


def get_json(s3, key):
    try:
        return json.loads(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())
    except Exception:  # noqa: BLE001 - missing or unreadable
        return None


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parts_of(manifest):
    """(name, key, size, sha256 or None) for every downloadable file in a manifest."""
    for name, component in sorted(manifest.get("components", {}).items()):
        parts = component.get("parts") or ([component] if component.get("key") else [])
        for part in parts:
            if part.get("key"):
                yield part.get("name") or f"{name}.zip", part["key"], part.get("size"), part.get("sha256")


def download_part(s3, key, size, sha, target):
    """Download one file with resume-by-skip; returns True when it matches the manifest."""
    if target.is_file() and target.stat().st_size == size and (sha is None or sha256_file(target) == sha):
        return True
    partial = target.with_suffix(target.suffix + ".part")
    s3.download_file(BUCKET, key, str(partial))
    if partial.stat().st_size != size or (sha is not None and sha256_file(partial) != sha):
        partial.unlink(missing_ok=True)
        return False
    partial.replace(target)
    return True


def pull_day(s3, day, dest, mark):
    manifest = get_json(s3, f"{PREFIX}/{day}/manifest.json")
    if manifest is None:
        print(f"{day}: no manifest in R2")
        return False
    folder = dest / day
    folder.mkdir(parents=True, exist_ok=True)
    ok, names = True, []
    for name, key, size, sha in parts_of(manifest):
        if download_part(s3, key, size, sha, folder / name):
            names.append(name)
            print(f"{day}: {name} ok ({size / 1e6:.1f} MB)")
        else:
            ok = False
            print(f"{day}: {name} FAILED verification", file=sys.stderr)
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not manifest.get("complete"):
        print(f"{day}: note - bundle is incomplete; missing/partial: "
              + ", ".join(n for n, c in manifest["components"].items() if c.get("status") != "present"))
    if ok and mark:
        receipt = {"day": day, "downloaded_at": datetime.now(timezone.utc).isoformat(),
                   "downloaded_by": f"{getpass.getuser()}@{socket.gethostname()}", "verified": True, "files": names}
        s3.put_object(Bucket=BUCKET, Key=f"{PREFIX}/receipts/{day}.json", ContentType="application/json",
                      Body=json.dumps(receipt, indent=2).encode())
        print(f"{day}: receipt recorded in R2")
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", type=Path, help="local folder to download into")
    parser.add_argument("--list", action="store_true", help="list bundles in R2 and exit")
    parser.add_argument("--date", help="one day, YYYY-MM-DD")
    parser.add_argument("--since", help="every available day from YYYY-MM-DD")
    parser.add_argument("--all-pending", action="store_true", help="every bundle not yet downloaded")
    parser.add_argument("--no-mark", action="store_true", help="do not record a download receipt")
    args = parser.parse_args()

    s3 = client()
    ledger = get_json(s3, f"{PREFIX}/ledger.json") or {}
    receipts = {Path(o["Key"]).stem for page in s3.get_paginator("list_objects_v2").paginate(
        Bucket=BUCKET, Prefix=f"{PREFIX}/receipts/") for o in page.get("Contents", [])}
    available = {d: e for d, e in ledger.items() if not e.get("deleted")}

    if args.list or not (args.date or args.since or args.all_pending):
        for day, entry in sorted(available.items()):
            done = "downloaded" if entry.get("downloaded_verified") or day in receipts else "PENDING"
            print(f"{day}  {entry.get('bytes', 0) / 1e9:6.2f} GB  "
                  f"{'complete' if entry.get('complete') else 'incomplete':10}  expires {str(entry.get('expires'))[:10]}  {done}")
        return 0

    if not args.dest:
        parser.error("--dest is required to download")
    if args.date:
        days = [args.date]
    elif args.since:
        days = [d for d in sorted(available) if d >= args.since]
    else:
        days = [d for d, e in sorted(available.items()) if not e.get("downloaded_verified") and d not in receipts]
    failed = [d for d in days if not pull_day(s3, d, args.dest, not args.no_mark)]
    print(f"Done: {len(days) - len(failed)} of {len(days)} day(s) verified" + (f"; FAILED: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
