"""Operate the versioned archive without starting the API."""
import argparse
import json
import sys
import tempfile
import zipfile
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parents[1] / '.env')
from services.archive_store import ArchiveStore, now


def import_zip(store, path, locked=False):
    # Leave the original ZIP untouched, including unknown legacy members.
    count = 0
    with (nullcontext() if locked else store.lock()), zipfile.ZipFile(path) as zf, tempfile.TemporaryDirectory() as tmp:
        for info in zf.infolist():
            name = Path(info.filename).name
            folder = Path(info.filename).parent.name.lower()
            source = ('forecasts' if name.startswith('station_forecasts_') else
                      'observations' if name.startswith('raw_data_') else
                      folder if folder in ('hrrr', 'rrfs') else None)
            if info.is_dir() or source is None:
                continue
            dest = Path(tmp) / name
            import shutil
            import os
            from datetime import datetime, timezone
            with zf.open(info) as src, dest.open('wb') as out:
                shutil.copyfileobj(src, out)
            stamp = datetime(*info.date_time, tzinfo=timezone.utc).timestamp()
            os.utime(dest, (stamp, stamp))
            store.ingest(dest, source)
            dest.unlink()
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['inventory', 'sync', 'status', 'restore', 'rebuild', 'import-zip', 'import-r2-zips'])
    parser.add_argument('--date', help='YYYY-MM-DD for restore')
    parser.add_argument('--zip', type=Path, help='Local legacy ZIP to import')
    parser.add_argument('--prune', action='store_true', help='Remove verified local files older than retention')
    args = parser.parse_args()
    store = ArchiveStore()
    if args.command in ('inventory', 'sync'):
        result = store.sync(inventory_only=args.command == 'inventory', prune=args.prune)
    elif args.command == 'status':
        result = store.status()
    elif args.command == 'restore':
        if not args.date:
            parser.error('--date is required')
        result = {'restored': store.restore(args.date)}
    elif args.command == 'rebuild':
        result = {'indexed': store.rebuild()}
    elif args.command == 'import-zip':
        if not args.zip:
            parser.error('--zip is required')
        if not store.configured:
            parser.error('R2 credentials are required for ZIP migration')
        result = {'imported': import_zip(store, args.zip)}
    else:
        with store.lock():
            s3 = store.client()
            count = completed = failed = 0
            objects = [item for page in s3.get_paginator('list_objects_v2').paginate(
                Bucket=store.bucket, Prefix='data-archive/') for item in page.get('Contents', [])
                if item['Key'].endswith('.zip')]
            with store.db() as db:
                run_id = db.execute("INSERT INTO runs(started_at,status,total) VALUES (?, 'running', ?)", (now(), len(objects))).lastrowid
            # One ZIP at a time; neither the old objects nor working files are deleted.
            for item in objects:
                with store.db() as db:
                    db.execute('UPDATE runs SET current_file=? WHERE id=?', (item['Key'], run_id))
                try:
                    with tempfile.TemporaryDirectory() as tmp:
                        path = Path(tmp) / 'legacy.zip'
                        s3.download_file(store.bucket, item['Key'], str(path))
                        count += import_zip(store, path, locked=True)
                    completed += 1
                except Exception as exc:
                    failed += 1
                    with store.db() as db:
                        db.execute('UPDATE runs SET error=? WHERE id=?', (f"{item['Key']}: {exc}", run_id))
                with store.db() as db:
                    db.execute('UPDATE runs SET completed=?,failed=? WHERE id=?', (completed, failed, run_id))
            with store.db() as db:
                db.execute('UPDATE runs SET status=?,finished_at=?,current_file=NULL WHERE id=?',
                           ('failed' if failed else 'completed', now(), run_id))
            result = {'imported': count, 'completed': completed, 'failed': failed}
    print(json.dumps(result, indent=2))
    if isinstance(result, dict) and result.get('failed'):
        sys.exit(1)


if __name__ == '__main__':
    main()
