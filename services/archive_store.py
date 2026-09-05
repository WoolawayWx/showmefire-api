"""Versioned R2 archive with a rebuildable SQLite inventory.

R2 owns immutable data and manifests. SQLite is an operational index, not the
only copy of metadata. Live files remain a working cache until explicit pruning.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import shutil
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

log = logging.getLogger(__name__)
SOURCES = {'hrrr': 'cache/hrrr', 'rrfs': 'cache/RRFS',
           'forecasts': 'archive/forecasts', 'observations': 'archive/raw_data',
           'forecasts_beta': 'data/testbed/forecast/archive/forecasts',
           'hrrr_beta': 'data/testbed/forecast/cache/hrrr'}
DATE = re.compile(r'(20\d{6})')


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def summarize(path, source):
    result = {'stations': None, 'records': None, 'start': None, 'end': None,
              'channel': 'beta' if 'beta' in path.name.lower() else 'primary'}
    times = []
    if source.startswith('forecasts') or source == 'observations':
        data = json.loads(path.read_text())
        if source.startswith('forecasts'):
            stations = data['stations']
            rows = stations.values() if isinstance(stations, dict) else stations
            for station in rows:
                times.extend(row['time'] for row in station.get('forecasts', []) if row.get('time'))
        else:
            stations = data['STATION']
            for station in stations:
                times.extend(station.get('OBSERVATIONS', {}).get('date_time', []))
        result.update(stations=len(stations), records=len(times))
        if times:
            normalized = [datetime.fromisoformat(t.replace('Z', '+00:00')).astimezone(timezone.utc).isoformat() for t in times]
            result.update(start=min(normalized), end=max(normalized))
    return result


class ArchiveStore:
    def __init__(self, root=None, client=None):
        self.root = Path(root or os.getenv('SMF_ARCHIVE_ROOT') or
                         ('/app' if Path('/app').exists() else Path(__file__).resolve().parents[1]))
        self.bucket = os.getenv('R2_ARCHIVE_BUCKET') or os.getenv('R2_BUCKET', 'cdn-showmefire')
        self.prefix = os.getenv('R2_DATA_PREFIX', 'data-v2').strip('/')
        namespace = hashlib.sha256(f'{self.bucket}/{self.prefix}'.encode()).hexdigest()[:12]
        self.db_path = self.root / 'data' / f'archive_inventory_{namespace}.sqlite3'
        self.client_override = client
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS objects (
                    key TEXT PRIMARY KEY, source TEXT NOT NULL, date TEXT NOT NULL,
                    name TEXT NOT NULL, sha256 TEXT NOT NULL, size INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL, status TEXT NOT NULL, error TEXT,
                    manifest TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS objects_date ON objects(date, source);
                CREATE TABLE IF NOT EXISTS transfer_progress (
                    run_id INTEGER PRIMARY KEY, uploaded INTEGER NOT NULL,
                    total INTEGER NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, started_at TEXT NOT NULL,
                    finished_at TEXT, status TEXT NOT NULL, total INTEGER DEFAULT 0,
                    completed INTEGER DEFAULT 0, failed INTEGER DEFAULT 0,
                    current_file TEXT, error TEXT);
            ''')

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.db_path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    @contextmanager
    def lock(self):
        import fcntl
        with self.db_path.with_suffix('.lock').open('a') as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError('An archive operation is already running') from exc
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    @property
    def configured(self):
        return self.client_override is not None or all(os.getenv(k) for k in
            ('R2_ACCOUNT_ID', 'R2_ACCESS_KEY_ID', 'R2_SECRET_ACCESS_KEY'))

    def client(self):
        if self.client_override is not None:
            return self.client_override
        if not self.configured:
            raise RuntimeError('R2 archive credentials are not configured')
        from services.archive_bundler import _r2_client
        return _r2_client()

    def record(self, manifest, status, error=None):
        with self.db() as db:
            db.execute('''INSERT INTO objects VALUES (?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(key) DO UPDATE SET status=excluded.status,
                error=excluded.error, updated_at=excluded.updated_at,
                mtime_ns=excluded.mtime_ns, manifest=excluded.manifest''',
                (manifest['key'], manifest['source'], manifest['date'], manifest['name'],
                 manifest['sha256'], manifest['size'], manifest['mtime_ns'], status,
                 error, json.dumps(manifest), now()))

    def ingest(self, path, source, inventory_only=False, progress_callback=None):
        path = Path(path)
        if source not in SOURCES or path.suffix.lower() not in ('.json', '.nc', '.grib2', '.grb2'):
            raise ValueError(f'Unsupported archive source or file: {source}/{path.name}')
        match = DATE.search(path.name)
        if not match:
            raise ValueError(f'No date in filename: {path.name}')
        date = datetime.strptime(match[1], '%Y%m%d').strftime('%Y-%m-%d')
        before = path.stat()
        with self.db() as db:
            cached = db.execute("SELECT manifest FROM objects WHERE source=? AND name=? AND size=? AND mtime_ns=? AND status='verified' ORDER BY updated_at DESC LIMIT 1",
                                (source, path.name, before.st_size, before.st_mtime_ns)).fetchone()
        if cached:
            return json.loads(cached['manifest'])
        sha = digest(path)
        key = f'{self.prefix}/objects/{date}/{source}/{sha}/{path.name}'
        manifest = dict(schema_version=2, key=key, source=source, date=date,
                        name=path.name, sha256=sha, size=before.st_size,
                        mtime_ns=before.st_mtime_ns, captured_at=now())
        with self.db() as db:
            previous = db.execute('SELECT status FROM objects WHERE key=?', (key,)).fetchone()
        try:
            manifest.update(summarize(path, source))
            self.record(manifest, 'pending')
            if inventory_only or not self.configured:
                if previous and previous['status'] == 'verified':
                    self.record(manifest, 'verified')
                return manifest
            s3 = self.client()
            if not previous or previous['status'] != 'verified':
                self.record(manifest, 'uploading')
                # Upload a stable snapshot so concurrent writers cannot damage an
                # already-existing content-addressed object during a retry.
                with tempfile.TemporaryDirectory(prefix='smf-archive-') as temp_dir:
                    snapshot = Path(temp_dir) / path.name
                    shutil.copyfile(path, snapshot)
                    if digest(snapshot) != sha:
                        raise RuntimeError('Source changed before upload; retained for retry')
                    s3.upload_file(str(snapshot), self.bucket, key,
                                   ExtraArgs={'Metadata': {'sha256': sha}}, Callback=progress_callback)
                # Read back the bytes: multipart ETags are not content hashes.
                body = s3.get_object(Bucket=self.bucket, Key=key)['Body']
                remote_hash = hashlib.sha256()
                try:
                    for chunk in iter(lambda: body.read(8 * 1024 * 1024), b''):
                        remote_hash.update(chunk)
                finally:
                    body.close()
                if remote_hash.hexdigest() != sha:
                    raise RuntimeError('R2 content verification failed')
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or digest(path) != sha:
                raise RuntimeError('Source changed during upload; retained for retry')
            manifest_key = key.replace('/objects/', '/manifests/', 1) + '.json'
            s3.put_object(Bucket=self.bucket, Key=manifest_key,
                          Body=json.dumps(manifest).encode(), ContentType='application/json')
            self.record(manifest, 'verified')
            return manifest
        except Exception as exc:
            self.record(manifest, 'failed', str(exc))
            raise

    def sync(self, inventory_only=False, prune=False):
        with self.lock():
            with self.db() as db:
                db.execute("UPDATE runs SET status='interrupted', finished_at=? WHERE status='running'", (now(),))
                run_id = db.execute("INSERT INTO runs(started_at,status) VALUES (?, 'running')", (now(),)).lastrowid
            failed = completed = 0
            try:
                settle = int(os.getenv('SMF_ARCHIVE_SETTLE_SECONDS', '300'))
                cutoff = datetime.now(timezone.utc).timestamp() - settle
                files = [(p, source) for source, folder in SOURCES.items()
                         for p in sorted((self.root / folder).glob('*'))
                         if p.is_file() and p.suffix.lower() in ('.json', '.nc', '.grib2', '.grb2')
                         and DATE.search(p.name) and p.stat().st_mtime <= cutoff]
                with self.db() as db:
                    db.execute('UPDATE runs SET total=? WHERE id=?', (len(files), run_id))
                for path, source in files:
                    with self.db() as db:
                        db.execute('UPDATE runs SET current_file=? WHERE id=?', (path.name, run_id))
                    uploaded = 0
                    last_report = 0.0
                    transfer_lock = threading.Lock()
                    total_bytes = path.stat().st_size
                    with self.db() as db:
                        db.execute('INSERT OR REPLACE INTO transfer_progress VALUES (?,?,?,?)', (run_id, 0, total_bytes, now()))

                    def report_transfer(amount):
                        nonlocal uploaded, last_report
                        with transfer_lock:
                            uploaded += amount
                            if time.monotonic() - last_report < 1 and uploaded < total_bytes:
                                return
                            last_report = time.monotonic()
                            with self.db() as db:
                                db.execute('UPDATE transfer_progress SET uploaded=?, updated_at=? WHERE run_id=?', (uploaded, now(), run_id))

                    try:
                        manifest = self.ingest(path, source, inventory_only, report_transfer)
                        if prune and not inventory_only and self.configured:
                            retention = max(1, int(os.getenv('SMF_ARCHIVE_LOCAL_DAYS', '7')))
                            expiry = datetime.now(timezone.utc) - timedelta(days=retention)
                            if path.stat().st_mtime < expiry.timestamp() and manifest['date'] < expiry.date().isoformat() and digest(path) == manifest['sha256']:
                                path.unlink()
                        completed += 1
                    except Exception as exc:
                        failed += 1
                        log.exception('Archive failed for %s', path)
                        with self.db() as db:
                            db.execute('UPDATE runs SET error=? WHERE id=?', (f'{path.name}: {exc}', run_id))
                    with self.db() as db:
                        db.execute('UPDATE runs SET completed=?, failed=? WHERE id=?', (completed, failed, run_id))
                status = 'failed' if failed else ('inventoried' if inventory_only or not self.configured else 'completed')
                with self.db() as db:
                    db.execute('UPDATE runs SET status=?,finished_at=?,current_file=NULL WHERE id=?', (status, now(), run_id))
            except Exception as exc:
                with self.db() as db:
                    db.execute("UPDATE runs SET status='failed',finished_at=?,error=? WHERE id=?", (now(), str(exc), run_id))
                raise
            return {'run_id': run_id, 'status': status, 'total': len(files), 'completed': completed, 'failed': failed}

    def remote_manifests(self, date=None):
        prefix = f'{self.prefix}/manifests/' + (date + '/' if date else '')
        s3 = self.client()
        for page in s3.get_paginator('list_objects_v2').paginate(Bucket=self.bucket, Prefix=prefix):
            for item in page.get('Contents', []):
                body = s3.get_object(Bucket=self.bucket, Key=item['Key'])['Body']
                try:
                    manifest = json.loads(body.read())
                finally:
                    body.close()
                self.validate_manifest(manifest)
                yield manifest

    def validate_manifest(self, m):
        if m.get('schema_version') != 2 or m.get('source') not in SOURCES:
            raise ValueError('Unsupported archive manifest')
        datetime.strptime(m['date'], '%Y-%m-%d')
        if Path(m['name']).name != m['name'] or m['name'] in ('.', '..') or '\\' in m['name']:
            raise ValueError('Invalid archive filename')
        if not re.fullmatch('[0-9a-f]{64}', m['sha256']):
            raise ValueError('Invalid archive digest')
        expected = f"{self.prefix}/objects/{m['date']}/{m['source']}/{m['sha256']}/{m['name']}"
        if m['key'] != expected:
            raise ValueError('Archive key does not match manifest')

    def rebuild(self):
        with self.lock():
            count = 0
            for m in self.remote_manifests():
                self.record(m, 'verified')
                count += 1
            return count

    def restore(self, date):
        datetime.strptime(date, '%Y-%m-%d')
        latest = {}
        for m in self.remote_manifests(date):
            identity = (m['source'], m['name'])
            if identity not in latest or m['mtime_ns'] > latest[identity]['mtime_ns']:
                latest[identity] = m
        with self.lock():
            for m in latest.values():
                dest = self.root / SOURCES[m['source']] / m['name']
                if dest.exists():
                    if digest(dest) == m['sha256']:
                        continue
                    raise RuntimeError(f'Restore would overwrite different local data: {dest.name}')
                dest.parent.mkdir(parents=True, exist_ok=True)
                temp = dest.with_suffix(dest.suffix + '.restore.tmp')
                try:
                    self.client().download_file(self.bucket, m['key'], str(temp))
                    if temp.stat().st_size != m['size'] or digest(temp) != m['sha256']:
                        raise RuntimeError(f'Restore checksum mismatch: {m["name"]}')
                    temp.replace(dest)
                    os.utime(dest, ns=(m['mtime_ns'], m['mtime_ns']))
                    self.record(m, 'verified')
                finally:
                    temp.unlink(missing_ok=True)
        return len(latest)

    def list_observations(self):
        with self.db() as db:
            rows = db.execute("SELECT manifest FROM objects WHERE source='observations' AND status='verified' ORDER BY mtime_ns DESC").fetchall()
        latest = {}
        for row in rows:
            m = json.loads(row['manifest'])
            latest.setdefault(m['name'], m)
        return list(latest.values())

    def read_observation(self, name):
        if Path(name).name != name or not name.startswith('raw_data_') or not name.endswith('.json'):
            raise ValueError('Invalid observation filename')
        manifest = next((m for m in self.list_observations() if m['name'] == name), None)
        if manifest is None:
            return None
        self.validate_manifest(manifest)
        body = self.client().get_object(Bucket=self.bucket, Key=manifest['key'])['Body']
        try:
            content = body.read()
        finally:
            body.close()
        if len(content) != manifest['size'] or hashlib.sha256(content).hexdigest() != manifest['sha256']:
            raise RuntimeError('Observation archive checksum mismatch')
        return json.loads(content)

    def status(self, days=30):
        start = (datetime.now(timezone.utc).date() - timedelta(days=days-1)).isoformat()
        with self.db() as db:
            rows = db.execute('''SELECT * FROM (SELECT *, ROW_NUMBER() OVER
                (PARTITION BY source,name ORDER BY mtime_ns DESC,updated_at DESC) AS rank
                FROM objects) WHERE rank=1 AND date>=? ORDER BY date DESC''', (start,)).fetchall()
            runs = [dict(r) for r in db.execute('SELECT runs.*, transfer_progress.uploaded AS current_bytes, transfer_progress.total AS current_size FROM runs LEFT JOIN transfer_progress ON runs.id=transfer_progress.run_id ORDER BY runs.id DESC LIMIT 10')]
        dates = {}
        for offset in range(days):
            day = (datetime.now(timezone.utc).date() - timedelta(days=offset)).isoformat()
            dates[day] = {'date': day, 'sources': {source: {'files': 0, 'bytes': 0, 'verified': 0, 'failed': 0, 'pending': 0, 'stations': 0, 'records': 0} for source in SOURCES}}
        errors = []
        for row in rows:
            if row['date'] not in dates:
                continue
            group = dates[row['date']]['sources'][row['source']]
            m = json.loads(row['manifest'])
            group['files'] += 1
            group['bytes'] += row['size']
            group['verified' if row['status'] == 'verified' else 'failed' if row['status'] == 'failed' else 'pending'] += 1
            group['stations'] += m.get('stations') or 0
            group['records'] += m.get('records') or 0
            if row['error']:
                errors.append({'file': row['name'], 'error': row['error'], 'date': row['date']})
        return {'configured': self.configured, 'bucket': self.bucket, 'prefix': self.prefix,
                'days': list(dates.values()), 'runs': runs, 'errors': errors[:100],
                'note': 'Counts describe the latest revision of each file. Observation records are timestamp samples, not unique measurements. Missing means not inventoried; legacy ZIPs require import.'}
