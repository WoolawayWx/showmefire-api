import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch

from services.archive_store import ArchiveStore


class FakeR2:
    def __init__(self):
        self.objects = {}
        self.uploads = 0
        self.fail = False
        self.corrupt = False

    def upload_file(self, path, bucket, key, ExtraArgs=None, Callback=None):
        self.uploads += 1
        if self.fail:
            raise RuntimeError('Network unavailable')
        self.objects[key] = b'corrupt' if self.corrupt else Path(path).read_bytes()
        if Callback:
            Callback(Path(path).stat().st_size)

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):
        return {'Body': io.BytesIO(self.objects[Key])}

    def download_file(self, bucket, key, path):
        Path(path).write_bytes(self.objects[key])

    def get_paginator(self, name):
        return self

    def paginate(self, Bucket, Prefix):
        # More than one page exercises restoration and inventory pagination.
        for key in self.objects:
            if key.startswith(Prefix):
                yield {'Contents': [{'Key': key}]}


class ArchiveStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.r2 = FakeR2()
        self.store = ArchiveStore(self.root, self.r2)
        self.date = datetime.now(timezone.utc).date().isoformat()
        self.path = self.root / f"archive/forecasts/station_forecasts_{self.date.replace('-', '')}_12.json"
        self.path.parent.mkdir(parents=True)
        self.write(3)

    def write(self, value):
        self.path.write_text(json.dumps({'stations': {'A': {'forecasts': [
            {'time': f'{self.date}T16:00:00Z', 'temp': value}]}}}))

    def test_upload_verify_retry_and_changed_revision(self):
        first = self.store.ingest(self.path, 'forecasts')
        self.assertEqual(first['records'], 1)
        self.assertEqual(first['stations'], 1)
        self.store.ingest(self.path, 'forecasts')
        self.assertEqual(self.r2.uploads, 1)
        self.write(4)
        second = self.store.ingest(self.path, 'forecasts')
        self.assertNotEqual(first['key'], second['key'])
        self.assertIn(first['key'], self.r2.objects)
        self.assertEqual(len(list(self.store.remote_manifests())), 2)

    def test_failed_upload_keeps_local_data_and_retries(self):
        self.r2.fail = True
        with self.assertRaises(RuntimeError):
            self.store.ingest(self.path, 'forecasts')
        self.assertTrue(self.path.exists())
        with self.store.db() as db:
            self.assertEqual(db.execute('SELECT status FROM objects').fetchone()[0], 'failed')
        self.r2.fail = False
        self.store.ingest(self.path, 'forecasts')
        self.assertEqual(self.r2.uploads, 2)

    def test_corrupt_upload_never_publishes_manifest(self):
        self.r2.corrupt = True
        with self.assertRaisesRegex(RuntimeError, 'verification'):
            self.store.ingest(self.path, 'forecasts')
        self.assertTrue(self.path.exists())
        self.assertEqual(list(self.store.remote_manifests()), [])

    def test_restore_newest_revision_and_protect_local_changes(self):
        self.store.ingest(self.path, 'forecasts')
        self.write(4)
        latest = self.path.read_bytes()
        self.store.ingest(self.path, 'forecasts')
        self.path.unlink()
        self.assertEqual(self.store.restore(self.date), 1)
        self.assertEqual(self.path.read_bytes(), latest)
        self.write(8)
        with self.assertRaisesRegex(RuntimeError, 'overwrite'):
            self.store.restore(self.date)

    def test_corrupt_restore_does_not_create_destination(self):
        m = self.store.ingest(self.path, 'forecasts')
        self.path.unlink()
        self.r2.objects[m['key']] = b'bad'
        with self.assertRaisesRegex(RuntimeError, 'checksum'):
            self.store.restore(self.date)
        self.assertFalse(self.path.exists())

    def test_rebuild_and_missing_dates(self):
        self.store.ingest(self.path, 'forecasts')
        with self.store.db() as db:
            db.execute('DELETE FROM objects')
        self.assertEqual(self.store.rebuild(), 1)
        status = self.store.status(7)
        self.assertEqual(len(status['days']), 7)
        self.assertEqual(sum(d['sources']['forecasts']['files'] for d in status['days']), 1)

    def test_inventory_without_credentials_does_not_upload(self):
        with patch.dict(os.environ, {}, clear=True):
            store = ArchiveStore(self.root)
            store.ingest(self.path, 'forecasts')
        self.assertTrue(self.path.exists())
        with store.db() as db:
            self.assertEqual(db.execute('SELECT status FROM objects').fetchone()[0], 'pending')

    def test_run_persists_failures_and_excludes_metadata_files(self):
        metadata = self.root / 'archive/raw_data/synoptic_backfill_manifest.json'
        metadata.parent.mkdir(parents=True)
        metadata.write_text('{}')
        self.r2.fail = True
        with patch.dict(os.environ, {'SMF_ARCHIVE_SETTLE_SECONDS': '0'}):
            result = self.store.sync(prune=True)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['total'], 1)
        self.assertEqual(result['failed'], 1)
        self.assertTrue(self.path.exists())

    def test_verified_old_sources_can_be_pruned_and_restored(self):
        old = self.path.with_name('station_forecasts_20200101_12.json')
        self.path.rename(old)
        os.utime(old, (1, 1))
        with patch.dict(os.environ, {'SMF_ARCHIVE_SETTLE_SECONDS': '0'}):
            result = self.store.sync(prune=True)
        self.assertEqual(result['completed'], 1)
        self.assertFalse(old.exists())
        self.assertEqual(self.store.restore('2020-01-01'), 1)
        self.assertTrue(old.exists())

    def test_legacy_import_keeps_zip_and_publishes_known_members(self):
        import zipfile
        from scripts.archive_storage import import_zip
        path = self.root / 'legacy.zip'
        with zipfile.ZipFile(path, 'w') as zf:
            zf.writestr('forecasts/' + self.path.name, self.path.read_bytes())
            zf.writestr('rtma/rtma_20200101_12.nc', b'keep in legacy archive')
        before = path.read_bytes()
        self.assertEqual(import_zip(self.store, path), 1)
        self.assertEqual(before, path.read_bytes())
        self.assertEqual(len(list(self.store.remote_manifests())), 1)

    def test_manifest_path_traversal_is_rejected(self):
        m = self.store.ingest(self.path, 'forecasts')
        m['name'] = '../escape.json'
        with self.assertRaises(ValueError):
            self.store.validate_manifest(m)

    def test_lock_excludes_other_workers(self):
        with self.store.lock():
            with self.assertRaisesRegex(RuntimeError, 'already running'):
                self.store.sync()


if __name__ == '__main__':
    unittest.main()
