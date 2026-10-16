"""Daily CDN map archive: zip, prune only behind a good zip, expire with a reminder (fake R2, no network)."""
import io
import sys
import tempfile
import unittest
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from services import map_archive as ma


class FakeS3:
    """Just enough of boto3's S3 client for services/map_archive.py."""

    def __init__(self, objects):
        self.objects = dict(objects)  # key -> bytes

    def get_paginator(self, _name):
        store = self

        class Paginator:
            def paginate(self, Bucket, Prefix, Delimiter=None):
                keys = sorted(k for k in store.objects if k.startswith(Prefix))
                if Delimiter:
                    prefixes = sorted({Prefix + k[len(Prefix):].split(Delimiter)[0] + Delimiter
                                       for k in keys if Delimiter in k[len(Prefix):]})
                    yield {"CommonPrefixes": [{"Prefix": p} for p in prefixes]}
                else:
                    yield {"Contents": [{"Key": k, "Size": len(store.objects[k]),
                                         "LastModified": datetime(2026, 10, 1, tzinfo=timezone.utc)} for k in keys]}
        return Paginator()

    def download_fileobj(self, Bucket, Key, handle):
        handle.write(self.objects[Key])

    def upload_file(self, path, Bucket, Key, ExtraArgs=None):
        self.objects[Key] = Path(path).read_bytes()

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise KeyError(Key)
        return {"ContentLength": len(self.objects[Key])}

    def delete_objects(self, Bucket, Delete):
        for item in Delete["Objects"]:
            self.objects.pop(item["Key"], None)
        return {}


def bucket(days):
    objects = {"latest/mo-rh.png": b"latest"}
    for day in days:
        path = f"{day[:4]}/{day[4:6]}/{day[6:]}"
        objects[f"{path}/1200/mo-rh.png"] = b"png-1200-" + day.encode()
        objects[f"{path}/1215/mo-rh.png"] = b"png-1215-" + day.encode()
        objects[f"{path}/forecast/mo-forecastfiredanger.png"] = b"forecast"
    return FakeS3(objects)


class MapArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        patches = [mock.patch.object(ma, "LEDGER_PATH", root / "ledger.json"),
                   mock.patch.object(ma, "WORK_DIR", root / "work"),
                   mock.patch.object(ma, "notify", return_value=True)]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        self.addCleanup(self.tmp.cleanup)

    def test_zips_completed_days_only_and_never_forecast(self):
        s3 = bucket(["20261014", "20261015", "20261016"])
        now = datetime(2026, 10, 16, 8, tzinfo=timezone.utc)
        result = ma.run(now=now, s3=s3)
        self.assertEqual(result["created"], ["20261014", "20261015"])  # today (16th) is still running
        with zipfile.ZipFile(io.BytesIO(s3.objects["map-archive/20261015.zip"])) as archive:
            self.assertEqual(sorted(archive.namelist()), ["1200/mo-rh.png", "1215/mo-rh.png"])
            self.assertEqual(archive.read("1200/mo-rh.png"), b"png-1200-20261015")
        self.assertIn("2026/10/15/1200/mo-rh.png", s3.objects)  # still within the 14-day window

    def test_prunes_old_folders_behind_a_good_zip_and_keeps_forecast(self):
        s3 = bucket(["20261001"])
        ma.run(now=datetime(2026, 10, 2, 8, tzinfo=timezone.utc), s3=s3)
        result = ma.run(now=datetime(2026, 10, 16, 8, tzinfo=timezone.utc), s3=s3)
        self.assertEqual(result["pruned"], ["20261001"])
        self.assertNotIn("2026/10/01/1200/mo-rh.png", s3.objects)
        self.assertIn("2026/10/01/forecast/mo-forecastfiredanger.png", s3.objects)
        self.assertIn("latest/mo-rh.png", s3.objects)

    def test_never_prunes_without_an_intact_zip(self):
        s3 = bucket(["20261001"])
        ma.run(now=datetime(2026, 10, 2, 8, tzinfo=timezone.utc), s3=s3)
        del s3.objects["map-archive/20261001.zip"]
        result = ma.run(now=datetime(2026, 10, 16, 8, tzinfo=timezone.utc), s3=s3)
        self.assertEqual(result["pruned"], [])
        self.assertIn("2026/10/01/1200/mo-rh.png", s3.objects)

    def test_zip_expires_only_after_a_reminder(self):
        s3 = bucket(["20261001"])
        start = datetime(2026, 10, 2, 8, tzinfo=timezone.utc)
        ma.run(now=start, s3=s3)
        expiry = start + timedelta(days=ma.zip_keep_days())
        late = ma.run(now=expiry + timedelta(hours=1), s3=s3)  # never reminded yet: remind, keep
        self.assertEqual(late["reminded"], ["20261001"])
        self.assertIn("map-archive/20261001.zip", s3.objects)
        final = ma.run(now=expiry + timedelta(days=1), s3=s3)
        self.assertEqual(final["expired"], ["20261001"])
        self.assertNotIn("map-archive/20261001.zip", s3.objects)
        self.assertEqual(ma.available(), [])

    def test_backlog_is_limited_per_run(self):
        days = [f"202609{d:02d}" for d in range(1, 11)]
        s3 = bucket(days)
        with mock.patch.dict("os.environ", {"SMF_MAP_ARCHIVE_DAYS_PER_RUN": "3"}):
            result = ma.run(now=datetime(2026, 10, 16, 8, tzinfo=timezone.utc), s3=s3)
        self.assertEqual(result["created"], days[:3])  # oldest first

    def test_available_lists_download_links(self):
        s3 = bucket(["20261015"])
        ma.run(now=datetime(2026, 10, 16, 8, tzinfo=timezone.utc), s3=s3)
        rows = ma.available()
        self.assertEqual(rows[0]["day"], "20261015")
        self.assertTrue(rows[0]["url"].endswith("/map-archive/20261015.zip"))
        self.assertEqual(rows[0]["images"], 2)


if __name__ == "__main__":
    unittest.main()
