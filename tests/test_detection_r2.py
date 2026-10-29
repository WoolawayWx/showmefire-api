import json
import shutil
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from services import detection_r2 as r2


class FakeS3:
    def __init__(self):
        self.objects = {}

    def upload_file(self, path, bucket, key, ExtraArgs=None):
        with open(path, "rb") as f:
            self.objects[key] = f.read()

    def put_object(self, Bucket, Key, Body, **_):
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):
        raise AssertionError("not used")

    def list_objects_v2(self, Bucket, Prefix, Delimiter=None, ContinuationToken=None):
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        if Delimiter:
            prefixes = sorted({Prefix + k[len(Prefix):].split(Delimiter)[0] + Delimiter for k in keys})
            return {"CommonPrefixes": [{"Prefix": p} for p in prefixes], "IsTruncated": False}
        return {"Contents": [{"Key": k} for k in keys], "IsTruncated": False}

    def delete_objects(self, Bucket, Delete):
        for item in Delete["Objects"]:
            self.objects.pop(item["Key"], None)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "showmefire.db"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE fire_events (id INTEGER PRIMARY KEY, source TEXT, external_id TEXT, satellite TEXT, occurred_at TEXT,
            latitude REAL, longitude REAL, frp REAL, incident_id INTEGER, reporter_contact TEXT);
        CREATE TABLE fire_incidents (id INTEGER PRIMARY KEY, centroid_latitude REAL, centroid_longitude REAL, status TEXT);
        CREATE TABLE fire_incident_feedback (id INTEGER PRIMARY KEY, incident_id INTEGER, classification TEXT, status TEXT,
            note TEXT, submitter_ip_hash TEXT);
        INSERT INTO fire_events VALUES (1, 'viirs', 'a', 'N', '2026-01-05T19:12:00Z', 38.5, -92.3, 4.2, 1, 'secret@example.com');
        INSERT INTO fire_events VALUES (2, 'user_submission', 'b', NULL, '2026-01-05T20:00:00Z', 38.5, -92.3, NULL, NULL, 'secret@example.com');
        INSERT INTO fire_incidents VALUES (1, 38.5, -92.3, 'active');
        INSERT INTO fire_incident_feedback VALUES (1, 1, 'confirmed_fire', 'approved', 'a note', 'iphash');
    """)
    conn.commit()
    conn.close()
    return path


def test_publish_uploads_zip_then_pointer_without_pii(db, monkeypatch):
    monkeypatch.setenv("R2_TRAINING_BUCKET", "b")
    s3 = FakeS3()
    pointer = r2.publish(db, client=s3, now=datetime(2026, 10, 27, 12, 0, tzinfo=timezone.utc))
    assert pointer["snapshot"] == "20261027T120000Z"
    assert pointer["rows"]["detections"] == 1 and pointer["rows"]["feedback"] == 1
    assert json.loads(s3.objects["training/detections/latest.json"])["zip_key"] == pointer["zip_key"]
    blob = s3.objects[pointer["zip_key"]]
    assert b"secret@example.com" not in blob and b"iphash" not in blob  # contact/IP columns are never exported (and the CSVs are stored, not compressed away)
    import hashlib
    assert hashlib.sha256(blob).hexdigest() == pointer["zip_sha256"]


def test_old_snapshots_are_pruned_but_the_latest_is_kept(db, monkeypatch):
    monkeypatch.setenv("R2_TRAINING_BUCKET", "b")
    s3 = FakeS3()
    start = datetime(2026, 10, 1, tzinfo=timezone.utc)
    for day in range(4):
        pointer = r2.publish(db, keep=2, client=s3, now=start + timedelta(days=day))
    assert r2.list_snapshots(s3) == ["20261004T000000Z", "20261003T000000Z"]
    assert pointer["pruned"] == ["20261002T000000Z"]  # the 4th publish prunes the one that just fell out of the newest two
    assert not [k for k in s3.objects if "20261001T000000Z" in k]
    assert "training/detections/latest.json" in s3.objects


def test_publish_job_skips_quietly_without_credentials(monkeypatch):
    for name in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(name, raising=False)
    r2.publish_job()  # must not raise
