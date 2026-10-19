import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from services import cdn_datasets as cd


class MissingKey(Exception):
    response = {"Error": {"Code": "NoSuchKey"}}


class FakeBody:
    def __init__(self, data):
        self.data = data

    def read(self):
        return self.data


class FakeR2:
    """Just enough of the S3 client: objects live in a dict, remembering how each was uploaded."""

    def __init__(self):
        self.objects = {}
        self.writes = []   # keys in the order they were written

    def upload_file(self, filename, bucket, key, ExtraArgs=None):
        self.objects[key] = {"body": Path(filename).read_bytes(), **(ExtraArgs or {})}
        self.writes.append(key)

    def put_object(self, Bucket, Key, Body, ContentType=None, CacheControl=None):
        self.objects[Key] = {"body": Body, "ContentType": ContentType, "CacheControl": CacheControl}
        self.writes.append(Key)

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise MissingKey()
        return {"Body": FakeBody(self.objects[Key]["body"])}

    def get_paginator(self, name):
        client = self

        class Pager:
            def paginate(self, Bucket, Prefix):
                yield {"Contents": [{"Key": k} for k in sorted(client.objects) if k.startswith(Prefix)]}
        return Pager()

    def delete_objects(self, Bucket, Delete):
        for item in Delete["Objects"]:
            self.objects.pop(item["Key"], None)


def make_run(root: Path, name: str) -> Path:
    run = root / name
    (run / "forecast.zarr" / "mixing").mkdir(parents=True, exist_ok=True)
    (run / "forecast.zarr" / ".zmetadata").write_text("{}")
    (run / "forecast.zarr" / "mixing" / ".zarray").write_text("{}")
    (run / "forecast.zarr" / "mixing" / "0.0.0").write_bytes(b"chunk-" + name.encode())
    (run / "manifest.json").write_text(json.dumps({"cycle": name}))
    (run / "timings.json").write_text("{}")          # internal; must not be published
    (run / "manifest.json.tmp").write_text("partial")  # leftover temp file; must not be published
    return run


def summary(run_id: str, day: int) -> dict:
    return {"id": run_id, "cycle": run_id, "label": run_id, "time_utc": f"2026-10-{day:02d}T12:00:00Z"}


def publish(client, root, run_id, day, now_day, **extra):
    return cd.publish_run("demo", run_id, make_run(root, run_id), summary(run_id, day), title="Demo", kind="grid",
                          schema_version=1, include=["forecast.zarr", "manifest.json"], client=client,
                          now=datetime(2026, 10, now_day, 15, tzinfo=timezone.utc), **extra)


class PublishTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.client = FakeR2()

    def tearDown(self):
        self.tmp.cleanup()

    def doc(self, key):
        return json.loads(self.client.objects[key]["body"])

    def test_files_land_under_the_run_with_the_right_cache_and_types(self):
        result = publish(self.client, self.root, "20261016_12z", 16, 16)
        base = "datasets/demo/runs/20261016_12z/"
        chunk = self.client.objects[base + "forecast.zarr/mixing/0.0.0"]
        self.assertEqual(chunk["ContentType"], "application/octet-stream")
        self.assertEqual(chunk["CacheControl"], cd.IMMUTABLE_CACHE)
        self.assertEqual(self.client.objects[base + "forecast.zarr/.zmetadata"]["ContentType"], "application/json")
        self.assertEqual(self.client.objects[base + "forecast.zarr/mixing/.zarray"]["ContentType"], "application/json")
        self.assertEqual(self.client.objects[base + "manifest.json"]["CacheControl"], cd.IMMUTABLE_CACHE)
        self.assertNotIn(base + "timings.json", self.client.objects)
        self.assertNotIn(base + "manifest.json.tmp", self.client.objects)
        self.assertEqual(result["uploaded"], 4)  # three zarr files and the manifest

    def test_pointers_are_written_last_with_a_short_cache(self):
        publish(self.client, self.root, "20261016_12z", 16, 16)
        writes = self.client.writes
        manifest_at = writes.index("datasets/demo/runs/20261016_12z/manifest.json")
        self.assertTrue(all(i < manifest_at for i, key in enumerate(writes) if "forecast.zarr" in key))
        self.assertEqual(writes[-3:], ["datasets/demo/runs.json", "datasets/demo/latest.json", "datasets/index.json"])
        self.assertEqual(self.client.objects["datasets/demo/latest.json"]["CacheControl"], cd.POINTER_CACHE)
        self.assertEqual(self.doc("datasets/demo/latest.json")["run"], "20261016_12z")
        self.assertEqual(self.doc("datasets/demo/latest.json")["manifest"], "runs/20261016_12z/manifest.json")

    def test_registry_keeps_other_datasets(self):
        self.client.put_object("b", "datasets/index.json", json.dumps({"datasets": {"other": {"title": "Other"}}}).encode())
        publish(self.client, self.root, "20261016_12z", 16, 16)
        datasets = self.doc("datasets/index.json")["datasets"]
        self.assertEqual(set(datasets), {"other", "demo"})
        self.assertEqual(datasets["demo"]["latest"], "demo/latest.json")
        self.assertEqual(datasets["demo"]["kind"], "grid")

    def test_runs_list_is_newest_first_and_old_runs_are_deleted_after_the_pointer_moves(self):
        publish(self.client, self.root, "20261001_12z", 1, 1)
        early = publish(self.client, self.root, "20261008_12z", 8, 8)      # 7 days back is Oct 1 15:00: the 1st is out
        self.assertEqual(early["pruned"], ["20261001_12z"])
        result = publish(self.client, self.root, "20261016_12z", 16, 16)   # 7 days back is Oct 9: the 8th is out
        self.assertEqual(result["pruned"], ["20261008_12z"])
        runs = self.doc("datasets/demo/runs.json")
        self.assertEqual([r["id"] for r in runs["runs"]], ["20261016_12z"])
        self.assertEqual(runs["latest"], "20261016_12z")
        self.assertFalse([k for k in self.client.objects if "20261001_12z" in k or "20261008_12z" in k])

    def test_runs_inside_the_window_are_kept(self):
        publish(self.client, self.root, "20261014_12z", 14, 14)
        result = publish(self.client, self.root, "20261016_12z", 16, 16)
        self.assertEqual(result["pruned"], [])
        self.assertEqual([r["id"] for r in self.doc("datasets/demo/runs.json")["runs"]], ["20261016_12z", "20261014_12z"])
        self.assertIn("datasets/demo/runs/20261014_12z/manifest.json", self.client.objects)

    def test_republishing_a_run_does_not_duplicate_it(self):
        publish(self.client, self.root, "20261016_12z", 16, 16)
        publish(self.client, self.root, "20261016_12z", 16, 16)
        self.assertEqual(len(self.doc("datasets/demo/runs.json")["runs"]), 1)

    def test_the_newest_run_is_never_pruned_even_if_old(self):
        result = publish(self.client, self.root, "20261001_12z", 1, 20)   # published late: older than the window
        self.assertEqual(result["pruned"], [])
        self.assertEqual(self.doc("datasets/demo/latest.json")["run"], "20261001_12z")

    def test_missing_manifest_or_bad_summary_is_rejected(self):
        run = make_run(self.root, "20261016_12z")
        (run / "manifest.json").unlink()
        with self.assertRaises(FileNotFoundError):
            cd.publish_run("demo", "20261016_12z", run, summary("20261016_12z", 16), title="d", kind="k",
                           schema_version=1, client=self.client)
        (run / "manifest.json").write_text("{}")
        with self.assertRaises(ValueError):
            cd.publish_run("demo", "20261016_12z", run, {"id": "other"}, title="d", kind="k", schema_version=1,
                           client=self.client)

    def test_try_publish_never_raises_and_skips_without_credentials(self):
        class Exploding(FakeR2):
            def upload_file(self, *args, **kwargs):
                raise RuntimeError("R2 is down")

        run = make_run(self.root, "20261016_12z")
        args = ("demo", "20261016_12z", run, summary("20261016_12z", 16))
        kwargs = dict(title="d", kind="k", schema_version=1)
        self.assertIsNone(cd.try_publish_run(*args, client=Exploding(), **kwargs))
        import os
        saved = {k: os.environ.pop(k, None) for k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY")}
        try:
            self.assertIsNone(cd.try_publish_run(*args, **kwargs))
        finally:
            os.environ.update({k: v for k, v in saved.items() if v is not None})

    def test_hrrr_summary_matches_the_apis_run_rows(self):
        from services.hrrr_hourly import publish as hrrr_publish
        manifest = {"cycle": "20261016_12z", "label": "Forecast from the 12z HRRR run",
                    "cycle_time_utc": "2026-10-16T12:00:00Z", "issued_at_utc": "2026-10-16T16:00:00Z",
                    "forecast_hours": [0, 1], "missing_hours": [], "variables": {"a": {}, "b": {}}}
        row = hrrr_publish.run_summary(manifest)
        self.assertEqual(row["id"], "20261016_12z")
        self.assertEqual(row["time_utc"], "2026-10-16T12:00:00Z")
        for field in ("cycle", "label", "cycle_time_utc", "issued_at_utc", "forecast_hours", "missing_hours"):
            self.assertEqual(row[field], manifest[field])
        self.assertEqual(row["variables"], 2)


if __name__ == "__main__":
    unittest.main()


class HourlyRunsTests(unittest.TestCase):
    """Every hourly HRRR run goes to the CDN: 24 a day for 7 days must all stay listed and uploaded."""

    def test_a_week_of_hourly_runs_is_not_cut_by_the_run_count_cap(self):
        from datetime import timedelta

        assert cd.RUNS_LISTED >= 24 * cd.keep_days() * 2          # headroom over runs per day x days kept
        with tempfile.TemporaryDirectory() as tmp:
            root, client = Path(tmp), FakeR2()
            start = datetime(2026, 10, 9, 0, tzinfo=timezone.utc)
            for hour in range(0, 24 * 7 + 1):                       # 7 days of hourly runs, oldest first
                when = start + timedelta(hours=hour)
                run_id = f"{when:%Y%m%d_%H}z"
                run = make_run(root, run_id)
                cd.publish_run("demo", run_id, run, {"id": run_id, "time_utc": f"{when:%Y-%m-%dT%H:%M:%SZ}"},
                                        title="d", kind="k", schema_version=1, include=["forecast.zarr", "manifest.json"],
                                        client=client, now=when + timedelta(hours=1, minutes=15))
            runs = json.loads(client.objects["datasets/demo/runs.json"]["body"])["runs"]
            ids = [r["id"] for r in runs]
            self.assertEqual(ids[0], f"{start + timedelta(hours=24 * 7):%Y%m%d_%H}z")        # newest first
            # 169 runs were published over 7 days; only the two that had aged past the 7-day cutoff are gone
            self.assertEqual(len(runs), 24 * 7 + 1 - 2)
            self.assertTrue(all(f"datasets/demo/runs/{i}/manifest.json" in client.objects for i in ids))
            self.assertFalse([k for k in client.objects if "/runs/20261009_00z/" in k or "/runs/20261009_01z/" in k])
