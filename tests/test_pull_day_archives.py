import json

from scripts import pull_day_archives as cli
from services import day_archive as da
from tests.test_day_archive import DAY, NOW, env  # noqa: F401 - fixture


def _setup(env, monkeypatch):
    root, s3 = env
    monkeypatch.setattr(cli, "BUCKET", "test-bucket")
    monkeypatch.setattr(cli, "PREFIX", "day-archive")
    da.build_day(s3, DAY, now=NOW)
    return s3


def test_pull_verifies_downloads_and_records_receipt(env, monkeypatch, tmp_path):
    s3 = _setup(env, monkeypatch)
    dest = tmp_path / "out"
    assert cli.pull_day(s3, DAY, dest, mark=True) is True
    assert (dest / DAY / "forecasts.zip").is_file() and (dest / DAY / "manifest.json").is_file()
    receipt = s3.json(f"day-archive/receipts/{DAY}.json")
    assert receipt["verified"] is True and "forecasts.zip" in receipt["files"]


def test_pull_second_run_skips_matching_files(env, monkeypatch, tmp_path):
    s3 = _setup(env, monkeypatch)
    cli.pull_day(s3, DAY, tmp_path, mark=False)
    calls = []
    real = s3.download_file
    s3.download_file = lambda *a, **k: (calls.append(a), real(*a, **k))
    assert cli.pull_day(s3, DAY, tmp_path, mark=False) is True
    assert calls == []


def test_corrupt_download_fails_and_records_no_receipt(env, monkeypatch, tmp_path):
    s3 = _setup(env, monkeypatch)
    s3.objects[f"day-archive/{DAY}/forecasts.zip"] += b"corrupt"
    assert cli.pull_day(s3, DAY, tmp_path, mark=True) is False
    assert f"day-archive/receipts/{DAY}.json" not in s3.objects
    assert not (tmp_path / DAY / "forecasts.zip").exists()


def test_no_mark_leaves_no_receipt(env, monkeypatch, tmp_path):
    s3 = _setup(env, monkeypatch)
    assert cli.pull_day(s3, DAY, tmp_path, mark=False) is True
    assert f"day-archive/receipts/{DAY}.json" not in s3.objects


def test_missing_manifest_is_reported(env, monkeypatch, tmp_path):
    root, s3 = env
    monkeypatch.setattr(cli, "BUCKET", "test-bucket")
    monkeypatch.setattr(cli, "PREFIX", "day-archive")
    assert cli.pull_day(s3, "2000-01-01", tmp_path, mark=True) is False
