import os
import sqlite3
import time
from collections import namedtuple
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from forecast_v1.repository import ensure_schema, prune_download_cache
from services import storage_cleanup as sc
from tests.day_archive_fakes import FakeS3

Usage = namedtuple("Usage", "total used free")
NOW = datetime(2026, 10, 22, 15, tzinfo=timezone.utc)


def _cycle(root, name, files=3):
    folder = root / "data" / "forecast-v1" / "download-cache" / name / "hrrr"
    folder.mkdir(parents=True)
    for index in range(files):
        (folder / f"part{index}.grib2").write_bytes(b"x")
    return folder.parent


def _run(db, run_id, cycle, status):
    with sqlite3.connect(db) as connection:
        connection.execute(
            "INSERT INTO forecast_runs (run_id,cycle_time_utc,status,grid_id,horizon_hours,schema_version,config_version,"
            "model_version,created_at_utc) VALUES (?,?,?,?,?,?,?,?,?)",
            (run_id, cycle, status, "g", 73, "1", "c", "m", "2026-10-22T00:00:00Z"))


@pytest.fixture
def env(tmp_path, monkeypatch):
    db = tmp_path / "smf.db"
    ensure_schema(db)
    monkeypatch.setattr("forecast_v1.repository.get_db_path", lambda: db)
    monkeypatch.setenv("SMF_ARCHIVE_ROOT", str(tmp_path))
    return tmp_path, db


def test_download_cache_of_finished_runs_goes_unfinished_recent_stays(env):
    root, db = env
    done = _cycle(root, "2026102012")
    superseded = _cycle(root, "2026101912")
    running = _cycle(root, "2026102112")
    failed_old = _cycle(root, "2026101712")
    _run(db, "20261020T1200Z-v1", "2026-10-20T12:00:00Z", "complete")
    _run(db, "20261019T1200Z-v1", "2026-10-19T12:00:00Z", "superseded")
    _run(db, "20261021T1200Z-v1", "2026-10-21T12:00:00Z", "staging")
    _run(db, "20261017T1200Z-v1", "2026-10-17T12:00:00Z", "failed")
    old = NOW.timestamp() - 3 * 86400
    for path in failed_old.rglob("*"):
        os.utime(path, (old, old))

    removed = prune_download_cache(root / "data" / "forecast-v1", now=NOW)

    assert removed == 9
    assert not done.exists() and not superseded.exists() and not failed_old.exists()
    assert len(list(running.rglob("*.grib2"))) == 3


def test_stale_staging_removed_only_after_a_day(env):
    root, _ = env
    stale = root / "gis" / ".publish-abc"
    fresh = root / "gis" / ".publish-def"
    work = root / "cache" / "day-archive"
    for path in (stale, fresh, work):
        path.mkdir(parents=True)
    (work / "2026-10-20_hrrr.zip").write_bytes(b"x")
    (root / "gis" / "latest").mkdir()
    old = time.time() - 2 * 86400
    os.utime(stale, (old, old))
    os.utime(work / "2026-10-20_hrrr.zip", (old, old))

    removed = sc.prune_stale_staging(root)

    assert sorted(removed) == ["cache/day-archive/2026-10-20_hrrr.zip", "gis/.publish-abc"]
    assert fresh.exists() and (root / "gis" / "latest").exists()


def test_low_disk_warns_and_critical_alerts_staff_once(env, caplog):
    root, _ = env
    alerts = []
    with patch("shutil.disk_usage", return_value=Usage(100 * sc.GB, 90 * sc.GB, 10 * sc.GB)):
        rows = sc.check_disks(root)
    assert rows[0]["level"] == "low" and "Disk space low" in caplog.text

    with patch("shutil.disk_usage", return_value=Usage(100 * sc.GB, 97 * sc.GB, 3 * sc.GB)), \
            patch("services.discord_notifier.notify_staff_alert", side_effect=lambda **kw: alerts.append(kw)):
        assert sc.check_disks(root, now=1000)[0]["level"] == "critical"
        sc.check_disks(root, now=1000 + 3600)            # within 6 h: no second alert
        sc.check_disks(root, now=1000 + 7 * 3600)
    assert len(alerts) == 2 and alerts[0]["alert_type"] == "system_issue"
    assert "Disk almost full" in caplog.text


def test_run_reports_disks_and_survives_missing_forecast_v1(tmp_path, monkeypatch):
    monkeypatch.setenv("SMF_ARCHIVE_ROOT", str(tmp_path))
    with patch("shutil.disk_usage", return_value=Usage(100 * sc.GB, 10 * sc.GB, 90 * sc.GB)):
        result = sc.run(tmp_path)
    assert result["forecast_v1_download_files"] == 0 and result["disks"][0]["level"] == "ok"


def test_archive_folder_verifies_then_deletes_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setenv("R2_BUCKET", "bucket")
    folder = tmp_path / "legacy"
    folder.mkdir()
    (folder / "20260125.zip").write_bytes(b"a" * 10)
    (folder / "20260126.zip").write_bytes(b"b" * 20)
    s3 = FakeS3()

    first = sc.archive_folder(folder, "legacy-archive/day", s3=s3, log=lambda _: None)
    assert first["uploaded"] == 2 and not first["failed"] and (folder / "20260125.zip").exists()
    assert s3.objects["legacy-archive/day/20260126.zip"] == b"b" * 20
    assert set(s3.json("legacy-archive/day/manifest.json")["files"]) == {"20260125.zip", "20260126.zip"}

    second = sc.archive_folder(folder, "legacy-archive/day", s3=s3, delete=True, log=lambda _: None)
    assert second["already_verified"] == 2 and second["uploaded"] == 0 and second["deleted"] == 2
    assert not any(folder.iterdir())


def test_archive_folder_keeps_file_when_read_back_differs(tmp_path, monkeypatch):
    monkeypatch.setenv("R2_BUCKET", "bucket")
    folder = tmp_path / "legacy"
    folder.mkdir()
    (folder / "20260125.zip").write_bytes(b"a" * 10)
    s3 = FakeS3()
    upload = s3.upload_file
    s3.upload_file = lambda filename, Bucket, Key, ExtraArgs=None: (upload(filename, Bucket, Key), s3.objects.update({Key: b"corrupt"}))

    summary = sc.archive_folder(folder, "legacy-archive/day", s3=s3, delete=True, log=lambda _: None)

    assert summary["deleted"] == 0 and len(summary["failed"]) == 1
    assert (folder / "20260125.zip").exists()


def _legacy(tmp_path, monkeypatch, files=2):
    monkeypatch.setenv("R2_BUCKET", "bucket")
    folder = tmp_path / "legacy"
    folder.mkdir()
    for index in range(files):
        (folder / f"2026012{index}.zip").write_bytes(b"z" * (index + 1))
    monkeypatch.setenv("SMF_LEGACY_ARCHIVE_DIR", str(folder))
    return folder


def test_legacy_archive_moves_files_then_does_nothing(tmp_path, monkeypatch):
    folder = _legacy(tmp_path, monkeypatch)
    s3 = FakeS3()

    first = sc.run_legacy_archive(s3=s3, log=lambda _: None)
    assert first["uploaded"] == 2 and first["deleted"] == 2 and not first["failed"]
    assert not any(path.is_file() for path in folder.rglob("*"))
    assert "legacy-archive/data_archive_day/20260121.zip" in s3.objects

    assert sc.run_legacy_archive(s3=s3, log=lambda _: None) is None   # empty folder: nothing to do


def test_legacy_archive_is_skipped_when_folder_not_mounted_or_r2_unset(tmp_path, monkeypatch):
    monkeypatch.setenv("SMF_LEGACY_ARCHIVE_DIR", str(tmp_path / "missing"))
    assert sc.run_legacy_archive(s3=FakeS3()) is None

    folder = _legacy(tmp_path, monkeypatch)
    for name in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert sc.run_legacy_archive() is None and len(list(folder.iterdir())) == 2


def test_legacy_archive_can_upload_without_deleting(tmp_path, monkeypatch):
    folder = _legacy(tmp_path, monkeypatch, files=1)
    monkeypatch.setenv("SMF_LEGACY_ARCHIVE_DELETE", "false")
    summary = sc.run_legacy_archive(s3=FakeS3(), log=lambda _: None)
    assert summary["uploaded"] == 1 and summary["deleted"] == 0 and (folder / "20260120.zip").exists()


def test_archive_folder_stops_starting_files_after_its_time_budget(tmp_path, monkeypatch):
    folder = _legacy(tmp_path, monkeypatch, files=3)
    summary = sc.archive_folder(folder, "p", s3=FakeS3(), log=lambda _: None, max_seconds=-1)
    assert summary["stopped_early"] and summary["uploaded"] == 0
