import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from services import day_archive as da
from tests.day_archive_fakes import FakeS3

DAY = "2026-10-10"
NOW = datetime(2026, 10, 12, 12, tzinfo=timezone.utc)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("SMF_ARCHIVE_ROOT", str(tmp_path))
    monkeypatch.setenv("R2_BUCKET", "test-bucket")
    monkeypatch.delenv("SMF_DAY_ARCHIVE_DELETE", raising=False)
    monkeypatch.delenv("SMF_DAY_ARCHIVE_PRUNE_LOCAL", raising=False)
    old = NOW.timestamp() - 3600
    files = {
        "archive/forecasts/station_forecasts_20261010_09.json": b'{"f":1}',
        "archive/raw_data/raw_data_20261010.json": b'{"o":1}',
        "archive/raw_data_full/raw_data_full_20261010.json": b'{"full":1}',
        "cache/ensemble/runs/member_hrrr_0_hrrr_20261010_12z_f04-15.nc": b"ncdata",
        "reports/2026-10-10/validation_summary.json": b"{}",
        "gis/forecast_peak/archive/2026-10-10.tif": b"tif",
        "archive/forecasts/station_forecasts_20261011_09.json": b"other-day",
    }
    for rel, data in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        os.utime(path, (old, old))
    # RTMA hours of the Central day (CDT: 05Z on the 10th through 04Z on the 11th)
    for hour in da.central_day_hours_utc(DAY):
        path = tmp_path / "cache" / "rtma" / f"rtma_{hour:%Y%m%d_%H}z.nc"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"rtma")
    s3 = FakeS3()
    for utc_day in da.map_zip_days(DAY):
        s3.objects[f"map-archive/{utc_day}.zip"] = b"mapzip-" + utc_day.encode()
    return tmp_path, s3


def test_central_day_has_24_hours_and_map_zips_span_two_utc_days():
    assert len(da.central_day_hours_utc(DAY)) == 24
    assert len(da.central_day_hours_utc("2026-11-01")) == 25  # DST ends
    assert da.map_zip_days(DAY) == ["20261010", "20261011"]


def test_build_day_uploads_verified_components_and_manifest(env):
    root, s3 = env
    manifest = da.build_day(s3, DAY, now=NOW)
    assert manifest["complete"] is True
    comps = manifest["components"]
    assert comps["forecasts"]["status"] == "present"
    assert [m["name"] for m in comps["forecasts"]["members"]] == ["archive/forecasts/station_forecasts_20261010_09.json"]
    assert comps["rtma"]["status"] == "present" and len(comps["rtma"]["members"]) == 24
    assert comps["hrrr"]["status"] == "missing"            # optional, nothing on disk
    assert comps["maps"]["status"] == "present" and len(comps["maps"]["parts"]) == 2
    assert s3.json(f"day-archive/{DAY}/manifest.json")["day"] == DAY
    assert f"day-archive/{DAY}/forecasts.zip" in s3.objects
    assert not list((root / "cache" / "day-archive").glob("*.zip"))   # work dir cleaned


def test_recent_files_are_not_bundled_until_settled(env):
    root, s3 = env
    fresh = root / "archive/forecasts/station_forecasts_20261010_21.json"
    fresh.write_bytes(b"still being written")
    os.utime(fresh, (NOW.timestamp() - 10, NOW.timestamp() - 10))
    manifest = da.build_day(s3, DAY, now=NOW)
    names = [m["name"] for m in manifest["components"]["forecasts"]["members"]]
    assert all("_21" not in n for n in names)


def test_missing_map_zip_makes_bundle_incomplete_then_completes(env):
    root, s3 = env
    del s3.objects["map-archive/20261011.zip"]
    first = da.build_day(s3, DAY, now=NOW)
    assert first["complete"] is False and first["components"]["maps"]["status"] == "partial"
    s3.objects["map-archive/20261011.zip"] = b"later"
    second = da.build_day(s3, DAY, now=NOW, previous=first)
    assert second["complete"] is True


def test_nothing_to_archive_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("SMF_ARCHIVE_ROOT", str(tmp_path))
    assert da.build_day(FakeS3(), DAY, now=NOW) is None


def test_readback_mismatch_marks_component_failed(env):
    root, s3 = env
    real = s3.upload_file

    def corrupt(filename, Bucket, Key, ExtraArgs=None):
        real(filename, Bucket, Key, ExtraArgs)
        if Key.endswith("forecasts.zip"):
            s3.objects[Key] = s3.objects[Key][:-1] + b"X"

    s3.upload_file = corrupt
    manifest = da.build_day(s3, DAY, now=NOW)
    assert manifest["components"]["forecasts"]["status"] == "failed"
    assert manifest["complete"] is False


# --- window, reminders, deletion -----------------------------------------------------------

def _entry(created, downloaded=False):
    entry = {"created": created.isoformat(), "expires": (created + timedelta(days=30)).isoformat(), "bytes": 1}
    if downloaded:
        entry["downloaded_verified"] = True
    return entry


def test_undownloaded_bundle_is_never_deletable():
    created = NOW - timedelta(days=400)
    assert da.eligible_for_delete(_entry(created), NOW) is False


def test_downloaded_bundle_deletable_only_after_window():
    assert da.eligible_for_delete(_entry(NOW - timedelta(days=10), True), NOW) is False
    assert da.eligible_for_delete(_entry(NOW - timedelta(days=31), True), NOW) is True
    done = _entry(NOW - timedelta(days=31), True)
    done["deleted"] = "x"
    assert da.eligible_for_delete(done, NOW) is False


def test_reminder_cadence_fires_once_per_threshold():
    created = NOW - timedelta(days=23, hours=1)       # ~7 days left
    entry = _entry(created)
    assert da.reminder_due(entry, NOW) == "7"
    entry["reminders"] = ["7"]
    assert da.reminder_due(entry, NOW) is None
    assert da.reminder_due(entry, NOW + timedelta(days=4)) == "3"
    entry["reminders"] = ["7", "3"]
    assert da.reminder_due(entry, NOW + timedelta(days=6, hours=2)) == "1"
    entry["reminders"] = ["7", "3", "1"]
    assert da.reminder_due(entry, NOW + timedelta(days=6, hours=2)) is None


def test_overdue_notice_once_a_day_and_none_when_downloaded():
    entry = _entry(NOW - timedelta(days=40))
    assert da.reminder_due(entry, NOW) == "overdue"
    entry["overdue_notified"] = NOW.date().isoformat()
    assert da.reminder_due(entry, NOW) is None
    assert da.reminder_due(entry, NOW + timedelta(days=1)) == "overdue"
    assert da.reminder_due(_entry(NOW - timedelta(days=40), True), NOW) is None


def test_run_builds_notifies_and_never_deletes_by_default(env):
    root, s3 = env
    with patch.object(da, "notify") as notify, patch.dict(os.environ, {"SMF_DAY_ARCHIVE_BACKFILL_DAYS": "3"}):
        result = da.run(now=NOW, s3=s3)
    assert DAY in result["built"]
    assert notify.call_count >= 1
    ledger = s3.json("day-archive/ledger.json")
    assert ledger[DAY]["complete"] is True and ledger[DAY]["expires"]
    # an old, undownloaded bundle stays, and triggers an overdue notice, not a delete
    ledger["2026-08-01"] = _entry(NOW - timedelta(days=70))
    s3.objects["day-archive/ledger.json"] = json.dumps(ledger).encode()
    s3.objects["day-archive/2026-08-01/manifest.json"] = b"{}"
    with patch.object(da, "notify") as notify:
        result = da.run(now=NOW, s3=s3)
    assert "2026-08-01" in result["overdue"]
    assert "day-archive/2026-08-01/manifest.json" in s3.objects


def test_receipt_enables_delete_only_with_flag(env):
    root, s3 = env
    ledger = {"2026-08-01": _entry(NOW - timedelta(days=70))}
    s3.objects["day-archive/ledger.json"] = json.dumps(ledger).encode()
    s3.objects["day-archive/2026-08-01/manifest.json"] = b"{}"
    s3.objects["day-archive/2026-08-01/forecasts.zip"] = b"z"
    da.write_receipt(s3, "2026-08-01", "tester")
    with patch.object(da, "notify"), patch.dict(os.environ, {"SMF_DAY_ARCHIVE_BACKFILL_DAYS": "1"}):
        da.run(now=NOW, s3=s3)
        assert "day-archive/2026-08-01/forecasts.zip" in s3.objects      # flag off: kept
        with patch.dict(os.environ, {"SMF_DAY_ARCHIVE_DELETE": "true"}):
            result = da.run(now=NOW, s3=s3)
    assert result["deleted"] == ["2026-08-01"]
    assert "day-archive/2026-08-01/forecasts.zip" not in s3.objects
    assert s3.json("day-archive/ledger.json")["2026-08-01"]["deleted"]


def test_unverified_receipt_does_not_count(env):
    root, s3 = env
    s3.objects["day-archive/ledger.json"] = json.dumps({"2026-08-01": _entry(NOW - timedelta(days=70))}).encode()
    da.write_receipt(s3, "2026-08-01", "tester", verified=False)
    ledger = da.load_ledger(s3)
    assert da.merge_receipts(s3, ledger) == []
    assert not ledger["2026-08-01"].get("downloaded_verified")


# --- local pruning ---------------------------------------------------------------------------

def test_prune_local_requires_verified_present_and_matching_sha(env):
    root, s3 = env
    manifest = da.build_day(s3, DAY, now=NOW)
    today = datetime(2026, 10, 20, tzinfo=timezone.utc).date()
    target = root / "archive/forecasts/station_forecasts_20261010_09.json"
    keep_report = root / "reports/2026-10-10/validation_summary.json"
    assert da.prune_local(manifest, today, dry_run=True)["would_delete"] > 0 and target.exists()
    target.write_bytes(b'{"f":"changed after bundling"}')           # no longer matches the manifest
    outcome = da.prune_local(manifest, today, dry_run=False)
    assert target.exists() and outcome["kept"] >= 1                 # changed file kept
    assert not (root / "archive/raw_data/raw_data_20261010.json").exists()   # verified source removed
    assert keep_report.exists()                                      # derived is never pruned


def test_prune_local_skips_days_inside_local_window(env):
    root, s3 = env
    manifest = da.build_day(s3, DAY, now=NOW)
    today = datetime(2026, 10, 14, tzinfo=timezone.utc).date()      # 4 days old < 7
    assert da.prune_local(manifest, today, dry_run=False) == {"deleted": 0, "would_delete": 0, "kept": 0}


def test_prune_local_ignores_failed_components(env):
    root, s3 = env
    manifest = da.build_day(s3, DAY, now=NOW)
    manifest["components"]["observations"]["status"] = "failed"
    da.prune_local(manifest, datetime(2026, 10, 30, tzinfo=timezone.utc).date(), dry_run=False)
    assert (root / "archive/raw_data/raw_data_20261010.json").exists()


def test_presigned_links_cover_every_part(env):
    root, s3 = env
    da.build_day(s3, DAY, now=NOW)
    names = {link["name"] for link in da.presigned_links(s3, DAY, expires_in=600)}
    assert {"manifest.json", "forecasts.zip", "rtma.zip", "maps_utc_20261010.zip", "maps_utc_20261011.zip"} <= names
    assert da.presigned_links(s3, "2000-01-01") is None


def test_target_days_skips_complete_and_retries_incomplete_recent():
    ledger = {"2026-10-11": {"complete": True}, "2026-10-10": {"complete": False}, "2026-10-05": {"complete": False}}
    with patch.dict(os.environ, {"SMF_DAY_ARCHIVE_BACKFILL_DAYS": "7", "SMF_DAY_ARCHIVE_RETRY_DAYS": "4"}):
        days = da.target_days(NOW, ledger)
    assert "2026-10-11" not in days and "2026-10-10" in days
    assert "2026-10-05" not in days          # incomplete but past the retry window
    assert "2026-10-09" in days              # never bundled
