"""live.json on the `registry` GitHub release: written after a promotion, merged per family, never in the way."""
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from models import live_record, versioning


class FakeGitHub:
    """A tiny in-memory GitHub: releases with assets. Understands exactly the `gh release ...` calls live_record makes."""

    def __init__(self, releases=None, fail=None, raises=None):
        self.releases = releases if releases is not None else {}      # {tag: {asset_name: bytes}}
        self.calls, self.fail, self.raises = [], set(fail or ()), raises

    def __call__(self, command, **kwargs):
        self.calls.append(command)
        if self.raises:
            raise self.raises
        verb, args = command[2], command[3:]
        if verb in self.fail:
            return SimpleNamespace(returncode=1, stderr=f"{verb} failed", stdout="")
        tag = args[0]
        if verb == "view":
            return SimpleNamespace(returncode=0 if tag in self.releases else 1, stderr="", stdout="")
        if verb == "create":
            self.releases.setdefault(tag, {})
            return SimpleNamespace(returncode=0, stderr="", stdout="")
        if verb == "download":
            directory, pattern = Path(args[args.index("--dir") + 1]), args[args.index("--pattern") + 1]
            data = self.releases.get(tag, {}).get(pattern)
            if data is None:
                return SimpleNamespace(returncode=1, stderr="no asset", stdout="")
            (directory / pattern).write_bytes(data)
            return SimpleNamespace(returncode=0, stderr="", stdout="")
        if verb == "upload":
            files = [Path(a) for a in args[1:] if not a.startswith("--") and Path(a).is_file()]
            for f in files:
                self.releases.setdefault(tag, {})[f.name] = f.read_bytes()
            return SimpleNamespace(returncode=0, stderr="", stdout="")
        raise AssertionError(f"unexpected gh call: {command}")

    def live(self):
        return json.loads(self.releases["registry"]["live.json"])


def stable(version="1.0.1", sha="7dd2467624e9a3303f2ea460a16070f8746558581fe3d2698922ecfb8dee735d", tag=None, identity=None):
    record = {"version": version, "sha256": sha, "file": "models/versions/x.json", "promoted_at": "2026-10-03 02:45:05",
              "performance": {"source_release_tag": tag} if tag else {}}
    if identity:
        record["identity"] = identity
    return record


class PublishLiveRecordTests(unittest.TestCase):
    def setUp(self):
        patcher = patch.dict(os.environ, {"SMF_GITHUB_REPO": "owner/models"})
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("PYTEST_CURRENT_TEST", None)
        os.environ.pop("SMF_MARK_LIVE_RELEASE", None)

    def test_a_release_backed_model_is_recorded_by_tag_without_uploading_a_model_file(self):
        gh = FakeGitHub()
        out = live_record.publish_live_record("fuel_moisture", stable(tag="fuel_moisture-v0.0.1-beta.6", identity="fuel_moisture@2026.08.16-7dd24676"),
                                              None, run=gh)
        self.assertTrue(out["published"], out)
        entry = gh.live()["families"]["fuel_moisture"]
        self.assertEqual((entry["identity"], entry["source_release_tag"], entry["server_version"]),
                         ("fuel_moisture@2026.08.16-7dd24676", "fuel_moisture-v0.0.1-beta.6", "1.0.1"))
        self.assertIsNone(entry["asset"])
        self.assertEqual(list(gh.releases["registry"]), ["live.json"])               # no model file for a release-backed model

    def test_a_server_trained_model_has_its_file_uploaded_so_it_is_always_fetchable(self):
        gh = FakeGitHub()
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "fuel_moisture_1.0.2.json"
            artifact.write_bytes(b"SERVER TRAINED MODEL")
            out = live_record.publish_live_record("fuel_moisture", stable(version="1.0.2", sha="ab" * 32), None, artifact_path=artifact, run=gh)
        self.assertTrue(out["published"], out)
        entry = gh.live()["families"]["fuel_moisture"]
        self.assertTrue(entry["asset"].startswith("fuel_moisture-") and entry["asset"].endswith("-abababab.json"), entry["asset"])
        self.assertEqual(gh.releases["registry"][entry["asset"]], b"SERVER TRAINED MODEL")

    def test_each_family_has_its_own_entry_and_updating_one_leaves_the_others(self):
        gh = FakeGitHub()
        live_record.publish_live_record("fuel_moisture", stable(tag="t1"), None, run=gh)
        live_record.publish_live_record("fire_weather_index", stable(version="0.0.1", sha="cd" * 32, tag="fwi-1"), None, run=gh)
        live_record.publish_live_record("fuel_moisture", stable(version="1.0.2", sha="ef" * 32, tag="t2"), stable(identity="fuel_moisture@2026.08.16-7dd24676"), run=gh)
        families = gh.live()["families"]
        self.assertEqual(sorted(families), ["fire_weather_index", "fuel_moisture"])
        self.assertEqual(families["fuel_moisture"]["source_release_tag"], "t2")
        self.assertEqual(families["fuel_moisture"]["previous_identity"], "fuel_moisture@2026.08.16-7dd24676")
        self.assertEqual(families["fire_weather_index"]["source_release_tag"], "fwi-1")

    def test_the_registry_release_is_created_once(self):
        gh = FakeGitHub()
        live_record.publish_live_record("fuel_moisture", stable(tag="t1"), None, run=gh)
        live_record.publish_live_record("fuel_moisture", stable(tag="t2"), None, run=gh)
        self.assertEqual(len([c for c in gh.calls if c[2] == "create"]), 1)

    def test_nothing_is_written_when_disabled_unconfigured_or_there_is_no_checksum(self):
        gh = FakeGitHub()
        with patch.dict(os.environ, {"SMF_MARK_LIVE_RELEASE": "false"}):
            self.assertIn("disabled", live_record.publish_live_record("m", stable(), None, run=gh)["skipped"])
        with patch.dict(os.environ, {"SMF_GITHUB_REPO": ""}):
            self.assertIn("SMF_GITHUB_REPO", live_record.publish_live_record("m", stable(), None, run=gh)["skipped"])
        self.assertIn("checksum", live_record.publish_live_record("m", {"version": "1", "assets": {}}, None, run=gh)["skipped"])
        self.assertEqual(gh.calls, [])

    def test_failures_are_reported_never_raised(self):
        self.assertIn("upload failed", live_record.publish_live_record("m", stable(tag="t"), None, run=FakeGitHub(fail=["upload"]))["error"])
        self.assertIn("could not create", live_record.publish_live_record("m", stable(tag="t"), None, run=FakeGitHub(fail=["create"]))["error"])
        self.assertIn("not installed", live_record.publish_live_record("m", stable(tag="t"), None, run=FakeGitHub(raises=FileNotFoundError()))["skipped"])
        self.assertIn("did not answer", live_record.publish_live_record("m", stable(tag="t"), None, run=FakeGitHub(raises=subprocess.TimeoutExpired("gh", 20)))["error"])
        self.assertIn("RuntimeError", live_record.publish_live_record("m", stable(tag="t"), None, run=FakeGitHub(raises=RuntimeError("x")))["error"])

    def test_a_corrupt_existing_live_json_is_replaced_not_fatal(self):
        gh = FakeGitHub({"registry": {"live.json": b"not json{"}})
        out = live_record.publish_live_record("fuel_moisture", stable(tag="t1"), None, run=gh)
        self.assertTrue(out["published"])
        self.assertEqual(list(gh.live()["families"]), ["fuel_moisture"])


class WiredIntoPromotionTests(unittest.TestCase):
    def test_promote_writes_the_record_after_saving_and_a_failure_cannot_break_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / "asset.json"
            asset.write_text("{}")
            with patch.multiple(versioning, API_DIR=root, MODELS_DIR=root / "models", CONFIG_PATH=root / "models" / "config.json",
                                VERSIONS_DIR=root / "models" / "versions"):
                versioning.register_trained_model("fire_weather_index", channel="beta", assets={"factor_weights": str(asset)},
                                                  metadata={"model_family": "x", "advisory_only": True})
                with patch("models.live_record.publish_live_record", side_effect=RuntimeError("github down")) as boom:
                    version = versioning.promote("fire_weather_index")
                    self.assertTrue(boom.called)
                self.assertEqual(versioning.get_model_entry("fire_weather_index")["stable"]["version"], version)

    def test_promote_hands_the_previous_and_new_stable_to_the_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / "asset.json"
            asset.write_text("{}")
            with patch.multiple(versioning, API_DIR=root, MODELS_DIR=root / "models", CONFIG_PATH=root / "models" / "config.json",
                                VERSIONS_DIR=root / "models" / "versions"):
                versioning.register_trained_model("fire_weather_index", channel="beta", assets={"factor_weights": str(asset)},
                                                  metadata={"model_family": "x", "advisory_only": True})
                with patch("models.live_record.publish_live_record") as record:
                    versioning.promote("fire_weather_index")
                args = record.call_args
                self.assertEqual(args.args[0], "fire_weather_index")
                self.assertEqual(args.args[1]["version"], versioning.get_model_entry("fire_weather_index")["stable"]["version"])
                self.assertIsNone(args.args[2])                                          # nothing was live before


if __name__ == "__main__":
    unittest.main()
