"""The live-release marker: after a promotion/rollback, GitHub is told which release is live. It must never get in the way."""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from models import live_marker, versioning


def record(tag=None, version="1.0.0"):
    return {"version": version, "performance": {"mae": 2.5, **({"source_release_tag": tag} if tag else {})}}


class Recorder:
    """A fake `gh`: remembers every command, returns a configurable exit code."""

    def __init__(self, returncode=0, raises=None):
        self.calls, self.returncode, self.raises = [], returncode, raises

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if self.raises:
            raise self.raises
        return SimpleNamespace(returncode=self.returncode, stderr="boom" if self.returncode else "", stdout="")


class MarkLiveReleaseTests(unittest.TestCase):
    def setUp(self):
        # pytest sets PYTEST_CURRENT_TEST, which switches the marker off on purpose; these tests exercise the real logic.
        patcher = patch.dict(os.environ, {"SMF_GITHUB_REPO": "owner/models"})
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("PYTEST_CURRENT_TEST", None)
        os.environ.pop("SMF_MARK_LIVE_RELEASE", None)

    def commands(self, run):
        return [call[0] for call in run.calls]

    def test_release_tag_is_read_from_the_registry_record(self):
        self.assertEqual(live_marker.release_tag(record("fuel_moisture-v1")), "fuel_moisture-v1")
        self.assertIsNone(live_marker.release_tag(record()))
        self.assertIsNone(live_marker.release_tag(None))
        self.assertIsNone(live_marker.release_tag({"performance": {"source_release_tag": "  "}}))

    def test_promotion_marks_the_new_release_live_and_demotes_the_old_one(self):
        run = Recorder()
        out = live_marker.mark_live_release("fuel_moisture", record("fuel_moisture-v2"), record("fuel_moisture-v1"), run=run)
        self.assertEqual(self.commands(run), [
            ["gh", "release", "edit", "fuel_moisture-v1", "--repo", "owner/models", "--prerelease=true"],
            ["gh", "release", "edit", "fuel_moisture-v2", "--repo", "owner/models", "--prerelease=false"]])
        self.assertTrue(out["marked_live"]["ok"] and out["unmarked"]["ok"])
        self.assertIsNone(out["error"])

    def test_the_same_release_is_not_demoted_and_remarked(self):
        run = Recorder()
        live_marker.mark_live_release("fuel_moisture", record("fuel_moisture-v1"), record("fuel_moisture-v1"), run=run)
        self.assertEqual(self.commands(run), [["gh", "release", "edit", "fuel_moisture-v1", "--repo", "owner/models", "--prerelease=false"]])

    def test_a_server_trained_model_replacing_a_release_demotes_the_old_release_only(self):
        run = Recorder()
        out = live_marker.mark_live_release("fuel_moisture", record(), record("fuel_moisture-v1"), run=run)
        self.assertEqual(self.commands(run), [["gh", "release", "edit", "fuel_moisture-v1", "--repo", "owner/models", "--prerelease=true"]])
        self.assertIn("trained on this server", out["skipped"])

    def test_nothing_to_mark_means_no_github_calls(self):
        run = Recorder()
        out = live_marker.mark_live_release("fuel_moisture", record(), record(), run=run)
        self.assertEqual(run.calls, [])
        self.assertIn("neither", out["skipped"])

    def test_no_repo_configured_means_no_github_calls(self):
        run = Recorder()
        with patch.dict(os.environ, {"SMF_GITHUB_REPO": ""}):
            out = live_marker.mark_live_release("fuel_moisture", record("t"), record("u"), run=run)
        self.assertEqual(run.calls, [])
        self.assertIn("SMF_GITHUB_REPO", out["skipped"])

    def test_the_off_switch_and_the_pytest_guard_both_stop_it(self):
        run = Recorder()
        with patch.dict(os.environ, {"SMF_MARK_LIVE_RELEASE": "false"}):
            self.assertIn("disabled", live_marker.mark_live_release("m", record("t"), None, run=run)["skipped"])
        with patch.dict(os.environ, {"PYTEST_CURRENT_TEST": "x"}):
            self.assertIn("disabled", live_marker.mark_live_release("m", record("t"), None, run=run)["skipped"])
        self.assertEqual(run.calls, [])

    def test_failures_are_reported_not_raised(self):
        out = live_marker.mark_live_release("m", record("new"), record("old"), run=Recorder(returncode=1))
        self.assertFalse(out["marked_live"]["ok"])
        out = live_marker.mark_live_release("m", record("new"), None, run=Recorder(raises=FileNotFoundError()))
        self.assertIn("not installed", out["skipped"])
        out = live_marker.mark_live_release("m", record("new"), None, run=Recorder(raises=subprocess.TimeoutExpired("gh", 20)))
        self.assertIn("did not answer", out["error"])
        out = live_marker.mark_live_release("m", record("new"), None, run=Recorder(raises=RuntimeError("weird")))
        self.assertIn("RuntimeError", out["error"])

    def test_every_call_has_a_timeout_and_only_edits_the_prerelease_flag(self):
        run = Recorder()
        live_marker.mark_live_release("m", record("new"), record("old"), run=run)
        for command, kwargs in run.calls:
            self.assertEqual(kwargs["timeout"], live_marker.TIMEOUT_SECONDS)
            self.assertEqual(command[:3], ["gh", "release", "edit"])
            self.assertEqual(len([a for a in command if a.startswith("--")]), 2)     # --repo and --prerelease, nothing else


class WiredIntoPromotionTests(unittest.TestCase):
    """promote()/rollback() announce the change AFTER saving, and nothing the announcement does can break them."""

    def isolated(self, root):
        return patch.multiple(versioning, API_DIR=root, MODELS_DIR=root / "models", CONFIG_PATH=root / "models" / "config.json",
                              VERSIONS_DIR=root / "models" / "versions")

    def register(self, root, tag):
        asset = root / f"asset_{tag}.json"
        asset.write_text("{}")
        versioning.register_trained_model("fire_weather_index", channel="beta", assets={"factor_weights": str(asset)},
                                          performance={"source_release_tag": tag}, metadata={"model_family": "x", "advisory_only": True})

    def test_promote_and_rollback_announce_previous_and_new(self):
        with tempfile.TemporaryDirectory() as directory, self.isolated(Path(directory)), patch("models.live_marker.mark_live_release") as mark:
            root = Path(directory)
            self.register(root, "fire_weather_index-v1")
            versioning.promote("fire_weather_index")
            first = mark.call_args.args
            self.assertEqual((first[0], first[2]), ("fire_weather_index", None))
            self.assertEqual(first[1]["performance"]["source_release_tag"], "fire_weather_index-v1")
            self.register(root, "fire_weather_index-v2")
            versioning.promote("fire_weather_index")
            _, new, previous = mark.call_args.args
            self.assertEqual((new["performance"]["source_release_tag"], previous["performance"]["source_release_tag"]),
                             ("fire_weather_index-v2", "fire_weather_index-v1"))
            versioning.rollback("fire_weather_index")
            _, new, previous = mark.call_args.args
            self.assertEqual((new["performance"]["source_release_tag"], previous["performance"]["source_release_tag"]),
                             ("fire_weather_index-v1", "fire_weather_index-v2"))
            self.assertEqual(mark.call_count, 3)

    def test_a_marker_that_explodes_cannot_break_a_promotion_or_rollback(self):
        with tempfile.TemporaryDirectory() as directory, self.isolated(Path(directory)), \
                patch("models.live_marker.mark_live_release", side_effect=RuntimeError("github is down")):
            root = Path(directory)
            self.register(root, "fire_weather_index-v1")
            version = versioning.promote("fire_weather_index")
            self.register(root, "fire_weather_index-v2")
            versioning.promote("fire_weather_index")
            versioning.rollback("fire_weather_index")
            entry = versioning.get_model_entry("fire_weather_index")
        self.assertEqual(entry["stable"]["performance"]["source_release_tag"], "fire_weather_index-v1")
        self.assertTrue(version)

    def test_the_registry_is_saved_before_the_announcement(self):
        seen = {}
        def peek(model_type, new, previous):
            seen["stable_on_disk"] = versioning.get_model_entry(model_type)["stable"]["performance"]["source_release_tag"]
        with tempfile.TemporaryDirectory() as directory, self.isolated(Path(directory)), patch("models.live_marker.mark_live_release", side_effect=peek):
            self.register(Path(directory), "fire_weather_index-v1")
            versioning.promote("fire_weather_index")
        self.assertEqual(seen["stable_on_disk"], "fire_weather_index-v1")


if __name__ == "__main__":
    unittest.main()
