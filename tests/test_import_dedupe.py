"""Importing a release whose bytes are already registered returns the existing version instead of a duplicate beta."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from models import versioning
from pipelines import import_model


def fake_gh(release_files):
    """`gh release download <tag> --dir <dir>` that writes the given {filename: bytes} into <dir>."""
    def run(command, **kwargs):
        directory = Path(command[command.index("--dir") + 1])
        for name, data in release_files.items():
            (directory / name).write_bytes(data)
        return SimpleNamespace(returncode=0)
    return run


class ImportDedupeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        patcher = patch.multiple(versioning, API_DIR=root, MODELS_DIR=root / "models", CONFIG_PATH=root / "models" / "config.json",
                                 VERSIONS_DIR=root / "models" / "versions")
        patcher.start()
        self.addCleanup(patcher.stop)
        self.meta = json.dumps({"performance": {"mae": 2.9}, "model_metadata": {"feature_columns": ["t"]}}).encode()

    def do_import(self, model_bytes, tag, meta=None):
        with patch.object(import_model.subprocess, "run", fake_gh({"fuel_moisture_x.json": model_bytes, "metadata.json": meta or self.meta})):
            return import_model.import_release("fuel_moisture", tag, "o/r")

    def meta_with_identity(self, identity):
        return json.dumps({"identity": identity, "performance": {"mae": 2.9}, "model_metadata": {"feature_columns": ["t"]}}).encode()

    def test_the_same_bytes_imported_twice_is_one_version(self):
        first = self.do_import(b"MODEL A", "fuel_moisture-v0.0.1-beta.6")
        second = self.do_import(b"MODEL A", "fuel_moisture-v0.0.1-beta.6")
        self.assertEqual(first, second)
        entry = versioning.get_model_entry("fuel_moisture")
        betas = [entry.get("beta"), *[h for h in entry.get("history", []) if h.get("channel") == "beta"]]
        self.assertEqual(len({b["version"] for b in betas if b}), 1)

    def test_the_same_bytes_under_a_different_tag_is_still_the_same_model(self):
        first = self.do_import(b"MODEL A", "fuel_moisture-v0.0.1-beta.6")
        self.assertEqual(self.do_import(b"MODEL A", "fuel_moisture-v9.9.9"), first)

    def test_a_different_model_is_imported_as_a_new_beta(self):
        first = self.do_import(b"MODEL A", "fuel_moisture-v0.0.1-beta.6")
        second = self.do_import(b"MODEL B", "fuel_moisture-v0.0.1-beta.2")
        self.assertNotEqual(first, second)
        self.assertEqual(versioning.get_model_entry("fuel_moisture")["beta"]["version"], second)

    def test_a_promoted_model_is_recognised_too(self):
        first = self.do_import(b"MODEL A", "fuel_moisture-v0.0.1-beta.6")
        entry = versioning.get_model_entry("fuel_moisture")
        stable_sha = entry["beta"]["sha256"]
        # promote without running real gates: move beta to stable the way promote() records it
        config = versioning._load_config()
        config["fuel_moisture"]["stable"] = {**config["fuel_moisture"]["beta"], "version": "1.0.1"}
        config["fuel_moisture"]["beta"] = None
        versioning._save_config(config)
        self.assertEqual(import_model.already_registered("fuel_moisture", file_sha=stable_sha), "1.0.1")
        self.assertEqual(self.do_import(b"MODEL A", "fuel_moisture-v0.0.1-beta.6"), "1.0.1")
        self.assertTrue(first)


    def test_the_identity_in_the_release_is_kept_so_both_machines_use_the_same_name(self):
        import hashlib
        sha = hashlib.sha256(b"MODEL A").hexdigest()
        identity = f"fuel_moisture@2026.11.18-{sha[:8]}"
        self.do_import(b"MODEL A", "fuel_moisture-2026.11.18-" + sha[:8], meta=self.meta_with_identity(identity))
        self.assertEqual(versioning.get_model_entry("fuel_moisture")["beta"]["identity"], identity)

    def test_an_identity_that_does_not_match_the_files_is_refused_and_nothing_is_registered(self):
        wrong = "fuel_moisture@2026.11.18-deadbeef"
        with self.assertRaises(import_model.ImportValidationError) as context:
            self.do_import(b"MODEL A", "fuel_moisture-2026.11.18-deadbeef", meta=self.meta_with_identity(wrong))
        self.assertIn("does not match the downloaded files", str(context.exception))
        self.assertIsNone(versioning.get_model_entry("fuel_moisture").get("beta"))

    def test_another_familys_identity_is_refused(self):
        with self.assertRaises(import_model.ImportValidationError):
            self.do_import(b"MODEL A", "t", meta=self.meta_with_identity("fire_danger@2026.11.18-12345678"))

    def test_a_release_without_an_identity_still_imports_and_gets_one_from_its_checksum(self):
        self.do_import(b"MODEL A", "fuel_moisture-v0.0.1-beta.6")
        self.assertRegex(versioning.get_model_entry("fuel_moisture")["beta"]["identity"], r"^fuel_moisture@\d{4}\.\d{2}\.\d{2}-[0-9a-f]{8}$")


if __name__ == "__main__":
    unittest.main()
