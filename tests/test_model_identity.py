"""models/identity.py: one checksum-based name per model (mirrored byte-for-byte in api/models/identity.py)."""
import hashlib
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from models import identity

SHA = "7dd2467624e9a3303f2ea460a16070f8746558581fe3d2698922ecfb8dee735d"       # the live fuel_moisture model


class IdentityTests(unittest.TestCase):
    def test_the_live_models_identity_is_built_from_its_checksum(self):
        self.assertEqual(identity.make_identity("fuel_moisture", SHA, "2026-08-16 13:57:02"), "fuel_moisture@2026.08.16-7dd24676")
        self.assertEqual(identity.make_identity("fuel_moisture", SHA, datetime(2026, 11, 18)), "fuel_moisture@2026.11.18-7dd24676")

    def test_tag_and_identity_convert_both_ways_and_old_tags_are_not_identities(self):
        ident = "fuel_moisture@2026.11.18-7dd24676"
        tag = identity.to_tag(ident)
        self.assertEqual(tag, "fuel_moisture-2026.11.18-7dd24676")
        self.assertNotIn("@", tag)
        self.assertEqual(identity.from_tag(tag), ident)
        self.assertIsNone(identity.from_tag("fuel_moisture-v0.0.1-beta.6"))
        self.assertIsNone(identity.from_tag(""))
        with self.assertRaises(ValueError):
            identity.to_tag("not an identity")

    def test_the_same_bytes_are_the_same_model_whatever_the_date(self):
        self.assertTrue(identity.same_model("fuel_moisture@2026.08.16-7dd24676", "fuel_moisture-2026.11.18-7dd24676"))
        self.assertFalse(identity.same_model("fuel_moisture@2026.08.16-7dd24676", "fuel_moisture@2026.08.16-aaaaaaaa"))
        self.assertFalse(identity.same_model("fuel_moisture@2026.08.16-7dd24676", "fire_danger@2026.08.16-7dd24676"))
        self.assertFalse(identity.same_model("fuel_moisture-v0.0.1-beta.6", "fuel_moisture@2026.08.16-7dd24676"))

    def test_a_file_is_hashed_by_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a.json", Path(tmp) / "b.json"
            a.write_bytes(b"MODEL"); b.write_bytes(b"MODEL")
            expected = hashlib.sha256(b"MODEL").hexdigest()[:8]
            self.assertEqual(identity.identity_from_file("fuel_moisture", a, "2026-01-02"), f"fuel_moisture@2026.01.02-{expected}")
            self.assertEqual(identity.identity_from_file("fuel_moisture", a, "2026-01-02"), identity.identity_from_file("fuel_moisture", b, "2026-01-02"))
            b.write_bytes(b"MODEL2")
            self.assertNotEqual(identity.sha256_file(a), identity.sha256_file(b))

    def test_a_bundle_identity_depends_on_every_asset_but_not_on_their_order(self):
        one = identity.bundle_sha256({"weights": "a" * 64, "thresholds": "b" * 64})
        self.assertEqual(one, identity.bundle_sha256({"thresholds": "b" * 64, "weights": "a" * 64}))
        self.assertNotEqual(one, identity.bundle_sha256({"weights": "a" * 64, "thresholds": "c" * 64}))
        self.assertTrue(identity.identity_from_assets("fire_weather_index", {"w": "a" * 64}, "2026-09-20").startswith("fire_weather_index@2026.09.20-"))

    def test_bad_input_is_refused(self):
        for family, sha in (("Fuel Moisture", SHA), ("fuel_moisture", "short"), ("", SHA), ("fuel_moisture", SHA.upper())):
            with self.assertRaises(ValueError):
                identity.make_identity(family, sha)

    def test_a_missing_date_falls_back_to_today_not_a_crash(self):
        self.assertRegex(identity.make_identity("fuel_moisture", SHA, None), r"^fuel_moisture@\d{4}\.\d{2}\.\d{2}-7dd24676$")


if __name__ == "__main__":
    unittest.main()
