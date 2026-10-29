import base64
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from ai import clef, clef_tasks
from ai.cloudflare import CloudflareAIError
from core import database

REPORT_REPLY = {
    "model": "clef",
    "answers": {
        "shows_fire_or_smoke": {"type": "noul", "noul": 0.93},
        "burn_type": {"type": "choice", "choice": "wildfire", "confidence": 0.8,
                      "probabilities": {"wildfire": 0.8, "unclear": 0.2, "made_up": 0.0}},
        "urgency": {"type": "score", "score": 2.4, "legend": {"0": "x"}, "probabilities": {"2": 1.0}, "confidence": 0.7},
        "likely_spam_or_prank": {"type": "noul", "noul": 1.7},  # out of range: clamped
    },
    "usage": {"input_tokens": 321, "output_tokens": 0},
}


class FakeClient:
    model = "clef"
    configured = True

    def __init__(self, reply=REPORT_REPLY, fail_with_images=False):
        self.reply, self.fail_with_images, self.calls = reply, fail_with_images, []

    def run(self, state, questions, images=()):
        self.calls.append({"state": state, "questions": questions, "images": list(images)})
        if images and self.fail_with_images:
            raise CloudflareAIError("Clef request failed with HTTP 400")
        return self.reply


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status = payload, status

    def raise_for_status(self):
        if self.status >= 400:
            raise httpx.HTTPStatusError("bad", request=httpx.Request("POST", "https://x"), response=httpx.Response(self.status))

    def json(self):
        return self.payload


class FakeHttp:
    def __init__(self, response):
        self.response, self.calls = response, []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return self.response


class PureTests(unittest.TestCase):
    def test_prepare_images_drops_what_clef_would_refuse(self):
        png = b"\x89PNG" + b"x" * 10
        files = [{"content_type": "image/png", "data": png},
                 {"content_type": "application/pdf", "data": b"%PDF"},
                 {"content_type": "image/jpeg", "data": b"x" * (clef.MAX_IMAGE_BYTES + 1)},
                 {"content_type": "image/webp", "data": b""}]
        out = clef.prepare_images(files)
        self.assertEqual(out, [{"content_type": "image/png", "base64": base64.b64encode(png).decode()}])

    def test_prepare_images_caps_count(self):
        files = [{"content_type": "image/png", "data": b"a" * 10}] * 7
        self.assertEqual(len(clef.prepare_images(files)), clef.MAX_IMAGES)

    def test_clean_answers_checks_against_the_questions_asked(self):
        cleaned = clef.clean_answers(REPORT_REPLY["answers"], clef_tasks.FIRE_REPORT_QUESTIONS)
        self.assertEqual(cleaned["shows_fire_or_smoke"]["noul"], 0.93)
        self.assertEqual(cleaned["likely_spam_or_prank"]["noul"], 1.0)  # clamped
        self.assertEqual(cleaned["burn_type"]["probabilities"], {"wildfire": 0.8, "unclear": 0.2})  # unknown option dropped
        self.assertEqual(cleaned["urgency"], {"type": "score", "score": 2.4, "confidence": 0.7, "levels": 4})

    def test_clean_answers_drops_unknown_choice_and_wrong_shape(self):
        questions = clef_tasks.FIRE_REPORT_QUESTIONS
        cleaned = clef.clean_answers({"burn_type": {"type": "choice", "choice": "nope", "probabilities": {}},
                                      "urgency": "high", "shows_fire_or_smoke": {"type": "noul"}}, questions)
        self.assertEqual(cleaned, {})

    def test_off_by_default(self):
        with patch.dict("os.environ", {}, clear=False):
            self.assertFalse(clef.enabled())

    def test_client_sends_documented_body_and_unwraps_result(self):
        http = FakeHttp(FakeResponse({"success": True, "result": REPORT_REPLY}))
        c = clef.ClefClient(api_key="k", account_id="acct", http_client=http)
        image = {"content_type": "image/png", "base64": "AAAA"}
        reply = c.run({"a": 1}, {"q": {"type": "noul", "instructions": "?"}}, [image])
        self.assertEqual(reply["answers"]["urgency"]["score"], 2.4)
        sent = http.calls[0]
        self.assertTrue(sent["url"].endswith("/accounts/acct/ai/run/@cf/cloudflare/clef"))
        self.assertEqual((sent["json"]["model"], sent["json"]["images"]), ("clef", [image]))
        self.assertEqual(sent["headers"]["Authorization"], "Bearer k")

    def test_client_surfaces_http_errors_and_missing_answers(self):
        c = clef.ClefClient(api_key="k", account_id="a", http_client=FakeHttp(FakeResponse({}, status=500)))
        with self.assertRaisesRegex(CloudflareAIError, "HTTP 500"):
            c.run("s", {"q": {}})
        c = clef.ClefClient(api_key="k", account_id="a", http_client=FakeHttp(FakeResponse({"result": {"model": "clef"}})))
        with self.assertRaisesRegex(CloudflareAIError, "no answers"):
            c.run("s", {"q": {}})


class DbTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.db = self.dir / "smf.db"
        for target in (database, clef, clef_tasks):
            patcher = patch.object(target, "get_db_path", return_value=self.db)
            patcher.start()
            self.addCleanup(patcher.stop)
        for name, value in (("MEDIA_DIR", self.dir / "media"), ("PROOF_DIR", self.dir / "proofs")):
            patcher = patch.object(clef_tasks, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        (self.dir / "media").mkdir()
        (self.dir / "proofs").mkdir()
        database.init_database()

    def tearDown(self):
        try:
            self.tmp.cleanup()
        except PermissionError:
            pass

    def add_report(self, source="user_submission"):
        with sqlite3.connect(self.db) as conn:
            cur = conn.execute(
                "INSERT INTO fire_events (source, status, latitude, longitude, occurred_at, description, "
                "reporter_name, reporter_contact, county_name) VALUES (?, 'pending', 37, -92, '2026-10-01T12:00:00Z', "
                "'Grass fire near a barn', 'Jane Q Public', 'jane@example.com', 'Phelps')", (source,))
            return cur.lastrowid

    def add_photo(self, event_id, data=b"\x89PNG\r\n\x1a\nxxxx", name="p.png"):
        (self.dir / "media" / name).write_bytes(data)
        with sqlite3.connect(self.db) as conn:
            conn.execute("INSERT INTO fire_event_media (event_id, stored_filename, content_type, kind) "
                         "VALUES (?, ?, 'image/png', 'photo')", (event_id, name))


class AssessTests(DbTestCase):
    def test_skipped_when_off_and_nothing_is_stored(self):
        event = self.add_report()
        with patch.dict("os.environ", {"CLEF_ENABLED": ""}):
            self.assertEqual(clef_tasks.assess_fire_report(event), "skipped")
        self.assertIsNone(clef.get_assessment("fire_event", event, "fire_report"))

    def test_report_is_assessed_with_photos_and_without_personal_details(self):
        event = self.add_report()
        self.add_photo(event)
        fake = FakeClient()
        self.assertEqual(clef_tasks.assess_fire_report(event, client=fake), "stored")
        call = fake.calls[0]
        self.assertEqual(len(call["images"]), 1)
        self.assertEqual(call["state"]["photos_attached"], 1)
        sent = str(call["state"])
        self.assertNotIn("Jane", sent)
        self.assertNotIn("jane@example.com", sent)
        stored = clef.get_assessment("fire_event", event, "fire_report")
        self.assertEqual(stored["answers"]["shows_fire_or_smoke"]["noul"], 0.93)
        self.assertEqual(stored["images_used"], 1)

    def test_same_input_is_not_asked_again_but_a_new_photo_is(self):
        event = self.add_report()
        fake = FakeClient()
        self.assertEqual(clef_tasks.assess_fire_report(event, client=fake), "stored")
        self.assertEqual(clef_tasks.assess_fire_report(event, client=fake), "unchanged")
        self.add_photo(event)
        self.assertEqual(clef_tasks.assess_fire_report(event, client=fake), "stored")
        self.assertEqual(len(fake.calls), 2)
        with sqlite3.connect(self.db) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM clef_assessments").fetchone()[0], 1)  # replaced, not appended

    def test_a_refused_image_falls_back_to_the_text(self):
        event = self.add_report()
        self.add_photo(event)
        fake = FakeClient(fail_with_images=True)
        self.assertEqual(clef_tasks.assess_fire_report(event, client=fake), "stored")
        self.assertEqual([len(c["images"]) for c in fake.calls], [1, 0])
        self.assertEqual(clef.get_assessment("fire_event", event, "fire_report")["images_used"], 0)

    def test_failures_never_raise_and_store_nothing(self):
        event = self.add_report()
        broken = FakeClient()
        broken.run = lambda *a, **k: (_ for _ in ()).throw(CloudflareAIError("down"))
        self.assertEqual(clef_tasks.assess_fire_report(event, client=broken), "failed")
        self.assertIsNone(clef.get_assessment("fire_event", event, "fire_report"))

    def test_only_public_submissions_are_assessed(self):
        event = self.add_report(source="viirs")
        self.assertEqual(clef_tasks.assess_fire_report(event, client=FakeClient()), "skipped")

    def test_burn_ban_asks_proof_questions_only_when_there_is_proof(self):
        with sqlite3.connect(self.db) as conn:
            conn.execute(
                "INSERT INTO burn_ban_submissions (county_fips, county_name, submitter_name, submitter_contact, effective_at, "
                "submitter_comment) VALUES ('29161', 'Phelps', 'Pat Person', 'pat@example.com', '2026-10-20T00:00:00', 'ban started')")
        reply = {"model": "clef", "answers": {"dates_consistent": {"type": "noul", "noul": 0.9},
                                              "likely_spam": {"type": "noul", "noul": 0.02},
                                              "proof_supports_request": {"type": "noul", "noul": 0.6}}, "usage": {}}
        fake = FakeClient(reply)
        self.assertEqual(clef_tasks.assess_burn_ban(1, client=fake), "stored")
        self.assertNotIn("proof_supports_request", fake.calls[0]["questions"])
        self.assertNotIn("Pat", str(fake.calls[0]["state"]))
        (self.dir / "proofs" / "1-a.png").write_bytes(b"\x89PNG\r\n\x1a\nxx")
        with sqlite3.connect(self.db) as conn:
            conn.execute("UPDATE burn_ban_submissions SET proof_stored_filename = '1-a.png', proof_content_type = 'image/png'")
        self.assertEqual(clef_tasks.assess_burn_ban(1, client=fake), "stored")
        self.assertIn("proof_supports_request", fake.calls[1]["questions"])
        self.assertEqual(len(fake.calls[1]["images"]), 1)
        self.assertEqual(clef.get_assessment("burn_ban", 1, "burn_ban")["answers"]["proof_supports_request"]["noul"], 0.6)


if __name__ == "__main__":
    unittest.main()
