import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError

from core import database
from core.security import create_access_token
from routers import burn_bans as burn_bans_router


class _FakeRequest:
    def __init__(self, headers=None, host="203.0.113.9"):
        self.headers = headers or {"cf-connecting-ip": host}
        self.client = SimpleNamespace(host=host)


def _valid_payload(**overrides):
    now = datetime.now(timezone.utc)
    base = dict(
        county_fips="29019",
        submitter_name="Jane Doe",
        submitter_contact="chief@example.com",
        proof_url="https://example.gov/burn-ban",
        effective_at=(now + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        expires_at=(now + timedelta(days=8)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        consent_acknowledged=True,
        turnstile_token="test-token",
        website="",
    )
    base.update(overrides)
    return base


class BurnBanValidationTests(unittest.TestCase):
    def test_accepts_valid_payload(self):
        payload = burn_bans_router.BurnBanCreate(**_valid_payload())
        self.assertEqual(payload.county_fips, "29019")

    def test_rejects_unknown_county(self):
        with self.assertRaises(ValidationError):
            burn_bans_router.BurnBanCreate(**_valid_payload(county_fips="99999"))

    def test_rejects_expires_before_effective(self):
        now = datetime.now(timezone.utc)
        with self.assertRaises(ValidationError):
            burn_bans_router.BurnBanCreate(**_valid_payload(
                effective_at=(now + timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%SZ"),
                expires_at=(now + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            ))

    def test_allows_missing_expiration(self):
        payload = burn_bans_router.BurnBanCreate(**_valid_payload(expires_at=""))
        self.assertEqual(payload.expires_at, "")

    def test_accepts_lift_request(self):
        payload = burn_bans_router.BurnBanCreate(**_valid_payload(request_type="lift", expires_at=""))
        self.assertEqual(payload.request_type, "lift")
        self.assertEqual(payload.expires_at, "")

    def test_admin_create_allows_missing_source_and_end_date(self):
        now = datetime.now(timezone.utc)
        payload = burn_bans_router.BurnBanAdminCreate(
            county_fips="29019",
            proof_url="",
            effective_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            expires_at="",
        )
        self.assertEqual(payload.proof_url, "")
        self.assertEqual(payload.expires_at, "")

    def test_rejects_honeypot(self):
        with self.assertRaises(ValidationError):
            burn_bans_router.BurnBanCreate(**_valid_payload(website="spam"))

    def test_rejects_invalid_proof_url(self):
        with self.assertRaises(ValidationError):
            burn_bans_router.BurnBanCreate(**_valid_payload(proof_url="not-a-url"))


class BurnBanWorkflowTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = Path(self._tmpdir.name) / "test.db"
        self._db_patcher = patch.object(database, "get_db_path", return_value=self._db_path)
        self._db_patcher.start()
        database.init_database()
        self.token = create_access_token({"sub": "staff@showmefire.org"})

    def tearDown(self):
        self._db_patcher.stop()
        self._tmpdir.cleanup()

    @patch.object(burn_bans_router, "verify_turnstile", return_value=(True, "success"))
    def test_submission_and_public_redaction(self, _turnstile):
        result = burn_bans_router.submit_burn_ban(
            burn_bans_router.BurnBanCreate(**_valid_payload()),
            _FakeRequest(),
        )
        submission_id = result["submission"]["id"]
        public = database.get_burn_ban_submission(submission_id, admin=False)
        self.assertNotIn("submitter_contact", public)
        admin = database.get_burn_ban_submission(submission_id, admin=True)
        self.assertEqual(admin["submitter_contact"], "chief@example.com")

    @patch.object(burn_bans_router, "verify_turnstile", return_value=(True, "success"))
    @patch.object(burn_bans_router, "_maybe_regenerate_map")
    def test_confirm_makes_ban_active(self, _map, _turnstile):
        created = burn_bans_router.submit_burn_ban(
            burn_bans_router.BurnBanCreate(**_valid_payload(
                effective_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )),
            _FakeRequest(),
        )
        submission_id = created["submission"]["id"]
        confirmed = burn_bans_router.admin_confirm_burn_ban(
            submission_id,
            burn_bans_router.BurnBanModeration(),
            self.token,
        )
        self.assertEqual(confirmed["submission"]["status"], "confirmed")
        active = database.list_active_burn_bans()
        self.assertEqual(len(active), 1)

    @patch.object(burn_bans_router, "_maybe_regenerate_map")
    def test_admin_create_publishes_ban(self, _map):
        now = datetime.now(timezone.utc)
        result = burn_bans_router.admin_create_burn_ban(
            burn_bans_router.BurnBanAdminCreate(
                county_fips="29019",
                proof_url="https://example.gov/burn-ban",
                effective_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                expires_at=(now + timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            ),
            self.token,
        )
        self.assertEqual(result["submission"]["status"], "confirmed")
        self.assertEqual(len(database.list_active_burn_bans()), 1)

    @patch.object(burn_bans_router, "_maybe_regenerate_map")
    def test_admin_create_without_source_or_end_date(self, _map):
        now = datetime.now(timezone.utc)
        result = burn_bans_router.admin_create_burn_ban(
            burn_bans_router.BurnBanAdminCreate(
                county_fips="29019",
                proof_url="",
                effective_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                expires_at="",
            ),
            self.token,
        )
        self.assertEqual(result["submission"]["status"], "confirmed")
        self.assertEqual(result["submission"]["proof_url"], "")
        self.assertEqual(result["submission"]["expires_at"], "")
        self.assertEqual(len(database.list_active_burn_bans()), 1)

    @patch.object(burn_bans_router, "verify_turnstile", return_value=(True, "success"))
    @patch.object(burn_bans_router, "_maybe_regenerate_map")
    def test_confirming_lift_expires_active_ban(self, _map, _turnstile):
        now = datetime.now(timezone.utc)
        created = burn_bans_router.admin_create_burn_ban(
            burn_bans_router.BurnBanAdminCreate(
                county_fips="29019",
                proof_url="",
                effective_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                expires_at="",
            ),
            self.token,
        )
        self.assertEqual(len(database.list_active_burn_bans()), 1)
        lift = burn_bans_router.submit_burn_ban(
            burn_bans_router.BurnBanCreate(**_valid_payload(
                request_type="lift",
                effective_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                expires_at="",
            )),
            _FakeRequest(),
        )
        confirmed = burn_bans_router.admin_confirm_burn_ban(
            lift["submission"]["id"],
            burn_bans_router.BurnBanModeration(),
            self.token,
        )
        self.assertEqual(confirmed["submission"]["status"], "confirmed")
        self.assertEqual(confirmed["submission"]["request_type"], "lift")
        self.assertEqual(database.list_active_burn_bans(), [])
        original = database.get_burn_ban_submission(created["submission"]["id"], admin=True)
        self.assertEqual(original["status"], "expired")

    def test_admin_regenerate_map(self):
        now = datetime.now(timezone.utc)
        database.create_burn_ban_submission(
            county_fips="29019",
            county_name="Boone",
            submitter_name="Jane",
            submitter_contact="jane@example.com",
            proof_url="https://example.gov/ban",
            effective_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            expires_at=(now + timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            submitter_ip_hash="hash",
            upload_token_hash="token",
            captcha_verdict="success",
            consent_version="test",
        )
        with patch("services.burn_ban_map.generate_burn_ban_map") as mock_generate:
            mock_generate.return_value = {
                "active_counties": 1,
                "image_path": "mo-burnban.png",
                "updated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }
            result = burn_bans_router.admin_regenerate_burn_ban_map(self.token)
        self.assertTrue(result["success"])
        self.assertEqual(result["active_count"], 1)
        mock_generate.assert_called_once()

    def test_expire_stale_burn_bans(self):
        now = datetime.now(timezone.utc)
        submission = database.create_burn_ban_submission(
            county_fips="29019",
            county_name="Boone",
            submitter_name="Jane",
            submitter_contact="jane@example.com",
            proof_url="https://example.gov/ban",
            effective_at=(now - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            expires_at=(now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            submitter_ip_hash="hash",
            upload_token_hash="token",
            captcha_verdict="success",
            consent_version="test",
        )
        database.moderate_burn_ban_submission(
            submission["id"], to_status="confirmed", actor="admin@test",
        )
        expired_count = database.expire_stale_burn_bans(now=now)
        self.assertEqual(expired_count, 1)
        self.assertEqual(database.list_active_burn_bans(now=now), [])

    def test_active_geojson_contains_county_geometry_and_no_private_fields(self):
        now = datetime.now(timezone.utc)
        created = database.create_burn_ban_submission(
            county_fips="29019", county_name="Boone", submitter_name="Private Name",
            submitter_contact="private@example.com", proof_url="https://example.gov/ban",
            effective_at=now.strftime("%Y-%m-%dT%H:%M:%SZ"), expires_at="",
            submitter_ip_hash="hash", upload_token_hash="token", captcha_verdict="success",
            consent_version="test",
        )
        database.moderate_burn_ban_submission(created["id"], to_status="confirmed", actor="test")
        response = SimpleNamespace(headers={})
        result = burn_bans_router.list_public_active_burn_bans_geojson(response)
        self.assertEqual(result["features"][0]["geometry"]["type"], "Polygon")
        properties = result["features"][0]["properties"]
        self.assertEqual(properties["county_fips"], "29019")
        self.assertNotIn("submitter_name", properties)
        self.assertNotIn("submitter_contact", properties)


class BurnBanNotesBulkHistoryTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = Path(self._tmpdir.name) / "test.db"
        self._db_patcher = patch.object(database, "get_db_path", return_value=self._db_path)
        self._db_patcher.start()
        self._map_patcher = patch.object(burn_bans_router, "_maybe_regenerate_map")
        self._map_patcher.start()
        database.init_database()
        self.token = create_access_token({"sub": "staff@showmefire.org"})
        self.now = datetime.now(timezone.utc)

    def tearDown(self):
        self._map_patcher.stop()
        self._db_patcher.stop()
        self._tmpdir.cleanup()

    def _iso(self, delta=timedelta()):
        return (self.now + delta).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _bulk(self, *items):
        return burn_bans_router.admin_bulk_burn_bans(
            burn_bans_router.BurnBanBulkRequest(items=list(items)), self.token,
        )

    @patch.object(burn_bans_router, "verify_turnstile", return_value=(True, "success"))
    def test_public_comment_is_admin_only(self, _turnstile):
        result = burn_bans_router.submit_burn_ban(
            burn_bans_router.BurnBanCreate(**_valid_payload(comment="Fire chief announced on radio.")),
            _FakeRequest(),
        )
        submission_id = result["submission"]["id"]
        admin = database.get_burn_ban_submission(submission_id, admin=True)
        self.assertEqual(admin["submitter_comment"], "Fire chief announced on radio.")
        public = database.get_burn_ban_submission(submission_id, admin=False)
        self.assertNotIn("submitter_comment", public)

    def _submit_public(self, **overrides):
        with patch.object(burn_bans_router, "verify_turnstile", return_value=(True, "success")):
            return burn_bans_router.submit_burn_ban(
                burn_bans_router.BurnBanCreate(**_valid_payload(**overrides)), _FakeRequest(),
            )["submission"]["id"]

    def test_admin_can_set_edit_and_clear_both_notes(self):
        created = burn_bans_router.admin_create_burn_ban(
            burn_bans_router.BurnBanAdminCreate(county_fips="29019", effective_at=self._iso()),
            self.token,
        )
        submission_id = created["submission"]["id"]
        set_notes = lambda **kw: burn_bans_router.admin_set_burn_ban_notes(
            submission_id, burn_bans_router.BurnBanNotes(**kw), self.token,
        )["submission"]
        result = set_notes(public_note="Includes fireworks.", internal_note="Called clerk.")
        self.assertEqual(result["public_note"], "Includes fireworks.")
        self.assertEqual(result["moderator_note"], "Called clerk.")
        result = set_notes(public_note="Includes fireworks and campfires.")
        self.assertEqual(result["moderator_note"], "Called clerk.")
        result = set_notes(public_note="", internal_note="")
        self.assertEqual(result["public_note"], "")
        self.assertEqual(result["moderator_note"], "")
        logged = [e for e in result["moderation_history"] if e["action"] == "notes"]
        self.assertEqual(len(logged), 3)

    def test_public_note_is_public_once_confirmed_and_internal_note_is_not(self):
        submission_id = self._submit_public(public_note="Ag burns still allowed with permit.",
                                            effective_at=self._iso())
        burn_bans_router.admin_set_burn_ban_notes(
            submission_id, burn_bans_router.BurnBanNotes(internal_note="Verified by phone"), self.token,
        )
        # Confirming without a moderator note must not wipe the internal note.
        confirmed = burn_bans_router.admin_confirm_burn_ban(
            submission_id, burn_bans_router.BurnBanModeration(), self.token,
        )["submission"]
        self.assertEqual(confirmed["moderator_note"], "Verified by phone")
        payload = burn_bans_router._public_ban_payload(database.list_active_burn_bans()[0])
        self.assertEqual(payload["public_note"], "Ag burns still allowed with permit.")
        self.assertNotIn("moderator_note", payload)
        history = burn_bans_router.list_public_burn_ban_history(SimpleNamespace(headers={}), county_fips="29019")
        self.assertEqual(history["events"][0]["public_note"], "Ag burns still allowed with permit.")
        self.assertNotIn("note", history["events"][0])

    def test_public_update_request_changes_current_ban(self):
        ban = burn_bans_router.admin_create_burn_ban(
            burn_bans_router.BurnBanAdminCreate(county_fips="29019", effective_at=self._iso(timedelta(days=-3))),
            self.token,
        )["submission"]
        new_expiry = self._iso(timedelta(days=10))
        update_id = self._submit_public(
            request_type="update", effective_at=self._iso(), expires_at=new_expiry,
            proof_url="https://example.gov/extended", public_note="Extended through the holiday.",
        )
        self.assertEqual(len(database.list_active_burn_bans()), 1)  # pending update isn't a ban
        burn_bans_router.admin_confirm_burn_ban(update_id, burn_bans_router.BurnBanModeration(), self.token)
        updated = database.get_burn_ban_submission(ban["id"], admin=True)
        self.assertEqual(updated["expires_at"], new_expiry)
        self.assertEqual(updated["proof_url"], "https://example.gov/extended")
        self.assertEqual(len(database.list_active_burn_bans()), 1)
        events = database.list_burn_ban_county_events(county_fips="29019")
        self.assertEqual([e["event_type"] for e in events], ["updated", "issued"])
        self.assertEqual(events[0]["public_note"], "Extended through the holiday.")

    def test_update_request_without_current_ban_is_rejected(self):
        update_id = self._submit_public(request_type="update", effective_at=self._iso())
        with self.assertRaises(burn_bans_router.HTTPException) as ctx:
            burn_bans_router.admin_confirm_burn_ban(update_id, burn_bans_router.BurnBanModeration(), self.token)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(database.get_burn_ban_submission(update_id)["status"], "pending")

    def test_public_submission_pings_staff(self):
        with patch.object(burn_bans_router, "notify_staff_alert") as notify:
            submission_id = self._submit_public(comment="Heard on the radio", public_note="Countywide")
        kwargs = notify.call_args.kwargs
        self.assertEqual(kwargs["alert_type"], "burn_ban")
        self.assertEqual(kwargs["admin_path"], f"/admin/burn-bans/{submission_id}")
        values = {f["name"]: f["value"] for f in kwargs["fields"]}
        self.assertEqual(values["Public note"], "Countywide")
        self.assertNotIn("chief@example.com", str(kwargs))

    def test_bulk_add_update_remove(self):
        added = self._bulk(
            {"action": "add", "county_fips": "29019", "effective_at": self._iso(), "note": "Order 12"},
            {"action": "add", "county_fips": "29051", "effective_at": self._iso()},
        )
        self.assertEqual(added["applied"], 2)
        self.assertEqual(len(database.list_active_burn_bans()), 2)

        new_expiry = self._iso(timedelta(days=3))
        self._bulk(
            {"action": "update", "county_fips": "29019", "expires_at": new_expiry, "note": "Extended",
             "public_note": "Extended three days"},
            {"action": "remove", "county_fips": "29051", "note": "Rain"},
        )
        active = database.list_active_burn_bans()
        self.assertEqual([b["county_fips"] for b in active], ["29019"])
        self.assertEqual(active[0]["expires_at"], new_expiry)
        boone = database.list_burn_ban_county_events(county_fips="29019", admin=True)
        self.assertEqual([e["event_type"] for e in boone], ["updated", "issued"])
        self.assertEqual(boone[0]["public_note"], "Extended three days")
        self.assertEqual(boone[0]["note"], "Extended")

    def test_bulk_rejects_whole_batch_on_any_invalid_row(self):
        with self.assertRaises(burn_bans_router.HTTPException) as ctx:
            self._bulk(
                {"action": "add", "county_fips": "29019", "effective_at": self._iso()},
                {"action": "remove", "county_fips": "29051"},
            )
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail["errors"][0]["index"], 1)
        self.assertEqual(database.list_active_burn_bans(), [])

    def test_bulk_rejects_duplicate_and_existing_counties(self):
        self._bulk({"action": "add", "county_fips": "29019", "effective_at": self._iso()})
        with self.assertRaises(burn_bans_router.HTTPException) as ctx:
            self._bulk(
                {"action": "add", "county_fips": "29019"},
                {"action": "update", "county_fips": "29019"},
            )
        self.assertEqual(len(ctx.exception.detail["errors"]), 2)

    def test_county_history_records_lifecycle(self):
        self._bulk({"action": "add", "county_fips": "29019", "effective_at": self._iso(timedelta(days=-5)), "note": "internal"})
        self._bulk({"action": "remove", "county_fips": "29019", "effective_at": self._iso(timedelta(days=-2))})
        database.moderate_burn_ban_submission(
            burn_bans_router._publish_admin_ban(
                actor="x", county_fips="29051",
                effective_at=self._iso(timedelta(days=-3)), expires_at=self._iso(timedelta(hours=-1)),
            )["id"],
            to_status="confirmed", actor="x",
        )
        database.expire_stale_burn_bans(now=self.now)

        response = SimpleNamespace(headers={})
        boone = burn_bans_router.list_public_burn_ban_history(response, county_fips="29019")
        self.assertEqual([e["event_type"] for e in boone["events"]], ["lifted", "issued"])
        self.assertNotIn("note", boone["events"][0])
        self.assertEqual(boone["summary"]["29019"]["bans_issued"], 1)

        cole = burn_bans_router.list_public_burn_ban_history(response, county_fips="29051")
        self.assertEqual([e["event_type"] for e in cole["events"]], ["expired", "issued"])

        admin = burn_bans_router.admin_burn_ban_history(self.token, county_fips="29019")
        self.assertEqual(admin["events"][-1]["note"], "internal")

    def test_deleting_submission_removes_it_from_history(self):
        result = burn_bans_router.admin_create_burn_ban(
            burn_bans_router.BurnBanAdminCreate(county_fips="29019", effective_at=self._iso()),
            self.token,
        )
        burn_bans_router.admin_delete_burn_ban(result["submission"]["id"], self.token, reason="error")
        self.assertEqual(database.list_burn_ban_county_events(county_fips="29019"), [])

    def test_backfill_seeds_history_for_existing_bans(self):
        created = burn_bans_router.admin_create_burn_ban(
            burn_bans_router.BurnBanAdminCreate(county_fips="29019", effective_at=self._iso()),
            self.token,
        )
        import sqlite3
        conn = sqlite3.connect(self._db_path)
        conn.execute("DROP TABLE burn_ban_county_events")
        conn.commit()
        conn.close()
        database.init_database()
        events = database.list_burn_ban_county_events(county_fips="29019")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["submission_id"], created["submission"]["id"])


class StaffAlertTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = Path(self._tmpdir.name) / "test.db"
        self._db_patcher = patch.object(database, "get_db_path", return_value=self._db_path)
        self._db_patcher.start()
        database.init_database()
        from services import discord_notifier
        self.notifier = discord_notifier

    def tearDown(self):
        self._db_patcher.stop()
        self._tmpdir.cleanup()

    def _build(self, **overrides):
        kwargs = dict(alert_type="fire_report", title="New fire report", admin_path="/admin/fires/1")
        kwargs.update(overrides)
        return self.notifier.build_staff_alert_payload(**kwargs)

    def test_skipped_without_staff_channel_even_if_default_channel_set(self):
        database.update_discord_admin_settings(channel_id="111", channel_name="public-updates")
        self.assertIsNone(self._build())

    def test_routes_to_staff_channel_with_role_mentions(self):
        database.update_discord_admin_settings(
            channel_id="111", staff_channel_id="222", staff_role_ids="333,444",
        )
        payload = self._build(fields=[{"name": "County", "value": "Boone"}, {"name": "Empty", "value": ""}])
        self.assertEqual(payload["event_type"], "staff_alert")
        self.assertEqual(payload["target_channel_id"], "222")
        self.assertEqual(payload["mention_role_ids"], ["333", "444"])
        self.assertEqual([f["name"] for f in payload["fields"]], ["County"])
        self.assertTrue(payload["url"].endswith("/admin/fires/1"))

    def test_disabled_alert_type_is_skipped_unless_forced(self):
        database.update_discord_admin_settings(staff_channel_id="222", staff_alert_types="burn_ban")
        self.assertIsNone(self._build())
        self.assertIsNotNone(self._build(alert_type="burn_ban"))
        self.assertIsNotNone(self._build(force=True))

    def test_defaults_enable_every_alert_type(self):
        settings = database.get_discord_admin_settings()
        self.assertEqual(set(settings["staff_alert_types"].split(",")), set(self.notifier.STAFF_ALERT_TYPES))

    def test_notify_never_raises(self):
        with patch.object(self.notifier, "build_staff_alert_payload", side_effect=RuntimeError("boom")):
            self.assertFalse(self.notifier.notify_staff_alert(alert_type="burn_ban", title="x"))


class DiscordFireAlertTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = Path(self._tmpdir.name) / "test.db"
        self._db_patcher = patch.object(database, "get_db_path", return_value=self._db_path)
        self._db_patcher.start()
        database.init_database()
        from services import discord_notifier
        self.notifier = discord_notifier

    def tearDown(self):
        self._db_patcher.stop()
        self._tmpdir.cleanup()

    @staticmethod
    def _alert(alert_id, event="Red Flag Warning"):
        return {"id": alert_id, "event": event, "headline": f"{event} {alert_id}", "areaDescription": "Boone"}

    def test_first_run_seeds_then_only_new_alerts_post(self):
        with patch.object(self.notifier, "_send_event", return_value=True) as send:
            self.assertEqual(self.notifier.process_fire_weather_alerts_for_discord([self._alert("a")]), 0)
            send.assert_not_called()
            posted = self.notifier.process_fire_weather_alerts_for_discord(
                [self._alert("a"), self._alert("b", "Fire Weather Watch")]
            )
            self.assertEqual(posted, 1)
            self.assertEqual(send.call_args.args[0]["alert_id"], "b")
            self.assertEqual(self.notifier.process_fire_weather_alerts_for_discord([self._alert("b")]), 0)

    def test_failed_delivery_retries_next_poll(self):
        self.notifier.process_fire_weather_alerts_for_discord([])  # baseline
        with patch.object(self.notifier, "_send_event", return_value=False):
            self.assertEqual(self.notifier.process_fire_weather_alerts_for_discord([self._alert("c")]), 0)
        with patch.object(self.notifier, "_send_event", return_value=True):
            self.assertEqual(self.notifier.process_fire_weather_alerts_for_discord([self._alert("c")]), 1)

    def test_routes_to_fire_alert_channel_with_default_fallback(self):
        database.update_discord_admin_settings(channel_id="111")
        with patch.object(self.notifier, "_send_event", return_value=True) as send:
            self.notifier.notify_fire_weather_alert(self._alert("d"))
            self.assertEqual(send.call_args.args[0]["target_channel_id"], "111")
            database.update_discord_admin_settings(fire_alert_channel_id="555", fire_alert_role_ids="777")
            self.notifier.notify_fire_weather_alert(self._alert("e"))
        payload = send.call_args.args[0]
        self.assertEqual(payload["event_type"], "fire_alert")
        self.assertEqual(payload["target_channel_id"], "555")
        self.assertEqual(payload["mention_role_ids"], ["777"])
        self.assertIn("mo-firewx-alerts.png", payload["image_url"])


if __name__ == "__main__":
    unittest.main()
