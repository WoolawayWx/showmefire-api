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


if __name__ == "__main__":
    unittest.main()
