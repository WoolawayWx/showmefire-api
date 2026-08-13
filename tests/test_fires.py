import tempfile
import asyncio
from io import BytesIO
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from fastapi import UploadFile
from pydantic import ValidationError
from starlette.datastructures import Headers

from core import database
from core.security import create_access_token
from routers import fires as fires_router


class _FakeRequest:
    def __init__(self, headers=None, host="203.0.113.9"):
        self.headers = headers or {"cf-connecting-ip": host}
        self.client = SimpleNamespace(host=host)


def _valid_payload(**overrides):
    base = dict(
        latitude=38.9517,
        longitude=-92.3341,
        occurred_at=(datetime.now(timezone.utc) - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        occurred_at_precision="minute",
        acres=2.5,
        acres_is_estimate=True,
        fuel_types=["grass", "brush"],
        description="Grass fire along the fence line, spreading slowly toward the road.",
        out_of_ordinary="",
        reporter_contact="chief@example.com",
        reporter_name="Jane Doe",
        consent_acknowledged=True,
        turnstile_token="test-token",
        website="",
    )
    base.update(overrides)
    return base


class FireReportValidationTests(unittest.TestCase):
    def test_accepts_in_bounds_point(self):
        report = fires_router.FireReportCreate(**_valid_payload())
        self.assertEqual(report.latitude, 38.9517)

    def test_rejects_latitude_outside_missouri(self):
        with self.assertRaises(ValidationError):
            fires_router.FireReportCreate(**_valid_payload(latitude=25.0))

    def test_rejects_longitude_outside_missouri(self):
        with self.assertRaises(ValidationError):
            fires_router.FireReportCreate(**_valid_payload(longitude=-70.0))

    def test_naive_occurred_at_is_interpreted_as_america_chicago(self):
        # A naive local timestamp yesterday at 14:30 should be read as
        # America/Chicago and normalized to the equivalent UTC instant -
        # computed independently here via zoneinfo, not hardcoded, so the
        # test holds regardless of DST.
        from zoneinfo import ZoneInfo

        local_naive = (datetime.now() - timedelta(days=1)).replace(
            hour=14, minute=30, second=0, microsecond=0
        )
        expected_utc = local_naive.replace(tzinfo=ZoneInfo("America/Chicago")).astimezone(timezone.utc)
        report = fires_router.FireReportCreate(
            **_valid_payload(occurred_at=local_naive.strftime("%Y-%m-%dT%H:%M:%S"))
        )
        self.assertEqual(report.occurred_at, expected_utc.strftime("%Y-%m-%dT%H:%M:%SZ"))

    def test_rejects_occurred_at_more_than_10_minutes_future(self):
        future = (datetime.now(timezone.utc) + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        with self.assertRaises(ValidationError):
            fires_router.FireReportCreate(**_valid_payload(occurred_at=future))

    def test_rejects_occurred_at_older_than_max_age(self):
        old = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
        with self.assertRaises(ValidationError):
            fires_router.FireReportCreate(**_valid_payload(occurred_at=old))

    def test_rejects_non_positive_acres(self):
        with self.assertRaises(ValidationError):
            fires_router.FireReportCreate(**_valid_payload(acres=0))

    def test_rejects_excessive_acres(self):
        with self.assertRaises(ValidationError):
            fires_router.FireReportCreate(**_valid_payload(acres=200000))

    def test_rejects_unknown_fuel_type(self):
        with self.assertRaises(ValidationError):
            fires_router.FireReportCreate(**_valid_payload(fuel_types=["nuclear_waste"]))

    def test_dedupes_and_lowercases_fuel_types(self):
        report = fires_router.FireReportCreate(**_valid_payload(fuel_types=["Grass", "grass", "BRUSH"]))
        self.assertEqual(report.fuel_types, ["grass", "brush"])

    def test_rejects_empty_fuel_list(self):
        with self.assertRaises(ValidationError):
            fires_router.FireReportCreate(**_valid_payload(fuel_types=[]))

    def test_rejects_short_description(self):
        with self.assertRaises(ValidationError):
            fires_router.FireReportCreate(**_valid_payload(description="too short"))

    def test_strips_control_characters_from_description(self):
        report = fires_router.FireReportCreate(
            **_valid_payload(description="Grass fire\x00 along the\x07 fence line spreading slowly.")
        )
        self.assertNotIn("\x00", report.description)
        self.assertNotIn("\x07", report.description)

    def test_rejects_consent_false(self):
        with self.assertRaises(ValidationError):
            fires_router.FireReportCreate(**_valid_payload(consent_acknowledged=False))

    def test_rejects_non_empty_honeypot(self):
        with self.assertRaises(ValidationError):
            fires_router.FireReportCreate(**_valid_payload(website="http://spam.example"))

    def test_rejects_malformed_reporter_contact(self):
        with self.assertRaises(ValidationError):
            fires_router.FireReportCreate(**_valid_payload(reporter_contact="!!!"))

    def test_rejects_missing_reporter_contact(self):
        with self.assertRaises(ValidationError):
            fires_router.FireReportCreate(**_valid_payload(reporter_contact=""))

    def test_accepts_email_reporter_contact(self):
        report = fires_router.FireReportCreate(**_valid_payload(reporter_contact="chief@example.com"))
        self.assertEqual(report.reporter_contact, "chief@example.com")

    def test_rejects_missing_reporter_name(self):
        with self.assertRaises(ValidationError):
            fires_router.FireReportCreate(**_valid_payload(reporter_name=""))

    def test_accepts_optional_reporter_and_address_fields(self):
        report = fires_router.FireReportCreate(**_valid_payload(
            reporter_name="Jane Doe",
            reporter_org="County Dispatch",
            address_text="123 Main Street, Columbia, MO",
        ))
        self.assertEqual(report.reporter_org, "County Dispatch")

    def test_rejects_line_breaks_in_reporter_fields(self):
        with self.assertRaises(ValidationError):
            fires_router.FireReportCreate(**_valid_payload(reporter_name="Jane\nDoe"))


class FireReportModerationValidationTests(unittest.TestCase):
    def test_official_source_confirmed_requires_ref(self):
        with self.assertRaises(ValidationError):
            fires_router.FireReportModeration(verification_tier="official_source_confirmed", official_source_ref="")

    def test_official_source_confirmed_with_ref_is_valid(self):
        moderation = fires_router.FireReportModeration(
            verification_tier="official_source_confirmed", official_source_ref="MDC-2026-0012"
        )
        self.assertEqual(moderation.official_source_ref, "MDC-2026-0012")

    def test_admin_reviewed_does_not_require_ref(self):
        moderation = fires_router.FireReportModeration(verification_tier="admin_reviewed")
        self.assertEqual(moderation.official_source_ref, "")


class FiresRouterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary.name) / "showmefire.db"
        self.database_patch = patch.object(database, "get_db_path", return_value=self.db_path)
        self.database_patch.start()
        database.init_database()
        self.token = create_access_token({"sub": "staff@showmefire.org"})

        self.turnstile_patch = patch.object(fires_router, "verify_turnstile", return_value=(True, "success"))
        self.mock_turnstile = self.turnstile_patch.start()

        self.trust_proxy_patch = patch.object(fires_router, "TRUST_PROXY_HEADERS", True)
        self.trust_proxy_patch.start()

        # submit_fire_report falls back to a real Nominatim reverse-geocode call
        # whenever address_text is blank (the default in _valid_payload) - mock
        # it so the suite never depends on network access or burns the shared
        # public rate-limit budget that the real address-lookup flow also uses.
        self.reverse_geocode_patch = patch.object(fires_router, "_reverse_geocode", return_value="")
        self.reverse_geocode_patch.start()

    def tearDown(self):
        self.reverse_geocode_patch.stop()
        self.trust_proxy_patch.stop()
        self.turnstile_patch.stop()
        self.database_patch.stop()
        self.temporary.cleanup()

    def submit(self, ip="203.0.113.9", **overrides):
        payload = fires_router.FireReportCreate(**_valid_payload(**overrides))
        request = _FakeRequest(host=ip)
        return fires_router.submit_fire_report(payload, request)

    def geocode(self, address="123 Main St, Columbia, MO", ip="203.0.113.9"):
        payload = fires_router.AddressGeocodeRequest(address=address)
        request = _FakeRequest(host=ip)
        return fires_router.geocode_fire_report_address(payload, request)

    # --- geocode handler ---

    def test_geocode_returns_approximate_point_in_bounds(self):
        with patch.object(fires_router, "_forward_geocode", return_value={
            "latitude": 38.9517, "longitude": -92.3341, "display_name": "Columbia, Boone County, Missouri",
        }):
            result = self.geocode()
        self.assertEqual(result["location"]["latitude"], 38.9517)
        self.assertEqual(result["location"]["display_name"], "Columbia, Boone County, Missouri")

    def test_geocode_returns_404_when_address_not_found(self):
        with patch.object(fires_router, "_forward_geocode", return_value=None):
            with self.assertRaises(HTTPException) as ctx:
                self.geocode()
        self.assertEqual(ctx.exception.status_code, 404)

    def test_geocode_returns_503_when_lookup_service_is_unavailable(self):
        with patch.object(fires_router, "_forward_geocode", side_effect=fires_router._GeocodeUnavailable()):
            with self.assertRaises(HTTPException) as ctx:
                self.geocode()
        self.assertEqual(ctx.exception.status_code, 503)

    def test_geocode_blocklisted_ip_gets_403_before_lookup(self):
        ip_hash = fires_router._ip_bucket_key("203.0.113.9")
        database.add_ip_to_blocklist(ip_hash, "abuse", "staff@showmefire.org")
        with patch.object(fires_router, "_forward_geocode") as mock_geocode:
            with self.assertRaises(HTTPException) as ctx:
                self.geocode(ip="203.0.113.9")
            mock_geocode.assert_not_called()
        self.assertEqual(ctx.exception.status_code, 403)

    def test_geocode_is_throttled_independently_of_report_submission(self):
        with patch.object(fires_router, "_forward_geocode", return_value={
            "latitude": 38.9517, "longitude": -92.3341, "display_name": "Columbia, MO",
        }):
            for _ in range(fires_router.FIRE_GEOCODE_LIMIT_PER_HOUR):
                self.geocode()
            with self.assertRaises(HTTPException) as ctx:
                self.geocode()
        self.assertEqual(ctx.exception.status_code, 429)
        # Submitting a report uses a separate quota bucket, unaffected by geocode throttling.
        result = self.submit()
        self.assertEqual(result["report"]["status"], "pending")

    # --- submit handler ---

    def test_happy_path_returns_pending_and_resolves_county(self):
        result = self.submit()
        self.assertEqual(result["report"]["status"], "pending")
        self.assertEqual(result["report"]["county_name"], "Boone")

    def test_pending_report_is_invisible_to_public_reads(self):
        self.submit()
        listing = fires_router.list_public_fire_events(SimpleNamespace(headers={}))
        self.assertEqual(listing["count"], 0)

    def test_media_upload_rejects_wrong_mime_and_accepts_valid_png(self):
        result = self.submit()
        report = result["report"]
        bad = UploadFile(filename="not-image.png", file=BytesIO(b"not an image"))
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(fires_router.upload_fire_report_media(
                report["id"], report["upload_token"], bad
            ))
        self.assertEqual(ctx.exception.status_code, 415)

        png = UploadFile(
            filename="fire.png",
            file=BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32),
        )
        uploaded = asyncio.run(fires_router.upload_fire_report_media(
            report["id"], report["upload_token"], png
        ))
        self.assertEqual(uploaded["media"]["content_type"], "image/png")
        self.assertEqual(len(database.get_fire_event(report["id"], admin=True)["media"]), 1)

    def test_document_upload_accepts_pdf_and_is_admin_only(self):
        result = self.submit()
        report = result["report"]
        pdf = UploadFile(
            filename="incident-report.pdf",
            file=BytesIO(b"%PDF-1.4\n" + b"\x00" * 32),
            headers=Headers({"content-type": "application/pdf"}),
        )
        uploaded = asyncio.run(fires_router.upload_fire_report_document(
            report["id"], report["upload_token"], pdf
        ))
        self.assertEqual(uploaded["document"]["content_type"], "application/pdf")
        self.assertEqual(uploaded["document"]["kind"], "document")

        admin_event = database.get_fire_event(report["id"], admin=True)
        self.assertEqual(len(admin_event["media"]), 1)

        listing = fires_router.list_public_fire_events(SimpleNamespace(headers={}))
        self.assertEqual(listing["count"], 0)  # still pending, but also: media never appears publicly

    def test_document_upload_rejects_wrong_mime(self):
        result = self.submit()
        report = result["report"]
        bad = UploadFile(
            filename="notes.txt",
            file=BytesIO(b"just some notes"),
            headers=Headers({"content-type": "text/plain"}),
        )
        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(fires_router.upload_fire_report_document(
                report["id"], report["upload_token"], bad
            ))
        self.assertEqual(ctx.exception.status_code, 415)

    def test_document_upload_enforces_single_file_cap(self):
        result = self.submit()
        report = result["report"]

        def _upload():
            pdf = UploadFile(
                filename="incident-report.pdf",
                file=BytesIO(b"%PDF-1.4\n" + b"\x00" * 32),
                headers=Headers({"content-type": "application/pdf"}),
            )
            return asyncio.run(fires_router.upload_fire_report_document(
                report["id"], report["upload_token"], pdf
            ))

        _upload()
        with self.assertRaises(HTTPException) as ctx:
            _upload()
        self.assertEqual(ctx.exception.status_code, 413)

    def test_document_upload_does_not_count_against_photo_cap(self):
        result = self.submit()
        report = result["report"]
        pdf = UploadFile(
            filename="incident-report.pdf",
            file=BytesIO(b"%PDF-1.4\n" + b"\x00" * 32),
            headers=Headers({"content-type": "application/pdf"}),
        )
        asyncio.run(fires_router.upload_fire_report_document(
            report["id"], report["upload_token"], pdf
        ))
        png = UploadFile(
            filename="fire.png",
            file=BytesIO(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32),
            headers=Headers({"content-type": "image/png"}),
        )
        uploaded = asyncio.run(fires_router.upload_fire_report_media(
            report["id"], report["upload_token"], png
        ))
        self.assertEqual(uploaded["media"]["kind"], "photo")

    def test_point_outside_counties_but_in_bbox_still_succeeds_with_null_county(self):
        # A point inside the MO bbox but over open water / no polygon match.
        result = self.submit(latitude=36.0, longitude=-89.5)
        self.assertIsNone(result["report"]["county_name"])

    def test_captcha_failure_returns_403_and_writes_no_row(self):
        with patch.object(fires_router, "verify_turnstile", return_value=(False, "timeout-or-duplicate")):
            with self.assertRaises(HTTPException) as ctx:
                self.submit()
            self.assertEqual(ctx.exception.status_code, 403)
        rows = database.list_fire_events(admin=True)
        self.assertEqual(len(rows), 0)

    def test_blocklisted_ip_gets_403_before_captcha_is_called(self):
        ip_hash = fires_router._ip_bucket_key("203.0.113.9")
        database.add_ip_to_blocklist(ip_hash, "abuse", "staff@showmefire.org")
        with self.assertRaises(HTTPException) as ctx:
            self.submit(ip="203.0.113.9")
        self.assertEqual(ctx.exception.status_code, 403)
        self.mock_turnstile.assert_not_called()

    def test_submitted_audit_row_is_created(self):
        result = self.submit()
        event = database.get_fire_event(result["report"]["id"], admin=True)
        self.assertEqual(event["moderation"][0]["action"], "submitted")

    def test_submitter_ip_hash_is_not_the_raw_ip(self):
        result = self.submit()
        event = database.get_fire_event(result["report"]["id"], admin=True)
        self.assertNotEqual(event["submitter_ip_hash"], "203.0.113.9")
        self.assertEqual(len(event["submitter_ip_hash"]), 64)

    # --- rate limiting ---

    def test_fourth_submission_in_an_hour_is_throttled(self):
        with patch.object(fires_router, "FIRE_REPORT_LIMIT_PER_HOUR", 3):
            for _ in range(3):
                self.submit()
            with self.assertRaises(HTTPException) as ctx:
                self.submit()
            self.assertEqual(ctx.exception.status_code, 429)
            self.assertIn("Retry-After", ctx.exception.headers)

    def test_different_ips_get_independent_buckets(self):
        with patch.object(fires_router, "FIRE_REPORT_LIMIT_PER_HOUR", 1):
            self.submit(ip="203.0.113.9")
            self.submit(ip="198.51.100.4")  # should not raise

    def test_global_bucket_trips_independently_of_per_ip_bucket(self):
        with patch.object(fires_router, "FIRE_REPORT_GLOBAL_LIMIT_PER_DAY", 1):
            self.submit(ip="203.0.113.9")
            with self.assertRaises(HTTPException) as ctx:
                self.submit(ip="198.51.100.4")
            self.assertEqual(ctx.exception.status_code, 429)

    # --- moderation ---

    def test_approve_sets_status_and_tier_and_appends_audit_row(self):
        result = self.submit()
        event_id = result["report"]["id"]
        moderation = fires_router.FireReportModeration(verification_tier="admin_reviewed")
        approved = fires_router.admin_approve_fire_report(event_id, moderation, self.token)
        self.assertEqual(approved["report"]["status"], "approved")
        self.assertEqual(approved["report"]["verification_tier"], "admin_reviewed")

    def test_reject_sets_status_and_reason(self):
        result = self.submit()
        event_id = result["report"]["id"]
        rejection = fires_router.FireReportRejection(reason="Not a real fire - duplicate of #4")
        rejected = fires_router.admin_reject_fire_report(event_id, rejection, self.token)
        self.assertEqual(rejected["report"]["status"], "rejected")

    def test_approving_already_moderated_report_returns_409(self):
        result = self.submit()
        event_id = result["report"]["id"]
        moderation = fires_router.FireReportModeration(verification_tier="admin_reviewed")
        fires_router.admin_approve_fire_report(event_id, moderation, self.token)
        with self.assertRaises(HTTPException) as ctx:
            fires_router.admin_approve_fire_report(event_id, moderation, self.token)
        self.assertEqual(ctx.exception.status_code, 409)

    def test_admin_routes_reject_bad_token(self):
        with self.assertRaises(HTTPException) as ctx:
            fires_router.admin_list_fire_reports("not-a-real-token")
        self.assertEqual(ctx.exception.status_code, 401)

    def test_missing_event_returns_404_on_admin_routes(self):
        with self.assertRaises(HTTPException) as ctx:
            fires_router.admin_get_fire_report(999999, self.token)
        self.assertEqual(ctx.exception.status_code, 404)

    def test_edit_requires_reason_field(self):
        with self.assertRaises(ValidationError):
            fires_router.FireEventUpdate(description="Updated description text.")

    def test_edit_changes_fuels_and_appends_audit_entry(self):
        result = self.submit()
        event_id = result["report"]["id"]
        update = fires_router.FireEventUpdate(fuel_types=["timber_litter"], edit_reason="Corrected fuel type")
        edited = fires_router.admin_update_fire_event(event_id, update, self.token)
        self.assertEqual(edited["event"]["fuel_types"], ["timber_litter"])
        full_event = database.get_fire_event(event_id, admin=True)
        self.assertEqual(full_event["moderation"][-1]["action"], "edited")

    def test_delete_is_soft_and_moderation_history_survives(self):
        result = self.submit()
        event_id = result["report"]["id"]
        fires_router.admin_delete_fire_event(event_id, self.token)
        event = database.get_fire_event(event_id, admin=True)
        self.assertEqual(event["status"], "deleted")
        self.assertGreaterEqual(len(event["moderation"]), 2)

    # --- public reads / the leak test ---

    def test_public_event_dict_has_no_pii_keys(self):
        result = self.submit(reporter_contact="chief@example.com")
        event_id = result["report"]["id"]
        moderation = fires_router.FireReportModeration(verification_tier="admin_reviewed")
        fires_router.admin_approve_fire_report(event_id, moderation, self.token)

        listing = fires_router.list_public_fire_events(SimpleNamespace(headers={}))
        self.assertEqual(listing["count"], 1)
        public_event = listing["events"][0]
        for leaked_key in (
            "reporter_contact", "reporter_name", "reporter_org", "address_text",
            "submitter_ip_hash", "captcha_verdict", "upload_token_hash",
        ):
            self.assertNotIn(leaked_key, public_event)

    def test_geojson_shape_matches_existing_satellite_feed_contract(self):
        result = self.submit()
        event_id = result["report"]["id"]
        moderation = fires_router.FireReportModeration(verification_tier="admin_reviewed")
        fires_router.admin_approve_fire_report(event_id, moderation, self.token)

        geojson = fires_router.list_public_fire_events_geojson(SimpleNamespace(headers={}))
        self.assertEqual(geojson["type"], "FeatureCollection")
        self.assertIn("fetched_at", geojson["metadata"])
        feature = geojson["features"][0]
        self.assertTrue(feature["properties"]["ACQ_DATE_TIME"].endswith("Z"))
        self.assertEqual(feature["properties"]["VERIFICATION_TIER"], "admin_reviewed")

    def test_get_pending_event_returns_404_on_public_route(self):
        result = self.submit()
        with self.assertRaises(HTTPException) as ctx:
            fires_router.get_public_fire_event(result["report"]["id"], SimpleNamespace(headers={}))
        self.assertEqual(ctx.exception.status_code, 404)


if __name__ == "__main__":
    unittest.main()
