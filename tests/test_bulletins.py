import tempfile
import unittest
from unittest.mock import patch

from core.database import (
    claim_newsletter_delivery,
    complete_newsletter_delivery,
    get_newsletter_preferences,
    init_database,
    list_matching_newsletter_forecasts,
    upsert_county_forecast_day,
    upsert_newsletter_subscriber,
    replace_newsletter_preferences,
)
from routers import bulletins
from services import newsletter_delivery
from services import graphics_email


class BulletinTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.environment = patch.dict("os.environ", {"DATA_DIR": self.temporary.name})
        self.environment.start()
        init_database()

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    def test_preferences_store_canonical_fips_and_internal_threshold(self):
        upsert_newsletter_subscriber("User@Example.com", "contact-id")
        saved = replace_newsletter_preferences("User@Example.com", [{
            "county_fips": "29001",
            "min_danger_level": 3,
        }])
        self.assertEqual(saved[0]["county_fips"], "29001")
        self.assertEqual(saved[0]["min_danger_level"], 3)
        self.assertEqual(get_newsletter_preferences("user@example.com")[0]["email"], "user@example.com")

    def test_forecast_threshold_matches_and_delivery_is_idempotent(self):
        upsert_newsletter_subscriber("user@example.com", subscription_types=["fire-weather-forecasts"])
        replace_newsletter_preferences("user@example.com", [{
            "county_fips": "29001",
            "min_danger_level": 2,
        }])
        upsert_county_forecast_day("2026-09-21", "29001", 3, "Critical conditions")
        matches = list_matching_newsletter_forecasts("2026-09-21")
        self.assertEqual(len(matches), 1)
        self.assertTrue(claim_newsletter_delivery("user@example.com", "29001", "2026-09-21"))
        self.assertFalse(claim_newsletter_delivery("user@example.com", "29001", "2026-09-21"))

    def test_non_forecast_list_does_not_receive_forecast_delivery(self):
        upsert_newsletter_subscriber("user@example.com", subscription_types=["show-me-fire-newsletter"])
        replace_newsletter_preferences("user@example.com", [{
            "county_fips": "29001",
            "min_danger_level": 0,
        }])
        upsert_county_forecast_day("2026-09-21", "29001", 4)
        self.assertEqual(list_matching_newsletter_forecasts("2026-09-21"), [])

    def test_signup_validates_counties_and_calls_resend(self):
        payload = bulletins.NewsletterSignup(
            email="USER@example.com",
            name="Test User",
            affiliation="Test Agency",
            lists=["fire-weather-forecasts"],
            counties=[{"county_fips": "29001", "level": 4}],
        )
        with patch.object(bulletins, "county_catalog", return_value=[{"fips": "29001", "name": "Adair"}]), \
             patch.object(bulletins, "upsert_audience_contact", return_value="contact-id"), \
             patch.object(bulletins, "set_audience_contact_unsubscribed"), \
             patch.object(bulletins, "set_audience_contact_properties"):
            result = bulletins.signup_newsletter(payload)
        self.assertEqual(result["email"], "user@example.com")
        self.assertEqual(result["counties"][0]["level"], 4)
        self.assertEqual(result["counties"][0]["level_name"], "Critical")

    def test_daily_delivery_uses_provider_and_records_success(self):
        upsert_newsletter_subscriber("user@example.com")
        replace_newsletter_preferences("user@example.com", [{
            "county_fips": "29001",
            "min_danger_level": 0,
        }, {
            "county_fips": "29003",
            "min_danger_level": 0,
        }])
        upsert_county_forecast_day("2026-09-21", "29001", 0)
        upsert_county_forecast_day("2026-09-21", "29003", 4)
        with patch.object(newsletter_delivery, "send_daily_forecast_email", return_value="email-id") as send:
            result = newsletter_delivery.run_daily_forecast_delivery("2026-09-21")
        self.assertEqual(result["matched"], 2)
        self.assertEqual(result["recipients"], 1)
        self.assertEqual(result["sent"], 1)
        self.assertEqual(send.call_args.args[0], "user@example.com")
        self.assertIn("2 counties", send.call_args.args[1])

    def test_failed_delivery_can_be_retried(self):
        upsert_newsletter_subscriber("user@example.com")
        replace_newsletter_preferences("user@example.com", [{
            "county_fips": "29001",
            "min_danger_level": 0,
        }])
        upsert_county_forecast_day("2026-09-21", "29001", 0)
        with patch.object(newsletter_delivery, "send_daily_forecast_email", side_effect=RuntimeError("temporary")):
            self.assertEqual(newsletter_delivery.run_daily_forecast_delivery("2026-09-21")["failed"], 1)
        with patch.object(newsletter_delivery, "send_daily_forecast_email", return_value="retry-id"):
            self.assertEqual(newsletter_delivery.run_daily_forecast_delivery("2026-09-21")["sent"], 1)

    def test_broadcast_uses_segment_and_resend_unsubscribe_placeholder(self):
        class Response:
            def __init__(self, data):
                self.data = data

            def raise_for_status(self):
                return None

            def json(self):
                return self.data

        with patch.dict("os.environ", {
            "RESEND_API_KEY": "key",
            "RESEND_SEGMENT_ID": "segment",
            "BULLETIN_EMAIL_FROM": "fire@example.com",
        }), patch.object(graphics_email.requests, "request", side_effect=[
            Response({"id": "broadcast-id"}),
        ]) as request:
            self.assertEqual(
                graphics_email.send_bulletin_broadcast("Subject", "<p>Body</p>", "Body"),
                "broadcast-id",
            )
        payload = request.call_args.kwargs["json"]
        self.assertEqual(payload["segment_id"], "segment")
        self.assertTrue(payload["send"])
        self.assertIn("contact.manage_url", payload["html"])
        self.assertIn("RESEND_UNSUBSCRIBE_URL", payload["html"])
        self.assertIn("contact.manage_url", payload["text"])
        self.assertIn("RESEND_UNSUBSCRIBE_URL", payload["text"])


if __name__ == "__main__":
    unittest.main()
