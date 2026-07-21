import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

from pydantic import ValidationError

from core import database
from routers import mobile as mobile_router
from routers.mobile import PushSubscriptionRequest
from services import mobile_content, mobile_push


class MobileContentTests(unittest.TestCase):
    def test_forecast_map_urls_are_versioned_with_the_forecast_revision(self):
        forecast = {
            "id": 42,
            "title": "Daily Forecast",
            "discussion": "Elevated fire danger.",
            "valid_time": "2026-07-21T12:00:00Z",
            "updated_at": "2026-07-21T11:45:00+00:00",
        }
        with patch.object(mobile_content, "get_latest_forecast", return_value=forecast):
            content = mobile_content.build_mobile_content("https://api.example", "https://cdn.example/latest")

        self.assertEqual(content["forecast"]["updatedAt"], forecast["updated_at"])
        self.assertTrue(all("?v=2026-07-21T11%3A45%3A00%2B00%3A00" in item["url"] for item in content["forecast"]["maps"]))

    def test_county_mapping_uses_same_codes_and_fire_zones(self):
        fips = mobile_content.county_fips_for_alert({
            "geocode": {"SAME": ["029019"], "UGC": ["MOZ053"]},
            "affectedZones": [],
        })
        self.assertIn("29019", fips)
        self.assertIn("29013", fips)  # Bates County is fire zone 053.

    def test_active_feed_filters_products_and_expired_alerts(self):
        now = datetime(2026, 7, 20, 12, tzinfo=timezone.utc)
        document = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {
                    "id": "active-red-flag", "event": "Red Flag Warning",
                    "headline": "Red Flag Warning", "ends": "2026-07-20T18:00:00Z",
                    "geocode": {"SAME": ["029019"]},
                }},
                {"type": "Feature", "properties": {
                    "id": "expired-watch", "event": "Fire Weather Watch",
                    "ends": "2026-07-19T18:00:00Z", "geocode": {},
                }},
                {"type": "Feature", "properties": {
                    "id": "heat", "event": "Heat Advisory",
                    "ends": "2026-07-20T18:00:00Z", "geocode": {},
                }},
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "active.json"
            path.write_text(json.dumps(document))
            alerts = mobile_content.active_fire_weather_alerts(path, now)
        self.assertEqual([alert["id"] for alert in alerts], ["active-red-flag"])
        self.assertEqual(alerts[0]["countyFips"], ["29019"])

    def test_subscription_model_rejects_bad_tokens_and_counties(self):
        with self.assertRaises(ValidationError):
            PushSubscriptionRequest.model_validate({
                "expoPushToken": "not-a-token",
                "platform": "ios",
                "preferences": {},
            })
        with self.assertRaises(ValidationError):
            PushSubscriptionRequest.model_validate({
                "expoPushToken": "ExpoPushToken[abcdefghijklmnopqrstuv]",
                "platform": "ios",
                "preferences": {"countyFips": ["99999"]},
            })


class MobilePushTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary.name) / "showmefire.db"
        self.database_patch = patch.object(database, "get_db_path", return_value=self.db_path)
        self.push_patch = patch.object(mobile_push, "get_db_path", return_value=self.db_path)
        self.database_patch.start()
        self.push_patch.start()
        database.init_database()

    def tearDown(self):
        self.push_patch.stop()
        self.database_patch.stop()
        self.temporary.cleanup()

    def register(self, installation_id="11111111-1111-4111-8111-111111111111"):
        return mobile_push.upsert_subscription(
            installation_id=installation_id,
            expo_push_token=f"ExpoPushToken[{installation_id.replace('-', '')}]",
            platform="ios",
            app_version="1.0.0",
            forecast=True,
            sitrep=True,
            fire_weather=True,
            county_fips=["29019"],
        )

    def test_subscription_upsert_delete_and_recipient_filtering(self):
        self.register()
        self.assertEqual(len(mobile_push._eligible_subscriptions("forecast")), 1)
        self.assertEqual(len(mobile_push._eligible_subscriptions("fire_weather", ["29019"])), 1)
        self.assertEqual(len(mobile_push._eligible_subscriptions("fire_weather", ["29001"])), 0)
        mobile_push.delete_subscription("11111111-1111-4111-8111-111111111111")
        mobile_push.delete_subscription("11111111-1111-4111-8111-111111111111")
        self.assertEqual(mobile_push._eligible_subscriptions("forecast"), [])

    def test_subscription_with_no_enabled_categories_is_deleted(self):
        self.register()
        result = mobile_push.upsert_subscription(
            installation_id="11111111-1111-4111-8111-111111111111",
            expo_push_token="ExpoPushToken[11111111111141118111111111111111]",
            platform="ios",
            app_version="1.0.0",
            forecast=False,
            sitrep=False,
            fire_weather=False,
            county_fips=["29019"],
        )

        self.assertFalse(result["registered"])
        self.assertEqual(mobile_push._eligible_subscriptions("forecast"), [])

    def test_delete_endpoint_is_idempotent(self):
        self.register()
        installation_id = "11111111-1111-4111-8111-111111111111"

        first = mobile_router.delete_push_subscription(UUID(installation_id))
        second = mobile_router.delete_push_subscription(UUID(installation_id))

        self.assertEqual((first.status_code, second.status_code), (204, 204))

    def test_delete_subscription_removes_delivery_data_but_preserves_events(self):
        self.register()
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "INSERT INTO mobile_push_events (event_key, event_type) VALUES (?, ?)",
                ("forecast:privacy-test", "forecast"),
            )
            connection.execute(
                "INSERT INTO mobile_push_tickets (ticket_id, installation_id, event_key) VALUES (?, ?, ?)",
                ("privacy-ticket", "11111111-1111-4111-8111-111111111111", "forecast:privacy-test"),
            )
            connection.execute(
                "INSERT INTO mobile_push_receipts (ticket_id, status) VALUES (?, ?)",
                ("privacy-ticket", "ok"),
            )

        mobile_push.delete_subscription("11111111-1111-4111-8111-111111111111")

        with sqlite3.connect(self.db_path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM mobile_push_subscriptions").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM mobile_push_tickets").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM mobile_push_receipts").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM mobile_push_events").fetchone()[0], 1)

    def test_delivery_record_cleanup_keeps_recent_records(self):
        self.register()
        with sqlite3.connect(self.db_path) as connection:
            connection.executemany(
                "INSERT INTO mobile_push_tickets (ticket_id, installation_id, event_key, created_at) VALUES (?, ?, ?, ?)",
                [
                    ("old-ticket", "11111111-1111-4111-8111-111111111111", "forecast:old", "2020-01-01 00:00:00"),
                    ("recent-ticket", "11111111-1111-4111-8111-111111111111", "forecast:recent", "2099-01-01 00:00:00"),
                ],
            )
            connection.executemany(
                "INSERT INTO mobile_push_receipts (ticket_id, status) VALUES (?, ?)",
                [("old-ticket", "ok"), ("recent-ticket", "ok")],
            )

        deleted = mobile_push.purge_delivery_records()

        self.assertEqual(deleted, {"receipts": 1, "tickets": 1})
        with sqlite3.connect(self.db_path) as connection:
            self.assertEqual(
                connection.execute("SELECT ticket_id FROM mobile_push_tickets").fetchall(),
                [("recent-ticket",)],
            )
            self.assertEqual(
                connection.execute("SELECT ticket_id FROM mobile_push_receipts").fetchall(),
                [("recent-ticket",)],
            )

    def test_event_delivery_is_deduplicated_and_stores_ticket(self):
        self.register()
        with patch.object(mobile_push, "_send_batch", return_value=[{"status": "ok", "id": "ticket-1"}]) as sender:
            first = mobile_push.send_mobile_event(
                event_type="forecast", event_key="forecast:1", title="Forecast", body="Ready", url="/forecasts"
            )
            second = mobile_push.send_mobile_event(
                event_type="forecast", event_key="forecast:1", title="Forecast", body="Ready", url="/forecasts"
            )
        self.assertEqual((first, second), (1, 0))
        sender.assert_called_once()
        with sqlite3.connect(self.db_path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM mobile_push_tickets").fetchone()[0], 1)

    def test_immediate_device_not_registered_error_deletes_subscription(self):
        self.register()
        with patch.object(
            mobile_push,
            "_send_batch",
            return_value=[{"status": "error", "details": {"error": "DeviceNotRegistered"}}],
        ):
            sent = mobile_push.send_mobile_event(
                event_type="forecast",
                event_key="forecast:unregistered",
                title="Forecast",
                body="Ready",
                url="/forecasts",
            )

        self.assertEqual(sent, 0)
        with sqlite3.connect(self.db_path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM mobile_push_subscriptions").fetchone()[0], 0)

    def test_forecast_notification_includes_revision_and_rich_image(self):
        self.register()
        with patch.object(mobile_push, "_send_batch", return_value=[{"status": "ok", "id": "ticket-image"}]) as sender:
            sent = mobile_push.notify_forecast(
                {"id": 42, "title": "Updated forecast", "updated_at": "2026-07-21T11:45:00Z"},
                "https://api.example/images/mo-forecastfiredanger.png?v=42",
            )

        self.assertEqual(sent, 1)
        message = sender.call_args.args[0][0]
        self.assertEqual(message["data"]["forecastRevision"], "2026-07-21T11:45:00Z")
        self.assertEqual(message["richContent"]["image"], "https://api.example/images/mo-forecastfiredanger.png?v=42")
        self.assertTrue(message["mutableContent"])

    def test_fire_alerts_seed_without_sending_then_deliver_new_products(self):
        self.register()
        existing = {"id": "old", "event": "Red Flag Warning", "headline": "Old", "countyFips": ["29019"]}
        new = {"id": "new", "event": "Fire Weather Watch", "headline": "New", "countyFips": ["29019"]}
        with patch.object(mobile_push, "_send_batch", return_value=[{"status": "ok", "id": "ticket-new"}]) as sender:
            self.assertEqual(mobile_push.process_fire_weather_alerts([existing]), 0)
            self.assertEqual(mobile_push.process_fire_weather_alerts([existing, new]), 1)
        sender.assert_called_once()

    def test_device_not_registered_receipt_deletes_subscription_and_delivery_data(self):
        self.register()
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                "INSERT INTO mobile_push_tickets (ticket_id, installation_id, event_key, created_at) VALUES (?, ?, ?, ?)",
                ("dead-ticket", "11111111-1111-4111-8111-111111111111", "forecast:old", "2020-01-01 00:00:00"),
            )

        class FakeResponse:
            def raise_for_status(self):
                return None
            def json(self):
                return {"data": {"dead-ticket": {"status": "error", "details": {"error": "DeviceNotRegistered"}}}}

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return None
            def post(self, *args, **kwargs):
                return FakeResponse()

        with patch.object(mobile_push.httpx, "Client", FakeClient):
            self.assertEqual(mobile_push.check_push_receipts(), 1)
        with sqlite3.connect(self.db_path) as connection:
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM mobile_push_subscriptions").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM mobile_push_tickets").fetchone()[0], 0)
            self.assertEqual(connection.execute("SELECT COUNT(*) FROM mobile_push_receipts").fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()
