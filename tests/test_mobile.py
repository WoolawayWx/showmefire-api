import json
import sqlite3
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from core import database
from routers.mobile import PushSubscriptionRequest
from services import mobile_content, mobile_push


class MobileContentTests(unittest.TestCase):
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

    def test_subscription_upsert_disable_and_recipient_filtering(self):
        self.register()
        self.assertEqual(len(mobile_push._eligible_subscriptions("forecast")), 1)
        self.assertEqual(len(mobile_push._eligible_subscriptions("fire_weather", ["29019"])), 1)
        self.assertEqual(len(mobile_push._eligible_subscriptions("fire_weather", ["29001"])), 0)
        mobile_push.disable_subscription("11111111-1111-4111-8111-111111111111")
        self.assertEqual(mobile_push._eligible_subscriptions("forecast"), [])

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

    def test_fire_alerts_seed_without_sending_then_deliver_new_products(self):
        self.register()
        existing = {"id": "old", "event": "Red Flag Warning", "headline": "Old", "countyFips": ["29019"]}
        new = {"id": "new", "event": "Fire Weather Watch", "headline": "New", "countyFips": ["29019"]}
        with patch.object(mobile_push, "_send_batch", return_value=[{"status": "ok", "id": "ticket-new"}]) as sender:
            self.assertEqual(mobile_push.process_fire_weather_alerts([existing]), 0)
            self.assertEqual(mobile_push.process_fire_weather_alerts([existing, new]), 1)
        sender.assert_called_once()

    def test_device_not_registered_receipt_disables_token(self):
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
            enabled = connection.execute(
                "SELECT enabled FROM mobile_push_subscriptions WHERE installation_id = ?",
                ("11111111-1111-4111-8111-111111111111",),
            ).fetchone()[0]
            receipt = connection.execute("SELECT error FROM mobile_push_receipts WHERE ticket_id = 'dead-ticket'").fetchone()[0]
        self.assertEqual(enabled, 0)
        self.assertEqual(receipt, "DeviceNotRegistered")


if __name__ == "__main__":
    unittest.main()
