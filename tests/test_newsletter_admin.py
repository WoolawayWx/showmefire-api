import tempfile
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from core.database import init_database, replace_newsletter_preferences, upsert_newsletter_subscriber
from routers import newsletter_admin as router
from services import newsletter_admin as service


class NewsletterAdminTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.environment = patch.dict("os.environ", {"DATA_DIR": self.temporary.name})
        self.environment.start()
        init_database()
        upsert_newsletter_subscriber("jane@example.org", name="Jane Doe", affiliation="Central FPD", subscription_types=["fire-weather-forecasts"])
        upsert_newsletter_subscriber("bob@example.org", name="Bob", subscription_types=["fire-weather-forecasts", "show-me-fire-newsletter"])
        replace_newsletter_preferences("bob@example.org", [{"county_fips": "29001", "min_danger_level": 2}])

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    def test_mask_hides_local_part_but_keeps_domain(self):
        self.assertEqual(service.mask_email("jane@example.org"), "j***@example.org")
        self.assertEqual(service.mask_email("nonsense"), "***")

    def test_summary_counts_graphic_only_and_lists(self):
        result = service.summary()
        self.assertEqual(result["active"], 2)
        self.assertEqual(result["graphic_only"], 1)
        self.assertEqual(result["per_list"]["fire-weather-forecasts"], 2)
        self.assertEqual(result["top_counties"][0]["county_fips"], "29001")

    def test_list_is_masked_and_never_leaks_names_or_full_emails(self):
        result = service.list_subscribers("admin@x.org")
        text = str(result)
        self.assertEqual(result["total"], 2)
        self.assertNotIn("jane@example.org", text)
        self.assertNotIn("Jane Doe", text)
        self.assertTrue(any(row["daily_graphic"] for row in result["subscribers"]))

    def test_search_filters_and_is_audited_without_the_query_text(self):
        result = service.list_subscribers("admin@x.org", query="jane")
        self.assertEqual(result["total"], 1)
        entries = service.recent_audit()
        self.assertEqual(entries[0]["action"], "search_list")
        self.assertNotIn("jane", str(entries))

    def test_list_filter_and_status(self):
        self.assertEqual(service.list_subscribers("a", list_id="show-me-fire-newsletter")["total"], 1)
        self.assertEqual(service.list_subscribers("a", status="unsubscribed")["total"], 0)

    def test_reveal_returns_full_record_and_audits(self):
        subscriber_id = service.list_subscribers("a", query="jane")["subscribers"][0]["id"]
        result = service.reveal("admin@x.org", subscriber_id)
        self.assertEqual(result["email"], "jane@example.org")
        self.assertEqual(result["name"], "Jane Doe")
        self.assertEqual(service.recent_audit()[0]["action"], "reveal")
        self.assertIsNone(service.reveal("admin@x.org", 99999))

    def test_unsubscribe_syncs_resend_then_marks_and_audits(self):
        subscriber_id = service.list_subscribers("a", query="jane")["subscribers"][0]["id"]
        with patch.object(router, "set_audience_contact_unsubscribed") as sync:
            self.assertEqual(router.newsletter_unsubscribe(subscriber_id, admin="admin@x.org"), {"unsubscribed": True})
        sync.assert_called_once_with("jane@example.org", True)
        self.assertEqual(service.list_subscribers("a", status="unsubscribed")["total"], 1)
        self.assertEqual(service.recent_audit()[0]["action"], "unsubscribe")

    def test_unsubscribe_leaves_subscriber_active_when_resend_fails(self):
        subscriber_id = service.list_subscribers("a", query="jane")["subscribers"][0]["id"]
        with patch.object(router, "set_audience_contact_unsubscribed", side_effect=RuntimeError("down")):
            with self.assertRaises(HTTPException) as caught:
                router.newsletter_unsubscribe(subscriber_id, admin="admin@x.org")
        self.assertEqual(caught.exception.status_code, 502)
        self.assertEqual(service.list_subscribers("a", status="unsubscribed")["total"], 0)

    def test_routes_require_admin(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        app = FastAPI()
        app.include_router(router.router)
        client = TestClient(app)
        for path in ("/summary", "/subscribers", "/audit"):
            self.assertEqual(client.get(f"/api/admin/newsletter{path}").status_code, 401)
        self.assertEqual(client.post("/api/admin/newsletter/subscribers/1/reveal").status_code, 401)


if __name__ == "__main__":
    unittest.main()
