import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from core import database
from core.security import create_access_token
from routers import forecast_discussions as router


class ForecastDiscussionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary.name) / "showmefire.db"
        self.database_patch = patch.object(database, "get_db_path", return_value=self.db_path)
        self.database_patch.start()
        database.init_database()
        self.token = create_access_token({"sub": "staff@showmefire.org"})

    def tearDown(self):
        self.database_patch.stop()
        self.temporary.cleanup()

    def create(self, title="Tonight", status="draft"):
        return router.admin_create_forecast_discussion(
            router.ForecastDiscussionCreate(
                title=title, body="Dry air and gusty winds.", author_name="Show Me Fire", status=status
            ),
            self.token,
        )["discussion"]

    def test_drafts_are_hidden_from_public(self):
        discussion = self.create()
        with self.assertRaises(HTTPException) as ctx:
            router.public_get_forecast_discussion(discussion["id"])
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(router.public_list_forecast_discussions()["discussions"], [])

    def test_publishing_archives_previous_discussion(self):
        first = self.create("First", "published")
        second = self.create("Second", "published")

        self.assertEqual(router.public_latest_forecast_discussion()["discussion"]["id"], second["id"])
        archived = router.public_get_forecast_discussion(first["id"])["discussion"]
        self.assertEqual(archived["status"], "archived")
        self.assertEqual(len(router.public_list_forecast_discussions()["discussions"]), 2)

    def test_explicit_archive_removes_current(self):
        discussion = self.create("Current", "published")
        archived = router.admin_archive_forecast_discussion(discussion["id"], self.token)["discussion"]
        self.assertEqual(archived["status"], "archived")
        with self.assertRaises(HTTPException):
            router.public_latest_forecast_discussion()

    def test_admin_can_edit_current_and_archived_discussions(self):
        first = self.create("First", "published")
        second = self.create("Second", "published")
        archived = router.public_get_forecast_discussion(first["id"])["discussion"]
        current = router.public_latest_forecast_discussion()["discussion"]
        self.assertEqual(archived["status"], "archived")
        self.assertEqual(current["id"], second["id"])

        updated_archived = router.admin_update_forecast_discussion(
            archived["id"],
            router.ForecastDiscussionUpdate(title="Edited archive", body="Updated archive body."),
            self.token,
        )["discussion"]
        updated_current = router.admin_update_forecast_discussion(
            current["id"],
            router.ForecastDiscussionUpdate(title="Edited current", body="Updated current body."),
            self.token,
        )["discussion"]

        self.assertEqual(updated_archived["status"], "archived")
        self.assertEqual(updated_archived["title"], "Edited archive")
        self.assertEqual(updated_archived["body"], "Updated archive body.")
        self.assertEqual(updated_current["status"], "published")
        self.assertEqual(updated_current["title"], "Edited current")
        self.assertEqual(router.public_latest_forecast_discussion()["discussion"]["title"], "Edited current")

    def test_admin_can_delete_current_and_archived_discussions(self):
        first = self.create("First", "published")
        second = self.create("Second", "published")
        archived_id = first["id"]
        current_id = second["id"]

        self.assertTrue(router.admin_delete_forecast_discussion(archived_id, self.token)["success"])
        with self.assertRaises(HTTPException) as archived_ctx:
            router.public_get_forecast_discussion(archived_id)
        self.assertEqual(archived_ctx.exception.status_code, 404)

        self.assertTrue(router.admin_delete_forecast_discussion(current_id, self.token)["success"])
        with self.assertRaises(HTTPException):
            router.public_latest_forecast_discussion()
        self.assertEqual(router.public_list_forecast_discussions()["discussions"], [])


if __name__ == "__main__":
    unittest.main()
