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


if __name__ == "__main__":
    unittest.main()
