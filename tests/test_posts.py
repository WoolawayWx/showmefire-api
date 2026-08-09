import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from core import database
from core.security import create_access_token
from routers import posts as posts_router


class PostsTests(unittest.TestCase):
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

    def create_post(self, title="Red flag conditions", body="Elevated winds statewide.", author_name="Jane - Boone County FPD", tags=None):
        payload = posts_router.PostCreate(title=title, body=body, author_name=author_name, tags=tags or ["boone-county"])
        result = posts_router.admin_create_post(payload, self.token)
        return result["post"]

    def test_create_post_returns_post_with_tags_and_zero_comments(self):
        post = self.create_post()
        self.assertEqual(post["title"], "Red flag conditions")
        self.assertEqual(post["tags"], ["boone-county"])
        self.assertEqual(post["comment_count"], 0)

    def test_list_posts_orders_newest_first_and_reports_available_tags(self):
        self.create_post(title="First", tags=["boone-county"])
        self.create_post(title="Second", tags=["callaway-county"])

        result = posts_router.admin_list_posts(self.token)
        titles = [p["title"] for p in result["posts"]]

        self.assertEqual(titles, ["Second", "First"])
        self.assertEqual(sorted(result["available_tags"]), ["boone-county", "callaway-county"])

    def test_list_posts_filters_by_tag(self):
        self.create_post(title="First", tags=["boone-county"])
        self.create_post(title="Second", tags=["callaway-county"])

        result = posts_router.admin_list_posts(self.token, tag="callaway-county")

        self.assertEqual([p["title"] for p in result["posts"]], ["Second"])

    def test_public_list_and_detail_do_not_require_admin_token(self):
        post = self.create_post()

        listing = posts_router.public_list_posts()
        self.assertEqual([p["id"] for p in listing["posts"]], [post["id"]])

        posts_router.admin_create_comment(
            post["id"], posts_router.CommentCreate(author_name="Mark", body="A comment"), self.token
        )
        detail = posts_router.public_get_post(post["id"])
        self.assertEqual(detail["post"]["title"], post["title"])
        self.assertNotIn("comments", detail["post"])

    def test_get_post_includes_ordered_comments(self):
        post = self.create_post()
        posts_router.admin_create_comment(
            post["id"], posts_router.CommentCreate(author_name="Mark", body="First comment"), self.token
        )
        posts_router.admin_create_comment(
            post["id"], posts_router.CommentCreate(author_name="Sue", body="Second comment"), self.token
        )

        result = posts_router.admin_get_post(post["id"], self.token)

        self.assertEqual(
            [c["body"] for c in result["post"]["comments"]],
            ["First comment", "Second comment"],
        )

    def test_get_missing_post_raises_404(self):
        with self.assertRaises(HTTPException) as ctx:
            posts_router.admin_get_post(999, self.token)
        self.assertEqual(ctx.exception.status_code, 404)

    def test_update_post_replaces_title_and_tags(self):
        post = self.create_post(tags=["boone-county"])
        payload = posts_router.PostUpdate(title="Updated title", tags=["callaway-county", "incident-42"])

        result = posts_router.admin_update_post(post["id"], payload, self.token)

        self.assertEqual(result["post"]["title"], "Updated title")
        self.assertEqual(result["post"]["body"], post["body"])
        self.assertEqual(sorted(result["post"]["tags"]), ["callaway-county", "incident-42"])

    def test_update_missing_post_raises_404(self):
        payload = posts_router.PostUpdate(title="Doesn't matter")
        with self.assertRaises(HTTPException) as ctx:
            posts_router.admin_update_post(999, payload, self.token)
        self.assertEqual(ctx.exception.status_code, 404)

    def test_delete_post_cascades_tags_and_comments(self):
        post = self.create_post()
        posts_router.admin_create_comment(
            post["id"], posts_router.CommentCreate(author_name="Mark", body="A comment"), self.token
        )

        result = posts_router.admin_delete_post(post["id"], self.token)
        self.assertTrue(result["success"])

        with self.assertRaises(HTTPException) as ctx:
            posts_router.admin_get_post(post["id"], self.token)
        self.assertEqual(ctx.exception.status_code, 404)

    def test_delete_missing_post_raises_404(self):
        with self.assertRaises(HTTPException) as ctx:
            posts_router.admin_delete_post(999, self.token)
        self.assertEqual(ctx.exception.status_code, 404)

    def test_create_comment_on_missing_post_raises_404(self):
        payload = posts_router.CommentCreate(author_name="Mark", body="A comment")
        with self.assertRaises(HTTPException) as ctx:
            posts_router.admin_create_comment(999, payload, self.token)
        self.assertEqual(ctx.exception.status_code, 404)

    def test_delete_comment_removes_it(self):
        post = self.create_post()
        comment = posts_router.admin_create_comment(
            post["id"], posts_router.CommentCreate(author_name="Mark", body="A comment"), self.token
        )["comment"]

        result = posts_router.admin_delete_comment(post["id"], comment["id"], self.token)
        self.assertTrue(result["success"])

        detail = posts_router.admin_get_post(post["id"], self.token)
        self.assertEqual(detail["post"]["comments"], [])

    def test_delete_missing_comment_raises_404(self):
        post = self.create_post()
        with self.assertRaises(HTTPException) as ctx:
            posts_router.admin_delete_comment(post["id"], 999, self.token)
        self.assertEqual(ctx.exception.status_code, 404)

    def test_invalid_token_is_rejected(self):
        with self.assertRaises(HTTPException) as ctx:
            posts_router.admin_list_posts("not-a-real-token")
        self.assertEqual(ctx.exception.status_code, 401)


if __name__ == "__main__":
    unittest.main()
