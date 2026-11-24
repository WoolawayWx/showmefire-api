import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from core import database
from routers import articles
from services import newsletter_web

PDF = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF\n"
HTML = b"<html><body><h1>Spring issue</h1><p>Burn season is here.</p><script>alert(1)</script></body></html>"


class NewsletterUploadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        for module in (database, newsletter_web):
            patcher = patch.object(module, "get_db_path", return_value=Path(self.tmp.name) / "smf.db")
            patcher.start()
            self.addCleanup(patcher.stop)
        database.init_database()
        app = FastAPI()
        app.include_router(articles.router)
        self.http = TestClient(app)

    def tearDown(self):
        try:
            self.tmp.cleanup()
        except PermissionError:
            pass

    def _upload(self, **kw):
        args = dict(title="Spring 2026", issue_date="2026-03-01", summary="Burn season.", filename="spring.pdf", content=PDF, actor="a@x.org")
        args.update(kw)
        return newsletter_web.create_upload(**args)

    def test_pdf_upload_is_public_with_a_pdf_link_and_the_file_is_served(self):
        row = self._upload()
        self.assertEqual((row["kind"], row["uploaded"], row["state"]), ("pdf", True, "public"))
        items = self.http.get("/api/newsletters").json()["newsletters"]
        self.assertEqual([(i["subject"], i["kind"], i["excerpt"]) for i in items], [("Spring 2026", "pdf", "Burn season.")])
        detail = self.http.get(f"/api/newsletters/{row['id']}").json()["newsletter"]
        self.assertEqual(detail["pdfUrl"], f"/api/newsletters/{row['id']}/pdf")
        served = self.http.get(detail["pdfUrl"])
        self.assertEqual((served.status_code, served.headers["content-type"], served.content[:5]), (200, "application/pdf", b"%PDF-"))
        self.assertIn("inline", served.headers["content-disposition"])

    def test_html_upload_is_cleaned(self):
        row = self._upload(filename="issue.html", content=HTML, summary="")
        detail = self.http.get(f"/api/newsletters/{row['id']}").json()["newsletter"]
        self.assertEqual(detail["kind"], "html")
        self.assertIn("Burn season", detail["html"])
        self.assertNotIn("script", detail["html"].lower())
        self.assertIsNone(detail["pdfUrl"])

    def test_rejects_bad_uploads(self):
        for kw in ({"content": b"not a pdf"}, {"filename": "x.exe"}, {"title": " "}, {"issue_date": "2999-01-01"},
                   {"issue_date": "soon"}, {"content": b""}, {"filename": "a.html", "content": b"<script>1</script>"}):
            with self.assertRaises(newsletter_web.UploadError, msg=str(kw)):
                self._upload(**kw)

    def test_hidden_upload_is_gone_from_the_public_and_delete_removes_file(self):
        row = self._upload()
        newsletter_web.set_hidden(row["id"], True, "a@x.org")
        self.assertEqual(self.http.get("/api/newsletters").json()["total"], 0)
        self.assertEqual(self.http.get(f"/api/newsletters/{row['id']}/pdf").status_code, 404)
        path = next((Path(self.tmp.name) / "newsletter_uploads").glob("*.pdf"))
        newsletter_web.delete_upload(row["id"], "a@x.org")
        self.assertFalse(path.exists())
        self.assertEqual(newsletter_web.list_admin(), [])

    def test_uploads_stay_out_of_email_studio_and_emailed_ones_cannot_be_deleted(self):
        self._upload()
        self.assertEqual(database.list_bulletins(), [])
        emailed = database.create_bulletin("Sent one", "<p>x</p>", "x", "show-me-fire-newsletter", [])["id"]
        database.claim_bulletin_send(emailed)
        database.mark_bulletin_sent(emailed, "targeted:1")
        with self.assertRaises(LookupError):
            newsletter_web.delete_upload(emailed, "a@x.org")
        self.assertEqual(len(database.list_bulletins()), 1)

    def test_newest_issue_date_first(self):
        self._upload(title="Old", issue_date="2025-01-01")
        self._upload(title="New", issue_date="2026-02-01")
        self.assertEqual([i["subject"] for i in self.http.get("/api/newsletters").json()["newsletters"]], ["New", "Old"])


if __name__ == "__main__":
    unittest.main()
