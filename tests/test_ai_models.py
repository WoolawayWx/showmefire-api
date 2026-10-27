import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai import models
from core import database


class TempSettingsDb(unittest.TestCase):
    """Every test here reads/writes model settings in a throwaway database, with the cache reset."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Path(self.tmp.name) / "smf.db"
        for target in (database, models):
            patcher = patch.object(target, "get_db_path", return_value=self.db)
            patcher.start()
            self.addCleanup(patcher.stop)
        database.init_database()
        models._cache["at"] = 0.0
        self.addCleanup(lambda: models._cache.update(at=0.0, values={}))

    def tearDown(self):
        try:
            self.tmp.cleanup()
        except PermissionError:
            pass


class ModelAwareClient:
    """Fails (or returns junk) for chosen models; records which models it was asked for."""
    configured = True

    def __init__(self, replies):
        self.replies, self.asked = replies, []

    def generate_text(self, prompt, system=None, *, model=None, max_tokens=0):
        self.asked.append(model)
        reply = self.replies[model]
        if isinstance(reply, Exception):
            raise reply
        return reply


class ModelSettingsTests(TempSettingsDb):
    def test_defaults_to_gemma_for_every_task(self):
        with patch.dict("os.environ", {}, clear=True):
            for task in models.TASK_ENV:
                self.assertEqual(models.model_for(task), models.DEFAULT_TEXT_MODEL)

    def test_a_task_setting_beats_the_shared_text_model_and_tasks_are_independent(self):
        env = {"CLOUDFLARE_AI_TEXT_MODEL": "shared", "CLOUDFLARE_AI_FEEDBACK_MODEL": "gpt-oss"}
        with patch.dict("os.environ", env, clear=True):
            self.assertEqual(models.model_for("feedback_analysis"), "gpt-oss")
            self.assertEqual(models.model_for("report_features"), "shared")

    def test_the_older_feature_model_variable_still_works(self):
        with patch.dict("os.environ", {"CLOUDFLARE_AI_FEATURE_MODEL": "old"}, clear=True):
            self.assertEqual(models.model_for("report_features"), "old")
        with patch.dict("os.environ", {"CLOUDFLARE_AI_FEATURE_MODEL": "old",
                                       "CLOUDFLARE_AI_REPORT_FEATURES_MODEL": "new"}, clear=True):
            self.assertEqual(models.model_for("report_features"), "new")


class EscalationTests(TempSettingsDb):
    def setUp(self):
        super().setUp()
        patcher = patch.dict("os.environ", {"CLOUDFLARE_AI_REPORT_FEATURES_MODEL": "small",
                                            "CLOUDFLARE_AI_ESCALATION_MODEL": "big"}, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_good_first_reply_never_touches_the_big_model(self):
        client = ModelAwareClient({"small": '{"a": 1}'})
        self.assertEqual(models.generate_json(client, "report_features", "p"), ({"a": 1}, "small"))
        self.assertEqual(client.asked, ["small"])

    def test_an_unparseable_reply_is_retried_once_on_the_big_model(self):
        client = ModelAwareClient({"small": "I cannot do that", "big": '{"a": 2}'})
        self.assertEqual(models.generate_json(client, "report_features", "p"), ({"a": 2}, "big"))
        self.assertEqual(client.asked, ["small", "big"])

    def test_an_api_error_such_as_a_wrong_model_id_also_escalates(self):
        client = ModelAwareClient({"small": RuntimeError("model not found"), "big": "a summary"})
        self.assertEqual(models.generate_prose(client, "report_features", "p"), ("a summary", "big"))

    def test_both_failing_raises_and_equal_models_do_not_retry_the_same_one(self):
        client = ModelAwareClient({"small": "junk", "big": "junk"})
        with self.assertRaises(ValueError):
            models.generate_json(client, "report_features", "p")
        self.assertEqual(client.asked, ["small", "big"])
        with patch.dict("os.environ", {"CLOUDFLARE_AI_REPORT_FEATURES_MODEL": "same",
                                       "CLOUDFLARE_AI_ESCALATION_MODEL": "same"}, clear=True):
            client = ModelAwareClient({"same": "junk"})
            with self.assertRaises(ValueError):
                models.generate_json(client, "report_features", "p")
            self.assertEqual(client.asked, ["same"])

    def test_a_pinned_model_goes_first(self):
        client = ModelAwareClient({"other": '{"a": 3}'})
        self.assertEqual(models.generate_json(client, "report_features", "p", model="other")[1], "other")


class AdminSettingTests(TempSettingsDb):
    def test_admin_setting_beats_env_beats_default_and_clearing_restores_the_next_level(self):
        env = {"CLOUDFLARE_AI_FEEDBACK_MODEL": "@cf/env/model"}
        with patch.dict("os.environ", env, clear=True):
            self.assertEqual(models.resolve_task("feedback_analysis"), ("@cf/env/model", "env"))
            models.save_overrides({"feedback_analysis": "@cf/admin/model"}, "me@example.com")
            self.assertEqual(models.resolve_task("feedback_analysis"), ("@cf/admin/model", "admin"))
            models.save_overrides({"feedback_analysis": ""}, "me@example.com")
            self.assertEqual(models.resolve_task("feedback_analysis"), ("@cf/env/model", "env"))
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(models.resolve_task("feedback_analysis"), (models.DEFAULT_TEXT_MODEL, "default"))
            self.assertEqual(models.resolve_escalation()[1], "default")

    def test_a_change_applies_immediately_even_with_the_cache(self):
        with patch.dict("os.environ", {}, clear=True):
            models.model_for("report_features")  # warms the cache
            models.save_overrides({"report_features": "@cf/new/one"}, "a")
            self.assertEqual(models.model_for("report_features"), "@cf/new/one")

    def test_bad_ids_and_unknown_keys_are_rejected_and_nothing_is_saved(self):
        for changes in ({"report_features": "gemma"}, {"report_features": "@cf/x/y; DROP"}, {"bogus": "@cf/a/b"}):
            with self.assertRaises(ValueError):
                models.save_overrides(changes, "a")
        self.assertEqual(models.overrides(), {})

    def test_a_broken_database_falls_back_instead_of_raising(self):
        with patch.object(models, "get_db_path", return_value=Path(self.tmp.name) / "missing" / "x.db"):
            models._cache["at"] = 0.0
            with patch.dict("os.environ", {}, clear=True):
                self.assertEqual(models.model_for("report_features"), models.DEFAULT_TEXT_MODEL)

    def test_escalation_uses_the_admin_setting(self):
        client = ModelAwareClient({"@cf/small/m": "junk", "@cf/big/m": '{"a": 1}'})
        with patch.dict("os.environ", {}, clear=True):
            models.save_overrides({"report_features": "@cf/small/m", "escalation": "@cf/big/m"}, "a")
            self.assertEqual(models.generate_json(client, "report_features", "p"), ({"a": 1}, "@cf/big/m"))


class AdminEndpointTests(TempSettingsDb):
    def setUp(self):
        super().setUp()
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from core.security import require_admin
        from routers import ai_admin

        app = FastAPI()
        app.include_router(ai_admin.router)
        app.dependency_overrides[require_admin] = lambda: "admin@example.com"
        self.http = TestClient(app)
        self.ai_admin = ai_admin
        env = patch.dict("os.environ", {}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def test_get_lists_each_task_with_its_source_and_suggestions(self):
        body = self.http.get("/api/admin/ai/models").json()
        self.assertEqual([t["key"] for t in body["tasks"]], ["report_features", "feedback_analysis", "report_summary"])
        self.assertTrue(all(t["source"] == "default" and t["model"] == models.DEFAULT_TEXT_MODEL for t in body["tasks"]))
        self.assertEqual(body["escalation"]["source"], "default")
        self.assertTrue(all(models.MODEL_ID.fullmatch(m["id"]) for m in body["suggested"]))

    def test_put_saves_then_clears_and_rejects_bad_input(self):
        body = self.http.put("/api/admin/ai/models", json={"settings": {"feedback_analysis": "@cf/openai/gpt-oss-20b"}}).json()
        task = next(t for t in body["tasks"] if t["key"] == "feedback_analysis")
        self.assertEqual((task["model"], task["source"], task["override"]), ("@cf/openai/gpt-oss-20b", "admin", "@cf/openai/gpt-oss-20b"))
        body = self.http.put("/api/admin/ai/models", json={"settings": {"feedback_analysis": None}}).json()
        self.assertEqual(next(t for t in body["tasks"] if t["key"] == "feedback_analysis")["source"], "default")
        self.assertEqual(self.http.put("/api/admin/ai/models", json={"settings": {"feedback_analysis": "nope"}}).status_code, 422)

    def test_test_endpoint_reports_success_failure_and_missing_credentials(self):
        class Good:
            configured = True

            def generate_text(self, prompt, system=None, *, model=None, max_tokens=0):
                return '{"ok": true}'

        class Bad(Good):
            def generate_text(self, *args, **kwargs):
                raise RuntimeError("model not found")

        class Off(Good):
            configured = False

        url = "/api/admin/ai/models/test"
        with patch.object(self.ai_admin, "CloudflareAIClient", Good):
            body = self.http.post(url, json={"model": "@cf/a/b"}).json()
            self.assertEqual((body["ok"], body["valid_json"]), (True, True))
        with patch.object(self.ai_admin, "CloudflareAIClient", Bad):
            body = self.http.post(url, json={"model": "@cf/a/b"}).json()
            self.assertEqual((body["ok"], body["error"]), (False, "model not found"))
        with patch.object(self.ai_admin, "CloudflareAIClient", Off):
            self.assertEqual(self.http.post(url, json={"model": "@cf/a/b"}).status_code, 409)
        self.assertEqual(self.http.post(url, json={"model": "bad id"}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
