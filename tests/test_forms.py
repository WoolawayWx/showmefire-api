"""The form builder: question validation, form storage, the public endpoints and the admin endpoints."""
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core.security import require_admin
from routers import forms as public_module
from routers import forms_admin as admin_module
from services.forms import schema, store

EVERY_TYPE = [
    {"id": "name", "type": "text", "label": "Name", "required": True},
    {"id": "bio", "type": "textarea", "label": "About you", "max_length": 50},
    {"id": "mail", "type": "email", "label": "Email"},
    {"id": "phone", "type": "tel", "label": "Phone"},
    {"id": "site", "type": "url", "label": "Website"},
    {"id": "acres", "type": "number", "label": "Acres", "min": 0, "max": 1000},
    {"id": "kind", "type": "select", "label": "Kind", "options": [{"value": "a", "label": "Alpha"}, {"value": "b", "label": "Beta"}]},
    {"id": "size", "type": "radio", "label": "Size", "options": [{"value": "s", "label": "Small"}, {"value": "l", "label": "Large"}]},
    {"id": "tools", "type": "multiselect", "label": "Tools", "required": True, "min_choices": 1, "max_choices": 2,
     "options": [{"value": "x", "label": "Ex"}, {"value": "y", "label": "Why"}, {"value": "z", "label": "Zed"}]},
    {"id": "agree", "type": "checkbox", "label": "I agree", "required": True},
    {"id": "when", "type": "date", "label": "Date", "min_date": "2026-01-01", "max_date": "2026-12-31"},
    {"id": "at", "type": "time", "label": "Time"},
    {"id": "stars", "type": "rating", "label": "Stars", "rating_max": 5},
    {"id": "nps", "type": "scale", "label": "Likely?", "min": 0, "max": 10, "min_label": "No", "max_label": "Yes"},
    {"id": "intro", "type": "heading", "label": "Heading"},
    {"id": "note", "type": "paragraph", "label": "Some words."},
]
GOOD = {"name": "  Pat  Chief ", "bio": "Hi", "mail": "Pat@Example.GOV", "phone": "573-555-0100", "site": "https://example.gov", "acres": "12.5",
        "kind": "a", "size": "l", "tools": ["x", "z"], "agree": True, "when": "2026-06-01", "at": "7:05", "stars": 4, "nps": "9"}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    store.reset_for_tests()
    alerts = []
    monkeypatch.setattr(public_module, "notify_staff_alert", lambda **kwargs: alerts.append(kwargs) or True)
    monkeypatch.setattr(public_module, "verify_turnstile", lambda token, ip: (token == "good", "test"))
    public_module.form_limiter.reset()
    public_module.submit_limiter.reset()
    app = FastAPI()
    app.include_router(public_module.router)
    app.include_router(admin_module.router)
    app.dependency_overrides[require_admin] = lambda: "admin-user"
    return SimpleNamespace(client=TestClient(app), alerts=alerts)


def make(env, status="published", **extra):
    body = {"title": "Burn Survey", "fields": EVERY_TYPE}
    created = env.client.post("/admin/form-builder", json=body)
    assert created.status_code == 201, created.text
    form = created.json()["form"]
    updated = env.client.put(f"/admin/form-builder/{form['id']}", json={"status": status, **extra})
    assert updated.status_code == 200, updated.text
    return updated.json()["form"]


def send(env, slug, answers=None, token="good", **extra):
    return env.client.post(f"/v1/forms/{slug}/responses", json={"answers": answers or GOOD, "turnstile_token": token, **extra})


# --- the question model ------------------------------------------------------------------------------

def test_every_question_type_is_accepted_and_tidied():
    fields = schema.clean_fields(EVERY_TYPE)
    assert [f["type"] for f in fields] == [f["type"] for f in EVERY_TYPE]
    assert {t for t in schema.TYPES} == {f["type"] for f in fields}


def test_a_bad_definition_is_refused():
    base = [{"id": "q1", "type": "text", "label": "One"}]
    bad = [
        [],
        [{"id": "Q1", "type": "text", "label": "One"}],
        base + base,
        [{"id": "q1", "type": "blob", "label": "One"}],
        [{"id": "q1", "type": "text", "label": " "}],
        [{"id": "q1", "type": "select", "label": "Pick", "options": []}],
        [{"id": "q1", "type": "select", "label": "Pick", "options": [{"value": "a", "label": "A"}, {"value": "a", "label": "B"}]}],
        [{"id": "q1", "type": "rating", "label": "R", "rating_max": 20}],
        [{"id": "q1", "type": "scale", "label": "S", "min": 5, "max": 2}],
        [{"id": "q1", "type": "number", "label": "N", "min": 9, "max": 1}],
        [{"id": "q1", "type": "date", "label": "D", "min_date": "soon"}],
        [{"id": "q1", "type": "heading", "label": "Only a heading"}],
        base + [{"id": "q2", "type": "text", "label": "Two", "show_if": {"field": "nope", "equals": "x"}}],
        [{"id": "h", "type": "heading", "label": "H"}, {"id": "q2", "type": "text", "label": "Two", "show_if": {"field": "h", "equals": "x"}}] + base,
        [{"id": "q1", "type": "multiselect", "label": "M", "min_choices": 3, "options": [{"value": "a", "label": "A"}]}],
    ]
    for fields in bad:
        with pytest.raises(schema.FormError):
            schema.clean_fields(fields)


def test_answers_are_cleaned_and_snapshotted():
    fields = schema.clean_fields(EVERY_TYPE)
    answers, errors = schema.validate_answers(GOOD, fields)
    assert errors == {}
    by_id = {a["id"]: a for a in answers}
    assert by_id["name"]["value"] == "Pat Chief"
    assert by_id["mail"]["value"] == "pat@example.gov"
    assert by_id["acres"]["value"] == 12.5
    assert by_id["stars"]["value"] == 4 and by_id["nps"]["value"] == 9
    assert by_id["kind"]["display"] == "Alpha"
    assert by_id["tools"]["value"] == ["x", "z"] and by_id["tools"]["display"] == ["Ex", "Zed"]
    assert by_id["at"]["value"] == "07:05"
    assert "intro" not in by_id and "note" not in by_id                 # display-only questions are never answers
    assert by_id["name"]["label"] == "Name"


@pytest.mark.parametrize("field,value", [
    ("name", ""), ("bio", "x" * 51), ("mail", "nope"), ("phone", "12"), ("site", "ftp://x"), ("acres", "lots"), ("acres", "5000"),
    ("kind", "zzz"), ("size", "zzz"), ("tools", []), ("tools", ["x", "y", "z"]), ("tools", ["x", "x"]), ("tools", ["nope"]), ("agree", False),
    ("when", "2027-02-01"), ("when", "tomorrow"), ("at", "25:00"), ("stars", 6), ("stars", 0), ("stars", "2.5"), ("nps", 11), ("nps", "x"),
    ("acres", "nan"), ("acres", "inf"),
])
def test_bad_answers_are_reported_by_field(field, value):
    fields = schema.clean_fields(EVERY_TYPE)
    answers, errors = schema.validate_answers({**GOOD, field: value}, fields)
    assert field in errors, errors


def test_optional_questions_may_be_blank_and_hidden_ones_are_not_checked():
    fields = schema.clean_fields([
        {"id": "kind", "type": "radio", "label": "Kind", "options": [{"value": "fire", "label": "Fire"}, {"value": "other", "label": "Other"}]},
        {"id": "why", "type": "text", "label": "Why", "required": True, "show_if": {"field": "kind", "equals": "other"}},
        {"id": "deeper", "type": "text", "label": "Deeper", "required": True, "show_if": {"field": "why", "equals": "x"}},
        {"id": "opt", "type": "text", "label": "Optional"},
    ])
    answers, errors = schema.validate_answers({"kind": "fire", "why": "ignored", "deeper": "ignored"}, fields)
    assert errors == {} and [a["id"] for a in answers] == ["kind"]
    _, errors = schema.validate_answers({"kind": "other"}, fields)
    assert set(errors) == {"why"}


def test_a_question_can_depend_on_a_multiselect():
    fields = schema.clean_fields([
        {"id": "tools", "type": "multiselect", "label": "Tools", "options": [{"value": "a", "label": "A"}, {"value": "b", "label": "B"}]},
        {"id": "more", "type": "text", "label": "More", "required": True, "show_if": {"field": "tools", "equals": "b"}},
    ])
    assert schema.validate_answers({"tools": ["a"]}, fields)[1] == {}
    assert set(schema.validate_answers({"tools": ["a", "b"]}, fields)[1]) == {"more"}


def test_non_object_and_oversized_bodies_are_refused():
    fields = schema.clean_fields(EVERY_TYPE)
    assert "_form" in schema.validate_answers(["x"], fields)[1]
    assert "_form" in schema.validate_answers({"name": "x" * 70_000}, fields)[1]


# --- admin -------------------------------------------------------------------------------------------

def test_create_gives_a_unique_address_and_a_starter_question(env):
    first = env.client.post("/admin/form-builder", json={"title": "Burn Survey!"}).json()["form"]
    second = env.client.post("/admin/form-builder", json={"title": "Burn Survey!"}).json()["form"]
    assert first["slug"] == "burn-survey" and second["slug"] == "burn-survey-2"
    assert first["status"] == "draft" and len(first["fields"]) == 1
    assert env.client.post("/admin/form-builder", json={"title": "Burn ban"}).json()["form"]["slug"] == "burn-ban-2"
    reserved = env.client.post("/admin/form-builder", json={"title": "x", "slug": "burn-ban"})
    assert reserved.status_code == 422


def test_update_bumps_the_version_only_when_questions_change(env):
    form = make(env, status="draft")
    url = f"/admin/form-builder/{form['id']}"
    assert env.client.put(url, json={"title": "Renamed"}).json()["form"]["version"] == 1
    assert env.client.put(url, json={"fields": EVERY_TYPE}).json()["form"]["version"] == 1       # same questions
    changed = [dict(EVERY_TYPE[0], label="Full name"), *EVERY_TYPE[1:]]
    assert env.client.put(url, json={"fields": changed}).json()["form"]["version"] == 2


@pytest.mark.parametrize("change", [
    {"status": "live"}, {"slug": "Bad Slug"}, {"title": " "}, {"max_responses": 0}, {"closes_at": "someday"}, {"fields": []},
    {"fields": [{"id": "a", "type": "text", "label": ""}]},
])
def test_invalid_updates_are_422_and_change_nothing(env, change):
    form = make(env, status="draft")
    url = f"/admin/form-builder/{form['id']}"
    assert env.client.put(url, json=change).status_code == 422
    assert env.client.get(url).json()["form"]["title"] == form["title"]


def test_slug_change_must_be_free(env):
    one, two = make(env, status="draft"), make(env, status="draft")
    taken = env.client.put(f"/admin/form-builder/{two['id']}", json={"slug": one["slug"]})
    assert taken.status_code == 422
    assert env.client.put(f"/admin/form-builder/{two['id']}", json={"slug": "my-form"}).json()["form"]["slug"] == "my-form"


def test_limits_can_be_set_and_cleared(env):
    form = make(env, max_responses=3, closes_at="2099-01-01T00:00:00Z")
    assert form["max_responses"] == 3 and form["closes_at"] == "2099-01-01T00:00:00Z"
    cleared = env.client.put(f"/admin/form-builder/{form['id']}", json={"max_responses": None, "closes_at": None}).json()["form"]
    assert cleared["max_responses"] is None and cleared["closes_at"] is None


def test_duplicate_makes_a_draft_copy_without_responses(env):
    form = make(env)
    send(env, form["slug"])
    copy_ = env.client.post(f"/admin/form-builder/{form['id']}/duplicate").json()["form"]
    assert copy_["status"] == "draft" and copy_["title"] == "Burn Survey (copy)" and copy_["slug"] != form["slug"]
    assert copy_["fields"] == form["fields"] and copy_["response_count"] == 0


def test_delete_removes_the_form_and_its_responses(env):
    form = make(env)
    send(env, form["slug"])
    assert env.client.delete(f"/admin/form-builder/{form['id']}").status_code == 200
    assert env.client.get(f"/admin/form-builder/{form['id']}").status_code == 404
    assert env.client.get(f"/v1/forms/{form['slug']}").status_code == 404
    assert env.client.delete(f"/admin/form-builder/{form['id']}").status_code == 404


def test_list_shows_counts(env):
    form = make(env)
    send(env, form["slug"])
    send(env, form["slug"])
    row = env.client.get("/admin/form-builder").json()["forms"][0]
    assert row["response_count"] == 2 and row["last_response_at"] and "fields" not in row


def test_admin_routes_need_an_admin(env):
    env.client.app.dependency_overrides.clear()
    for method, path in (("get", "/admin/form-builder"), ("post", "/admin/form-builder"), ("get", "/admin/form-builder/1"),
                         ("put", "/admin/form-builder/1"), ("delete", "/admin/form-builder/1"), ("get", "/admin/form-builder/1/responses"),
                         ("get", "/admin/form-builder/1/responses.csv")):
        assert getattr(env.client, method)(path).status_code == 401, path


# --- public ------------------------------------------------------------------------------------------

def test_public_form_hides_drafts_and_staff_settings(env):
    form = make(env, status="draft", notify_staff=True)
    assert env.client.get(f"/v1/forms/{form['slug']}").status_code == 404
    assert send(env, form["slug"]).status_code == 404
    env.client.put(f"/admin/form-builder/{form['id']}", json={"status": "published"})
    shown = env.client.get(f"/v1/forms/{form['slug']}").json()["form"]
    assert shown["title"] == "Burn Survey" and len(shown["fields"]) == len(EVERY_TYPE)
    assert not ({"id", "notify_staff", "max_responses", "created_by", "response_count"} & set(shown))
    assert env.client.get("/v1/forms/nope").status_code == 404


def test_closed_forms_say_why(env):
    form = make(env, status="closed")
    seen = env.client.get(f"/v1/forms/{form['slug']}")
    assert seen.status_code == 410 and seen.json()["detail"]["title"] == "Burn Survey"
    assert send(env, form["slug"]).status_code == 410
    past = make(env, closes_at="2020-01-01T00:00:00Z")
    assert env.client.get(f"/v1/forms/{past['slug']}").status_code == 410


def test_the_response_limit_closes_the_form(env):
    form = make(env, max_responses=2)
    assert send(env, form["slug"]).status_code == 201
    assert send(env, form["slug"]).status_code == 201
    assert send(env, form["slug"]).status_code == 410
    assert env.client.get(f"/v1/forms/{form['slug']}").status_code == 410


def test_a_response_is_stored_with_its_version_and_snapshot(env):
    form = make(env)
    sent = send(env, form["slug"])
    assert sent.status_code == 201 and sent.json()["reference"].startswith("FR-")
    stored = env.client.get(f"/admin/form-builder/{form['id']}/responses").json()
    assert stored["total"] == 1
    row = stored["responses"][0]
    assert row["form_version"] == 1 and row["reference"] == sent.json()["reference"]
    assert next(a for a in row["answers"] if a["id"] == "name")["value"] == "Pat Chief"
    assert "ip_hash" not in row


def test_old_responses_keep_the_label_they_were_sent_with(env):
    form = make(env)
    send(env, form["slug"])
    changed = [dict(EVERY_TYPE[0], label="Full name"), *EVERY_TYPE[1:]]
    env.client.put(f"/admin/form-builder/{form['id']}", json={"fields": changed})
    row = env.client.get(f"/admin/form-builder/{form['id']}/responses").json()["responses"][0]
    assert next(a for a in row["answers"] if a["id"] == "name")["label"] == "Name"


def test_invalid_answers_get_a_422_with_field_errors_and_store_nothing(env):
    form = make(env)
    bad = send(env, form["slug"], {**GOOD, "mail": "nope", "tools": []})
    assert bad.status_code == 422
    assert set(bad.json()["detail"]["errors"]) == {"mail", "tools"}
    assert env.client.get(f"/admin/form-builder/{form['id']}/responses").json()["total"] == 0


def test_the_human_check_is_enforced_unless_the_form_turns_it_off(env):
    form = make(env)
    assert send(env, form["slug"], token="bad").status_code == 400
    off = make(env, require_captcha=False)
    assert send(env, off["slug"], token="").status_code == 201


def test_honeypot_looks_like_success_but_stores_nothing(env):
    form = make(env)
    assert send(env, form["slug"], website="http://spam").status_code == 201
    assert env.client.get(f"/admin/form-builder/{form['id']}/responses").json()["total"] == 0


def test_daily_cap_per_address(env, monkeypatch):
    monkeypatch.setattr(public_module, "DAILY_RESPONSES_PER_IP", 2)
    form = make(env)
    assert [send(env, form["slug"]).status_code for _ in range(3)] == [201, 201, 429]


def test_staff_alert_only_when_asked_and_a_failing_alert_does_not_lose_the_response(env, monkeypatch):
    quiet = make(env)
    send(env, quiet["slug"])
    assert env.alerts == []
    loud = make(env, notify_staff=True)
    assert send(env, loud["slug"]).status_code == 201
    assert env.alerts and env.alerts[0]["admin_path"] == f"/admin/forms/{loud['id']}"

    def boom(**kwargs):
        raise RuntimeError("discord down")
    monkeypatch.setattr(public_module, "notify_staff_alert", boom)
    assert send(env, loud["slug"]).status_code == 201
    assert env.client.get(f"/admin/form-builder/{loud['id']}/responses").json()["total"] == 2


def test_delete_one_response(env):
    form = make(env)
    send(env, form["slug"])
    row = env.client.get(f"/admin/form-builder/{form['id']}/responses").json()["responses"][0]
    assert env.client.delete(f"/admin/form-builder/{form['id']}/responses/{row['id']}").status_code == 200
    assert env.client.delete(f"/admin/form-builder/{form['id']}/responses/{row['id']}").status_code == 404
    assert env.client.get(f"/admin/form-builder/{form['id']}/responses").json()["total"] == 0


def test_csv_export_has_every_question_and_is_safe_in_a_spreadsheet(env):
    form = make(env)
    send(env, form["slug"], {**GOOD, "name": "=cmd|' /C calc'!A0"})
    csv_text = env.client.get(f"/admin/form-builder/{form['id']}/responses.csv")
    assert csv_text.status_code == 200 and csv_text.headers["content-type"].startswith("text/csv")
    assert "burn-survey-responses.csv" in csv_text.headers["content-disposition"]
    lines = csv_text.content.decode("utf-8-sig").splitlines()
    assert lines[0].startswith("Reference,Submitted (UTC),Form version,Name,About you")
    assert "Heading" not in lines[0] and "Some words." not in lines[0]
    assert "'=cmd" in lines[1] and ",=cmd" not in lines[1]
    assert "Ex; Zed" in lines[1] and ",Yes," in lines[1]


def test_csv_keeps_a_column_for_a_question_that_was_removed(env):
    form = make(env)
    send(env, form["slug"])
    env.client.put(f"/admin/form-builder/{form['id']}", json={"fields": [EVERY_TYPE[0]]})
    header = env.client.get(f"/admin/form-builder/{form['id']}/responses.csv").content.decode("utf-8-sig").splitlines()[0]
    assert "Kind" in header and "Stars" in header


def test_zero_is_a_real_answer_not_a_blank():
    fields = schema.clean_fields([{"id": "nps", "type": "scale", "label": "Likely?", "min": 0, "max": 10, "required": True},
                                  {"id": "n", "type": "number", "label": "Count", "required": True}])
    answers, errors = schema.validate_answers({"nps": 0, "n": 0}, fields)
    assert errors == {} and [a["value"] for a in answers] == [0, 0]
