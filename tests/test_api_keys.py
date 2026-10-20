"""API key applications: the editable form, the public request endpoint, the admin approval flow, the emails and the keys themselves."""
import copy
import json
import sqlite3
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core import rate_limit
from core.rate_limit import RateLimiter
from core.security import require_admin
from routers import api_keys as public_module
from routers import api_keys_admin as admin_module
from services.api_keys import form as key_form
from services.api_keys import mail, store

GOOD = {"name": "Pat Chief", "email": "Pat@Example.GOV", "phone": "573-555-0100", "role": "Fire chief", "agency": "Central Fire District",
        "agency_type": "fire", "website": "https://example.gov", "reason": "Residents ask about burning.", "plan": "Embed it on our site.",
        "interest": "burn_check", "integration": "embed", "volume": "under_100", "agree_terms": True}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    store.reset_for_tests()
    rate_limit.clear_database_key_cache()
    sent, alerts = [], []
    monkeypatch.setattr(mail, "send", lambda to, subject, text, html: sent.append(SimpleNamespace(to=to, subject=subject, text=text, html=html)) or "id")
    monkeypatch.setattr(mail, "configured", lambda: True)
    monkeypatch.setattr(public_module, "notify_staff_alert", lambda **kwargs: alerts.append(kwargs) or True)
    monkeypatch.setattr(public_module, "verify_turnstile", lambda token, ip: (token == "good", "test"))
    monkeypatch.setattr(public_module.threading, "Thread", lambda target, args=(), daemon=None: SimpleNamespace(start=lambda: target(*args)))
    public_module.form_limiter.reset()           # the route holds these objects from import time, so empty them rather than replace them
    public_module.submit_limiter.reset()
    app = FastAPI()
    app.include_router(public_module.router)
    app.include_router(admin_module.router)
    app.dependency_overrides[require_admin] = lambda: "admin-user"
    return SimpleNamespace(client=TestClient(app), app=app, sent=sent, alerts=alerts)


def submit(env, answers=None, token="good", **extra):
    return env.client.post("/v1/api-keys/requests", json={"answers": answers or GOOD, "turnstile_token": token, **extra})


# --- the form ----------------------------------------------------------------------------------------------

def test_the_shipped_form_is_valid_and_keeps_the_fields_the_emails_need():
    form = key_form.load_form()
    ids = [f["id"] for f in form["fields"]]
    assert {"name", "email", "agency", "interest", "reason", "plan", "agree_terms"} <= set(ids)
    assert not any(key.startswith("_") for key in key_form.public_form(form))


def test_a_bad_form_definition_is_refused():
    base = key_form.load_form()
    for change in (lambda f: f["fields"].pop(0), lambda f: f["fields"].append(dict(f["fields"][0])),
                   lambda f: f["fields"][0].update(type="blob"), lambda f: f["fields"][1].update(type="text"),
                   lambda f: f["fields"].append({"id": "pick", "label": "Pick", "type": "select"}),
                   lambda f: f["fields"].append({"id": "later", "label": "L", "type": "text", "show_if": {"field": "nope", "equals": "x"}})):
        broken = copy.deepcopy(base)
        change(broken)
        with pytest.raises(key_form.FormError):
            key_form.check_definition(broken)


def test_validation_cleans_and_reports_errors():
    answers, errors = key_form.validate_answers(GOOD)
    assert not errors
    by_id = {a["id"]: a for a in answers}
    assert by_id["email"]["value"] == "pat@example.gov" and by_id["agency_type"]["display"] == "Fire department or fire district"
    assert by_id["agree_terms"]["value"] is True and by_id["name"]["label"] == "Your full name"
    assert key_form.answer_value(answers, "agency_type") == "Fire department or fire district"

    _answers, errors = key_form.validate_answers({"name": " ", "email": "nope", "agency": "A", "agency_type": "wizard", "reason": "r", "plan": "p",
                                                  "interest": "forecasts", "integration": "embed", "website": "ftp://x", "phone": "abc", "agree_terms": False})
    assert set(errors) == {"name", "email", "agency_type", "website", "phone", "agree_terms"}
    assert key_form.validate_answers("not a dict")[1] == {"_form": "The request was not understood."}
    assert "reason" in key_form.validate_answers({**GOOD, "reason": "x" * 2001})[1]
    assert "_form" in key_form.validate_answers({**GOOD, "reason": "x" * 25_000})[1]
    answers, _ = key_form.validate_answers({**GOOD, "surprise": "ignored"})
    assert "surprise" not in {a["id"] for a in answers}


def test_adding_a_question_needs_only_a_change_to_the_form():
    form = copy.deepcopy(key_form.load_form())
    form["fields"] += [{"id": "fleet", "label": "Vehicles", "type": "number", "required": True, "min": 1, "max": 500},
                       {"id": "other_detail", "label": "Tell us more", "type": "text", "required": True, "show_if": {"field": "integration", "equals": "other"}}]
    assert "fleet" in key_form.validate_answers(GOOD, form)[1]
    answers, errors = key_form.validate_answers({**GOOD, "fleet": "12"}, form)
    assert not errors and next(a for a in answers if a["id"] == "fleet")["value"] == 12
    assert "fleet" in key_form.validate_answers({**GOOD, "fleet": "900"}, form)[1]
    assert "other_detail" in key_form.validate_answers({**GOOD, "fleet": "1", "integration": "other"}, form)[1]      # shown, so required


# --- the public request -------------------------------------------------------------------------------------

def test_form_route_serves_the_definition(env):
    body = env.client.get("/v1/api-keys/form").json()
    assert body["form"]["title"] and body["form"]["fields"][0]["id"] == "name" and "_how_to_edit" not in body["form"]


def test_a_request_is_stored_announced_and_confirmed(env):
    response = submit(env)
    body = response.json()
    assert response.status_code == 201 and body["reference"].startswith("KR-") and "reviewed" in body["message"].lower() or "review" in body["message"].lower()
    row = store.list_requests()[0]
    assert row["status"] == "pending" and row["email"] == "pat@example.gov" and row["agency"] == "Central Fire District"
    assert "ip_hash" not in row
    assert len(env.alerts) == 1 and env.alerts[0]["alert_type"] == "api_key_request" and env.alerts[0]["force"] is True
    assert env.alerts[0]["admin_path"] == "/admin/api-keys" and "Central Fire District" in env.alerts[0]["title"]
    assert {f["name"] for f in env.alerts[0]["fields"]} >= {"Agency", "Contact", "Wants to use", "Reason", "Plan"}
    assert [m.to for m in env.sent] == ["pat@example.gov"] and body["reference"] in env.sent[0].text


def test_validation_turnstile_honeypot_duplicates_and_daily_cap(env):
    bad = submit(env, {**GOOD, "email": "nope"})
    assert bad.status_code == 422 and "email" in bad.json()["detail"]["errors"]
    assert submit(env, token="bad").status_code == 400 and not store.list_requests()
    assert submit(env, website="http://spam").status_code == 201 and not store.list_requests() and not env.alerts       # honeypot
    assert submit(env).status_code == 201
    duplicate = submit(env)
    assert duplicate.status_code == 409 and "KR-" in duplicate.json()["detail"]
    for index in range(public_module.DAILY_REQUESTS_PER_IP - 1):
        assert submit(env, {**GOOD, "email": f"p{index}@example.gov"}).status_code == 201
    assert submit(env, {**GOOD, "email": "late@example.gov"}).status_code == 429


# --- approval ------------------------------------------------------------------------------------------------

def make_request(env, **changes):
    assert submit(env, {**GOOD, **changes}).status_code == 201
    return store.list_requests()[0]


def test_admin_routes_require_an_admin(env):
    del env.app.dependency_overrides[require_admin]
    for method, path in (("get", "/admin/api-key-requests/requests"), ("get", "/admin/api-key-requests/keys"), ("get", "/admin/api-key-requests/summary"),
                         ("post", "/admin/api-key-requests/requests/1/approve"), ("post", "/admin/api-key-requests/keys/1/revoke")):
        assert getattr(env.client, method)(path, **({"json": {}} if method == "post" else {})).status_code in (401, 403)


def test_approving_creates_a_key_emails_it_and_stores_only_its_hash(env):
    row = make_request(env)
    env.sent.clear()
    response = env.client.post(f"/admin/api-key-requests/requests/{row['id']}/approve", json={"note": "Welcome", "multiplier": 20, "expires_days": 365})
    body = response.json()
    assert response.status_code == 200 and body["key"].startswith("smf_") and body["email_sent"] is True
    assert body["request"]["status"] == "approved" and body["request"]["reviewed_by"] == "admin-user" and body["request"]["decision_email_sent"] is True
    message = env.sent[0]
    assert message.to == "pat@example.gov" and body["key"] in message.text and body["key"] in message.html and "Welcome" in message.text
    assert "20 times" in message.text and "terms" in message.text.lower() and "Show Me Fire API key" in message.subject + message.text
    assert "burn check api key" not in (message.subject + message.text).lower() and "whole public Show Me Fire API" in message.text
    with sqlite3.connect(str(store.get_db_path())) as conn:
        stored = json.dumps([list(r) for r in conn.execute("SELECT * FROM api_keys").fetchall()])
    assert body["key"] not in stored and store.hash_key(body["key"]) in stored
    assert body["key_info"]["multiplier"] == 20 and body["key_info"]["expires_at"] and "key_hash" not in body["key_info"]
    again = env.client.post(f"/admin/api-key-requests/requests/{row['id']}/approve", json={})
    assert again.status_code == 409 and len(store.list_keys()) == 1
    assert env.client.get("/admin/api-key-requests/summary").json() == {"pending": 0, "active_keys": 1, "email_configured": True}


def test_if_the_email_cannot_be_sent_the_admin_still_gets_the_key(env, monkeypatch):
    row = make_request(env)

    def down(*args):
        raise RuntimeError("resend is down")
    monkeypatch.setattr(mail, "send", down)
    body = env.client.post(f"/admin/api-key-requests/requests/{row['id']}/approve", json={}).json()
    assert body["key"].startswith("smf_") and body["email_sent"] is False and "resend" in body["email_error"]
    assert store.get_request(row["id"])["decision_email_sent"] is False
    skipped = make_request(env, email="two@example.gov")
    body = env.client.post(f"/admin/api-key-requests/requests/{skipped['id']}/approve", json={"send_email": False}).json()
    assert body["email_sent"] is False and body["email_error"] is None


def test_denying_sends_a_polite_email_without_a_key(env):
    row = make_request(env)
    env.sent.clear()
    body = env.client.post(f"/admin/api-key-requests/requests/{row['id']}/deny", json={"note": "internal only", "message": "Please reapply with a plan."}).json()
    assert body["request"]["status"] == "denied" and body["request"]["review_note"] == "internal only" and not store.list_keys()
    assert "Please reapply" in env.sent[0].text and "internal only" not in env.sent[0].text and "smf_" not in env.sent[0].text
    assert env.client.post(f"/admin/api-key-requests/requests/{row['id']}/approve", json={}).status_code == 409
    assert env.client.get("/admin/api-key-requests/requests/999").status_code == 404


def test_listing_puts_pending_requests_first_and_filters(env):
    first = make_request(env)
    second = make_request(env, email="b@example.gov")
    env.client.post(f"/admin/api-key-requests/requests/{first['id']}/deny", json={"send_email": False})
    listed = env.client.get("/admin/api-key-requests/requests").json()["requests"]
    assert [r["status"] for r in listed] == ["pending", "denied"] and listed[0]["id"] == second["id"]
    assert [r["id"] for r in env.client.get("/admin/api-key-requests/requests?status=denied").json()["requests"]] == [first["id"]]
    detail = env.client.get(f"/admin/api-key-requests/requests/{second['id']}").json()["request"]
    assert detail["answers"][0]["label"] == "Your full name" and detail["form_version"] == 1


def test_emails_escape_what_people_typed():
    _subject, _text, body = mail.approved("<script>alert(1)</script>", "<b>Agency</b>", "smf_key", 10, None, "<i>note</i>")
    assert "<script>" not in body and "&lt;script&gt;" in body and "<b>Agency</b>" not in body and "<i>note</i>" not in body
    assert "<script>" not in mail.received("<script>", "KR-1")[2] and "<script>" not in mail.denied("<script>", "<script>")[2]


# --- keys in use -----------------------------------------------------------------------------------------------

def request_with(key=None):
    headers = {"authorization": f"Bearer {key}"} if key else {}
    return SimpleNamespace(headers=headers, client=SimpleNamespace(host="203.0.113.7"))


def test_an_issued_key_raises_the_limit_until_it_is_revoked_or_expires(env, monkeypatch):
    row = make_request(env)
    key = env.client.post(f"/admin/api-key-requests/requests/{row['id']}/approve", json={"multiplier": 7}).json()["key"]
    assert rate_limit.key_multiplier(request_with(key)) == 7 and rate_limit.valid_api_key(request_with(key))
    assert rate_limit.key_multiplier(request_with()) == 0 and rate_limit.key_multiplier(request_with("smf_wrong")) == 0
    limiter = RateLimiter("t_keyed", 3)
    for _ in range(21):
        limiter(request_with(key))                                        # 3 x 7 = 21 a minute
    with pytest.raises(Exception) as caught:
        limiter(request_with(key))
    assert getattr(caught.value, "status_code", None) == 429

    key_id = store.list_keys()[0]["id"]
    assert env.client.post(f"/admin/api-key-requests/keys/{key_id}/revoke").status_code == 200
    assert rate_limit.key_multiplier(request_with(key)) == 0
    assert env.client.post(f"/admin/api-key-requests/keys/{key_id}/revoke").status_code == 404


def test_expired_keys_do_not_work_and_environment_keys_still_do(env, monkeypatch):
    key, _row = store.create_key(label="x", email="x@example.gov", expires_at="2020-01-01T00:00:00Z")
    assert rate_limit.key_multiplier(request_with(key)) == 0
    monkeypatch.setenv("API_READ_KEYS", "legacy-secret")
    assert rate_limit.key_multiplier(request_with("legacy-secret")) == rate_limit.KEY_TIER_MULTIPLIER


def test_a_database_failure_leaves_the_caller_unkeyed_not_broken(env, monkeypatch):
    def broken(_key):
        raise sqlite3.OperationalError("locked")
    monkeypatch.setattr(store, "verify_key", broken)
    assert rate_limit.key_multiplier(request_with("smf_anything")) == 0


def test_use_is_counted_and_rotating_replaces_the_key(env):
    row = make_request(env)
    key = env.client.post(f"/admin/api-key-requests/requests/{row['id']}/approve", json={}).json()["key"]
    for _ in range(3):
        rate_limit.key_multiplier(request_with(key))
    rate_limit.clear_database_key_cache()
    rate_limit.key_multiplier(request_with(key))                          # refresh writes the pending uses back
    info = store.list_keys()[0]
    assert info["use_count"] >= 4 and info["last_used_at"]
    env.sent.clear()
    rotated = env.client.post(f"/admin/api-key-requests/keys/{info['id']}/rotate", json={}).json()
    rate_limit.clear_database_key_cache()
    assert rotated["key"] != key and rate_limit.key_multiplier(request_with(key)) == 0 and rate_limit.key_multiplier(request_with(rotated["key"])) == 10
    assert rotated["email_sent"] and rotated["key"] in env.sent[0].text and "replaces your previous" in env.sent[0].text
    assert [k["active"] for k in store.list_keys()].count(True) == 1
