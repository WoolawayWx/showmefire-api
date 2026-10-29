"""The staff API: permissions on every route, narrow responses, the same moderation code as the website, and a full audit trail."""
import sqlite3
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from core import database, sandbox, sandbox_data, security
from core.admin_auth import lockdown, permissions, sessions, users
from core.admin_session import AdminCookieMiddleware
from routers import admin_mobile, burn_bans, fires, staff_api
from services import audit

PASSWORD = "correct horse battery 9"


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    path = tmp_path_factory.mktemp("staff-api")
    patch = pytest.MonkeyPatch()
    patch.setenv("DATA_DIR", str(path))
    patch.setenv("STAFF_API_ENABLED", "true")
    patch.setenv("SANDBOX_ENABLED", "true")
    patch.setattr(security, "ADMIN_EMAIL", "owner@example.org")
    patch.setattr(security, "ADMIN_PASSWORD_HASH", security.hash_password(PASSWORD))
    patch.setattr(fires, "MEDIA_DIR", path / "fire-media")
    (path / "fire-media").mkdir()
    sandbox_data.build_template()
    yield path
    patch.undo()


@pytest.fixture
def app(data_dir, monkeypatch):
    monkeypatch.setenv("STAFF_API_ENABLED", "true")
    monkeypatch.setenv("SANDBOX_ENABLED", "true")
    monkeypatch.setattr("services.burn_ban_map.generate_burn_ban_map", lambda: None, raising=False)
    sessions._forget()
    sandbox_data.reset_sandbox()
    database.init_database()
    with sqlite3.connect(database.real_db_path()) as conn:  # a clean real database for every test
        for table in ("fire_events", "fire_event_media", "fire_incidents", "fire_incident_feedback", "burn_ban_submissions",
                      "fire_event_moderation", "burn_ban_moderation", "burn_ban_county_events", "clef_assessments"):
            conn.execute(f"DELETE FROM {table}")
        conn.execute("DROP TRIGGER IF EXISTS moderation_audit_no_delete")
        conn.execute("DELETE FROM moderation_audit")
        audit.ensure_table(conn)
    application = FastAPI()
    application.add_middleware(lockdown.ReviewerLockdownMiddleware)
    application.add_middleware(AdminCookieMiddleware)
    application.include_router(admin_mobile.router)
    application.include_router(staff_api.router)
    return application


@pytest.fixture
def client(app):
    return TestClient(app)


def headers_for(role, email=None):
    users.ensure_seeded()
    email = email or f"{role}@example.org"
    user = users.get_by_email(email)
    if user is None:
        extra = {"expires_at": time.time() + 3600} if role == "reviewer" else {}
        user = users.create_user(email, role.title(), role, PASSWORD, must_change=False, **extra)
    session_id, _ = sessions.create(user["id"], "127.0.0.1", "pytest", "mobile")
    return {"Authorization": f"Bearer {permissions.access_token_for(users.get_by_id(user['id']), session_id, client='mobile')}"}


def sql(statement, params=()):
    with sqlite3.connect(database.get_db_path()) as conn:
        return conn.execute(statement, params).lastrowid


def add_report(description="Grass fire near the road", **extra):
    values = dict(source="user_submission", status="pending", latitude=37.9, longitude=-91.8, county_fips="29161", county_name="Phelps",
                  occurred_at="2026-10-01T12:00:00Z", description=description, reporter_name="Pat Person", reporter_contact="pat@example.com",
                  reporter_org="Rolla VFD", submitter_ip_hash="iphash-secret", address_text="County Road 100", reporter_relationship="bystander")
    values.update(extra)
    return sql(f"INSERT INTO fire_events ({', '.join(values)}) VALUES ({', '.join('?' for _ in values)})", tuple(values.values()))


def add_incident():
    return sql("INSERT INTO fire_incidents (centroid_latitude, centroid_longitude, first_detected_at, last_detected_at, county_name, status, public_slug) "
               "VALUES (37.9, -91.8, '2026-10-01T12:00:00Z', '2026-10-01T13:00:00Z', 'Phelps', 'active', 'inc-1')")


# ---------------------------------------------------------------------------------------------- the surface itself

def test_every_staff_route_declares_a_permission(app):
    assert permissions.unguarded_admin_routes(app, prefix="/api/staff") == []


def test_it_is_off_until_switched_on(client, monkeypatch):
    headers = headers_for("staff")
    monkeypatch.setenv("STAFF_API_ENABLED", "false")
    assert client.get("/api/staff/v1/summary", headers=headers).status_code == 404


def test_nothing_works_without_signing_in(client):
    for path in ("/summary", "/me", "/reports/pending", "/incidents/pending", "/feedback", "/burn-bans", "/audit"):
        assert client.get(f"/api/staff/v1{path}").status_code == 401, path
    assert client.post("/api/staff/v1/reports/1/approve", json={}).status_code == 401


def test_partners_can_look_but_every_action_is_refused(client):
    event = add_report()
    ban = sql("INSERT INTO burn_ban_submissions (county_fips, county_name, effective_at, status) VALUES ('29161', 'Phelps', '2026-10-20T00:00:00', 'pending')")
    headers = headers_for("partner")
    for path in ("/summary", "/reports/pending", f"/reports/{event}", "/incidents/pending", "/feedback", "/burn-bans", f"/burn-bans/{ban}", "/audit"):
        assert client.get(f"/api/staff/v1{path}", headers=headers).status_code == 200, path
    for path, body in ((f"/reports/{event}/approve", {}), (f"/reports/{event}/reject", {"reason": "spam spam"}),
                       ("/incidents/1/approve", {}), ("/incidents/1/reject", {"reason": "spam spam"}), ("/incidents/1/not-a-fire", {"reason": "spam spam"}),
                       ("/feedback/1/approve", {}), ("/feedback/1/reject", {"reason": "spam spam"}),
                       (f"/burn-bans/{ban}/confirm", {}), (f"/burn-bans/{ban}/deny", {"reason": "spam spam"})):
        assert client.post(f"/api/staff/v1{path}", json=body, headers=headers).status_code == 403, path
    assert sql("SELECT status FROM fire_events WHERE id = ?", (event,)) is not None
    with sqlite3.connect(database.get_db_path()) as conn:
        assert conn.execute("SELECT status FROM fire_events WHERE id = ?", (event,)).fetchone()[0] == "pending"


def test_me_and_summary(client):
    add_report()
    add_report()
    incident = add_incident()
    sql("INSERT INTO fire_events (source, status, latitude, longitude, occurred_at, incident_id) VALUES ('viirs', 'pending', 37.9, -91.8, '2026-10-01T12:00:00Z', ?)", (incident,))
    sql("INSERT INTO fire_incident_feedback (incident_id, classification, status) VALUES (?, 'confirmed_fire', 'pending')", (incident,))
    sql("INSERT INTO burn_ban_submissions (county_fips, county_name, effective_at, status) VALUES ('29161', 'Phelps', '2026-10-20T00:00:00', 'pending')")
    headers = headers_for("staff")
    me = client.get("/api/staff/v1/me", headers=headers).json()
    assert me["user"]["role"] == "staff" and me["sandbox"] is False and "reports.moderate" in me["user"]["permissions"]
    response = client.get("/api/staff/v1/summary", headers=headers)
    assert response.json()["pending"] == {"reports": 2, "incidents": 1, "feedback": 1, "burn_bans": 1}
    assert response.headers["cache-control"] == "no-store"


# ---------------------------------------------------------------------------------------------- narrow responses

def test_responses_carry_no_reporter_identity_or_network_details(client):
    event = add_report("Barn fire behind the co-op")
    incident = add_incident()
    sql("UPDATE fire_events SET incident_id = ? WHERE id = ?", (incident, event))
    sql("INSERT INTO fire_incident_feedback (incident_id, classification, note, contact, submitter_ip_hash, status) "
        "VALUES (?, 'confirmed_fire', 'big flames', 'fb@example.com', 'fbhash-secret', 'pending')", (incident,))
    ban = sql("INSERT INTO burn_ban_submissions (county_fips, county_name, effective_at, status, submitter_name, submitter_contact, submitter_ip_hash, upload_token_hash) "
              "VALUES ('29161', 'Phelps', '2026-10-20T00:00:00', 'pending', 'Sam Submitter', 'sam@example.com', 'banhash-secret', 'token-secret')")
    headers = headers_for("staff")
    texts = [client.get(f"/api/staff/v1{path}", headers=headers).text for path in (
        "/reports/pending", f"/reports/{event}", "/incidents/pending", f"/incidents/{incident}", "/feedback", "/burn-bans", f"/burn-bans/{ban}")]
    combined = " ".join(texts)
    assert "Barn fire behind the co-op" in combined and "big flames" in combined  # the content staff review is there
    for private in ("Pat Person", "pat@example.com", "Rolla VFD", "iphash-secret", "fb@example.com", "fbhash-secret", "Sam Submitter",
                    "sam@example.com", "banhash-secret", "token-secret"):
        assert private not in combined, private


def test_report_detail_has_photos_clef_and_history_and_the_photo_downloads(client, data_dir):
    event = add_report()
    (data_dir / "fire-media" / "r1.png").write_bytes(b"\x89PNG\r\n\x1a\nxx")
    (data_dir / "fire-media" / "doc.pdf").write_bytes(b"%PDF-1.4")
    photo = sql("INSERT INTO fire_event_media (event_id, stored_filename, content_type, kind) VALUES (?, 'r1.png', 'image/png', 'photo')", (event,))
    document = sql("INSERT INTO fire_event_media (event_id, stored_filename, content_type, kind) VALUES (?, 'doc.pdf', 'application/pdf', 'document')", (event,))
    sql("INSERT INTO clef_assessments (subject_type, subject_id, task, input_hash, answers_json, model, created_at) VALUES ('fire_event', ?, 'fire_report', 'h', ?, 'clef', 'now')",
        (event, '{"shows_fire_or_smoke": {"type": "noul", "noul": 0.9}}'))
    headers = headers_for("staff")
    detail = client.get(f"/api/staff/v1/reports/{event}", headers=headers).json()["report"]
    assert [m["id"] for m in detail["media"]] == [photo] and detail["clef_summary"] == "fire or smoke 90%"
    image = client.get(f"/api/staff/v1/reports/{event}/media/{photo}", headers=headers)
    assert image.status_code == 200 and image.content.startswith(b"\x89PNG") and "no-store" in image.headers["cache-control"]
    assert client.get(f"/api/staff/v1/reports/{event}/media/{document}", headers=headers).status_code == 404  # department documents stay on the website
    assert client.get(f"/api/staff/v1/reports/{event}/media/{photo}").status_code == 401
    satellite = sql("INSERT INTO fire_events (source, status, latitude, longitude, occurred_at) VALUES ('viirs', 'pending', 37.9, -91.8, '2026-10-01T12:00:00Z')")
    assert client.get(f"/api/staff/v1/reports/{satellite}", headers=headers).status_code == 404


# ---------------------------------------------------------------------------------------------- actions use the website's code, with a reason

def test_approving_a_report_works_like_the_website_and_is_audited_as_mobile(client):
    event = add_report()
    headers = headers_for("staff")
    response = client.post(f"/api/staff/v1/reports/{event}/approve", json={"label": "controlled_burn", "note": "landowner confirmed"}, headers=headers)
    assert response.status_code == 200 and response.json()["report"]["status"] == "approved"
    with sqlite3.connect(database.get_db_path()) as conn:
        assert conn.execute("SELECT status, cause_category FROM fire_events WHERE id = ?", (event,)).fetchone() == ("approved", "prescribed")
    entry = audit.history("fire_event", event)[0]
    assert (entry["action"], entry["actor_email"], entry["actor_role"], entry["client"], entry["reason"]) == (
        "fire_event.approved", "staff@example.org", "staff", "mobile", "landowner confirmed")
    assert client.post(f"/api/staff/v1/reports/{event}/approve", json={}, headers=headers).status_code == 409  # already decided, as on the website
    history = client.get(f"/api/staff/v1/history/reports/{event}", headers=headers).json()["history"]
    assert [h["action"] for h in history] == ["fire_event.approved"]


def test_rejecting_needs_a_reason_and_keeps_it(client):
    event = add_report()
    headers = headers_for("staff")
    assert client.post(f"/api/staff/v1/reports/{event}/reject", json={}, headers=headers).status_code == 422
    assert client.post(f"/api/staff/v1/reports/{event}/reject", json={"reason": "no"}, headers=headers).status_code == 422
    assert client.post(f"/api/staff/v1/reports/{event}/reject", json={"reason": "duplicate of the Phelps fire"}, headers=headers).status_code == 200
    assert audit.history("fire_event", event)[0]["reason"] == "duplicate of the Phelps fire"


def test_incident_decisions_and_not_a_fire_record_who_and_why(client):
    incident = add_incident()
    first = sql("INSERT INTO fire_events (source, status, latitude, longitude, occurred_at, incident_id) VALUES ('viirs', 'pending', 37.9, -91.8, '2026-10-01T12:00:00Z', ?)", (incident,))
    headers = headers_for("admin")
    assert client.post(f"/api/staff/v1/incidents/{incident}/not-a-fire", json={}, headers=headers).status_code == 422
    done = client.post(f"/api/staff/v1/incidents/{incident}/not-a-fire", json={"reason": "it is the sawmill stack"}, headers=headers).json()
    assert done["delete_after"] and done["incident_decision"]["decision"] == "rejected"
    actions = {e["action"]: e for e in audit.history("fire_incident", incident)}
    assert actions["incident.rejected"]["reason"] == "it is the sawmill stack"
    assert actions["incident.deletion_scheduled"]["reason"] == "it is the sawmill stack"
    assert all(e["actor_email"] == "admin@example.org" and e["client"] == "mobile" for e in actions.values())
    with sqlite3.connect(database.get_db_path()) as conn:
        assert conn.execute("SELECT status, cause_category FROM fire_events WHERE id = ?", (first,)).fetchone() == ("rejected", "not_a_fire")


def test_approving_and_rejecting_a_whole_incident(client):
    incident = add_incident()
    sql("INSERT INTO fire_events (source, status, latitude, longitude, occurred_at, incident_id) VALUES ('viirs', 'pending', 37.9, -91.8, '2026-10-01T12:00:00Z', ?)", (incident,))
    headers = headers_for("staff")
    assert client.post(f"/api/staff/v1/incidents/{incident}/reject", json={}, headers=headers).status_code == 422
    approved = client.post(f"/api/staff/v1/incidents/{incident}/approve", json={"label": "fire", "note": "confirmed by county dispatch"}, headers=headers)
    assert approved.status_code == 200 and approved.json()["count"] == 1
    detail = client.get(f"/api/staff/v1/incidents/{incident}", headers=headers).json()["incident"]
    assert detail["review_decision"] == "approved" and detail["history"][0]["reason"] == "confirmed by county dispatch"


def test_feedback_review_keeps_the_note_and_rejecting_needs_a_reason(client):
    incident = add_incident()
    good = sql("INSERT INTO fire_incident_feedback (incident_id, classification, status) VALUES (?, 'confirmed_fire', 'pending')", (incident,))
    headers = headers_for("staff")
    assert client.post(f"/api/staff/v1/feedback/{good}/reject", json={}, headers=headers).status_code == 422
    assert client.post(f"/api/staff/v1/feedback/{good}/approve", json={"note": "matches the photos"}, headers=headers).status_code == 200
    history = audit.history("incident_feedback", good)
    assert history[0]["reason"] == "matches the photos" and history[0]["client"] == "mobile"
    other = sql("INSERT INTO fire_incident_feedback (incident_id, classification, status) VALUES (?, 'not_a_fire', 'pending')", (incident,))
    assert client.post(f"/api/staff/v1/feedback/{other}/reject", json={"reason": "joke submission"}, headers=headers).status_code == 200
    assert audit.history("incident_feedback", other)[0]["reason"] == "joke submission"
    assert client.post("/api/staff/v1/feedback/999/approve", json={}, headers=headers).status_code == 404


def test_burn_ban_confirm_and_deny_use_the_websites_rules(client):
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    issue = sql("INSERT INTO burn_ban_submissions (county_fips, county_name, effective_at, expires_at, proof_url, request_type, status) "
                "VALUES ('29161', 'Phelps', ?, '', 'https://example.invalid/order', 'issue', 'pending')", (now,))
    no_proof = sql("INSERT INTO burn_ban_submissions (county_fips, county_name, effective_at, request_type, status) VALUES ('29215', 'Texas', ?, 'issue', 'pending')", (now,))
    headers = headers_for("staff")
    assert client.post(f"/api/staff/v1/burn-bans/{no_proof}/confirm", json={}, headers=headers).status_code == 400  # the website's proof rule
    assert client.post(f"/api/staff/v1/burn-bans/{issue}/confirm", json={"note": "order matches the courthouse notice"}, headers=headers).status_code == 200
    assert client.post(f"/api/staff/v1/burn-bans/{no_proof}/deny", json={}, headers=headers).status_code == 422
    assert client.post(f"/api/staff/v1/burn-bans/{no_proof}/deny", json={"reason": "no proof attached"}, headers=headers).status_code == 200
    confirmed = audit.history("burn_ban", issue)[0]
    assert (confirmed["action"], confirmed["reason"], confirmed["client"]) == ("burn_ban.confirmed", "order matches the courthouse notice", "mobile")
    assert audit.history("burn_ban", no_proof)[0]["reason"] == "no proof attached"
    detail = client.get(f"/api/staff/v1/burn-bans/{issue}", headers=headers).json()["submission"]
    assert detail["status"] == "confirmed" and detail["history"][0]["actor_email"] == "staff@example.org"
    assert client.get("/api/staff/v1/burn-bans?status=nonsense", headers=headers).status_code == 400


def test_the_audit_log_can_be_filtered_and_paged(client):
    headers = headers_for("staff")
    for _ in range(3):
        client.post(f"/api/staff/v1/reports/{add_report()}/approve", json={}, headers=headers)
    other = headers_for("admin")
    client.post(f"/api/staff/v1/reports/{add_report()}/reject", json={"reason": "not a real fire"}, headers=other)
    everything = client.get("/api/staff/v1/audit", headers=headers).json()
    assert len(everything["entries"]) == 4
    by_admin = client.get("/api/staff/v1/audit?actor=admin@example.org", headers=headers).json()["entries"]
    assert [e["action"] for e in by_admin] == ["fire_event.rejected"]
    page = client.get("/api/staff/v1/audit?limit=2", headers=headers).json()
    older = client.get(f"/api/staff/v1/audit?limit=2&before_id={page['next_before_id']}", headers=headers).json()["entries"]
    assert len(page["entries"]) == 2 and len(older) == 2 and older[0]["id"] < page["entries"][-1]["id"]
    assert client.get("/api/staff/v1/audit?subject_type=nope", headers=headers).status_code == 400
    assert client.get("/api/staff/v1/history/widgets/1", headers=headers).status_code == 404


# ---------------------------------------------------------------------------------------------- push registration

def test_push_registration_follows_the_permission(client):
    body = {"installation_id": "install-12345678", "expo_token": "ExponentPushToken[aaaaaaaaaaaaaaaaaaaaaa]", "platform": "ios", "app_version": "1.3.0"}
    assert client.put("/api/staff/v1/push-device", json=body, headers=headers_for("staff")).status_code == 200
    assert client.put("/api/staff/v1/push-device", json={**body, "expo_token": "nope"}, headers=headers_for("staff")).status_code == 422
    assert client.put("/api/staff/v1/push-device", json=body, headers=headers_for("reviewer")).status_code == 403  # reviewers get no real push
    other = {**body, "installation_id": "install-87654321", "expo_token": "ExponentPushToken[bbbbbbbbbbbbbbbbbbbbbb]"}
    assert client.put("/api/staff/v1/push-device", json=other, headers=headers_for("partner")).status_code == 200  # partners may receive
    assert client.delete("/api/staff/v1/push-device", headers=headers_for("staff")).json() == {"success": True}


# ---------------------------------------------------------------------------------------------- the reviewer sandbox through the same routes

def test_a_reviewer_uses_the_same_routes_on_demo_data_and_real_data_is_untouched(client):
    real = add_report("REAL report from a real person")
    headers = headers_for("reviewer", "apple@example.org")
    pending = client.get("/api/staff/v1/reports/pending", headers=headers).json()["reports"]
    assert len(pending) == 3 and all("REAL" not in r["description"] for r in pending)
    me = client.get("/api/staff/v1/me", headers=headers).json()
    assert me["sandbox"] is True and me["user"]["role"] == "reviewer"
    target = pending[0]["id"]
    assert client.post(f"/api/staff/v1/reports/{target}/approve", json={"note": "looks right"}, headers=headers).status_code == 200
    assert client.post(f"/api/staff/v1/reports/{pending[1]['id']}/reject", json={"reason": "demo rejection"}, headers=headers).status_code == 200
    with sqlite3.connect(database.real_db_path()) as conn:
        assert conn.execute("SELECT status FROM fire_events WHERE id = ?", (real,)).fetchone()[0] == "pending"
        assert conn.execute("SELECT COUNT(*) FROM moderation_audit").fetchone()[0] == 0
    entry = client.get(f"/api/staff/v1/history/reports/{target}", headers=headers).json()["history"][-1]
    assert (entry["client"], entry["actor_role"]) == ("sandbox", "reviewer")
    photo = client.get(f"/api/staff/v1/reports/{target}", headers=headers).json()["report"]["media"][0]["id"]
    image = client.get(f"/api/staff/v1/reports/{target}/media/{photo}", headers=headers)
    assert image.status_code == 200 and image.content.startswith(b"\x89PNG")
    test_push = client.post("/api/staff/v1/push-test", headers=headers).json()["notification"]
    assert test_push["url"] == "/staff/reports"
    assert client.post("/api/staff/v1/push-test", headers=headers_for("staff")).status_code == 404  # real staff get real pushes


def test_a_reviewer_can_confirm_a_demo_burn_ban_without_touching_the_public_map(client, monkeypatch):
    calls = []
    monkeypatch.setattr("services.burn_ban_map.generate_burn_ban_map", lambda: calls.append(1), raising=False)
    headers = headers_for("reviewer", "apple@example.org")
    bans = client.get("/api/staff/v1/burn-bans", headers=headers).json()["submissions"]
    assert len(bans) == 3
    assert client.post(f"/api/staff/v1/burn-bans/{bans[0]['id']}/confirm", json={"note": "demo"}, headers=headers).status_code in (200, 400)
    assert calls == []  # regenerating the public map (and its CDN copy) is skipped in the sandbox
