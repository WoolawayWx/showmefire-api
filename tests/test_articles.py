"""Feature articles (services/articles.py, services/article_blocks.py, routers/articles.py) and the newsletter web archive
(services/newsletter_web.py): block validation, the draft/publish rules per role, the public read model, and what the web copy of
a newsletter is allowed to contain."""
import sqlite3

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from core.admin_auth import permissions
from core.database import get_db_path, init_database
from routers import articles as articles_router
from services import articles, newsletter_web
from services.article_blocks import make_summary, reading_minutes, validate_blocks


@pytest.fixture()
def db(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    init_database()
    return tmp_path


@pytest.fixture()
def client(db, monkeypatch):
    """The real router with sign-in faked: `who["role"]` is the signed-in role (None = nobody), and every permission check
    still goes through the real ROLE_PERMISSIONS table."""
    who = {"role": "staff", "email": "writer@example.org"}

    def authorize_user(token, *needed):
        if who["role"] is None:
            raise HTTPException(status_code=401, detail="Unauthorized")
        user = {"email": who["email"], "role": who["role"]}
        if any(not permissions.has(user, permission) for permission in needed):
            raise HTTPException(status_code=403, detail="Your account does not have access to that.")
        return user

    monkeypatch.setattr(permissions, "authorize_user", authorize_user)
    monkeypatch.setattr(permissions, "authorize", lambda token=None, *needed: authorize_user(token, *needed)["email"])
    app = FastAPI()
    app.include_router(articles_router.router)
    test_client = TestClient(app)
    test_client.who = who  # type: ignore[attr-defined]
    return test_client


GOOD = {
    "title": "Why Missouri burns in spring",
    "subtitle": "Dormant grass, dry air and wind.",
    "tags": ["Spring", "spring", " Fire Weather "],
    "blocks": [
        {"type": "heading", "level": 2, "text": "The short version"},
        {"type": "paragraph", "text": "Dead grass dries **fast** when the [humidity](https://example.org/rh) drops."},
        {"type": "callout", "tone": "warning", "title": "Heads up", "text": "Check your county's burn ban."},
        {"type": "list", "style": "bullet", "items": ["Low humidity", "Wind"]},
    ],
}


# ------------------------------------------------------------------------------------------------------------ blocks

def test_valid_blocks_round_trip_and_summary():
    blocks = validate_blocks(GOOD["blocks"])
    assert [block["type"] for block in blocks] == ["heading", "paragraph", "callout", "list"]
    assert make_summary(blocks).startswith("The short version Dead grass dries fast when the humidity")
    assert reading_minutes(blocks) == 1


@pytest.mark.parametrize("bad, fragment", [
    ([{"type": "image", "url": "http://example.org/a.jpg", "alt": "x"}], "https"),
    ([{"type": "image", "url": "https://example.org/a.jpg"}], "alt"),
    ([{"type": "image", "url": "data:image/png;base64,AAAA", "alt": "x"}], "https"),
    ([{"type": "button", "label": "Go", "url": "javascript:alert(1)"}], "https"),
    ([{"type": "button", "label": "Go", "url": "//evil.example"}], "https"),
    ([{"type": "video", "id": "https://youtu.be/eNYjCM6hTwQ", "title": "v"}], "11-character"),
    ([{"type": "paragraph", "text": "x", "html": "<script>"}], "Extra"),
    ([{"type": "script", "text": "x"}], "script"),
    ([{"type": "list", "items": []}], "items"),
    ([{"type": "heading", "level": 1, "text": "x"}], "level"),
])
def test_unsafe_or_malformed_blocks_are_refused(bad, fragment):
    with pytest.raises(ValueError) as error:
        validate_blocks(bad)
    assert fragment.lower() in str(error.value).lower()
    assert "block 1" in str(error.value)


def test_relative_links_are_allowed_on_buttons_only():
    assert validate_blocks([{"type": "button", "label": "Help", "url": "/help"}])[0]["url"] == "/help"
    with pytest.raises(ValueError):
        validate_blocks([{"type": "image", "url": "/images/a.png", "alt": "x"}])


# --------------------------------------------------------------------------------------------------------- the service

def test_create_cleans_fields_and_makes_a_unique_slug(db):
    first = articles.create(GOOD, "writer@example.org")
    second = articles.create(GOOD, "writer@example.org")
    assert first["status"] == "draft" and first["slug"] == "why-missouri-burns-in-spring"
    assert second["slug"] == "why-missouri-burns-in-spring-2"
    assert first["tags"] == ["spring", "fire weather"]


def test_hero_needs_alt_text_and_https(db):
    with pytest.raises(articles.ArticleError, match="alt"):
        articles.create({**GOOD, "hero": {"url": "https://example.org/a.jpg"}}, "w")
    with pytest.raises(articles.ArticleError, match="https"):
        articles.create({**GOOD, "hero": {"url": "http://example.org/a.jpg", "alt": "a"}}, "w")


def test_weather_map_must_exist_and_is_resolved_for_readers(db):
    with pytest.raises(articles.ArticleError, match="not in the map catalog"):
        articles.create({**GOOD, "blocks": [{"type": "weather_map", "key": "nope"}]}, "w")
    key = next(iter(articles.map_lookup()))
    article = articles.create({**GOOD, "blocks": [{"type": "paragraph", "text": "Hi"}, {"type": "weather_map", "key": key}]}, "w")
    articles.publish(article["id"], "admin@example.org")
    read = articles.public_view(articles.get_public(article["slug"]))
    assert read["blocks"][1]["map"]["url"].startswith("https://")


def test_publish_needs_content_and_drafts_are_not_public(db):
    empty = articles.create({"title": "Empty", "blocks": []}, "w")
    with pytest.raises(articles.ArticleError, match="content"):
        articles.publish(empty["id"], "a")
    draft = articles.create(GOOD, "w")
    with pytest.raises(articles.NotFound):
        articles.get_public(draft["slug"])
    assert articles.list_public()[1] == 0
    articles.publish(draft["id"], "a")
    assert articles.list_public()[1] == 1
    articles.unpublish(draft["id"], "a")
    assert articles.list_public()[1] == 0


def test_publishing_and_deleting_are_written_to_the_moderation_log(db):
    article = articles.create(GOOD, "w@example.org")
    articles.publish(article["id"], "a@example.org")
    articles.delete(article["id"], "a@example.org")
    conn = sqlite3.connect(get_db_path())
    actions = [row[0] for row in conn.execute("SELECT action FROM moderation_audit WHERE subject_type = 'article' ORDER BY id")]
    conn.close()
    assert actions == ["article_create", "article_publish", "article_delete"]


# ---------------------------------------------------------------------------------------------------------- the routes

def test_staff_can_write_a_draft_but_not_publish_or_touch_a_live_article(client):
    created = client.post("/api/admin/articles", json=GOOD)
    assert created.status_code == 201
    article_id = created.json()["article"]["id"]
    assert client.put(f"/api/admin/articles/{article_id}", json={**GOOD, "title": "Edited draft"}).status_code == 200
    assert client.post(f"/api/admin/articles/{article_id}/publish").status_code == 403

    client.who["role"] = "admin"
    assert client.post(f"/api/admin/articles/{article_id}/publish").status_code == 200

    client.who["role"] = "staff"
    assert client.put(f"/api/admin/articles/{article_id}", json=GOOD).status_code == 403
    assert client.delete(f"/api/admin/articles/{article_id}").status_code == 403
    assert client.post(f"/api/admin/articles/{article_id}/unpublish").status_code == 403

    client.who["role"] = "owner"
    assert client.delete(f"/api/admin/articles/{article_id}").status_code == 200


def test_partners_reviewers_and_signed_out_visitors_cannot_use_the_admin_routes(client):
    for role, status in (("partner", 403), ("reviewer", 403), (None, 401)):
        client.who["role"] = role
        assert client.get("/api/admin/articles").status_code == status
        assert client.post("/api/admin/articles", json=GOOD).status_code == status
        assert client.get("/api/admin/newsletter-archive").status_code == status


def test_a_bad_block_comes_back_as_a_readable_400(client):
    response = client.post("/api/admin/articles", json={**GOOD, "blocks": [{"type": "image", "url": "http://x.org/a.png", "alt": "a"}]})
    assert response.status_code == 400
    assert "block 1" in response.json()["detail"] and "https" in response.json()["detail"]


def test_public_routes_show_only_published_articles_without_admin_fields(client):
    client.who["role"] = "admin"
    live = client.post("/api/admin/articles", json=GOOD).json()["article"]
    client.post("/api/admin/articles", json={**GOOD, "title": "Still a draft"})
    client.post(f"/api/admin/articles/{live['id']}/publish")
    client.who["role"] = None  # the public

    listing = client.get("/api/articles").json()
    assert listing["total"] == 1 and listing["articles"][0]["slug"] == live["slug"]
    assert "blocks" not in listing["articles"][0]
    assert {"tag": "spring", "count": 1} in listing["tags"]
    assert client.get("/api/articles?tag=nothing").json()["total"] == 0

    detail = client.get(f"/api/articles/{live['slug']}")
    assert detail.status_code == 200 and detail.headers["cache-control"] == "public, max-age=60"
    body = detail.json()["article"]
    assert body["blocks"][0]["text"] == "The short version"
    assert "created_by" not in body and "status" not in body
    assert client.get("/api/articles/still-a-draft").status_code == 404


# ------------------------------------------------------------------------------------------------- newsletter web archive

def _bulletin(db_path, subject, html_body, *, status="sent", list_id="show-me-fire-newsletter", hidden=0, sent_at="2026-11-20 14:00:00"):
    conn = sqlite3.connect(get_db_path())
    conn.execute("INSERT INTO bulletins (subject, html_body, text_body, audience_list_id, status, sent_at, web_hidden) VALUES (?, ?, ?, ?, ?, ?, ?)",
                 (subject, html_body, "text", list_id, status, sent_at, hidden))
    conn.commit()
    new_id = conn.execute("SELECT MAX(id) FROM bulletins").fetchone()[0]
    conn.close()
    return new_id


def test_sanitizer_removes_scripts_handlers_personal_links_and_merge_tags():
    cleaned = newsletter_web.sanitize_html(
        '<h1 style="color:#c00;position:fixed" onclick="x()">Hello</h1><p>Hi {{{contact.first_name}}}<script>alert(1)</script></p>'
        '<a href="javascript:alert(1)">bad</a><a href="https://showmefire.org/x">ok</a>'
        '<img src="http://t.example/p.gif"><img src="https://cdn.showmefire.org/a.png" alt="a" onerror="x()">'
        '<p><a href="{{{contact.manage_url}}}">Manage preferences</a> | <a href="{{{RESEND_UNSUBSCRIBE_URL}}}">Unsubscribe</a></p>'
        '<iframe src="https://evil.example"></iframe><form action="https://evil.example"><input></form>')
    assert "script" not in cleaned and "alert" not in cleaned and "onclick" not in cleaned and "onerror" not in cleaned
    assert "javascript:" not in cleaned and "position" not in cleaned and "iframe" not in cleaned and "<form" not in cleaned
    assert "{{" not in cleaned and "unsubscribe" not in cleaned.lower() and "Manage preferences" not in cleaned
    assert "http://t.example" not in cleaned and 'src="https://cdn.showmefire.org/a.png"' in cleaned
    assert 'href="https://showmefire.org/x"' in cleaned and "noopener" in cleaned


def test_archive_lists_only_sent_unhidden_list_newsletters(db):
    public = _bulletin(db, "November newsletter", "<p>Rain is coming.</p>")
    _bulletin(db, "Draft", "<p>x</p>", status="draft")
    _bulletin(db, "Hidden", "<p>x</p>", hidden=1)
    _bulletin(db, "Direct send", "<p>x</p>", list_id="")
    _bulletin(db, "Briefing packet", "<p>x</p>", list_id="show-me-fire-briefing-packet")
    items, total = newsletter_web.list_public()
    assert total == 1 and items[0]["id"] == public and items[0]["excerpt"] == "Rain is coming."
    assert newsletter_web.get_public(public)["html"] == "<p>Rain is coming.</p>"
    for private_id in range(public + 1, public + 5):
        assert newsletter_web.get_public(private_id) is None


def test_archive_detail_links_to_the_neighbouring_newsletters(db):
    older = _bulletin(db, "Older", "<p>a</p>", sent_at="2026-10-01 12:00:00")
    middle = _bulletin(db, "Middle", "<p>b</p>", sent_at="2026-11-01 12:00:00")
    newer = _bulletin(db, "Newer", "<p>c</p>", sent_at="2026-11-15 12:00:00")
    detail = newsletter_web.get_public(middle)
    assert detail["newer"]["id"] == newer and detail["older"]["id"] == older
    assert newsletter_web.get_public(newer)["newer"] is None and newsletter_web.get_public(older)["older"] is None


def test_only_admins_can_hide_a_newsletter_and_the_public_route_obeys(client, db):
    bulletin_id = _bulletin(db, "Spring burn tips", "<p>Be careful.</p>")
    assert client.get(f"/api/newsletters/{bulletin_id}").status_code == 200
    client.who["role"] = "staff"
    assert client.put(f"/api/admin/newsletter-archive/{bulletin_id}", json={"hidden": True}).status_code == 403
    client.who["role"] = "admin"
    hidden = client.put(f"/api/admin/newsletter-archive/{bulletin_id}", json={"hidden": True})
    assert hidden.status_code == 200 and hidden.json()["newsletter"]["state"] == "hidden"
    client.who["role"] = None
    assert client.get(f"/api/newsletters/{bulletin_id}").status_code == 404
    assert client.get("/api/newsletters").json()["total"] == 0
    assert client.put("/api/admin/newsletter-archive/999", json={"hidden": True}).status_code == 401
    client.who["role"] = "admin"
    assert client.put("/api/admin/newsletter-archive/999", json={"hidden": True}).status_code == 404


def test_browser_link_is_added_only_for_newsletters_that_will_be_public(monkeypatch):
    monkeypatch.setenv("PUBLIC_WEB_URL", "https://showmefire.org/")
    assert newsletter_web.web_url({"id": 7, "audience_list_id": "show-me-fire-newsletter"}) == "https://showmefire.org/comms/newsletters/7"
    assert newsletter_web.web_url({"id": 7, "audience_list_id": ""}) is None
    assert newsletter_web.web_url({"id": 7, "audience_list_id": "show-me-fire-briefing-packet"}) is None


def test_new_routes_all_declare_a_permission():
    app = FastAPI()
    app.include_router(articles_router.router)
    assert permissions.unguarded_admin_routes(app, prefix="/api/admin") == []
