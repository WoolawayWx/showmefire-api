"""Reviewer labels (fire / controlled burn) on approval, and the source log of everything that arrives."""
import json
import sqlite3
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from core import database
from routers import fires as fires_router
from routers import fires_ingest as ingest_router
from services import fire_ingest


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "showmefire.db"
    monkeypatch.setattr(database, "get_db_path", lambda: path)
    database.init_database()
    return path


def _rows(db, sql, *args):
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute(sql, args).fetchall()]
    finally:
        conn.close()


def _detection(n=0, **kwargs):
    return database.upsert_detection_event("viirs", f"d{n}", 38.9517 + n * 0.001, -92.3341, "2026-10-20T18:00:00Z",
                                           county_fips="29019", county_name="Boone", **kwargs)


def _report(db, **origin):
    return database.create_fire_report(
        latitude=38.9517, longitude=-92.3341, occurred_at="2026-10-20T14:30:00Z", occurred_at_precision="minute",
        acres=3.0, acres_is_estimate=True, fuel_types=["grass"], description="Grass fire along the fence line.",
        out_of_ordinary="", reporter_contact="chief@example.com", submitter_ip_hash="a" * 64,
        consent_version="v1", captcha_verdict="success", county_fips="29019", county_name="Boone", **origin)


def _cause(db, event_id):
    return _rows(db, "SELECT cause_category, verification_tier, label_revision, status FROM fire_events WHERE id = ?", event_id)[0]


# --- the label on approval -----------------------------------------------------------------------------------------

def test_the_label_only_allows_a_fire_or_a_controlled_burn():
    assert fires_router.FireReportModeration(label="fire").label == "fire"
    assert fires_router.FireReportModeration(label="controlled_burn").label == "controlled_burn"
    assert fires_router.FireReportModeration().label is None
    with pytest.raises(ValidationError):
        fires_router.FireReportModeration(label="bonfire")


@pytest.mark.parametrize("label,cause", [("fire", "wildfire"), ("controlled_burn", "prescribed")])
def test_approving_with_a_label_sets_the_cause_and_audits_it(db, label, cause):
    event_id = _report(db)["id"]
    result = fires_router.admin_approve_fire_report(event_id, fires_router.FireReportModeration(label=label), actor="staff@x.org")

    assert result["report"]["cause_category"] == cause
    row = _cause(db, event_id)
    assert row["cause_category"] == cause and row["verification_tier"] == "admin_reviewed" and row["status"] == "approved"
    assert row["label_revision"] == 2
    audit = _rows(db, "SELECT changed_fields_json FROM fire_event_moderation WHERE event_id = ? AND action = 'approved'", event_id)
    assert json.loads(audit[0]["changed_fields_json"]) == {"cause_category": cause}


def test_approving_without_a_label_leaves_the_cause_alone(db):
    event_id = _report(db)["id"]
    fires_router.admin_approve_fire_report(event_id, fires_router.FireReportModeration(), actor="staff@x.org")
    assert _cause(db, event_id)["cause_category"] == "unknown" and _cause(db, event_id)["label_revision"] == 1


def test_rejecting_never_labels(db):
    event_id = _report(db)["id"]
    database.set_fire_event_status(event_id, to_status="rejected", actor="staff@x.org", reason="duplicate", cause_category="prescribed")
    assert _cause(db, event_id)["cause_category"] == "unknown"


def test_incident_approval_labels_every_pending_member_and_later_arrivals(db):
    first, second = _detection(0)["event_id"], _detection(1)["event_id"]
    incident_id = _rows(db, "SELECT incident_id FROM fire_events WHERE id = ?", first)[0]["incident_id"]
    assert _rows(db, "SELECT incident_id FROM fire_events WHERE id = ?", second)[0]["incident_id"] == incident_id

    result = fires_router.admin_approve_all_incident_reports(
        incident_id, fires_router.FireReportModeration(label="controlled_burn"), actor="staff@x.org")
    assert result["count"] == 2 and result["label"] == "controlled_burn" and result["cause_category"] == "prescribed"
    assert _cause(db, first)["cause_category"] == "prescribed" and _cause(db, second)["cause_category"] == "prescribed"
    assert _rows(db, "SELECT review_label FROM fire_incidents WHERE id = ?", incident_id)[0]["review_label"] == "prescribed"

    late = _detection(2)["event_id"]            # joins the incident after the review
    row = _cause(db, late)
    assert row["status"] == "approved" and row["cause_category"] == "prescribed" and row["verification_tier"] == "admin_reviewed"
    assert any("inherited the label" in audit["reason"] for audit in
               _rows(db, "SELECT reason FROM fire_event_moderation WHERE event_id = ?", late))


def test_an_incident_approval_without_a_label_clears_the_standing_label(db):
    first = _detection(0)["event_id"]
    incident_id = _rows(db, "SELECT incident_id FROM fire_events WHERE id = ?", first)[0]["incident_id"]
    database.set_incident_review_decision(incident_id, "approved", "staff@x.org", cause_category="prescribed")
    database.set_incident_review_decision(incident_id, "approved", "staff@x.org")
    assert _rows(db, "SELECT review_label FROM fire_incidents WHERE id = ?", incident_id)[0]["review_label"] is None


# --- the source log ------------------------------------------------------------------------------------------------

def test_a_satellite_detection_records_its_origin_and_keeps_it_on_resighting(db):
    event_id = _detection(0, origin={"channel": "satfiredetection.geojson", "raw": {"SOURCE_ID": "d0", "FRP": 12.5}})["event_id"]
    _detection(0)                                   # the next ingest sees the same detection again
    with_origin = database.get_fire_event(event_id, admin=True)

    origin = with_origin["origin"]
    assert (origin["kind"], origin["feed"], origin["provider_ref"]) == ("satellite_feed", "NASA FIRMS (VIIRS)", "d0")
    assert origin["channel"] == "satfiredetection.geojson" and origin["raw"] == {"FRP": 12.5, "SOURCE_ID": "d0"}
    assert origin["receive_count"] == 2
    ingested = [row for row in with_origin["moderation"] if row["action"] == "ingested"][0]
    assert ingested["reason"] == "ingested from NASA FIRMS (VIIRS)"
    assert json.loads(ingested["changed_fields_json"])["origin"]["provider_ref"] == "d0"


def test_every_source_gets_a_default_origin_even_when_the_caller_says_nothing(db):
    event_id = database.upsert_detection_event("official", "fpa_fod:1", 38.9, -92.3, "2015-11-10T12:00:00Z")["event_id"]
    origin = database.get_fire_event(event_id, admin=True)["origin"]
    assert (origin["kind"], origin["feed"], origin["provider_ref"]) == ("official_import", "Official dataset", "fpa_fod:1")


def test_a_public_report_records_the_client_but_no_address(db):
    event = _report(db, origin={"client": {"user_agent": "Mozilla/5.0", "origin": "https://showmefire.org"}})
    origin = database.get_fire_event(event["id"], admin=True)["origin"]
    assert (origin["kind"], origin["feed"], origin["channel"]) == ("public_form", "Public fire report form", "POST /api/fires/reports")
    assert origin["client"] == {"user_agent": "Mozilla/5.0", "origin": "https://showmefire.org"}
    assert "a" * 64 not in json.dumps(origin)       # the IP hash stays on the event, not in the origin


def test_request_client_keeps_only_coarse_details(monkeypatch):
    request = SimpleNamespace(headers={"user-agent": "U" * 500, "referer": "https://showmefire.org/report",
                                       "cf-ipcountry": "US", "cf-region-code": "MO", "cf-connecting-ip": "203.0.113.9"})
    monkeypatch.setattr(fires_router, "TRUST_PROXY_HEADERS", True)
    client = fires_router._request_client(request)
    assert len(client["user_agent"]) == 200 and client["country"] == "US" and client["region"] == "MO"
    assert "203.0.113.9" not in json.dumps(client)
    monkeypatch.setattr(fires_router, "TRUST_PROXY_HEADERS", False)
    assert "country" not in fires_router._request_client(request)


def test_an_osint_lead_records_the_integration_the_link_and_the_lead_as_received(db):
    event, created = database.upsert_ingest_report(
        source="muse", external_id="post-1", title="Grass fire", description="Near Route 63", source_url="https://facebook.com/x/1",
        source_name="Boone County FPD", latitude=38.95, longitude=-92.33, occurred_at="2026-10-20T14:00:00Z",
        occurred_at_precision="day", acres=2.0, county_fips="29019", county_name="Boone",
        origin={"feed": "Muse", "kind": "osint_api", "source_url": "https://facebook.com/x/1", "raw": {"title": "Grass fire"}})
    assert created
    origin = database.get_fire_event(event["id"], admin=True)["origin"]
    assert (origin["kind"], origin["feed"], origin["source_url"]) == ("osint_api", "Muse", "https://facebook.com/x/1")
    assert origin["raw"] == {"title": "Grass fire"}


def test_the_ingest_endpoint_passes_the_integration_and_post_source(db, monkeypatch):
    captured = {}
    monkeypatch.setattr(ingest_router, "consume_fire_submission_quota", lambda *a, **k: {"allowed": True})
    monkeypatch.setattr(ingest_router, "county_for_point", lambda lat, lon: ("29019", "Boone"))
    monkeypatch.setattr(ingest_router, "add_ingest_media_links", lambda *a, **k: None)

    def fake_upsert(**kwargs):
        captured.update(kwargs)
        return {"id": 7, "status": "pending"}, True

    monkeypatch.setattr(ingest_router, "upsert_ingest_report", fake_upsert)
    payload = ingest_router.IngestReportCreate(
        external_id="p1", title="Grass fire", reported_at="2026-10-20T14:00:00Z", latitude=38.95, longitude=-92.33,
        source_url="https://facebook.com/x/1", source_name="Boone County FPD")
    source = {"source_slug": "muse", "source_name": "Muse", "daily_limit": 10, "source_id": 3, "api_key_id": 9}
    ingest_router.ingest_fire_report(payload, SimpleNamespace(status_code=0), ingest_source=source)

    origin = captured["origin"]
    assert origin["feed"] == "Muse" and origin["source_url"] == "https://facebook.com/x/1"
    assert origin["client"]["post_source"] == "Boone County FPD" and origin["client"]["api_key_id"] == 9
    assert origin["raw"]["external_id"] == "p1"


def _feature_file(path, features):
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    return path


def test_a_feed_run_is_logged_with_a_reason_for_everything_it_did_not_store(db, tmp_path):
    point = lambda lon, lat, **props: {"type": "Feature", "geometry": {"type": "Point", "coordinates": [lon, lat]},
                                       "properties": {"TYPENAME": "VIIRS", "SATELLITE": "N20", "ACQ_DATE_TIME": "2026-10-20T18:00:00Z", **props}}
    satdet = _feature_file(tmp_path / "satfiredetection.geojson", [
        point(-92.17, 38.57, SOURCE_ID="in-mo"),                    # Cole County
        point(-88.5, 38.6, SOURCE_ID="in-illinois"),                # outside Missouri
        point(-92.2, 38.6, SOURCE_ID="no-time", ACQ_DATE_TIME=""),  # no acquisition time
        {"type": "Feature", "geometry": None, "properties": {}},    # no coordinates
    ])
    ngfs = _feature_file(tmp_path / "missouri_fires.geojson", [])

    summary = fire_ingest.ingest_detection_files({"satdet": satdet, "ngfs": ngfs})
    assert summary["inserted"] == 1 and summary["skipped"] == 3
    assert summary["skipped_reasons"] == {"satdet:outside_missouri": 1, "satdet:no_acquisition_time": 1, "satdet:no_coordinates": 1}

    log = database.fire_origin_summary()
    run = [r for r in log["runs"] if r["job"] == "fire_ingest:satdet"][0]
    assert (run["feed"], run["input"], run["input_records"]) == ("NASA FIRMS (VIIRS/MODIS)", "satfiredetection.geojson", 4)
    assert (run["inserted"], run["skipped"]) == (1, 3) and run["finished_at"] and run["input_modified_at"]
    assert run["skipped_reasons"]["outside_missouri"] == 1
    source = [s for s in log["sources"] if s["feed"] == "NASA FIRMS (VIIRS/MODIS)"][0]
    assert source["kind"] == "satellite_feed" and source["events"] == 1
    origin = _rows(db, "SELECT run_id, provider_ref, channel FROM fire_event_origins")[0]
    assert (origin["provider_ref"], origin["channel"], origin["run_id"]) == ("in-mo", "satfiredetection.geojson", run["id"])


def test_a_dry_run_writes_no_run_records(db, tmp_path):
    satdet = _feature_file(tmp_path / "s.geojson", [])
    fire_ingest.ingest_detection_files({"satdet": satdet, "ngfs": _feature_file(tmp_path / "n.geojson", [])}, dry_run=True)
    assert database.fire_origin_summary()["runs"] == []


def test_the_origin_log_endpoint_summarises_sources_and_runs(db):
    _detection(0)
    _report(db)
    body = fires_router.admin_fire_origin_log(_admin="staff@x.org", days=7, runs=5)
    kinds = {(source["kind"], source["feed"]) for source in body["sources"]}
    assert ("satellite_feed", "NASA FIRMS (VIIRS)") in kinds and ("public_form", "Public fire report form") in kinds


def test_admin_event_lists_say_which_feed_each_event_came_through(db):
    _detection(0)
    _report(db)
    feeds = {event["source"]: event["origin_feed"] for event in database.list_fire_events(admin=True)}
    assert feeds == {"viirs": "NASA FIRMS (VIIRS)", "user_submission": "Public fire report form"}
    assert "origin_feed" not in database.list_fire_events(status="pending")[0]      # public lists never carry it
