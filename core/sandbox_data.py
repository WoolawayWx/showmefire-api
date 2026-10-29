"""Building, resetting and seeding the reviewer sandbox database (see core/sandbox.py for how requests reach it).

Everything in it is synthetic: invented reporters, `example.invalid` contacts, generated placeholder photos. It is built once
into a template (`sandbox.template.db`) and the live sandbox is a copy of that, made with SQLite's backup API so a stale
`-wal`/`-shm` file can never survive a reset. Reviewers may approve, reject and edit things in it; a reset puts everything
back. A reset happens when an owner asks, or when no reviewer has used the sandbox for a while (`reset_if_idle`), never in the
middle of a review.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import struct
import threading
import time
import zlib
from pathlib import Path
from typing import Optional

from core.sandbox import (
    sandbox_db_path, sandbox_media_dir, sandbox_root, sandbox_scope, sandbox_template_path,
)

logger = logging.getLogger(__name__)

IDLE_RESET_SECONDS = 30 * 60
_lock = threading.Lock()


def sandbox_enabled() -> bool:
    import os

    return os.getenv("SANDBOX_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")


def _png(width: int, height: int, top: tuple, bottom: tuple) -> bytes:
    """A small vertical-gradient PNG, with no imaging library."""
    rows = bytearray()
    for y in range(height):
        mix = y / max(1, height - 1)
        colour = bytes(int(top[i] + (bottom[i] - top[i]) * mix) for i in range(3))
        rows += b"\x00" + colour * width

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(rows), 9)) + chunk(b"IEND", b""))


def _write_photo(name: str, top: tuple, bottom: tuple) -> dict:
    folder = sandbox_media_dir()
    folder.mkdir(parents=True, exist_ok=True)
    data = _png(640, 400, top, bottom)
    (folder / name).write_bytes(data)
    return {"stored_filename": name, "size_bytes": len(data)}


def _insert(conn: sqlite3.Connection, table: str, **values) -> int:
    columns = ", ".join(values)
    marks = ", ".join("?" for _ in values)
    return conn.execute(f"INSERT INTO {table} ({columns}) VALUES ({marks})", tuple(values.values())).lastrowid


def _seed(conn: sqlite3.Connection) -> None:
    """Demo content: a pending queue, decided items, an incident with feedback, burn-ban requests, assessments and history."""
    from core import database

    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    reports = [
        ("Grass fire along a fence line, smoke drifting east toward the county road.", "Phelps", "29161", 37.93, -91.78, 12.0, "Demo Reporter A"),
        ("Brush pile burn that got away from the landowner; the volunteer fire department is on scene.", "Texas", "29215", 37.31, -91.97, 3.5, "Demo Reporter B"),
        ("Large column of smoke near a wooded ridge, no flames visible from the road.", "Reynolds", "29179", 37.37, -90.97, None, "Demo Reporter C"),
        ("Planned pasture burn, attended, with a firebreak mowed around it.", "Howell", "29091", 36.77, -91.88, 40.0, "Demo Reporter D"),
        ("Test report: nothing is burning here, just checking the form.", "Pulaski", "29169", 37.82, -92.21, None, "Demo Reporter E"),
    ]
    palette = [((250, 140, 40), (60, 40, 30)), ((200, 200, 205), (90, 95, 100)), ((120, 120, 125), (30, 30, 35)),
               ((240, 190, 90), (110, 150, 70)), ((140, 190, 235), (70, 140, 90))]
    event_ids = []
    for index, (text, county, fips, lat, lon, acres, name) in enumerate(reports):
        event_id = _insert(
            conn, "fire_events", source="user_submission", status="pending", verification_tier="unverified", latitude=lat,
            longitude=lon, county_fips=fips, county_name=county, occurred_at=now, acres=acres, description=text,
            out_of_ordinary="", reporter_name=name, reporter_contact="demo@example.invalid", address_text=f"County Road {100 + index}, {county} County, Missouri",
            reporter_relationship="bystander", cause_category="unknown")
        event_ids.append(event_id)
        if index < 4:
            photo = _write_photo(f"demo-report-{event_id}.png", *palette[index])
            _insert(conn, "fire_event_media", event_id=event_id, stored_filename=photo["stored_filename"],
                    original_filename=f"photo-{index + 1}.png", content_type="image/png", size_bytes=photo["size_bytes"],
                    sha256="0" * 64, review_state="pending", kind="photo")
        answers = {
            "shows_fire_or_smoke": {"type": "noul", "noul": 0.04 if index == 4 else 0.93},
            "burn_type": {"type": "choice", "choice": "not_a_fire" if index == 4 else "prescribed_or_agricultural" if index == 3 else "wildfire",
                          "confidence": 0.8, "probabilities": {"wildfire": 0.1 if index in (3, 4) else 0.8, "prescribed_or_agricultural": 0.8 if index == 3 else 0.1,
                                                                "not_a_fire": 0.8 if index == 4 else 0.02, "unclear": 0.05}},
            "urgency": {"type": "score", "score": 0.2 if index == 4 else 2.1, "confidence": 0.7, "levels": 4},
            "likely_spam_or_prank": {"type": "noul", "noul": 0.9 if index == 4 else 0.04},
        }
        _insert(conn, "clef_assessments", subject_type="fire_event", subject_id=event_id, task="fire_report", input_hash=f"demo-{event_id}",
                answers_json=json.dumps(answers), model="clef", images_used=0 if index == 4 else 1, input_tokens=0, created_at=now)

    # An incident with satellite detections, a public report and public feedback.
    incident_id = _insert(conn, "fire_incidents", centroid_latitude=37.37, centroid_longitude=-90.97, first_detected_at=now, last_detected_at=now,
                          detection_count=3, county_fips="29179", county_name="Reynolds", status="active", public_slug="demo-incident-1")
    for offset in range(3):
        _insert(conn, "fire_events", source="viirs", status="pending", verification_tier="unverified", latitude=37.37 + offset * 0.002,
                longitude=-90.97, county_fips="29179", county_name="Reynolds", occurred_at=now, confidence="nominal", frp=8.0 + offset,
                satellite="N20", description="", incident_id=incident_id, cause_category="unknown")
    conn.execute("UPDATE fire_events SET incident_id = ? WHERE id = ?", (incident_id, event_ids[2]))
    for classification, note in (("confirmed_fire", "Saw flames from Highway 49."), ("controlled_burn", "Landowner said it is a planned burn."),
                                 ("unsure", "")):
        _insert(conn, "fire_incident_feedback", incident_id=incident_id, classification=classification, note=note, contact="",
                submitter_ip_hash="demo", status="pending", submission_source="mobile")

    bans = [
        ("29161", "Phelps", "issue", "Fire chief announced a county-wide ban on the radio.", "pending"),
        ("29215", "Texas", "update", "Ban extended another week.", "pending"),
        ("29179", "Reynolds", "lift", "Rain over the weekend; the commission lifted the ban.", "pending"),
        ("29091", "Howell", "issue", "Commission order posted at the courthouse.", "pending"),
    ]
    ban_ids = []
    for fips, county, kind, comment, status in bans:
        ban_ids.append(_insert(conn, "burn_ban_submissions", county_fips=fips, county_name=county, submitter_name="Demo Submitter",
                               submitter_contact="demo@example.invalid", proof_url="https://example.invalid/order", request_type=kind,
                               effective_at=now, expires_at="", status=status, submitter_comment=comment, public_note=""))
    conn.commit()

    # Some history, written through the real functions so it is shaped exactly like live history.
    database.set_fire_event_status(event_ids[3], "approved", actor="demo.staff@example.invalid", reason="Attended pasture burn; labelled prescribed.",
                                   cause_category="prescribed")
    database.set_fire_event_status(event_ids[4], "rejected", actor="demo.staff@example.invalid", reason="Test submission, no fire.")
    database.moderate_burn_ban_submission(ban_ids[3], to_status="confirmed", actor="demo.staff@example.invalid", reason="Order matches the courthouse notice.")


def build_template() -> Path:
    """(Re)build the pristine sandbox database. Slow (it creates the whole schema), so it is done once and copied."""
    from core import database

    root = sandbox_root()
    root.mkdir(parents=True, exist_ok=True)
    template = sandbox_template_path()
    for leftover in (template, Path(f"{template}-wal"), Path(f"{template}-shm")):
        leftover.unlink(missing_ok=True)
    with sandbox_scope(template):
        database.init_database()
        with sqlite3.connect(template, timeout=30) as conn:
            _seed(conn)
        with sqlite3.connect(template, timeout=30) as conn:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    return template


def reset_sandbox() -> Path:
    """Put the live sandbox back to the pristine template."""
    with _lock:
        template = sandbox_template_path()
        if not template.exists():
            build_template()
        live = sandbox_db_path()
        live.parent.mkdir(parents=True, exist_ok=True)
        source = sqlite3.connect(template, timeout=30)
        target = sqlite3.connect(live, timeout=30)
        try:
            source.backup(target)
        finally:
            target.close()
            source.close()
        _marker("dirty").unlink(missing_ok=True)
        _marker("last_reset").write_text(str(time.time()))
        return live


def ensure_sandbox() -> Path:
    """The live sandbox database, created from the template on first use. Raises FileNotFoundError if it cannot be made."""
    live = sandbox_db_path()
    if live.exists():
        return live
    reset_sandbox()
    if not live.exists():
        raise FileNotFoundError("the reviewer sandbox could not be created")
    return live


def _marker(name: str) -> Path:
    return sandbox_root() / name


def touch_activity(modified: bool) -> None:
    """Note that a reviewer just used the sandbox (and whether they changed anything)."""
    try:
        sandbox_root().mkdir(parents=True, exist_ok=True)
        _marker("last_activity").write_text(str(time.time()))
        if modified:
            _marker("dirty").write_text("1")
    except OSError:
        logger.warning("could not record sandbox activity", exc_info=True)


def seconds_idle() -> Optional[float]:
    try:
        return time.time() - float(_marker("last_activity").read_text())
    except (OSError, ValueError):
        return None


def reset_if_idle(idle_seconds: int = IDLE_RESET_SECONDS) -> bool:
    """Reset the sandbox if a reviewer changed it and nobody has used it for `idle_seconds`. Run by the scheduler."""
    if not _marker("dirty").exists():
        return False
    idle = seconds_idle()
    if idle is not None and idle < idle_seconds:
        return False
    reset_sandbox()
    return True
