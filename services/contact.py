"""Contact tickets and a durable, independently retried email outbox."""
from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone

from fastapi import HTTPException

from core.database import get_db_path
from services import graphics_email
from services.api_keys.store import hash_ip

logger = logging.getLogger(__name__)
CONTACT_EMAIL = "contact@showmefire.org"


@contextmanager
def connect():
    conn = sqlite3.connect(get_db_path(), timeout=15)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS contact_tickets (
                ticket_id TEXT PRIMARY KEY, request_id TEXT NOT NULL UNIQUE,
                content_hash TEXT NOT NULL, name TEXT NOT NULL, email TEXT NOT NULL,
                subject TEXT NOT NULL, message TEXT NOT NULL, ip_hash TEXT NOT NULL,
                created_at REAL NOT NULL, resolved_at REAL
            );
            CREATE INDEX IF NOT EXISTS contact_tickets_ip ON contact_tickets(ip_hash, created_at);
            CREATE TABLE IF NOT EXISTS contact_email_outbox (
                ticket_id TEXT NOT NULL, kind TEXT NOT NULL, payload TEXT NOT NULL,
                sent_at REAL, next_attempt_at REAL NOT NULL DEFAULT 0,
                first_attempt_at REAL, attempts INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (ticket_id, kind)
            );
        """)
        yield conn
        conn.commit()
    finally:
        conn.close()


def sender() -> str:
    return (os.getenv("CONTACT_EMAIL_FROM") or os.getenv("API_KEY_EMAIL_FROM")
            or os.getenv("GRAPHICS_EMAIL_FROM") or os.getenv("BULLETIN_EMAIL_FROM") or "").strip()


def configured() -> bool:
    return bool(sender() and os.getenv("RESEND_API_KEY", "").strip())


def content_hash(fields: dict) -> str:
    return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()


def existing(request_id: str, digest: str):
    with connect() as conn:
        row = conn.execute("SELECT ticket_id, content_hash FROM contact_tickets WHERE request_id = ?", (request_id,)).fetchone()
    if row and row["content_hash"] != digest:
        raise HTTPException(409, "This submission reference was already used. Please reload the page and try again.")
    return row["ticket_id"] if row else None


def email_payloads(ticket_id: str, fields: dict) -> dict:
    details = (f"Ticket: {ticket_id}\nName: {fields['name']}\nEmail: {fields['email']}\n"
               f"Subject: {fields['subject']}\n\n{fields['message']}")
    receipt = (f"Hello {fields['name']},\n\nWe received your message. Your ticket ID is {ticket_id}.\n"
               "Our team will reply by email. You can reply to this email to add details; please keep the ticket ID in the subject.\n\n"
               f"Your message:\n{details}\n\nShow Me Fire")
    result = {}
    for kind, to, reply_to, text, subject in (
        ("staff", CONTACT_EMAIL, fields["email"], details, f"[{ticket_id}] {fields['subject']}"),
        ("receipt", fields["email"], CONTACT_EMAIL, receipt, f"[{ticket_id}] We received your message"),
    ):
        result[kind] = {"from": sender(), "to": [to], "reply_to": reply_to, "subject": subject,
                        "text": text, "html": '<div style="font-family:Arial,sans-serif;white-space:pre-wrap;line-height:1.6">'
                        + html.escape(text) + "</div>"}
    return result


def create(request_id: str, fields: dict, ip: str) -> str:
    """Atomically enforce the daily cap, deduplicate, and save the ticket with both emails."""
    digest = content_hash(fields)
    now = time.time()
    with connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT ticket_id, content_hash FROM contact_tickets WHERE request_id = ?", (request_id,)).fetchone()
        if row:
            if row["content_hash"] != digest:
                raise HTTPException(409, "This submission reference was already used. Please reload the page and try again.")
            return row["ticket_id"]
        ip_digest = hash_ip(ip)
        count = conn.execute("SELECT COUNT(*) FROM contact_tickets WHERE ip_hash = ? AND created_at > ?",
                             (ip_digest, now - 86400)).fetchone()[0]
        if count >= 5:
            raise HTTPException(429, "Too many messages from this address today. Please try again tomorrow.")
        date = datetime.now(timezone.utc).strftime("%Y%m%d")
        ticket_id = f"SMF-{date}-{secrets.token_hex(6).upper()}"
        conn.execute("INSERT INTO contact_tickets (ticket_id, request_id, content_hash, name, email, subject, message, ip_hash, created_at) "
                     "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     (ticket_id, request_id, digest, fields["name"], fields["email"], fields["subject"], fields["message"], ip_digest, now))
        for kind, payload in email_payloads(ticket_id, fields).items():
            conn.execute("INSERT INTO contact_email_outbox (ticket_id, kind, payload) VALUES (?, ?, ?)",
                         (ticket_id, kind, json.dumps(payload)))
    return ticket_id


def deliver(ticket_id: str) -> bool:
    """Claim each email before sending. A crash releases its lease after two minutes.

    Retries stop before Resend's 24-hour idempotency window expires; unsent rows remain for staff review.
    Never log email bodies, addresses or provider errors that might contain them.
    """
    for kind in ("staff", "receipt"):
        now = time.time()
        with connect() as conn:
            claimed = conn.execute(
                "UPDATE contact_email_outbox SET next_attempt_at = ?, first_attempt_at = COALESCE(first_attempt_at, ?), attempts = attempts + 1 "
                "WHERE ticket_id = ? AND kind = ? AND sent_at IS NULL AND next_attempt_at <= ? "
                "AND (first_attempt_at IS NULL OR first_attempt_at > ?)",
                (now + 120, now, ticket_id, kind, now, now - 23 * 3600)).rowcount
            if not claimed:
                continue
            row = conn.execute("SELECT payload FROM contact_email_outbox WHERE ticket_id = ? AND kind = ?", (ticket_id, kind)).fetchone()
        try:
            result = graphics_email._resend_request("POST", "/emails", json.loads(row["payload"]),
                                                     idempotency_key=f"contact/{ticket_id}/{kind}")
            if not result.get("id"):
                raise RuntimeError("No email ID returned")
        except Exception:
            logger.warning("Contact email pending: ticket=%s kind=%s", ticket_id, kind)
            with connect() as conn:
                conn.execute("UPDATE contact_email_outbox SET next_attempt_at = ? WHERE ticket_id = ? AND kind = ?",
                             (time.time() + 300, ticket_id, kind))
        else:
            with connect() as conn:
                conn.execute("UPDATE contact_email_outbox SET sent_at = ? WHERE ticket_id = ? AND kind = ?", (time.time(), ticket_id, kind))
    with connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM contact_email_outbox WHERE ticket_id = ? AND sent_at IS NULL", (ticket_id,)).fetchone()[0] == 0


def maintain() -> None:
    """API scheduler: purge resolved records after 90 days and retry pending delivery."""
    now = time.time()
    with connect() as conn:
        conn.execute("DELETE FROM contact_email_outbox WHERE ticket_id IN (SELECT ticket_id FROM contact_tickets WHERE resolved_at < ?)",
                     (now - 90 * 86400,))
        conn.execute("DELETE FROM contact_tickets WHERE resolved_at < ?", (now - 90 * 86400,))
        pending = conn.execute("SELECT DISTINCT ticket_id FROM contact_email_outbox WHERE sent_at IS NULL AND next_attempt_at <= ? "
                               "AND (first_attempt_at IS NULL OR first_attempt_at > ?) LIMIT 20", (now, now - 23 * 3600)).fetchall()
        stale = conn.execute("SELECT COUNT(*) FROM contact_email_outbox WHERE sent_at IS NULL AND first_attempt_at <= ?", (now - 23 * 3600,)).fetchone()[0]
    if stale:
        logger.error("Contact outbox has %d emails requiring staff review", stale)
    if configured():
        for row in pending:
            deliver(row["ticket_id"])
