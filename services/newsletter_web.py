"""The newsletter web archive: every sent newsletter can be read in a browser ("view in browser", and an archive page).

Which bulletins are public: status `sent`, not hidden by an admin (`web_hidden`), sent to a mailing list (a send to hand-picked
direct recipients has no list and stays private), and not on a private list (the Operations Briefing Packet).

What the web copy is NOT: the email. The stored HTML is cleaned before it leaves the API:
  * Resend merge tags ({{{contact.manage_url}}}, {{{RESEND_UNSUBSCRIBE_URL}}}, ...) and every link to a manage-preferences or
    unsubscribe address are removed, so no personal link ever appears on a public page;
  * only a safe set of tags, attributes, CSS properties and URL schemes survives (no scripts, forms, iframes, event handlers,
    javascript: links), because the site draws it as written.
PDF attachments are never exposed. The website shows the cleaned HTML inside a sandboxed frame, so the email's own styles cannot
leak into the page either.
"""
from __future__ import annotations

import re
import secrets
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import bleach
from bleach.css_sanitizer import CSSSanitizer
from bs4 import BeautifulSoup

from core.database import get_db_path
from services import audit

PRIVATE_LISTS = {"show-me-fire-briefing-packet"}
UPLOAD_LIST = "show-me-fire-newsletter"   # uploaded issues are filed under the main newsletter
PDF_MAX_BYTES = 25 * 1024 * 1024
HTML_MAX_BYTES = 2 * 1024 * 1024
LIST_LABELS = {
    "show-me-fire-newsletter": "Show Me Fire Newsletter",
    "fire-weather-forecasts": "Fire Weather Forecasts",
}

ALLOWED_TAGS = frozenset({
    "a", "b", "strong", "i", "em", "u", "s", "small", "sub", "sup", "br", "hr", "p", "div", "span", "center", "blockquote",
    "pre", "code", "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "img", "figure", "figcaption",
    "table", "thead", "tbody", "tfoot", "tr", "td", "th", "caption", "colgroup", "col",
})
_CELL = ["colspan", "rowspan", "align", "valign", "width", "height", "bgcolor", "style"]
ALLOWED_ATTRIBUTES = {
    "a": ["href", "title"],
    "img": ["src", "alt", "width", "height", "style", "align"],
    "table": ["width", "align", "bgcolor", "cellpadding", "cellspacing", "border", "style", "role"],
    "td": _CELL, "th": _CELL, "tr": ["style", "align", "valign", "bgcolor"],
    "col": ["width", "span"], "colgroup": ["span"],
    "p": ["style", "align"], "div": ["style", "align"], "span": ["style"], "center": [],
    "h1": ["style", "align"], "h2": ["style", "align"], "h3": ["style", "align"], "h4": ["style", "align"],
    "ul": ["style"], "ol": ["style", "start"], "li": ["style"], "blockquote": ["style"],
}
ALLOWED_CSS = [
    "color", "background-color", "font-size", "font-weight", "font-style", "font-family", "line-height", "text-align",
    "text-decoration", "letter-spacing", "text-transform", "margin", "margin-top", "margin-bottom", "margin-left",
    "margin-right", "padding", "padding-top", "padding-bottom", "padding-left", "padding-right", "border", "border-top",
    "border-bottom", "border-left", "border-right", "border-radius", "border-collapse", "border-color", "border-width",
    "border-style", "width", "max-width", "min-width", "height", "vertical-align", "display",
]
_CSS = CSSSanitizer(allowed_css_properties=ALLOWED_CSS)
MERGE_TAG = re.compile(r"\{\{\{?[^{}]*\}?\}\}")
PERSONAL_LINK = re.compile(r"unsubscribe|manage[_\- ]?(url|preferences|your)|preferences|\{\{", re.IGNORECASE)
_PUNCTUATION_ONLY = re.compile(r"^[\s|·•\-–—:;,.]*$")


def _iso(value: Optional[str]) -> Optional[str]:
    return str(value).replace(" ", "T") + "Z" if value else None


def sanitize_html(raw: str) -> str:
    """The newsletter's HTML as safe, personal-link-free markup for a public page."""
    soup = BeautifulSoup(raw or "", "lxml")
    # Elements whose CONTENT is code or markup, not words: remove them whole so nothing of them is left as visible text.
    for element in soup(["script", "style", "iframe", "object", "embed", "form", "noscript", "template", "title", "head"]):
        element.decompose()
    for anchor in list(soup.find_all("a")):
        href = anchor.get("href", "")
        if PERSONAL_LINK.search(href) or PERSONAL_LINK.search(anchor.get_text(" ")):
            parent = anchor.parent
            anchor.decompose()
            # A "Manage preferences | Unsubscribe" line is left empty (or just punctuation): drop it too.
            while parent is not None and parent.name in ("p", "div", "span", "center") and _PUNCTUATION_ONLY.match(parent.get_text(" ")):
                grandparent = parent.parent
                parent.decompose()
                parent = grandparent
    body = soup.body.decode_contents() if soup.body else str(soup)
    body = MERGE_TAG.sub("", body)
    cleaned = bleach.clean(body, tags=ALLOWED_TAGS, attributes=ALLOWED_ATTRIBUTES, protocols=["https", "http", "mailto"],
                           css_sanitizer=_CSS, strip=True, strip_comments=True)
    # Images are https only (no tracking over plain http, no data: URLs).
    fixed = BeautifulSoup(cleaned, "lxml")
    for image in fixed.find_all("img"):
        if not str(image.get("src", "")).lower().startswith("https://"):
            image.decompose()
    for anchor in fixed.find_all("a"):
        anchor["rel"] = "noopener noreferrer nofollow"
        anchor["target"] = "_blank"
    return fixed.body.decode_contents() if fixed.body else cleaned


def excerpt(cleaned_html: str, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", BeautifulSoup(cleaned_html, "lxml").get_text(" ")).strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0].rstrip(",;:.") + "…"


def is_public_list(bulletin: Dict[str, Any]) -> bool:
    """Would this bulletin be on the web archive once sent (ignoring a later admin hide)?"""
    list_id = str(bulletin.get("audience_list_id") or "")
    return bool(list_id) and list_id not in PRIVATE_LISTS


def web_url(bulletin: Dict[str, Any]) -> Optional[str]:
    """The address of the bulletin's web copy for the email's "view in your browser" line, or None when it won't be public."""
    import os

    if not is_public_list(bulletin):
        return None
    return f"{os.getenv('PUBLIC_WEB_URL', 'https://showmefire.org').rstrip('/')}/comms/newsletters/{bulletin['id']}"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(get_db_path(), timeout=15)
    conn.row_factory = sqlite3.Row
    return conn


_PUBLIC_WHERE = ("status = 'sent' AND web_hidden = 0 AND audience_list_id != '' AND audience_list_id NOT IN ({private})"
                 .format(private=",".join("?" for _ in PRIVATE_LISTS)))


def _kind(row: sqlite3.Row) -> str:
    return {"upload_pdf": "pdf", "upload_html": "html"}.get(row["source"], "email")


def _summary_text(row: sqlite3.Row, cleaned: str) -> str:
    """The description shown in the list: what the admin wrote for an upload, else the start of the text."""
    if row["source"] != "email" and (row["text_body"] or "").strip():
        return row["text_body"].strip()
    return excerpt(cleaned) if cleaned else ""


def _public_summary(row: sqlite3.Row) -> Dict[str, Any]:
    cleaned = sanitize_html(row["html_body"]) if row["html_body"] else ""
    return {"id": row["id"], "subject": row["subject"], "sentAt": _iso(row["sent_at"]), "kind": _kind(row),
            "list": LIST_LABELS.get(row["audience_list_id"], "Show Me Fire"), "excerpt": _summary_text(row, cleaned)}


def list_public(limit: int = 20, offset: int = 0) -> Tuple[List[Dict[str, Any]], int]:
    limit = max(1, min(int(limit), 50))
    offset = max(0, int(offset))
    conn = _connect()
    try:
        total = conn.execute(f"SELECT COUNT(*) FROM bulletins WHERE {_PUBLIC_WHERE}", tuple(PRIVATE_LISTS)).fetchone()[0]
        rows = conn.execute(
            f"SELECT * FROM bulletins WHERE {_PUBLIC_WHERE} ORDER BY sent_at DESC, id DESC LIMIT ? OFFSET ?",
            (*PRIVATE_LISTS, limit, offset)).fetchall()
    finally:
        conn.close()
    return [_public_summary(row) for row in rows], total


def get_public(bulletin_id: int) -> Optional[Dict[str, Any]]:
    """One newsletter with its cleaned HTML, plus the ids of the one before and after it. None if it isn't public."""
    conn = _connect()
    try:
        row = conn.execute(f"SELECT * FROM bulletins WHERE id = ? AND {_PUBLIC_WHERE}", (bulletin_id, *PRIVATE_LISTS)).fetchone()
        if row is None:
            return None
        newer = conn.execute(f"SELECT id, subject FROM bulletins WHERE {_PUBLIC_WHERE} AND (sent_at > ? OR (sent_at = ? AND id > ?)) "
                             "ORDER BY sent_at ASC, id ASC LIMIT 1", (*PRIVATE_LISTS, row["sent_at"], row["sent_at"], row["id"])).fetchone()
        older = conn.execute(f"SELECT id, subject FROM bulletins WHERE {_PUBLIC_WHERE} AND (sent_at < ? OR (sent_at = ? AND id < ?)) "
                             "ORDER BY sent_at DESC, id DESC LIMIT 1", (*PRIVATE_LISTS, row["sent_at"], row["sent_at"], row["id"])).fetchone()
    finally:
        conn.close()
    cleaned = sanitize_html(row["html_body"]) if row["html_body"] else ""
    kind = _kind(row)
    return {"id": row["id"], "subject": row["subject"], "sentAt": _iso(row["sent_at"]), "kind": kind,
            "list": LIST_LABELS.get(row["audience_list_id"], "Show Me Fire"), "excerpt": _summary_text(row, cleaned), "html": cleaned,
            "pdfUrl": f"/api/newsletters/{row['id']}/pdf" if kind == "pdf" else None,
            "pdfName": row["upload_original_name"] if kind == "pdf" else None,
            "newer": {"id": newer["id"], "subject": newer["subject"]} if newer else None,
            "older": {"id": older["id"], "subject": older["subject"]} if older else None}


# ----------------------------------------------------------------------------------------------------------- admin

def list_admin() -> List[Dict[str, Any]]:
    """Every sent newsletter with whether it is on the web, and why not when it isn't."""
    conn = _connect()
    try:
        rows = conn.execute("SELECT id, subject, audience_list_id, sent_at, web_hidden, source FROM bulletins WHERE status = 'sent' "
                            "ORDER BY sent_at DESC, id DESC").fetchall()
    finally:
        conn.close()
    result = []
    for row in rows:
        list_id = row["audience_list_id"]
        if row["web_hidden"]:
            state, reason = "hidden", "Hidden by an admin"
        elif not list_id:
            state, reason = "private", "Sent to chosen recipients, not a mailing list"
        elif list_id in PRIVATE_LISTS:
            state, reason = "private", "Operations briefing packets are never public"
        else:
            state, reason = "public", ""
        result.append({"id": row["id"], "subject": row["subject"], "list": LIST_LABELS.get(list_id, list_id or "Direct recipients"),
                       "sentAt": _iso(row["sent_at"]), "state": state, "reason": reason, "hidden": bool(row["web_hidden"]),
                       "kind": _kind(row), "uploaded": row["source"] != "email"})
    return result


def set_hidden(bulletin_id: int, hidden: bool, actor: str) -> Dict[str, Any]:
    conn = _connect()
    try:
        row = conn.execute("SELECT id, subject, status, web_hidden FROM bulletins WHERE id = ?", (bulletin_id,)).fetchone()
        if row is None or row["status"] != "sent":
            raise LookupError("newsletter not found")
        conn.execute("UPDATE bulletins SET web_hidden = ? WHERE id = ?", (1 if hidden else 0, bulletin_id))
        audit.record(conn, action="newsletter_hide" if hidden else "newsletter_show", subject_type="newsletter", subject_id=bulletin_id,
                     actor=actor, before={"hidden": bool(row["web_hidden"])}, after={"hidden": hidden, "subject": row["subject"]})
        conn.commit()
    finally:
        conn.close()
    return next(item for item in list_admin() if item["id"] == bulletin_id)


# --------------------------------------------------------------------------------------------------------- uploads

class UploadError(ValueError):
    """A rejected upload; the message is safe to show the admin."""


def _upload_dir() -> Path:
    return get_db_path().parent / "newsletter_uploads"


def create_upload(*, title: str, issue_date: str, summary: str, filename: str, content: bytes, actor: str) -> Dict[str, Any]:
    """File an already-written newsletter (a PDF or an HTML file) in the web archive, dated `issue_date`. Never emailed."""
    title = re.sub(r"\s+", " ", title or "").strip()
    summary = re.sub(r"\s+", " ", summary or "").strip()
    if not title:
        raise UploadError("Add a title.")
    if len(title) > 255:
        raise UploadError("The title must be 255 characters or fewer.")
    if len(summary) > 600:
        raise UploadError("The description must be 600 characters or fewer.")
    try:
        issued = date.fromisoformat((issue_date or "").strip())
    except ValueError:
        raise UploadError("Choose the date this issue was published.") from None
    if issued > datetime.now(timezone.utc).date():
        raise UploadError("The issue date cannot be in the future.")
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in (filename or "") else ""
    if not content:
        raise UploadError("That file is empty.")

    path: Optional[Path] = None
    if suffix == "pdf":
        if len(content) > PDF_MAX_BYTES:
            raise UploadError("PDFs must be 25 MB or smaller.")
        if not content.lstrip()[:5] == b"%PDF-":
            raise UploadError("That file is not a valid PDF.")
        source, html = "upload_pdf", ""
        _upload_dir().mkdir(parents=True, exist_ok=True)
        path = _upload_dir() / f"{secrets.token_urlsafe(12)}.pdf"
        path.write_bytes(content)
    elif suffix in ("html", "htm"):
        if len(content) > HTML_MAX_BYTES:
            raise UploadError("HTML files must be 2 MB or smaller.")
        try:
            html = content.decode("utf-8-sig")
        except UnicodeDecodeError:
            html = content.decode("latin-1")
        if not excerpt(sanitize_html(html), 50):
            raise UploadError("That HTML file has no readable text once scripts and unsafe markup are removed.")
        source = "upload_html"
    else:
        raise UploadError("Upload a PDF or an HTML file.")

    safe_name = re.sub(r"[^A-Za-z0-9._ -]+", "_", filename.rsplit("/", 1)[-1])[:120]
    conn = _connect()
    try:
        cursor = conn.execute(
            "INSERT INTO bulletins (subject, html_body, text_body, audience_list_id, status, sent_at, source, upload_path, "
            "upload_original_name) VALUES (?, ?, ?, ?, 'sent', ?, ?, ?, ?)",
            (title, html, summary, UPLOAD_LIST, f"{issued.isoformat()} 12:00:00", source, str(path) if path else None, safe_name))
        new_id = cursor.lastrowid
        audit.record(conn, action="newsletter_upload", subject_type="newsletter", subject_id=new_id, actor=actor,
                     after={"subject": title, "kind": "pdf" if path else "html", "file": safe_name, "issueDate": issued.isoformat()})
        conn.commit()
    except Exception:
        conn.rollback()
        if path:
            path.unlink(missing_ok=True)
        raise
    finally:
        conn.close()
    return next(item for item in list_admin() if item["id"] == new_id)


def delete_upload(bulletin_id: int, actor: str) -> None:
    """Remove an uploaded issue (and its PDF). Newsletters that were emailed can only be hidden, never deleted."""
    conn = _connect()
    try:
        row = conn.execute("SELECT id, subject, source, upload_path FROM bulletins WHERE id = ?", (bulletin_id,)).fetchone()
        if row is None or row["source"] == "email":
            raise LookupError("upload not found")
        audit.record(conn, action="newsletter_delete", subject_type="newsletter", subject_id=bulletin_id, actor=actor,
                     before={"subject": row["subject"], "source": row["source"]})
        conn.execute("DELETE FROM bulletins WHERE id = ?", (bulletin_id,))
        conn.commit()
    finally:
        conn.close()
    if row["upload_path"]:
        Path(row["upload_path"]).unlink(missing_ok=True)


def public_pdf(bulletin_id: int) -> Optional[Tuple[Path, str]]:
    """The file and download name of a public PDF issue, or None."""
    conn = _connect()
    try:
        row = conn.execute(f"SELECT upload_path, upload_original_name, subject FROM bulletins WHERE id = ? AND source = 'upload_pdf' "
                           f"AND {_PUBLIC_WHERE}", (bulletin_id, *PRIVATE_LISTS)).fetchone()
    finally:
        conn.close()
    if row is None or not row["upload_path"] or not Path(row["upload_path"]).is_file():
        return None
    return Path(row["upload_path"]), row["upload_original_name"] or "newsletter.pdf"
