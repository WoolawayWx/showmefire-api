"""QR codes for the admin tools: storage, validation, UTM tagging, SVG/PNG rendering and scan tracking (in the main showmefire.db).

A tracked code encodes a short address on our own site (<site>/q/<slug>). Scanning it lands on that route, which records the scan and sends the
visitor on to the saved destination, so the destination (and its UTM tags) can be edited after the code is printed. An untracked code encodes the
destination (or the text) itself, with any UTM tags already in the address. A scan row keeps the time, a device class, the referrer's host and a
salted hash of the visitor's IP (for the unique count only, never the address).
"""
from __future__ import annotations

import io
import os
import re
import secrets
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from core.database import get_db_path
from services.api_keys.store import hash_ip

KINDS = ("url", "text")
ERROR_LEVELS = ("L", "M", "Q", "H")
UTM_KEYS = ("utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content")
SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,39}$")
COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
SLUG_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"        # no look-alikes (i l o 0 1)
MAX_CONTENT = 1500
BOT_HINTS = ("bot", "crawl", "spider", "slurp", "preview", "facebookexternalhit", "headless", "curl/", "wget", "python-requests")
EDITABLE = ("name", "kind", "tracked", "destination", "slug", "error_level", "fg", "bg", "border", "active", *UTM_KEYS)
_lock = threading.Lock()
_ready_for: Optional[str] = None


class QRError(ValueError):
    """A problem with what was sent, worded for the admin who sent it."""


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def web_url() -> str:
    return os.getenv("PUBLIC_WEB_URL", "https://showmefire.org").rstrip("/")


@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    global _ready_for
    path = str(get_db_path())
    conn = sqlite3.connect(path, timeout=15)
    conn.row_factory = sqlite3.Row
    try:
        with _lock:
            if _ready_for != path:
                _create_tables(conn)
                _ready_for = path
        yield conn
        conn.commit()
    finally:
        conn.close()


def _create_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS qr_codes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            slug TEXT NOT NULL UNIQUE,
            name TEXT NOT NULL,
            kind TEXT NOT NULL DEFAULT 'url',
            tracked INTEGER NOT NULL DEFAULT 1,
            destination TEXT NOT NULL,
            utm_source TEXT NOT NULL DEFAULT '',
            utm_medium TEXT NOT NULL DEFAULT '',
            utm_campaign TEXT NOT NULL DEFAULT '',
            utm_term TEXT NOT NULL DEFAULT '',
            utm_content TEXT NOT NULL DEFAULT '',
            error_level TEXT NOT NULL DEFAULT 'M',
            fg TEXT NOT NULL DEFAULT '#000000',
            bg TEXT NOT NULL DEFAULT '#ffffff',
            border INTEGER NOT NULL DEFAULT 4,
            active INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL,
            created_by TEXT,
            updated_at TEXT NOT NULL,
            updated_by TEXT
        );
        CREATE TABLE IF NOT EXISTS qr_scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            qr_id INTEGER NOT NULL,
            scanned_at TEXT NOT NULL,
            device TEXT NOT NULL,
            is_bot INTEGER NOT NULL DEFAULT 0,
            referrer_host TEXT NOT NULL DEFAULT '',
            ip_hash TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_qr_scans_code ON qr_scans(qr_id, scanned_at DESC);
        """
    )


# --- validation and addresses ---------------------------------------------------------------------------

def clean_destination(kind: str, value: str) -> str:
    value = str(value or "").strip()
    if not value:
        raise QRError("enter the address or text the code should open")
    if len(value) > MAX_CONTENT:
        raise QRError(f"keep it under {MAX_CONTENT} characters")
    if kind == "url":
        parsed = urlparse(value)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise QRError("a web address must start with http:// or https://")
    return value


def clean_utm(value: Any) -> str:
    return " ".join(str(value or "").split())[:100]


def with_utm(url: str, utm: dict[str, str]) -> str:
    """Add the non-empty UTM tags to `url`, never replacing a tag the address already carries."""
    tags = {k: v for k, v in utm.items() if k in UTM_KEYS and v}
    if not tags:
        return url
    parsed = urlparse(url)
    query = parse_qsl(parsed.query, keep_blank_values=True)
    present = {k for k, _ in query}
    query += [(k, v) for k, v in tags.items() if k not in present]
    return urlunparse(parsed._replace(query=urlencode(query)))


def _utm(code: dict) -> dict[str, str]:
    return {k: code.get(k, "") for k in UTM_KEYS}


def target_url(code: dict) -> str:
    """Where a scan of a tracked code (or a visit to its short address) ends up."""
    return with_utm(code["destination"], _utm(code)) if code["kind"] == "url" else code["destination"]


def short_url(slug: str) -> str:
    return f"{web_url()}/q/{slug}"


def encoded_text(code: dict) -> str:
    """What the QR symbol itself holds."""
    if code["tracked"] and code["kind"] == "url":
        return short_url(code["slug"])
    return target_url(code)


# --- rendering ------------------------------------------------------------------------------------------

def _make(text: str, level: str, border: int):
    import qrcode
    import qrcode.constants

    qr = qrcode.QRCode(error_correction=getattr(qrcode.constants, f"ERROR_CORRECT_{level}"), box_size=10, border=border)
    qr.add_data(text)
    try:
        qr.make(fit=True)
    except Exception:
        raise QRError("that is too much to fit in one QR code, shorten it") from None
    return qr


def render_svg(text: str, *, level: str = "M", fg: str = "#000000", bg: str = "#ffffff", border: int = 4) -> str:
    """A clean vector file: one background rectangle and one path for all the dark modules, sized by the viewBox so it scales to any print size."""
    matrix = _make(text, level, 0).get_matrix()
    size = len(matrix)
    total = size + 2 * border
    cells = []
    for y, row in enumerate(matrix):
        x = 0
        while x < size:
            if row[x]:
                start = x
                while x < size and row[x]:
                    x += 1
                cells.append(f"M{start + border},{y + border}h{x - start}v1h-{x - start}z")
            else:
                x += 1
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total} {total}" width="{total * 10}" height="{total * 10}" shape-rendering="crispEdges">'
            f'<rect width="{total}" height="{total}" fill="{bg}"/><path d="{"".join(cells)}" fill="{fg}"/></svg>')


def render_png(text: str, *, level: str = "M", fg: str = "#000000", bg: str = "#ffffff", border: int = 4, scale: int = 10) -> bytes:
    from PIL import Image

    matrix = _make(text, level, 0).get_matrix()
    scale = max(1, min(scale, 40))
    total = len(matrix) + 2 * border
    image = Image.new("RGB", (total * scale, total * scale), bg)
    dark = Image.new("RGB", (scale, scale), fg)
    for y, row in enumerate(matrix):
        for x, filled in enumerate(row):
            if filled:
                image.paste(dark, ((x + border) * scale, (y + border) * scale))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def render_code(code: dict, fmt: str = "svg") -> tuple[bytes, str]:
    opts = {"level": code["error_level"], "fg": code["fg"], "bg": code["bg"], "border": code["border"]}
    text = encoded_text(code)
    if fmt == "png":
        return render_png(text, scale=30, **opts), "image/png"
    return render_svg(text, **opts).encode("utf-8"), "image/svg+xml"


# --- codes ----------------------------------------------------------------------------------------------

def _code_row(row: sqlite3.Row) -> dict:
    out = dict(row)
    out["tracked"] = bool(out["tracked"])
    out["active"] = bool(out["active"])
    out["encoded_text"] = encoded_text(out)
    out["target_url"] = target_url(out)
    return out


def _new_slug(conn: sqlite3.Connection) -> str:
    while True:
        slug = "".join(secrets.choice(SLUG_ALPHABET) for _ in range(6))
        if not conn.execute("SELECT 1 FROM qr_codes WHERE slug = ?", (slug,)).fetchone():
            return slug


def _check_fits(row: dict) -> None:
    """Refuse a code whose content cannot fit in a QR symbol at the chosen error correction, instead of failing later at download time."""
    _make(encoded_text({**row, "tracked": bool(row["tracked"])}), row["error_level"] if "error_level" in row else "M", 0)


def _validated(values: dict) -> dict:
    out = dict(values)
    if "kind" in out and out["kind"] not in KINDS:
        raise QRError("kind must be url or text")
    if "error_level" in out and out["error_level"] not in ERROR_LEVELS:
        raise QRError("error correction must be L, M, Q or H")
    for key in ("fg", "bg"):
        if key in out and not COLOR_RE.match(str(out[key])):
            raise QRError("colors must look like #1a2b3c")
    if "border" in out:
        out["border"] = int(out["border"])
        if not 0 <= out["border"] <= 10:
            raise QRError("the quiet zone must be between 0 and 10 modules")
    if "name" in out:
        out["name"] = " ".join(str(out["name"] or "").split())[:120]
        if not out["name"]:
            raise QRError("give the code a name")
    for key in UTM_KEYS:
        if key in out:
            out[key] = clean_utm(out[key])
    for flag in ("tracked", "active"):
        if flag in out:
            out[flag] = 1 if out[flag] else 0
    return out


def create_code(*, name: str, destination: str, kind: str = "url", slug: str = "", created_by: str = "", **options: Any) -> dict:
    values = _validated({"name": name, "kind": kind, **{k: v for k, v in options.items() if k in EDITABLE and v is not None}})
    values["destination"] = clean_destination(values["kind"], destination)
    if values["kind"] == "text":
        values["tracked"] = 0                      # plain text has nowhere to redirect to
    now = _now()
    with connect() as conn:
        slug = (slug or "").strip().lower()
        if slug and not SLUG_RE.match(slug):
            raise QRError("the short address can use lowercase letters, numbers and dashes (2 to 40 characters)")
        if slug and conn.execute("SELECT 1 FROM qr_codes WHERE slug = ?", (slug,)).fetchone():
            raise QRError("that short address is taken")
        values["slug"] = slug or _new_slug(conn)
        values.update(created_at=now, created_by=created_by, updated_at=now, updated_by=created_by)
        _check_fits(values)
        columns = ", ".join(values)
        cursor = conn.execute(f"INSERT INTO qr_codes ({columns}) VALUES ({', '.join('?' for _ in values)})", tuple(values.values()))
        return _code_row(conn.execute("SELECT * FROM qr_codes WHERE id = ?", (cursor.lastrowid,)).fetchone())


def get_code(code_id: int) -> Optional[dict]:
    with connect() as conn:
        row = conn.execute("SELECT * FROM qr_codes WHERE id = ?", (code_id,)).fetchone()
        return _code_row(row) if row else None


def get_code_by_slug(slug: str) -> Optional[dict]:
    with connect() as conn:
        row = conn.execute("SELECT * FROM qr_codes WHERE slug = ?", (slug.lower(),)).fetchone()
        return _code_row(row) if row else None


def list_codes() -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """SELECT c.*, COUNT(s.id) AS scan_count, MAX(s.scanned_at) AS last_scan_at
               FROM qr_codes c LEFT JOIN qr_scans s ON s.qr_id = c.id AND s.is_bot = 0
               GROUP BY c.id ORDER BY c.created_at DESC, c.id DESC"""
        ).fetchall()
        return [{**_code_row(r), "scan_count": r["scan_count"], "last_scan_at": r["last_scan_at"]} for r in rows]


def update_code(code_id: int, changes: dict, updated_by: str) -> Optional[dict]:
    changes = {k: v for k, v in changes.items() if k in EDITABLE}
    with connect() as conn:
        current = conn.execute("SELECT * FROM qr_codes WHERE id = ?", (code_id,)).fetchone()
        if current is None:
            return None
        values = _validated(changes)
        kind = values.get("kind", current["kind"])
        if "destination" in values or "kind" in values:
            values["destination"] = clean_destination(kind, values.get("destination", current["destination"]))
        if kind == "text":
            values["tracked"] = 0
        if "slug" in values:
            slug = str(values["slug"] or "").strip().lower()
            if not SLUG_RE.match(slug):
                raise QRError("the short address can use lowercase letters, numbers and dashes (2 to 40 characters)")
            if conn.execute("SELECT 1 FROM qr_codes WHERE slug = ? AND id != ?", (slug, code_id)).fetchone():
                raise QRError("that short address is taken")
            values["slug"] = slug
        if values:
            _check_fits({**dict(current), **values})
            values.update(updated_at=_now(), updated_by=updated_by)
            assignments = ", ".join(f"{k} = ?" for k in values)
            conn.execute(f"UPDATE qr_codes SET {assignments} WHERE id = ?", (*values.values(), code_id))
        return _code_row(conn.execute("SELECT * FROM qr_codes WHERE id = ?", (code_id,)).fetchone())


def delete_code(code_id: int) -> bool:
    with connect() as conn:
        conn.execute("DELETE FROM qr_scans WHERE qr_id = ?", (code_id,))
        return conn.execute("DELETE FROM qr_codes WHERE id = ?", (code_id,)).rowcount > 0


# --- scans ----------------------------------------------------------------------------------------------

def classify_device(user_agent: str) -> tuple[str, bool]:
    ua = (user_agent or "").lower()
    if not ua or any(hint in ua for hint in BOT_HINTS):
        return "bot", True
    if "iphone" in ua or "ipad" in ua or "ipod" in ua:
        return "ios", False
    if "android" in ua:
        return "android", False
    if any(token in ua for token in ("windows", "macintosh", "x11", "cros", "linux")):
        return "desktop", False
    return "other", False


def referrer_host(referrer: str) -> str:
    try:
        return (urlparse(referrer or "").hostname or "")[:100]
    except ValueError:
        return ""


def record_scan(code: dict, *, user_agent: str = "", referrer: str = "", ip: str = "") -> None:
    device, is_bot = classify_device(user_agent)
    with connect() as conn:
        conn.execute("INSERT INTO qr_scans (qr_id, scanned_at, device, is_bot, referrer_host, ip_hash) VALUES (?, ?, ?, ?, ?, ?)",
                     (code["id"], _now(), device, int(is_bot), referrer_host(referrer), hash_ip(ip) if ip and ip != "unknown" else None))


def stats(code_id: int, days: int = 30) -> dict:
    days = max(1, min(int(days), 365))
    since = (datetime.now(timezone.utc) - timedelta(days=days - 1)).strftime("%Y-%m-%dT00:00:00Z")
    with connect() as conn:
        total, unique, last = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT ip_hash), MAX(scanned_at) FROM qr_scans WHERE qr_id = ? AND is_bot = 0", (code_id,)).fetchone()
        bots = conn.execute("SELECT COUNT(*) FROM qr_scans WHERE qr_id = ? AND is_bot = 1", (code_id,)).fetchone()[0]
        per_day = {r[0]: r[1] for r in conn.execute(
            "SELECT substr(scanned_at, 1, 10), COUNT(*) FROM qr_scans WHERE qr_id = ? AND is_bot = 0 AND scanned_at >= ? GROUP BY 1", (code_id, since))}
        devices = {r[0]: r[1] for r in conn.execute(
            "SELECT device, COUNT(*) FROM qr_scans WHERE qr_id = ? AND is_bot = 0 GROUP BY 1 ORDER BY 2 DESC", (code_id,))}
        referrers = [{"host": r[0], "count": r[1]} for r in conn.execute(
            "SELECT referrer_host, COUNT(*) FROM qr_scans WHERE qr_id = ? AND is_bot = 0 AND referrer_host != '' GROUP BY 1 ORDER BY 2 DESC LIMIT 10", (code_id,))]
    today = datetime.now(timezone.utc).date()
    series = []
    for offset in range(days - 1, -1, -1):
        day = (today - timedelta(days=offset)).isoformat()
        series.append({"date": day, "count": per_day.get(day, 0)})
    return {"total": total, "unique": unique, "bots_ignored": bots, "last_scan_at": last, "days": days, "per_day": series, "devices": devices, "referrers": referrers}


def scans_csv_rows(code_id: int) -> list[list[Any]]:
    with connect() as conn:
        rows = conn.execute("SELECT scanned_at, device, referrer_host FROM qr_scans WHERE qr_id = ? AND is_bot = 0 ORDER BY scanned_at", (code_id,)).fetchall()
    return [[r["scanned_at"], r["device"], r["referrer_host"]] for r in rows]


def reset_for_tests() -> None:
    global _ready_for
    _ready_for = None
