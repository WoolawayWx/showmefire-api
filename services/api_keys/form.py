"""The API key application form: loaded from core/api_key_form.json and validated against it.

The file is the one place the questions live. The website draws the form from `GET /v1/api-keys/form` and the API checks every
submission against the same list, so adding a question means editing the JSON only. What is stored for a request is a snapshot of
each answer with the question's label as it was at the time, so later edits to the form never change how an old request reads.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional

FORM_PATH = Path(__file__).resolve().parent.parent.parent / "core" / "api_key_form.json"
TYPES = ("text", "email", "tel", "url", "textarea", "select", "radio", "checkbox", "number")
REQUIRED_IDS = ("name", "email", "agency")      # used for the review list and the emails
ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
EMAIL_PATTERN = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s]{2,}$")
DEFAULT_MAX = {"text": 200, "email": 200, "tel": 40, "url": 300, "textarea": 2000, "select": 100, "radio": 100}
MAX_BODY_CHARS = 20_000


class FormError(ValueError):
    """The form definition itself is wrong (caught by the tests and at start-up, never by a visitor)."""


def check_definition(form: dict) -> dict:
    fields = form.get("fields")
    if not isinstance(fields, list) or not fields:
        raise FormError("the form has no fields")
    seen: set[str] = set()
    for field in fields:
        fid = field.get("id", "")
        if not ID_PATTERN.match(str(fid)) or fid in seen:
            raise FormError(f"field id {fid!r} is missing, invalid or repeated")
        seen.add(fid)
        if field.get("type") not in TYPES:
            raise FormError(f"field {fid}: type must be one of {TYPES}")
        if not str(field.get("label") or "").strip():
            raise FormError(f"field {fid}: needs a label")
        if field["type"] in ("select", "radio"):
            options = field.get("options") or []
            if not options or any(not o.get("value") or not o.get("label") for o in options):
                raise FormError(f"field {fid}: needs options with a value and a label")
        show_if = field.get("show_if")
        if show_if and show_if.get("field") not in seen:
            raise FormError(f"field {fid}: show_if must refer to an earlier field")
    for needed in REQUIRED_IDS:
        field = next((f for f in fields if f["id"] == needed), None)
        if field is None or not field.get("required"):
            raise FormError(f"the form must keep a required {needed!r} field")
    if next(f for f in fields if f["id"] == "email")["type"] != "email":
        raise FormError("the 'email' field must have type email")
    return form


def load_form(path: Path = FORM_PATH) -> dict:
    return check_definition(json.loads(Path(path).read_text(encoding="utf-8")))


def public_form(form: Optional[dict] = None) -> dict:
    """The definition as the website needs it (everything except the editing notes)."""
    form = form or load_form()
    return {key: value for key, value in form.items() if not key.startswith("_")}


def _visible(field: dict, answers: dict) -> bool:
    show_if = field.get("show_if")
    if not show_if:
        return True
    return str(answers.get(show_if["field"], "")).strip().lower() == str(show_if.get("equals", "")).strip().lower()


def validate_answers(raw: Any, form: Optional[dict] = None) -> tuple[list[dict], dict[str, str]]:
    """(answers, errors). `answers` is the snapshot to store: [{id, label, type, value}] in form order, for the questions that were
    asked and answered. `errors` maps a field id to a message; when it is not empty nothing should be stored."""
    form = form or load_form()
    if not isinstance(raw, dict):
        return [], {"_form": "The request was not understood."}
    if len(json.dumps(raw, default=str)) > MAX_BODY_CHARS:
        return [], {"_form": "That is too much text. Please shorten your answers."}
    errors: dict[str, str] = {}
    answers: list[dict] = []
    for field in form["fields"]:
        if not _visible(field, raw):
            continue
        fid, kind, label = field["id"], field["type"], field["label"]
        value = raw.get(fid)
        required = bool(field.get("required"))
        if kind == "checkbox":
            checked = value is True or str(value).lower() in ("true", "on", "1", "yes")
            if required and not checked:
                errors[fid] = "You need to agree to continue."
            answers.append({"id": fid, "label": label, "type": kind, "value": checked})
            continue
        text = " ".join(str(value).split()) if kind != "textarea" and value is not None else (str(value).strip() if value is not None else "")
        if not text:
            if required:
                errors[fid] = "This is required."
            continue
        limit = int(field.get("max_length") or DEFAULT_MAX.get(kind, 200))
        if kind == "number":
            try:
                number = float(text)
            except ValueError:
                errors[fid] = "Enter a number."
                continue
            if ("min" in field and number < field["min"]) or ("max" in field and number > field["max"]):
                errors[fid] = f"Enter a number between {field.get('min', '-')} and {field.get('max', '-')}."
                continue
            answers.append({"id": fid, "label": label, "type": kind, "value": number})
            continue
        if len(text) > limit:
            errors[fid] = f"Please keep this under {limit} characters."
            continue
        if kind == "email":
            text = text.lower()
            if not EMAIL_PATTERN.match(text):
                errors[fid] = "Enter a valid email address."
                continue
        elif kind == "url":
            if not re.match(r"^https?://[^\s/$.?#][^\s]*$", text, re.IGNORECASE):
                errors[fid] = "Enter a web address starting with http:// or https://."
                continue
        elif kind == "tel":
            if not (7 <= len(re.sub(r"\D", "", text)) <= 15) or re.search(r"[^\d\s()+\-.x#ext]", text, re.IGNORECASE):
                errors[fid] = "Enter a phone number."
                continue
        elif kind in ("select", "radio"):
            match = next((o for o in field["options"] if o["value"] == text), None)
            if match is None:
                errors[fid] = "Choose one of the options."
                continue
            answers.append({"id": fid, "label": label, "type": kind, "value": text, "display": match["label"]})
            continue
        answers.append({"id": fid, "label": label, "type": kind, "value": text})
    return answers, errors


def answer_value(answers: list[dict], field_id: str) -> str:
    for answer in answers:
        if answer["id"] == field_id:
            return str(answer.get("display") or answer["value"])
    return ""
