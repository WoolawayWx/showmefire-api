"""The form builder's question model: checking a form definition and validating a visitor's answers against it.

A form is a list of fields. Every field has an `id` (stable, letters/digits/underscore), a `type` and a `label`. The website's builder
edits this list, the public page draws it, and `validate_answers` checks every submission against the same list. What is stored for a
response is a snapshot of each answer with the question's label as it was when it was sent, so editing a form later never rewrites how
an old response reads.

Field keys: id, type, label, required, help, placeholder, max_length (text/textarea), options [{value, label}] (select/radio/multiselect),
min/max (number), min_choices/max_choices (multiselect), rating_max (rating, 3-10), min/max + min_label/max_label (scale),
min_date/max_date (date, YYYY-MM-DD), section (heading to group fields), show_if {field, equals} (ask only when an earlier answer matches;
for a multiselect "matches" means "includes").
"""
from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

INPUT_TYPES = ("text", "textarea", "email", "tel", "url", "number", "select", "radio", "checkbox", "multiselect", "date", "time", "rating", "scale")
DISPLAY_TYPES = ("heading", "paragraph")                  # shown on the form, never answered
TYPES = INPUT_TYPES + DISPLAY_TYPES
CHOICE_TYPES = ("select", "radio", "multiselect")
ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,40}$")
SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{1,60}$")
EMAIL_PATTERN = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,255}\.[^@\s]{2,}$")
DEFAULT_MAX = {"text": 200, "email": 200, "tel": 40, "url": 300, "textarea": 2000, "select": 100, "radio": 100}
MAX_FIELDS = 100
MAX_OPTIONS = 100
MAX_BODY_CHARS = 60_000
MAX_LABEL = 300
MAX_HELP = 1000


class FormError(ValueError):
    """The form definition is wrong (shown to the staff member building it, never to a visitor)."""


def _text(value: Any) -> str:
    return " ".join(str("" if value is None else value).split())


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def clean_fields(raw: Any) -> list[dict]:
    """Check a field list from the builder and return it tidied (unknown keys dropped, text trimmed). Raises FormError."""
    if not isinstance(raw, list):
        raise FormError("fields must be a list")
    if len(raw) > MAX_FIELDS:
        raise FormError(f"a form can have at most {MAX_FIELDS} questions")
    fields: list[dict] = []
    answerable: dict[str, dict] = {}
    for position, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise FormError(f"question {position} is not an object")
        fid, kind = str(item.get("id") or ""), item.get("type")
        if not ID_PATTERN.match(fid) or any(f["id"] == fid for f in fields):
            raise FormError(f"question {position}: id {fid!r} is missing, invalid or repeated")
        if kind not in TYPES:
            raise FormError(f"question {position}: type must be one of {', '.join(TYPES)}")
        label = _text(item.get("label")) if kind != "paragraph" else " ".join(str(item.get("label") or "").split())
        if not label:
            raise FormError(f"question {position}: needs {'text' if kind in DISPLAY_TYPES else 'a label'}")
        if len(label) > MAX_LABEL * (3 if kind == "paragraph" else 1):
            raise FormError(f"question {position}: the label is too long")
        field: dict[str, Any] = {"id": fid, "type": kind, "label": label}
        if _text(item.get("section")):
            field["section"] = _text(item["section"])[:120]
        if kind in INPUT_TYPES:
            if item.get("required"):
                field["required"] = True
            if _text(item.get("help")):
                field["help"] = _text(item["help"])[:MAX_HELP]
        if kind in ("text", "textarea", "email", "tel", "url", "number") and _text(item.get("placeholder")):
            field["placeholder"] = _text(item["placeholder"])[:200]
        if kind in ("text", "textarea", "email", "tel", "url") and item.get("max_length") not in (None, ""):
            limit = item["max_length"]
            if not _is_int(limit) or not 1 <= limit <= 10_000:
                raise FormError(f"question {position}: max_length must be a whole number from 1 to 10000")
            field["max_length"] = limit
        if kind in CHOICE_TYPES:
            options, values = [], set()
            for option in item.get("options") or []:
                option = option if isinstance(option, dict) else {}
                value, option_label = _text(option.get("value")), _text(option.get("label"))
                if not value or not option_label or len(value) > 100 or len(option_label) > 200:
                    raise FormError(f"question {position}: every option needs a value and a label")
                if value in values:
                    raise FormError(f"question {position}: option value {value!r} is repeated")
                values.add(value)
                options.append({"value": value, "label": option_label})
            if not options:
                raise FormError(f"question {position}: needs at least one option")
            if len(options) > MAX_OPTIONS:
                raise FormError(f"question {position}: too many options")
            field["options"] = options
        if kind == "multiselect":
            for key in ("min_choices", "max_choices"):
                if item.get(key) not in (None, ""):
                    if not _is_int(item[key]) or not 0 <= item[key] <= MAX_OPTIONS:
                        raise FormError(f"question {position}: {key} must be a whole number")
                    field[key] = item[key]
            if field.get("min_choices", 0) > field.get("max_choices", MAX_OPTIONS) or field.get("min_choices", 0) > len(field["options"]):
                raise FormError(f"question {position}: the choice limits cannot be met")
        if kind == "number":
            for key in ("min", "max"):
                if item.get(key) not in (None, ""):
                    if not isinstance(item[key], (int, float)) or isinstance(item[key], bool):
                        raise FormError(f"question {position}: {key} must be a number")
                    field[key] = item[key]
            if "min" in field and "max" in field and field["min"] > field["max"]:
                raise FormError(f"question {position}: min is greater than max")
        if kind == "rating":
            top = item.get("rating_max", 5)
            if not _is_int(top) or not 3 <= top <= 10:
                raise FormError(f"question {position}: rating_max must be a whole number from 3 to 10")
            field["rating_max"] = top
        if kind == "scale":
            low, high = item.get("min", 1), item.get("max", 10)
            if not (_is_int(low) and _is_int(high)) or not 0 <= low < high <= 20:
                raise FormError(f"question {position}: a scale needs whole-number min and max (0-20, min below max)")
            field["min"], field["max"] = low, high
            for key in ("min_label", "max_label"):
                if _text(item.get(key)):
                    field[key] = _text(item[key])[:60]
        if kind == "date":
            for key in ("min_date", "max_date"):
                if _text(item.get(key)):
                    try:
                        date.fromisoformat(_text(item[key]))
                    except ValueError:
                        raise FormError(f"question {position}: {key} must look like 2026-01-31") from None
                    field[key] = _text(item[key])
            if "min_date" in field and "max_date" in field and field["min_date"] > field["max_date"]:
                raise FormError(f"question {position}: min_date is after max_date")
        show_if = item.get("show_if")
        if show_if:
            target = answerable.get(str(show_if.get("field") if isinstance(show_if, dict) else ""))
            if target is None:
                raise FormError(f"question {position}: show_if must refer to an earlier question that is answered")
            field["show_if"] = {"field": target["id"], "equals": _text(show_if.get("equals"))}
        fields.append(field)
        if kind in INPUT_TYPES:
            answerable[fid] = field
    if not any(f["type"] in INPUT_TYPES for f in fields):
        raise FormError("add at least one question")
    return fields


def normalize_slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", str(value or "").lower()).strip("-")[:60].strip("-")


def check_slug(slug: str) -> str:
    if not SLUG_PATTERN.match(slug or ""):
        raise FormError("the address can use lowercase letters, numbers and dashes (2-61 characters)")
    return slug


def _matches(field: dict, answers: dict) -> bool:
    show_if = field.get("show_if")
    if not show_if:
        return True
    given, wanted = answers.get(show_if["field"]), str(show_if.get("equals", "")).strip().lower()
    if isinstance(given, list):
        return wanted in {str(v).strip().lower() for v in given}
    return str(given if given is not None else "").strip().lower() == wanted


def visible_fields(fields: list[dict], raw: dict) -> list[dict]:
    """The questions a visitor sees given their answers (a question that depends on a hidden one is hidden too)."""
    shown: list[dict] = []
    hidden: set[str] = set()
    for field in fields:
        show_if = field.get("show_if")
        if (show_if and show_if["field"] in hidden) or not _matches(field, raw):
            hidden.add(field["id"])
            continue
        shown.append(field)
    return shown


def validate_answers(raw: Any, fields: list[dict]) -> tuple[list[dict], dict[str, str]]:
    """(answers, errors). `answers` is the snapshot to store: [{id, label, type, value[, display]}] in form order for the questions that
    were shown and answered. `errors` maps a field id to a message; when it is not empty nothing should be stored."""
    if not isinstance(raw, dict):
        return [], {"_form": "The response was not understood."}
    if len(json.dumps(raw, default=str)) > MAX_BODY_CHARS:
        return [], {"_form": "That is too much text. Please shorten your answers."}
    errors: dict[str, str] = {}
    answers: list[dict] = []
    for field in visible_fields(fields, raw):
        fid, kind, label = field["id"], field["type"], field["label"]
        if kind in DISPLAY_TYPES:
            continue
        value, required = raw.get(fid), bool(field.get("required"))
        entry: dict[str, Any] = {"id": fid, "label": label, "type": kind}
        if kind == "checkbox":
            checked = value is True or str(value).lower() in ("true", "on", "1", "yes")
            if required and not checked:
                errors[fid] = "You need to agree to continue."
            answers.append({**entry, "value": checked})
            continue
        if kind == "multiselect":
            chosen = value if isinstance(value, list) else ([] if value in (None, "") else [value])
            chosen = [str(v).strip() for v in chosen]
            known = {o["value"]: o["label"] for o in field["options"]}
            if len(set(chosen)) != len(chosen) or any(v not in known for v in chosen):
                errors[fid] = "Choose from the options shown."
                continue
            if not chosen:
                if required:
                    errors[fid] = "Choose at least one."
                continue
            low, high = field.get("min_choices", 0), field.get("max_choices", MAX_OPTIONS)
            if len(chosen) < low:
                errors[fid] = f"Choose at least {low}."
            elif len(chosen) > high:
                errors[fid] = f"Choose no more than {high}."
            else:
                answers.append({**entry, "value": chosen, "display": [known[v] for v in chosen]})
            continue
        text = (str(value).strip() if kind == "textarea" else _text(value)) if value is not None else ""
        if not text:
            if required:
                errors[fid] = "This is required."
            continue
        if kind in ("number", "rating", "scale"):
            try:
                number = float(text)
            except ValueError:
                errors[fid] = "Enter a number." if kind == "number" else "Choose one of the values."
                continue
            if number != number or number in (float("inf"), float("-inf")):
                errors[fid] = "Enter a number."
                continue
            if kind == "number":
                low, high = field.get("min"), field.get("max")
            elif kind == "rating":
                low, high = 1, field["rating_max"]
            else:
                low, high = field["min"], field["max"]
            if kind != "number" and number != int(number):
                errors[fid] = "Choose one of the values."
                continue
            if (low is not None and number < low) or (high is not None and number > high):
                errors[fid] = (f"Enter a number between {low if low is not None else '-'} and {high if high is not None else '-'}."
                               if kind == "number" else "Choose one of the values.")
                continue
            answers.append({**entry, "value": int(number) if number == int(number) else number})
            continue
        limit = int(field.get("max_length") or DEFAULT_MAX.get(kind, 200))
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
        elif kind == "date":
            try:
                when = date.fromisoformat(text)
            except ValueError:
                errors[fid] = "Enter a date."
                continue
            if (field.get("min_date") and text < field["min_date"]) or (field.get("max_date") and text > field["max_date"]):
                errors[fid] = "That date is outside the allowed range."
                continue
            text = when.isoformat()
        elif kind == "time":
            clock = re.match(r"^(\d{1,2}):(\d{2})(?::\d{2})?$", text)
            if not clock or int(clock.group(1)) > 23 or int(clock.group(2)) > 59:
                errors[fid] = "Enter a time."
                continue
            text = f"{int(clock.group(1)):02d}:{clock.group(2)}"
        elif kind in ("select", "radio"):
            match = next((o for o in field["options"] if o["value"] == text), None)
            if match is None:
                errors[fid] = "Choose one of the options."
                continue
            answers.append({**entry, "value": text, "display": match["label"]})
            continue
        answers.append({**entry, "value": text})
    return answers, errors


def answer_text(answer: dict) -> str:
    """One answer as plain text (for the CSV export and the staff alert)."""
    if answer.get("type") == "checkbox":
        return "Yes" if answer.get("value") else "No"
    shown = answer.get("display", answer.get("value"))
    if isinstance(shown, list):
        return "; ".join(str(v) for v in shown)
    return "" if shown is None else str(shown)
