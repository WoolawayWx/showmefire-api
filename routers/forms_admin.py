"""Admin side of the form builder: create, edit, publish and delete forms, read their responses, export them as CSV.

The prefix is /admin/form-builder (not /admin/forms) so the website's /admin/forms pages are not shadowed by it."""
from __future__ import annotations

import csv
import io
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field

from core.security import require_admin
from services.forms import schema, store

router = APIRouter(prefix="/admin/form-builder", tags=["forms-admin"])


class CreateBody(BaseModel):
    title: str = Field(..., max_length=200)
    slug: str = Field("", max_length=61)
    fields: Optional[list[dict[str, Any]]] = None


class UpdateBody(BaseModel):
    """Only the keys sent are changed (`max_responses` and `closes_at` accept null to clear them)."""
    title: Optional[str] = None
    slug: Optional[str] = None
    status: Optional[str] = None
    intro: Optional[str] = None
    submit_label: Optional[str] = None
    success_message: Optional[str] = None
    require_captcha: Optional[bool] = None
    notify_staff: Optional[bool] = None
    max_responses: Optional[int] = None
    closes_at: Optional[str] = None
    fields: Optional[list[dict[str, Any]]] = None


def _form_or_404(form_id: int) -> dict:
    form = store.get_form(form_id)
    if form is None:
        raise HTTPException(status_code=404, detail="Form not found")
    return form


def _bad(error: schema.FormError) -> HTTPException:
    return HTTPException(status_code=422, detail=str(error))


@router.get("/types")
def question_types(_admin: str = Depends(require_admin)):
    return {"input_types": list(schema.INPUT_TYPES), "display_types": list(schema.DISPLAY_TYPES), "max_fields": schema.MAX_FIELDS}


@router.get("")
def list_forms(_admin: str = Depends(require_admin)):
    return {"forms": store.list_forms()}


@router.post("", status_code=201)
def create_form(body: CreateBody, admin: str = Depends(require_admin)):
    try:
        return {"success": True, "form": store.create_form(title=body.title, slug=body.slug, fields=body.fields, created_by=admin)}
    except schema.FormError as error:
        raise _bad(error) from None


@router.get("/{form_id}")
def get_form(form_id: int, _admin: str = Depends(require_admin)):
    return {"form": _form_or_404(form_id)}


@router.put("/{form_id}")
def update_form(form_id: int, body: UpdateBody, admin: str = Depends(require_admin)):
    changes = body.model_dump(exclude_unset=True)
    try:
        form = store.update_form(form_id, changes, admin)
    except schema.FormError as error:
        raise _bad(error) from None
    if form is None:
        raise HTTPException(status_code=404, detail="Form not found")
    return {"success": True, "form": form}


@router.delete("/{form_id}")
def delete_form(form_id: int, _admin: str = Depends(require_admin)):
    if not store.delete_form(form_id):
        raise HTTPException(status_code=404, detail="Form not found")
    return {"success": True}


@router.post("/{form_id}/duplicate", status_code=201)
def duplicate_form(form_id: int, admin: str = Depends(require_admin)):
    form = store.duplicate_form(form_id, admin)
    if form is None:
        raise HTTPException(status_code=404, detail="Form not found")
    return {"success": True, "form": form}


@router.get("/{form_id}/responses")
def list_responses(form_id: int, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0), _admin: str = Depends(require_admin)):
    form = _form_or_404(form_id)
    return {"responses": store.list_responses(form_id, limit, offset), "total": form["response_count"]}


@router.delete("/{form_id}/responses/{response_id}")
def delete_response(form_id: int, response_id: int, _admin: str = Depends(require_admin)):
    if not store.delete_response(form_id, response_id):
        raise HTTPException(status_code=404, detail="Response not found")
    return {"success": True}


def _safe_cell(value: str) -> str:
    """Stop a spreadsheet from running an answer that starts like a formula."""
    return "'" + value if value[:1] in ("=", "+", "-", "@", "\t", "\r") else value


@router.get("/{form_id}/responses.csv")
def export_responses(form_id: int, _admin: str = Depends(require_admin)):
    form = _form_or_404(form_id)
    responses = store.all_responses(form_id)
    columns: dict[str, str] = {f["id"]: f["label"] for f in form["fields"] if f["type"] in schema.INPUT_TYPES}
    for response in reversed(responses):                       # questions removed since still get a column (oldest labels first)
        for answer in response["answers"]:
            columns.setdefault(answer["id"], answer["label"])
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["Reference", "Submitted (UTC)", "Form version", *[_safe_cell(label) for label in columns.values()]])
    for response in responses:
        by_id = {a["id"]: a for a in response["answers"]}
        writer.writerow([response["reference"], response["created_at"], response["form_version"],
                         *[_safe_cell(schema.answer_text(by_id[c])) if c in by_id else "" for c in columns]])
    return Response(content="\ufeff" + out.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{form["slug"]}-responses.csv"'})
