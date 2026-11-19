"""First-party contact form; no public ticket lookup or exposure of support records."""
import re
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from core.rate_limit import RateLimiter, client_ip
from services import contact
from services.turnstile import verify_turnstile

router = APIRouter(prefix="/v1/contact", tags=["contact"])
submit_limiter = RateLimiter("contact_submit", 5)


class ContactBody(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    request_id: UUID
    name: str = Field(min_length=1, max_length=120)
    email: str = Field(min_length=3, max_length=254)
    subject: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=10000)
    consent: Literal[True]
    website: str = Field(default="", max_length=200)
    turnstile_token: str = Field(default="", max_length=4096)

    @field_validator("email")
    @classmethod
    def valid_email(cls, value):
        if not re.fullmatch(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]{1,64}@[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?\.[A-Za-z]{2,63}", value):
            raise ValueError("Enter a valid email address")
        return value.lower()

    @field_validator("name", "subject", "email")
    @classmethod
    def single_line(cls, value):
        if any(ord(char) < 32 or ord(char) == 127 for char in value):
            raise ValueError("Use a single line without control characters")
        return value


@router.post("", status_code=201, dependencies=[Depends(submit_limiter)])
def submit(body: ContactBody, request: Request):
    if body.website:
        raise HTTPException(400, "We could not accept this submission.")
    fields = body.model_dump(include={"name", "email", "subject", "message"})
    request_id = str(body.request_id)
    ticket_id = contact.existing(request_id, contact.content_hash(fields))
    if not ticket_id:
        if not contact.configured():
            raise HTTPException(503, "Contact email is temporarily unavailable. Please email contact@showmefire.org directly.")
        ip = client_ip(request)
        ok, _ = verify_turnstile(body.turnstile_token, ip)
        if not ok:
            raise HTTPException(400, "The human check failed. Please try again.")
        ticket_id = contact.create(request_id, fields, ip)
    sent = contact.deliver(ticket_id)
    return {"success": True, "ticket_id": ticket_id, "email_status": "sent" if sent else "pending"}
