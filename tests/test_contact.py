"""Contact submissions, duplicate recovery, abuse protection and durable email delivery."""
import time
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from routers import contact as routes
from services import contact


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv('DATA_DIR', str(tmp_path))
    monkeypatch.setenv('RESEND_API_KEY', 'test-key')
    monkeypatch.setenv('CONTACT_EMAIL_FROM', 'Show Me Fire <contact@showmefire.org>')
    monkeypatch.setattr(routes, 'verify_turnstile', lambda token, ip: (token == 'good', 'test'))
    routes.submit_limiter.reset()
    sent = []
    def send(method, path, payload, **kwargs):
        sent.append((payload, kwargs))
        return {'id': str(uuid4())}
    monkeypatch.setattr(contact.graphics_email, '_resend_request', send)
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app), sent


def body(**changes):
    return dict(request_id=str(uuid4()), name=' Pat ', email='Pat@example.org', subject='A question',
                message='Hello <script>alert(1)</script>', consent=True, turnstile_token='good', **changes)


def test_ticket_is_saved_and_both_emails_have_reference_and_correct_reply_to(env):
    client, sent = env
    response = client.post('/v1/contact', json=body())
    assert response.status_code == 201, response.text
    ticket = response.json()['ticket_id']
    assert ticket.startswith('SMF-') and response.json()['email_status'] == 'sent'
    assert len(sent) == 2
    staff, receipt = (item[0] for item in sent)
    assert staff['to'] == ['contact@showmefire.org'] and staff['reply_to'] == 'pat@example.org'
    assert receipt['to'] == ['pat@example.org'] and receipt['reply_to'] == 'contact@showmefire.org'
    assert all(ticket in item[0]['subject'] for item in sent)
    assert '<script>' not in staff['html'] and '&lt;script&gt;' in staff['html']
    assert all(item[1]['idempotency_key'].startswith(f'contact/{ticket}/') for item in sent)
    with contact.connect() as conn:
        row = conn.execute('SELECT * FROM contact_tickets').fetchone()
        assert row['name'] == 'Pat' and row['email'] == 'pat@example.org'
        assert row['ip_hash'] != 'testclient'


def test_retry_recovers_same_ticket_without_reusing_captcha_or_resending(env):
    client, sent = env
    payload = body()
    first = client.post('/v1/contact', json=payload).json()
    payload['turnstile_token'] = 'expired'
    assert client.post('/v1/contact', json=payload).json() == first
    assert len(sent) == 2
    payload['message'] = 'Different message'
    assert client.post('/v1/contact', json=payload).status_code == 409


@pytest.mark.parametrize('changes', [
    {'name': ' '}, {'email': 'invalid'}, {'email': 'x@example.org,other@example.org'},
    {'subject': 'Hi\r\nBcc: spam@example.org'}, {'message': 'x' * 10001}, {'consent': False},
])
def test_invalid_entries_cannot_create_tickets_or_send_email(env, changes):
    client, sent = env
    assert client.post('/v1/contact', json={**body(), **changes}).status_code == 422
    assert not sent


@pytest.mark.parametrize('changes', [{'website': 'spam'}, {'turnstile_token': 'bad'}])
def test_abuse_checks_reject_without_storage(env, changes):
    client, sent = env
    assert client.post('/v1/contact', json={**body(), **changes}).status_code == 400
    assert not sent
    with contact.connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM contact_tickets').fetchone()[0] == 0


def test_missing_email_config_fails_before_ticket_creation(env, monkeypatch):
    client, sent = env
    monkeypatch.delenv('RESEND_API_KEY')
    assert client.post('/v1/contact', json=body()).status_code == 503
    assert not sent


def test_daily_cap_is_persisted_independently_of_minute_limiter(env):
    client, sent = env
    for _ in range(5):
        assert client.post('/v1/contact', json=body()).status_code == 201
    routes.submit_limiter.reset()
    assert client.post('/v1/contact', json=body()).status_code == 429
    assert len(sent) == 10


def test_partial_email_failure_is_saved_and_only_failed_email_retries(env, monkeypatch):
    client, sent = env
    original = contact.graphics_email._resend_request
    def partial(method, path, payload, **kwargs):
        if payload['to'] == ['pat@example.org']:
            raise RuntimeError('Temporary failure')
        return original(method, path, payload, **kwargs)
    monkeypatch.setattr(contact.graphics_email, '_resend_request', partial)
    response = client.post('/v1/contact', json=body()).json()
    assert response['email_status'] == 'pending' and len(sent) == 1
    with contact.connect() as conn:
        conn.execute('UPDATE contact_email_outbox SET next_attempt_at = 0')
    monkeypatch.setattr(contact.graphics_email, '_resend_request', original)
    contact.maintain()
    assert len(sent) == 2 and sent[-1][0]['to'] == ['pat@example.org']
    assert contact.deliver(response['ticket_id']) is True
    assert len(sent) == 2


def test_outbox_lease_and_expiry_prevent_concurrent_or_late_sends(env):
    client, sent = env
    fields = {'name': 'Pat', 'email': 'pat@example.org', 'subject': 'Hi', 'message': 'Hello'}
    ticket = contact.create(str(uuid4()), fields, '127.0.0.1')
    with contact.connect() as conn:
        conn.execute('UPDATE contact_email_outbox SET next_attempt_at = ?', (time.time() + 120,))
    assert contact.deliver(ticket) is False and not sent
    with contact.connect() as conn:
        conn.execute('UPDATE contact_email_outbox SET next_attempt_at = 0, first_attempt_at = ?', (time.time() - 24 * 3600,))
    contact.maintain()
    assert not sent


def test_resolved_tickets_and_email_copies_are_purged_after_90_days(env):
    client, sent = env
    old = client.post('/v1/contact', json=body()).json()['ticket_id']
    keep = client.post('/v1/contact', json=body()).json()['ticket_id']
    with contact.connect() as conn:
        conn.execute('UPDATE contact_tickets SET resolved_at = ? WHERE ticket_id = ?', (time.time() - 91 * 86400, old))
    contact.maintain()
    with contact.connect() as conn:
        assert [r[0] for r in conn.execute('SELECT ticket_id FROM contact_tickets')] == [keep]
        assert conn.execute('SELECT COUNT(*) FROM contact_email_outbox WHERE ticket_id = ?', (old,)).fetchone()[0] == 0
