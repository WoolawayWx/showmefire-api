import unittest
from datetime import timedelta
from unittest.mock import patch

from fastapi import FastAPI, HTTPException, Response
from fastapi.testclient import TestClient
from core import security
from core.admin_session import admin_cookie_context


class AdminSessionTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.middleware('http')(admin_cookie_context)

        @app.get('/protected')
        def protected(token: str | None = None):
            email = security.verify_token(token)
            if not email:
                raise HTTPException(401)
            return {'email': email}

        @app.post('/api/admin/logout')
        def logout(response: Response):
            response.delete_cookie(security.ACCESS_COOKIE_NAME)
            response.delete_cookie(security.REFRESH_COOKIE_NAME)
            return {'success': True}

        @app.post('/api/admin/refresh')
        def refresh():
            return {'explicit': True}

        self.client = TestClient(app)
        self.cookie_settings = patch.object(security, 'AUTH_COOKIE_SECURE', False)
        self.cookie_settings.start()
        self.addCleanup(self.cookie_settings.stop)

    def tokens(self, expired_access=True, expired_refresh=False):
        self.client.cookies.set(security.ACCESS_COOKIE_NAME, security.create_access_token(
            {'sub': 'admin@example.com'}, timedelta(hours=-1 if expired_access else 1)))
        self.client.cookies.set(security.REFRESH_COOKIE_NAME, security.create_refresh_token(
            {'sub': 'admin@example.com'}, timedelta(days=-1 if expired_refresh else 30)))

    def test_expired_access_renews_without_replaying_request(self):
        self.tokens()
        response = self.client.get('/protected')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['email'], 'admin@example.com')
        self.assertIn('HttpOnly', response.headers['set-cookie'])
        self.assertIn(security.ACCESS_COOKIE_NAME, response.headers['set-cookie'])
        self.assertNotIn(security.REFRESH_COOKIE_NAME, response.headers['set-cookie'])

    def test_missing_access_uses_remaining_refresh_session(self):
        self.tokens()
        self.client.cookies.delete(security.ACCESS_COOKIE_NAME)
        self.assertEqual(self.client.get('/protected').status_code, 200)

    def test_expired_session_still_requires_login(self):
        self.tokens(expired_refresh=True)
        response = self.client.get('/protected')
        self.assertEqual(response.status_code, 401)
        self.assertNotIn('set-cookie', response.headers)

    def test_valid_access_does_not_write_cookies_on_every_request(self):
        self.tokens(expired_access=False)
        response = self.client.get('/protected')
        self.assertEqual(response.status_code, 200)
        self.assertNotIn('set-cookie', response.headers)

    def test_logout_cannot_be_undone_by_automatic_renewal(self):
        self.tokens()
        response = self.client.post('/api/admin/logout')
        self.assertEqual(len(response.headers.get_list('set-cookie')), 2)
        self.assertTrue(all('Max-Age=0' in c for c in response.headers.get_list('set-cookie')))

    def test_refresh_token_cannot_act_as_access_token(self):
        self.client.cookies.set(security.ACCESS_COOKIE_NAME, security.create_refresh_token({'sub': 'admin@example.com'}))
        self.assertEqual(self.client.get('/protected').status_code, 401)

    def test_explicit_invalid_token_is_not_replaced_by_cookie_identity(self):
        self.tokens()
        self.assertEqual(self.client.get('/protected?token=invalid').status_code, 401)

    def test_default_refresh_lifetime_is_thirty_days(self):
        from jose import jwt
        from datetime import datetime, timezone
        with patch.object(security, 'REFRESH_TOKEN_EXPIRE_DAYS', 30):
            claims = jwt.get_unverified_claims(security.create_refresh_token({'sub': 'admin@example.com'}))
        remaining = claims['exp'] - datetime.now(timezone.utc).timestamp()
        self.assertAlmostEqual(remaining, 30 * 86400, delta=2)


if __name__ == '__main__':
    unittest.main()
