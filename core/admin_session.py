"""Renew expired admin access cookies before protected requests are handled."""
from core import security
from core.cookie_renewal import CookieRenewalMiddleware

# These routes explicitly manage authentication and must not get an implicit
# renewal (especially logout, which must be the last writer of its cookies).
AUTH_ACTIONS = {'/api/admin/login', '/api/admin/logout', '/api/admin/refresh'}


def set_auth_cookie(response, name, value, max_age):
    response.set_cookie(
        key=name, value=value, max_age=max_age, httponly=True,
        secure=security.AUTH_COOKIE_SECURE,
        samesite=security.AUTH_COOKIE_SAMESITE,
        domain=security.AUTH_COOKIE_DOMAIN, path='/',
    )


class AdminCookieMiddleware(CookieRenewalMiddleware):
    """Renews an expired admin access cookie from the refresh cookie and makes the token available to the request."""

    def resolve(self, request):
        access = request.cookies.get(security.ACCESS_COOKIE_NAME)
        if (request.url.path.rstrip('/') not in AUTH_ACTIONS
                and not security.verify_token(access)
                and request.cookies.get(security.REFRESH_COOKIE_NAME)):
            email = security.verify_token(
                request.cookies[security.REFRESH_COOKIE_NAME], expected_type='refresh',
            )
            if email:
                return security.create_access_token({'sub': email}), True
        return access, False

    def set_token(self, token):
        return security.set_request_token(token)

    def reset_token(self, context):
        security.reset_request_token(context)

    def set_cookie(self, response, access):
        set_auth_cookie(response, security.ACCESS_COOKIE_NAME, access, security.ACCESS_TOKEN_EXPIRE_MINUTES * 60)
