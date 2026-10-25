"""Renew expired admin access cookies before protected requests are handled."""
from core import security
from core.admin_auth import sessions
from core.cookie_renewal import CookieRenewalMiddleware

# These routes explicitly manage authentication and must not get an implicit
# renewal (especially logout, which must be the last writer of its cookies).
AUTH_ACTIONS = {'/api/admin/logout', '/api/admin/refresh'}


def is_auth_action(path: str) -> bool:
    path = path.rstrip('/')
    return path in AUTH_ACTIONS or path.startswith('/api/admin/login')


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
        refresh = request.cookies.get(security.REFRESH_COOKIE_NAME)
        if not is_auth_action(request.url.path) and refresh and not security.verify_token(access):
            renewed = sessions.renew(refresh)
            if renewed:
                email, session_id = renewed
                return security.create_access_token({'sub': email, 'sid': session_id}), True
        return access, False

    def set_token(self, token):
        return security.set_request_token(token)

    def reset_token(self, context):
        security.reset_request_token(context)

    def set_cookie(self, response, access):
        set_auth_cookie(response, security.ACCESS_COOKIE_NAME, access, security.ACCESS_TOKEN_EXPIRE_MINUTES * 60)
