"""Renew expired graphics access cookies before protected requests are handled."""
from core import security
from core.cookie_renewal import CookieRenewalMiddleware

# These routes explicitly manage authentication and must not get an implicit
# renewal (especially logout, which must be the last writer of its cookies).
AUTH_ACTIONS = {
    '/api/graphics/auth/request-code',
    '/api/graphics/auth/verify-code',
    '/api/graphics/auth/logout',
    '/api/graphics/auth/refresh',
}


def set_auth_cookie(response, name, value, max_age):
    response.set_cookie(
        key=name, value=value, max_age=max_age, httponly=True,
        secure=security.AUTH_COOKIE_SECURE,
        samesite=security.AUTH_COOKIE_SAMESITE,
        domain=security.AUTH_COOKIE_DOMAIN, path='/',
    )


class GraphicsCookieMiddleware(CookieRenewalMiddleware):
    """Renews an expired graphics access cookie from the refresh cookie and makes the token available to the request."""

    def resolve(self, request):
        access = request.cookies.get(security.GRAPHICS_ACCESS_COOKIE_NAME)
        if (request.url.path.rstrip('/') not in AUTH_ACTIONS
                and not security.verify_graphics_token(access)
                and request.cookies.get(security.GRAPHICS_REFRESH_COOKIE_NAME)):
            payload = security.verify_graphics_token(
                request.cookies[security.GRAPHICS_REFRESH_COOKIE_NAME], expected_type='graphics_refresh',
            )
            if payload:
                return security.create_graphics_access_token(
                    payload['sub'], payload['department_id'], payload['user_id'],
                ), True
        return access, False

    def set_token(self, token):
        return security.set_graphics_request_token(token)

    def reset_token(self, context):
        security.reset_graphics_request_token(context)

    def set_cookie(self, response, access):
        set_auth_cookie(response, security.GRAPHICS_ACCESS_COOKIE_NAME, access,
                        security.ACCESS_TOKEN_EXPIRE_MINUTES * 60)
