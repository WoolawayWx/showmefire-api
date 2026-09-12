"""Renew expired graphics access cookies before protected requests are handled."""
from core import security

# These routes explicitly manage authentication and must not get an implicit
# renewal (especially logout, which must be the last writer of its cookies).
AUTH_ACTIONS = {
    '/api/graphics/auth/login',
    '/api/graphics/auth/logout',
    '/api/graphics/auth/refresh',
    '/api/graphics/auth/verify-invite',
    '/api/graphics/auth/set-password',
}


def set_auth_cookie(response, name, value, max_age):
    response.set_cookie(
        key=name, value=value, max_age=max_age, httponly=True,
        secure=security.AUTH_COOKIE_SECURE,
        samesite=security.AUTH_COOKIE_SAMESITE,
        domain=security.AUTH_COOKIE_DOMAIN, path='/',
    )


async def graphics_cookie_context(request, call_next):
    access = request.cookies.get(security.GRAPHICS_ACCESS_COOKIE_NAME)
    renewed = False
    if (request.url.path.rstrip('/') not in AUTH_ACTIONS
            and not security.verify_graphics_token(access)
            and request.cookies.get(security.GRAPHICS_REFRESH_COOKIE_NAME)):
        payload = security.verify_graphics_token(
            request.cookies[security.GRAPHICS_REFRESH_COOKIE_NAME], expected_type='graphics_refresh',
        )
        if payload:
            access = security.create_graphics_access_token(
                payload['sub'], payload['department_id'], payload['user_id'],
            )
            renewed = True
    context = security.set_graphics_request_token(access)
    try:
        response = await call_next(request)
        if renewed:
            response.headers['Cache-Control'] = 'no-store'
            set_auth_cookie(response, security.GRAPHICS_ACCESS_COOKIE_NAME, access,
                            security.ACCESS_TOKEN_EXPIRE_MINUTES * 60)
        return response
    finally:
        security.reset_graphics_request_token(context)
