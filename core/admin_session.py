"""Renew expired admin access cookies before protected requests are handled."""
from core import security

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


async def admin_cookie_context(request, call_next):
    access = request.cookies.get(security.ACCESS_COOKIE_NAME)
    renewed = False
    if (request.url.path.rstrip('/') not in AUTH_ACTIONS
            and not security.verify_token(access)
            and request.cookies.get(security.REFRESH_COOKIE_NAME)):
        email = security.verify_token(
            request.cookies[security.REFRESH_COOKIE_NAME], expected_type='refresh',
        )
        if email:
            access = security.create_access_token({'sub': email})
            renewed = True
    context = security.set_request_token(access)
    try:
        response = await call_next(request)
        if renewed:
            response.headers['Cache-Control'] = 'no-store'
            set_auth_cookie(response, security.ACCESS_COOKIE_NAME, access,
                            security.ACCESS_TOKEN_EXPIRE_MINUTES * 60)
        return response
    finally:
        security.reset_request_token(context)
