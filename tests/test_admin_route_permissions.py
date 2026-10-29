"""Every admin route that moderates anything declares a permission (core/admin_auth/permissions.py).

The website's other admin pages still only check "signed in" (`require_admin`); `scripts/list_unguarded_admin_routes.py` lists them so
the gap stays visible while it is closed. This test covers the routers that hold the moderation surface, so a new route there cannot
be added without deciding who may use it."""
from fastapi import FastAPI

from core.admin_auth import permissions
from routers import admin_audit, admin_reviewers, burn_bans, feedback, fires, staff_api


def build_app() -> FastAPI:
    app = FastAPI()
    for module in (fires, burn_bans, feedback, admin_reviewers, admin_audit, staff_api):
        app.include_router(module.router)
    return app


def test_no_moderation_admin_route_is_missing_a_permission():
    missing = permissions.unguarded_admin_routes(build_app(), prefix="/api/admin")
    assert missing == [], f"admin routes with no permission: {missing}"


def test_no_staff_api_route_is_missing_a_permission():
    assert permissions.unguarded_admin_routes(build_app(), prefix="/api/staff") == []


def test_view_and_moderate_are_split_by_method():
    """Reading needs the view permission, anything that changes something needs the moderate one."""
    app = build_app()
    seen = {"view": 0, "moderate": 0}
    for route in app.routes:
        path = getattr(route, "path", "")
        if not path.startswith(("/api/admin/fires", "/api/admin/burn-bans")):
            continue
        declared = set(permissions.route_permissions(route))
        for method in route.methods - {"HEAD", "OPTIONS"}:
            if method == "GET":
                assert declared <= {"reports.view", "burnbans.view"}, (method, path, declared)
                seen["view"] += 1
            else:
                assert declared <= {"reports.moderate", "burnbans.moderate"}, (method, path, declared)
                seen["moderate"] += 1
    assert seen["view"] >= 15 and seen["moderate"] >= 25
