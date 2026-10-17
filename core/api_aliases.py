"""Unprefixed aliases (``/api/fires`` -> ``/fires``) for routes served from api.showmefire.org.

Every existing ``/api/...`` route keeps working exactly as before; this only
adds a second path to the same handler, with the same dependencies (auth, rate
limits), so the website can migrate one call at a time. Aliases are hidden from
the OpenAPI schema.

Not aliased, on purpose:
* ``/api/admin/*`` and ``/api/graphics/auth/*``: session middleware matches these
  exact paths (core/admin_session.py, core/graphics_session.py).
* Any alias that would collide with an existing route or mount, or that an
  existing route would already match (e.g. an existing ``/fires/{id}``). Those
  are reported so they can be renamed deliberately.
"""
from __future__ import annotations

import copy
import logging
import re
from typing import Dict, List

from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.routing import Mount, compile_path

logger = logging.getLogger(__name__)

API_PREFIX = "/api"
EXCLUDED_PREFIXES = ("/api/admin", "/api/graphics/auth")
_PARAM = re.compile(r"\{[^}]+\}")


def _excluded(path: str) -> bool:
    return any(path == p or path.startswith(p + "/") for p in EXCLUDED_PREFIXES)


def _alias_for(path: str) -> str | None:
    if path == API_PREFIX or not path.startswith(API_PREFIX + "/") or _excluded(path):
        return None
    return path[len(API_PREFIX):]


def _shadowed(alias: str, methods: set, routes: List[APIRoute], mounts: List[str]) -> str | None:
    """Reason an alias cannot be added, or None when it is safe."""
    for mount in mounts:
        if mount and (alias == mount or alias.startswith(mount + "/")):
            return f"mount {mount}"
    sample = _PARAM.sub("x", alias)
    for route in routes:
        if route.methods & methods and (route.path_format == alias or route.path_regex.match(sample)):
            return f"route {route.path}"
    return None


def add_unprefixed_aliases(app: FastAPI) -> Dict[str, List]:
    """Add ``/x`` aliases for ``/api/x`` routes. Call after all routers are included."""
    existing: List[APIRoute] = [r for r in app.router.routes if isinstance(r, APIRoute)]
    mounts = [r.path for r in app.router.routes if isinstance(r, Mount)]
    report: Dict[str, List] = {"added": [], "skipped": []}
    for route in list(existing):
        alias = _alias_for(route.path)
        if alias is None:
            continue
        reason = _shadowed(alias, set(route.methods), existing, mounts)
        if reason:
            report["skipped"].append({"path": route.path, "alias": alias, "reason": reason})
            continue
        clone = copy.copy(route)
        clone.path = alias
        clone.path_regex, clone.path_format, clone.param_convertors = compile_path(alias)
        clone.include_in_schema = False
        clone.name = f"{route.name}_alias"
        # Insert ahead of a root ("/") mount, which would otherwise swallow the alias.
        root_mount = next(
            (i for i, r in enumerate(app.router.routes) if isinstance(r, Mount) and r.path == ""), None
        )
        if root_mount is None:
            app.router.routes.append(clone)
        else:
            app.router.routes.insert(root_mount, clone)
        existing.append(clone)
        report["added"].append({"path": route.path, "alias": alias, "methods": sorted(route.methods)})
    logger.info(
        "API aliases: %d added, %d skipped (collisions)", len(report["added"]), len(report["skipped"])
    )
    for item in report["skipped"]:
        logger.warning("API alias skipped: %s -> %s (%s)", item["path"], item["alias"], item["reason"])
    return report
