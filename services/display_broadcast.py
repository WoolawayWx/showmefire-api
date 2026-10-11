"""Per-board live sync for the station-TV kiosk display (/display/dashboard).

Lets an admin changing a display code's settings - either from the kiosk's
own in-screen editor (routers/display_auth.py:display_settings_save) or from
the admin panel (routers/display_auth.py:admin_set_display_code_settings) -
push every already-running kiosk on that same code to reload immediately,
instead of relying on someone physically walking up to the TV. Scoped by
code_id (not a single global broadcast like services/broadcast.py) since one
code system can drive multiple independent kiosks (see display_auth.py's
module docstring) that shouldn't all reload just because one of them changed.
"""
import logging
from typing import Dict, Set

from fastapi import WebSocket

logger = logging.getLogger(__name__)

_connections: Dict[int, Set[WebSocket]] = {}


def register(code_id: int, websocket: WebSocket) -> None:
    _connections.setdefault(code_id, set()).add(websocket)


def unregister(code_id: int, websocket: WebSocket) -> None:
    sockets = _connections.get(code_id)
    if not sockets:
        return
    sockets.discard(websocket)
    if not sockets:
        _connections.pop(code_id, None)


async def notify_settings_changed(code_id: int) -> None:
    sockets = list(_connections.get(code_id) or ())
    if not sockets:
        return
    message = {"type": "settings_changed"}
    disconnected = []
    for socket in sockets:
        try:
            await socket.send_json(message)
        except Exception as exc:
            logger.debug("Dropping display socket for code %s: %s", code_id, exc)
            disconnected.append(socket)
    for socket in disconnected:
        unregister(code_id, socket)
