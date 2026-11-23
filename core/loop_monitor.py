"""Catch a blocked event loop and record where it is stuck.

The 2026-11-16 outage (API at ~100% CPU for ~35 min) never had its cause captured because nobody was there to send
USR1. A coroutine on the loop updates a heartbeat every half second; a plain daemon thread (which keeps running while
the loop is blocked) watches it. When the loop has not ticked for `threshold` seconds the thread dumps every thread's
stack to stderr (`docker logs`) with faulthandler and logs a warning once per episode, then logs again when the loop
comes back. Nothing here changes behaviour and it never raises.
"""
from __future__ import annotations

import asyncio
import faulthandler
import logging
import sys
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)

TICK_SECONDS = 0.5
CHECK_SECONDS = 1.0
THRESHOLD_SECONDS = 3.0


class LoopMonitor:
    def __init__(self, threshold: float = THRESHOLD_SECONDS, clock: Callable[[], float] = time.monotonic,
                 dump: Optional[Callable[[], None]] = None):
        self.threshold = threshold
        self.clock = clock
        self._dump = dump or (lambda: faulthandler.dump_traceback(file=sys.stderr, all_threads=True))
        self.last_beat = clock()
        self.blocked_since: Optional[float] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._task: Optional[asyncio.Task] = None

    def beat(self) -> None:
        self.last_beat = self.clock()

    def check(self) -> Optional[str]:
        """One look from the watching thread. Returns "blocked" / "recovered" when an episode starts / ends."""
        now = self.clock()
        lag = now - self.last_beat
        if lag >= self.threshold and self.blocked_since is None:
            self.blocked_since = self.last_beat
            logger.warning("Event loop has not responded for %.1f s; thread stacks follow on stderr", lag)
            try:
                self._dump()
            except Exception:  # noqa: BLE001 - diagnostics must never break anything
                logger.debug("stack dump failed", exc_info=True)
            return "blocked"
        if lag < self.threshold and self.blocked_since is not None:
            logger.warning("Event loop responding again after about %.1f s", self.last_beat - self.blocked_since + lag)
            self.blocked_since = None
            return "recovered"
        return None

    async def _tick(self) -> None:
        while True:
            self.beat()
            await asyncio.sleep(TICK_SECONDS)

    def _watch(self) -> None:
        while not self._stop.wait(CHECK_SECONDS):
            try:
                self.check()
            except Exception:  # noqa: BLE001
                logger.debug("loop monitor check failed", exc_info=True)

    def start(self) -> None:
        """Call from inside the running loop."""
        self.beat()
        self._task = asyncio.get_running_loop().create_task(self._tick())
        self._thread = threading.Thread(target=self._watch, name="loop-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._task:
            self._task.cancel()
