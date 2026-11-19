"""Remember what was last uploaded to each CDN key, so an identical file is not written again.

Every R2 write is a billed Class A operation. Map images are re-pushed every 15 minutes; when a run produces a byte-identical
`latest/` file there is nothing new to publish. The state is a small JSON file of {key: {"sha256", "at"}}.

Safety: an entry older than MAX_AGE_SECONDS is ignored, so a `latest/` object that was deleted from the bucket by hand is
restored within hours, and an unreadable or missing state file just means "upload everything". Only `latest/` keys use
this - dated archive keys are unique per run and the timeline links to them, so they are always written."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

MAX_AGE_SECONDS = 6 * 3600


def state_path(project_root):
    return Path(os.getenv("CDN_UPLOAD_STATE") or Path(project_root) / "cdn_upload_state.json")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class UploadState:
    def __init__(self, path, now=None):
        self.path = Path(path)
        self.now = now or time.time
        self.entries = {}
        self.skipped = 0
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                self.entries = loaded
        except (OSError, ValueError):
            pass

    def unchanged(self, key, file_path):
        """True when `file_path` is byte-identical to what was last uploaded to `key`, recently enough to trust it."""
        entry = self.entries.get(key)
        if not isinstance(entry, dict):
            return False
        try:
            fresh = self.now() - float(entry.get("at", 0)) < MAX_AGE_SECONDS
            same = entry.get("sha256") == file_sha256(file_path)
        except (OSError, ValueError, TypeError):
            return False
        if fresh and same:
            self.skipped += 1
        return fresh and same

    def remember(self, key, file_path):
        try:
            self.entries[key] = {"sha256": file_sha256(file_path), "at": self.now()}
        except OSError:
            pass

    def save(self):
        try:
            tmp = self.path.with_name(self.path.name + ".part")
            tmp.write_text(json.dumps(self.entries), encoding="utf-8")
            tmp.replace(self.path)
        except OSError:
            pass                                       # losing the state only costs extra uploads
