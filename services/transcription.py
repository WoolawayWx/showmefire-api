"""Broadcastify call ingestion, local transcription, and fire-signal filtering."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import os
import re
import time
from urllib.parse import urlparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import aiohttp

from core.config import (
    BCFY_API_BASE_URL,
    BCFY_AUDIO_DIR,
    BCFY_CALLS_URL,
    BCFY_ISSUER,
    BCFY_KEY_ID,
    BCFY_KEY_SECRET,
)
from core.database import (
    add_bcfy_job_event,
    create_bcfy_call,
    create_bcfy_fire_signal,
    get_bcfy_call,
    get_bcfy_config,
    list_bcfy_channels,
    reclaim_bcfy_stale_calls,
    update_bcfy_call,
    update_bcfy_channel,
)

logger = logging.getLogger(__name__)
_poll_lock = asyncio.Lock()

FIRE_TERMS = {
    "wildland", "wildland fire", "brush fire", "grass fire", "woods fire",
    "wooded fire", "forest fire", "vegetation fire", "field fire",
    "timber fire", "smoke in the area", "smoke showing", "flames in the woods",
}
NON_FIRE_TERMS = {
    "structure fire", "house fire", "building fire", "vehicle fire",
    "car fire", "medical", "ambulance", "false alarm", "controlled burn",
}


def configured() -> bool:
    return bool(BCFY_KEY_ID and BCFY_KEY_SECRET and BCFY_ISSUER)


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _mint_bcfy_jwt(ttl_seconds: int = 3600) -> str:
    """Self-signed HS256 JWT per Broadcastify's Calls API auth contract.

    There is no token-exchange endpoint: the app signs its own short-lived
    token with the API key as the HMAC secret and mints a fresh one per call.
    """
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT", "kid": BCFY_KEY_ID}
    payload = {"iss": BCFY_ISSUER, "iat": now, "exp": now + ttl_seconds}
    signing_input = (
        f"{_b64url(json.dumps(header, separators=(',', ':')).encode())}."
        f"{_b64url(json.dumps(payload, separators=(',', ':')).encode())}"
    )
    signature = hmac.new(BCFY_KEY_SECRET.encode(), signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url(signature)}"


def classify_transcript(text: str, threshold: float = 0.65) -> dict:
    normalized = re.sub(r"\s+", " ", (text or "").lower()).strip()
    fire_hits = sorted(term for term in FIRE_TERMS if term in normalized)
    non_fire_hits = sorted(term for term in NON_FIRE_TERMS if term in normalized)
    score = min(1.0, len(fire_hits) * 0.32 + (0.18 if "fire" in normalized else 0))
    if non_fire_hits:
        score = max(0.0, score - min(0.6, len(non_fire_hits) * 0.25))
    is_fire = bool(fire_hits) and score >= threshold and not (
        non_fire_hits and score < 0.8
    )
    return {
        "classification": "wildland_fire" if is_fire else "other",
        "confidence": round(score, 3),
        "evidence": fire_hits + [f"excluded:{term}" for term in non_fire_hits],
    }


def _parse_calls(payload: Any, fallback_group_id: str) -> tuple[list[dict], Optional[int]]:
    """Parse a Live Calls response into call records plus the new `pos` cursor.

    A Broadcastify call has no single opaque id -- it's identified by
    `groupId` + `ts` (unix seconds) -- and the audio link is just `url`.
    """
    if isinstance(payload, list):
        items, last_pos = payload, None
    else:
        items = payload.get("calls") or payload.get("data") or []
        last_pos = payload.get("lastPos") or payload.get("pos")
    result = []
    for item in items:
        if not isinstance(item, dict):
            continue
        group_id = item.get("groupId") or fallback_group_id
        ts = item.get("ts")
        audio_url = item.get("url")
        if not (group_id and ts and audio_url):
            continue
        started_ts = item.get("start_ts") or ts
        result.append({
            "external_id": f"{group_id}:{ts}",
            "channel_id": str(group_id),
            "started_at": datetime.fromtimestamp(int(started_ts), timezone.utc).isoformat(),
            "duration_seconds": item.get("duration"),
            "audio_url": str(audio_url),
        })
    return result, (int(last_pos) if last_pos else None)


class BroadcastifyClient:
    def __init__(self) -> None:
        self.calls_url = BCFY_CALLS_URL or f"{BCFY_API_BASE_URL}/calls/v1/live/"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {_mint_bcfy_jwt()}", "Accept": "application/json"}

    async def _fetch_calls_payload(self, group_id: str, pos: Optional[int]) -> Any:
        timeout = aiohttp.ClientTimeout(total=30)
        params: dict[str, str] = {"groups": group_id}
        if pos:
            params["pos"] = str(pos)
        else:
            params["init"] = "1"
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(self.calls_url, headers=self._headers(), params=params) as response:
                body = await response.text()
                if response.status >= 400:
                    raise RuntimeError(f"Broadcastify calls request failed ({response.status}): {body[:500]}")
                return json.loads(body)

    async def list_calls(self, group_id: str, pos: Optional[int]) -> tuple[list[dict], Optional[int]]:
        payload = await self._fetch_calls_payload(group_id, pos)
        return _parse_calls(payload, group_id)

    async def download(self, audio_url: str, target: Path) -> str:
        timeout = aiohttp.ClientTimeout(total=120)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(audio_url, headers=self._headers()) as response:
                if response.status >= 400:
                    raise RuntimeError(f"Broadcastify audio download failed ({response.status})")
                content = await response.read()
        if not content:
            raise RuntimeError("Broadcastify returned an empty audio file")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return hashlib.sha256(content).hexdigest()


def transcribe_audio(audio_path: str, model_name: str) -> dict:
    """Run local Whisper inference. Import lazily so the API can boot without a model."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise RuntimeError("Local transcription is unavailable; install faster-whisper") from exc
    model = WhisperModel(
        model_name,
        device=os.getenv("BCFY_WHISPER_DEVICE", "auto"),
        compute_type=os.getenv("BCFY_WHISPER_COMPUTE_TYPE", "int8"),
    )
    segments, info = model.transcribe(audio_path, vad_filter=True, beam_size=5)
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return {"text": text, "language": info.language or ""}


async def _transcribe_and_classify(call_id: str | int, audio_path: str, config: dict) -> dict:
    """Run Whisper + fire-keyword classification on audio already on disk, and persist the outcome."""
    call_id = int(call_id)
    try:
        add_bcfy_job_event(call_id, "transcribing", "Running local Whisper transcription")
        update_bcfy_call(call_id, status="transcribing")
        result = await asyncio.to_thread(transcribe_audio, audio_path, config["model_name"])
        decision = classify_transcript(result["text"], float(config["classification_threshold"]))
        status = "accepted" if decision["classification"] == "wildland_fire" else "rejected"
        update_bcfy_call(
            call_id, status="classifying", transcript=result["text"], language=result["language"],
            model_name=config["model_name"], classification=decision["classification"],
            confidence=decision["confidence"], evidence_json=json.dumps(decision["evidence"]),
        )
        if status == "accepted":
            signal = create_bcfy_fire_signal(
                call_id, decision["confidence"], ", ".join(decision["evidence"])
            )
            update_bcfy_call(call_id, fire_signal_id=signal["id"], status=status)
        else:
            update_bcfy_call(call_id, status=status)
        add_bcfy_job_event(call_id, status, f"Classified as {decision['classification']}")
    except Exception as exc:
        logger.exception("BCFY call %s failed", call_id)
        update_bcfy_call(call_id, status="failed", error=str(exc))
        add_bcfy_job_event(call_id, "failed", str(exc))
    return get_bcfy_call(call_id) or {}


async def _process_call(client: BroadcastifyClient, call: dict, config: dict) -> dict:
    call_id = int(call["id"])
    update_bcfy_call(call_id, status="downloading", attempts=int(call.get("attempts") or 0) + 1, error="")
    add_bcfy_job_event(call_id, "downloading", "Downloading audio")
    suffix = Path(urlparse(call["audio_url"]).path).suffix.lower()
    if suffix not in {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}:
        suffix = ".mp3"
    target = BCFY_AUDIO_DIR / f"{call['external_id']}{suffix}"
    try:
        digest = await client.download(call["audio_url"], target)
    except Exception as exc:
        logger.exception("BCFY call %s failed", call.get("external_id"))
        update_bcfy_call(call_id, status="failed", error=str(exc))
        add_bcfy_job_event(call_id, "failed", str(exc))
        return get_bcfy_call(call_id) or {}
    update_bcfy_call(call_id, audio_path=str(target), audio_sha256=digest)
    return await _transcribe_and_classify(call_id, str(target), config)


async def reprocess_call(call_id: int) -> dict:
    """Re-run a call that already failed or is stuck, reusing downloaded audio when it's still on disk.

    Unlike poll_bcfy_calls, this does not depend on the call still being inside
    the Broadcastify Live Calls window -- it works for calls that failed (e.g.
    before faster-whisper was installed) and have since aged out of that feed.
    """
    call = get_bcfy_call(call_id)
    if not call:
        raise ValueError("Call not found")
    config = get_bcfy_config()
    update_bcfy_call(call_id, attempts=int(call.get("attempts") or 0) + 1, error="")
    audio_path = call.get("audio_path")
    if audio_path and Path(audio_path).is_file():
        return await _transcribe_and_classify(call_id, audio_path, config)
    if not call.get("audio_url"):
        raise RuntimeError("No audio URL on file for this call; it cannot be redownloaded")
    if not configured():
        raise RuntimeError("BCFY credentials are incomplete; cannot redownload audio")
    client = BroadcastifyClient()
    return await _process_call(client, call, config)


async def poll_bcfy_calls() -> dict:
    config = get_bcfy_config()
    if not config.get("enabled"):
        return {"status": "disabled", "discovered": 0, "processed": 0}
    if not configured():
        return {"status": "not_configured", "reason": "BCFY credentials are incomplete"}
    channels = list_bcfy_channels(enabled_only=True)
    if not channels:
        return {"status": "no_channels", "discovered": 0, "processed": 0}
    async with _poll_lock:
        client = BroadcastifyClient()
        discovered = 0
        processed = 0
        per_channel = []
        for channel in channels:
            pos = channel.get("last_pos")
            try:
                calls, last_pos = await client.list_calls(channel["group_id"], int(pos) if pos else None)
            except Exception as exc:
                logger.exception("BCFY channel %s poll failed", channel["group_id"])
                per_channel.append({"group_id": channel["group_id"], "error": str(exc)})
                continue
            channel_discovered = 0
            channel_processed = 0
            for item in calls[:20]:
                record = create_bcfy_call(item)
                if not record or record["status"] not in {"discovered", "failed"}:
                    continue
                channel_discovered += 1
                await _process_call(client, record, config)
                channel_processed += 1
            if last_pos:
                update_bcfy_channel(channel["id"], last_pos=last_pos)
            discovered += channel_discovered
            processed += channel_processed
            per_channel.append({
                "group_id": channel["group_id"], "discovered": channel_discovered, "processed": channel_processed,
            })
        return {"status": "ok", "discovered": discovered, "processed": processed, "channels": per_channel}


async def test_bcfy_connection(group_id: str) -> dict:
    if not configured():
        return {"ok": False, "error": "BCFY_KEYID, BCFY_KEYSECRET, and BCFY_ISSUER are required"}
    try:
        client = BroadcastifyClient()
        payload = await client._fetch_calls_payload(group_id, None)
        calls, last_pos = _parse_calls(payload, group_id)
        raw_items = payload if isinstance(payload, list) else payload.get("calls") or payload.get("data") or []
        return {
            "ok": True,
            "calls_seen": len(calls),
            "calls_url": client.calls_url,
            "last_pos": last_pos,
            "raw_sample": raw_items[:2] if raw_items else payload,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def recover_bcfy_jobs() -> int:
    return reclaim_bcfy_stale_calls()


def purge_bcfy_audio(retention_days: int) -> int:
    """Remove raw audio older than the configured operational retention window."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=max(1, retention_days))
    removed = 0
    for path in BCFY_AUDIO_DIR.glob("*"):
        if not path.is_file():
            continue
        modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        if modified < cutoff:
            path.unlink(missing_ok=True)
            removed += 1
    return removed
