from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .artifacts import sha256_file


class ForecastR2Store:
    """Checksummed immutable-object publisher for the forecast-v1 namespace."""

    def __init__(self, client=None, bucket: str | None = None):
        self._client = client
        self.bucket = bucket or os.getenv("R2_FORECAST_BUCKET") or os.getenv("R2_BUCKET", "cdn-showmefire")

    @property
    def configured(self) -> bool:
        return self._client is not None or all(os.getenv(name) for name in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY"))

    @property
    def client(self):
        if self._client is None:
            import boto3
            from botocore.config import Config
            self._client = boto3.client(
                "s3", endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
                aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"], aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
                region_name="auto", config=Config(signature_version="s3v4"),
            )
        return self._client

    def upload_immutable(self, path: str | Path, key: str, checksum: str | None = None) -> dict:
        source = Path(path)
        digest = checksum or sha256_file(source)
        try:
            existing = self.client.head_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))
            if code not in {"404", "NoSuchKey", "NotFound"}:
                raise
        else:
            remote_digest = (existing.get("Metadata") or {}).get("sha256")
            if remote_digest == digest:
                return {"key": key, "sha256": digest, "bytes": source.stat().st_size, "deduplicated": True}
            raise RuntimeError(f"immutable R2 key already exists with a different checksum: {key}")
        self.client.upload_file(str(source), self.bucket, key, ExtraArgs={"Metadata": {"sha256": digest}})
        body = self.client.get_object(Bucket=self.bucket, Key=key)["Body"]
        remote = hashlib.sha256()
        try:
            for chunk in iter(lambda: body.read(8 * 1024 * 1024), b""):
                remote.update(chunk)
        finally:
            body.close()
        if remote.hexdigest() != digest:
            raise RuntimeError(f"R2 round-trip checksum failed: {key}")
        return {"key": key, "sha256": digest, "bytes": source.stat().st_size, "deduplicated": False}

    def put_latest(self, payload: dict, key: str = "forecast-v1/latest.json") -> None:
        body = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        self.client.put_object(Bucket=self.bucket, Key=key, Body=body, ContentType="application/json", CacheControl="no-cache")

    def put_alias(self, path: str | Path, key: str) -> None:
        source = Path(path)
        content_type = "image/webp" if source.suffix.lower() == ".webp" else "image/png"
        self.client.upload_file(
            str(source), self.bucket, key,
            ExtraArgs={"ContentType": content_type, "CacheControl": "public, max-age=900", "Metadata": {"sha256": sha256_file(source)}},
        )
