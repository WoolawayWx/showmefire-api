"""Install an immutable V5 shadow bundle into the persistent API data volume."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from services.v5_shadow import validate_bundle


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def _safe_target(root, name):
    target = (root / name).resolve()
    if root.resolve() not in target.parents and target != root.resolve():
        raise ValueError(f"unsafe archive member: {name}")
    return target


def extract(archive, destination):
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            for member in source.infolist(): _safe_target(destination, member.filename)
            source.extractall(destination)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as source:
            for member in source.getmembers():
                _safe_target(destination, member.name)
                if member.issym() or member.islnk(): raise ValueError("bundle links are not allowed")
            source.extractall(destination, filter="data")
    else:
        raise ValueError("bundle must be a zip or tar archive")


def install(archive, expected_sha256=None, destination_root=None):
    archive = Path(archive).resolve(); digest = sha256(archive)
    if expected_sha256 and digest.lower() != expected_sha256.lower():
        raise ValueError("V5 bundle archive checksum mismatch")
    root = Path(destination_root or (Path(os.getenv("DATA_DIR", "data")) / "model-bundles" / "v5")).resolve()
    final = root / digest
    if final.exists():
        validate_bundle(final)
        return final, digest, False
    root.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{digest[:12]}-", dir=root))
    try:
        extract(archive, temporary)
        candidates = [temporary] + [item for item in temporary.iterdir() if item.is_dir()]
        bundle = next((item for item in candidates if (item / "contract.json").exists()), None)
        if bundle is None: raise ValueError("archive does not contain a V5 bundle")
        validate_bundle(bundle)
        if bundle == temporary:
            temporary.replace(final)
        else:
            bundle.replace(final); shutil.rmtree(temporary, ignore_errors=True)
        (final / "installation.json").write_text(json.dumps({"archive_sha256": digest}, indent=2))
        return final, digest, True
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path); parser.add_argument("--sha256")
    parser.add_argument("--destination-root", type=Path)
    args = parser.parse_args()
    try: path, digest, installed = install(args.archive, args.sha256, args.destination_root)
    except Exception as error: raise SystemExit(str(error)) from error
    print(json.dumps({"bundle": str(path), "archive_sha256": digest, "installed": installed,
                      "configure": f"SMF_V5_SHADOW_BUNDLE={path}"}, indent=2))


if __name__ == "__main__": main()
