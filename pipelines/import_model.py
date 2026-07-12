"""
Pull a model artifact from a GitHub release published by ShowMeFire-Models'
publish_release.py, and register it into THIS server's own models/versioning.py
registry as a beta candidate. Only the server's existing promote_model.py
makes it live - this script never touches `stable`.

The imported model gets the next version in *this server's own* sequence,
independent of whatever version the training repo used - the two registries
are never shared directly. The source release tag is recorded in the
performance metadata for traceability.

Requires the `gh` CLI, authenticated - reuses .env's GITHUB_TOKEN automatically
if exported as GH_TOKEN, otherwise run `gh auth login` first.

Usage:
    python pipelines/import_model.py --model fuel_moisture --tag fuel_moisture-v1.5.0-beta.1 --repo youruser/ShowMeFire-Models
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.versioning import register_trained_model


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def _verify_spatial_assets(files, declarations):
    import numpy as np
    import onnxruntime as ort
    import xarray as xr
    required = {"model", "checkpoint", "static_bundle", "static_manifest", "evaluation", "smoke"}
    if missing := required - set(declarations): raise SystemExit(f"Spatial release assets missing: {sorted(missing)}")
    resolved = {}
    for role, declaration in declarations.items():
        path = files.get(declaration["filename"])
        if not path or _sha256(path) != declaration["sha256"]: raise SystemExit(f"Missing or invalid release asset: {role}")
        resolved[role] = path
    static_manifest = json.loads(resolved["static_manifest"].read_text())
    if static_manifest["sha256"] != _sha256(resolved["static_bundle"]): raise SystemExit("Static bundle/manifest checksum mismatch")
    with xr.open_dataset(resolved["static_bundle"]) as ds:
        if ds.sizes.get("x") != 256 or ds.sizes.get("y") != 256: raise SystemExit("Static bundle grid is not 256x256")
        if ds.attrs.get("grid_fingerprint") != declarations["static_bundle"].get("grid_fingerprint"): raise SystemExit("Static grid fingerprint mismatch")
    session = ort.InferenceSession(str(resolved["model"]), providers=["CPUExecutionProvider"])
    with np.load(resolved["smoke"]) as smoke:
        feed = {item.name: smoke[item.name] for item in session.get_inputs()}; expected = smoke["expected"]
    actual = session.run(None, feed)[0]
    if float(np.max(np.abs(actual - expected))) > 1e-4: raise SystemExit("ONNX release smoke test failed")
    return resolved


def import_release(model_type, tag, repo, bump="patch"):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        cmd = ["gh", "release", "download", tag, "--repo", repo, "--dir", str(tmp_path), "--clobber"]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        downloaded = list(tmp_path.iterdir())
        meta_files = [p for p in downloaded if p.name == "metadata.json"]
        model_files = [p for p in downloaded if p.name != "metadata.json"]

        if not model_files:
            raise SystemExit(f"No model artifact found in release assets for {tag}")
        performance = {}
        meta = {}
        if meta_files:
            meta = json.loads(meta_files[0].read_text())
            performance = meta.get("performance", {})
        performance = {**performance, "source_release_tag": tag}
        declarations = meta.get("assets") or {}
        if declarations:
            resolved = _verify_spatial_assets({path.name: path for path in model_files}, declarations)
            assets = {role: {"path": resolved[role], **{key: value for key, value in declaration.items() if key not in ("file", "filename", "sha256")}}
                      for role, declaration in declarations.items()}
            version = register_trained_model(model_type=model_type, performance=performance, bump=bump, channel="beta", assets=assets)
        else:
            if len(model_files) > 1: raise SystemExit(f"Expected exactly one model file asset, found {len(model_files)}")
            version = register_trained_model(model_type=model_type, source_path=model_files[0], performance=performance, bump=bump, channel="beta")

    print(f"\nImported release {tag!r} -> registered as {model_type} beta version {version} on this server.")
    print(f"Review it, then promote with: python pipelines/promote_model.py --model {model_type} --version {version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import a model from a GitHub release into this server's registry as a beta candidate")
    parser.add_argument("--model", required=True, choices=["fuel_moisture", "fire_danger", "fuel_moisture_spatial"])
    parser.add_argument("--tag", required=True, help="Release tag to import, e.g. fuel_moisture-v1.5.0-beta.1")
    parser.add_argument("--repo", default=None, help="owner/repo (defaults to SMF_GITHUB_REPO env var)")
    parser.add_argument("--bump", choices=["major", "minor", "patch"], default="patch",
                         help="Version segment to bump in this server's own sequence (default: patch)")
    args = parser.parse_args()

    repo = args.repo or os.getenv("SMF_GITHUB_REPO")
    if not repo:
        raise SystemExit("No source repo - pass --repo or set SMF_GITHUB_REPO")

    import_release(args.model, args.tag, repo, bump=args.bump)
