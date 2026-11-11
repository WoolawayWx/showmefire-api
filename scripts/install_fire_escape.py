"""Install, check or roll back the fire_escape model release on this server (services/fire_escape/install.py).

    python scripts/install_fire_escape.py --latest               # newest release on SMF_GITHUB_REPO
    python scripts/install_fire_escape.py --tag fire_escape-v1.0.0-ab12cd34
    python scripts/install_fire_escape.py --file fire_escape_release.zip     # an archive copied here by hand
    python scripts/install_fire_escape.py --status               # what is installed
    python scripts/install_fire_escape.py --rollback             # back to the release this one replaced

--latest and --tag need the `gh` CLI, authenticated (GH_TOKEN, as for pipelines/import_model.py). The release
is verified and test-loaded before anything is replaced; a failed check leaves the running release alone.
Run it where the API runs (inside the container: /opt/venv/bin/python), so it has the model libraries and writes
to the data folder the job reads. On a Python without them, --no-load-test installs on checksums alone.
Installing does not turn the daily job on: that is SMF_FIRE_ESCAPE_ENABLED.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

API_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(API_ROOT))

# cron starts jobs with a near-empty environment: read the API's own .env before anything looks at a setting.
try:
    from dotenv import load_dotenv
    load_dotenv(API_ROOT / ".env")
except ImportError:
    pass

from services import fire_escape
from services.fire_escape import install


def _summary(manifest: dict) -> dict:
    return {key: manifest.get(key) for key in ("tag", "created_at", "installed_at", "load_tested", "source", "models", "neris_2026_auc_10ac")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--latest", action="store_true")
    action.add_argument("--tag")
    action.add_argument("--file", type=Path)
    action.add_argument("--status", action="store_true")
    action.add_argument("--rollback", action="store_true")
    parser.add_argument("--repo", default=None, help="owner/repo (defaults to SMF_GITHUB_REPO)")
    parser.add_argument("--no-load-test", action="store_true",
                        help="do not test-load the models (a Python without numpy/pandas/xgboost); checksums are still verified")
    args = parser.parse_args()
    target = fire_escape.bundle_dir()
    try:
        if args.status:
            current = install.installed(target)
            print(json.dumps({"folder": str(target), "installed": _summary(current) if current else None,
                              "present": (target / "county" / "bundle.json").is_file(), "job_enabled": fire_escape.enabled()}, indent=1))
            return 0
        if args.rollback:
            print(json.dumps({"rolled_back_to": _summary(install.rollback(target))}, indent=1))
            return 0
        if args.file:
            manifest = install.install_archive(args.file, target, load_test=not args.no_load_test)
        else:
            repo = args.repo or os.getenv("SMF_GITHUB_REPO")
            if not repo:
                raise SystemExit("No source repo - pass --repo or set SMF_GITHUB_REPO (or use --file)")
            manifest = install.install_release(args.tag or install.latest_tag(repo), repo, target, load_test=not args.no_load_test)
        print(json.dumps({"installed": _summary(manifest), "folder": str(target)}, indent=1))
        return 0
    except install.InstallError as error:
        raise SystemExit(f"Not installed: {error}") from error


if __name__ == "__main__":
    raise SystemExit(main())
