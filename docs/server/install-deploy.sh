#!/bin/bash
# Install the deployment tooling for the existing split production layout.
set -euo pipefail

if [ "$#" -gt 1 ]; then
    echo "Usage: bash install-deploy.sh [release-ref]" >&2
    exit 2
fi

root=${SMF_DEPLOY_ROOT:-"$HOME/showmefire"}
cd "$root"
root=$(pwd -P)
repo=$root/showmefire-api
compose=$root/docker-compose.yml
live_deploy=$root/deploy.sh
default_ref='@{upstream}'
revision=$(git -C "$repo" rev-parse --verify --end-of-options "${1:-$default_ref}^{commit}")
git -C "$repo" merge-base --is-ancestor HEAD "$revision" ||
    { echo "The release must be a fast-forward from the server checkout." >&2; exit 1; }

exec 9>>"$root/.deploy.lock"
flock -n 9 || { echo "Another installation or deployment is in progress." >&2; exit 1; }

stage=$(mktemp -d "$root/.install-deploy.XXXXXX")
backup=
installing=false
cleanup() {
    local result=$?
    if $installing; then
        if cp -p "$backup/docker-compose.yml" "$compose" && cp -p "$backup/deploy.sh" "$live_deploy"; then
            echo "Restored the original deployment files." >&2
        else
            echo "Restore the deployment files from $backup." >&2
            result=1
        fi
    fi
    rm -rf -- "$stage"
    exit "$result"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

git -C "$repo" show "$revision:docs/server/deploy.sh" > "$stage/deploy.sh"
bash -n "$stage/deploy.sh"
chmod 755 "$stage/deploy.sh"
git -C "$repo" show "$revision:Dockerfile" | grep -F 'smf.long_job_lock=1' >/dev/null ||
    { echo "The release is missing the Dockerfile changes for the shared job lock." >&2; exit 1; }

# Keep the live Compose file and modify only the two service environment lists.
# The server's split layout uses block-style environment entries.
cp -p "$compose" "$stage/docker-compose.yml"
python3 - "$stage/docker-compose.yml" <<'PY'
from pathlib import Path
import re
import sys

path = Path(sys.argv[1])
lines = path.read_text().splitlines(keepends=True)
services = {"forecast", "worker-long"}
updated = set()

for index in range(len(lines) - 1, -1, -1):
    # Work backwards so insertions leave the remaining indices intact.
    match = re.fullmatch(r"  (forecast|worker-long):\s*(?:#.*)?", lines[index].rstrip())
    if not match:
        continue
    service = match.group(1)
    for field in range(index + 1, len(lines)):
        line = lines[field]
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if len(line) - len(line.lstrip()) <= 2:
            break
        if not re.fullmatch(r"    environment:\s*(?:#.*)?", line.rstrip()):
            continue
        entries = []
        for entry in lines[field + 1:]:
            if not entry.strip() or entry.lstrip().startswith("#"):
                continue
            if len(entry) - len(entry.lstrip()) <= 4:
                break
            entries.append(entry)
        if not entries or not all(entry.startswith("      - ") for entry in entries):
            break
        if not any(re.match(r"""      -\s*['"]?SMF_LONG_JOB_LOCK(?:=|['"\s#]|$)""", entry) for entry in entries):
            lines.insert(field + 1, "      - SMF_LONG_JOB_LOCK=/app/.locks/long-deploy.lock\n")
        updated.add(service)
        break

if updated != services:
    raise SystemExit("Expected block-style environment lists for forecast and worker-long in the active Compose file.")
path.write_text("".join(lines))
PY

verify_compose() {
    docker compose "$@" config --format json |
        jq -e --arg source "$repo/.locks" '
            [.services.forecast, .services["worker-long"]]
            | all(.environment.SMF_LONG_JOB_LOCK == "/app/.locks/long-deploy.lock" and
                  any(.volumes[]?; .type == "bind" and .source == $source and .target == "/app/.locks"))
        ' >/dev/null
}

verify_compose --project-directory "$root" -f "$stage/docker-compose.yml" ||
    { echo "The proposed Compose configuration failed validation. The live files are unchanged." >&2; exit 1; }

backup=$(mktemp -d "$root/deploy-backup.XXXXXX")
cp -p "$live_deploy" "$compose" "$backup/"
mkdir -p "$repo/.locks" "$repo/images"
installing=true
mv -f "$stage/docker-compose.yml" "$compose"
mv -f "$stage/deploy.sh" "$live_deploy"
verify_compose
installing=false

echo "Installed deploy.sh and updated docker-compose.yml for ${revision:0:12}."
echo "Backups saved in $backup"
echo "The next normal deploy will pick up the tracked Dockerfile from the release."
