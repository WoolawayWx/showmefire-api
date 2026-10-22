#!/usr/bin/env bash
# Host-level disk cleanup the API container can't do itself (Docker build cache, unused images,
# the systemd journal). Run on the server as a user that can use docker and sudo:
#
#   bash scripts/host_disk_cleanup.sh           # show what would be freed
#   bash scripts/host_disk_cleanup.sh --apply   # free it
#
# Safe: keeps the last week of build cache, removes only images no container uses, and keeps
# 500 MB of journal. The app's own data is handled by services/storage_cleanup.py.
set -euo pipefail

apply=false
[[ "${1:-}" == "--apply" ]] && apply=true

echo "== Disks"
df -h / /mnt/adrive 2>/dev/null || df -h /

echo "== Docker"
docker system df

echo "== Journal"
journalctl --disk-usage

if ! $apply; then
  echo
  echo "Dry run. With --apply this runs:"
  echo "  docker builder prune -f --filter until=168h   (build cache older than a week)"
  echo "  docker image prune -a -f --filter until=168h  (images no container uses, older than a week)"
  echo "  sudo journalctl --vacuum-size=500M"
  exit 0
fi

docker builder prune -f --filter until=168h
docker image prune -a -f --filter until=168h
sudo journalctl --vacuum-size=500M

echo "== After"
df -h / /mnt/adrive 2>/dev/null || df -h /
