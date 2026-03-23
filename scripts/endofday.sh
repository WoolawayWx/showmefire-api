#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Keep legacy entrypoint behavior but delegate to the canonical wrapper.
exec "$SCRIPT_DIR/validateForecast.sh" "$@"
