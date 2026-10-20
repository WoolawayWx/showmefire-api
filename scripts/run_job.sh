#!/bin/bash
# Runs one cron script and records the run for the admin Logs page.
#
#   run_job.sh <job-id> <script> [args...]
#
# Prints a UTC start/finish banner around the script's output (into whatever
# log cron redirects to) and stores start time, duration and exit code via
# `python -m services.job_runs record`. A script can leave a one-line
# summary for that record (e.g. "18 ok, 1 failed: fuelmoisturemap.py") by
# writing it to the file named in $JOB_SUMMARY_FILE. The job ids are listed
# in services/log_viewer.py's CRON_JOBS; the cron entries are in api/Dockerfile.

set -u

JOB_ID="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

if [ -f "/opt/venv/bin/python" ]; then
    PYTHON="/opt/venv/bin/python"
elif [ -f "$PROJECT_DIR/venv/bin/python" ]; then
    PYTHON="$PROJECT_DIR/venv/bin/python"
else
    PYTHON="python3"
fi

JOB_SUMMARY_FILE="$(mktemp)"
export JOB_SUMMARY_FILE
trap 'rm -f "$JOB_SUMMARY_FILE"' EXIT

started=$(date +%s)
echo "=== [$JOB_ID] started $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
/bin/bash "$@"
rc=$?
elapsed=$(( $(date +%s) - started ))

summary="$(head -c 400 "$JOB_SUMMARY_FILE" 2>/dev/null | tr '\n' ' ')"
if [ "$rc" -eq 0 ]; then
    echo "=== [$JOB_ID] finished OK in ${elapsed}s ${summary:+- $summary }==="
else
    echo "=== [$JOB_ID] FAILED with exit code $rc after ${elapsed}s ${summary:+- $summary }==="
fi

(cd "$PROJECT_DIR" && PYTHONPATH="$PROJECT_DIR" "$PYTHON" -m services.job_runs record "$JOB_ID" \
    --started "$started" --exit-code "$rc" --message "$summary") \
    || echo "(could not record this run for the Logs page)"

exit "$rc"
