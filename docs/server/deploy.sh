#!/bin/bash
# Deploy the latest showmefire-api code.
#
#   ./deploy.sh            pull, then restart/recreate containers once it is safe (waits for forecasts)
#   ./deploy.sh --check    only report whether it is safe to restart containers right now
#   ./deploy.sh --api-only pull, restart ONLY the backend, no wait (the heavy jobs live in the scheduler container)
#   ./deploy.sh --force    skip the safety guard (emergencies only; kills running jobs)
#
# Restarting the worker kills any cron job running in it (nothing resumes), restarting the scheduler kills the
# heavy jobs it is running (forecast-v1, archives...), and restarting the backend of an UNSPLIT deployment (no
# scheduler container yet) kills an in-progress forecast-v1 run. The guard waits for those to finish first.
# --api-only is only for code the scheduler and worker don't need to pick up; it refuses to run if the scheduler
# container isn't up.
# Env: GUARD_WAIT_MIN (default 90) = how long to wait before giving up; GUARD_LEAD_MIN (default 10)
# = how long before a daily job's start time we refuse to restart.
set -e
MODE=deploy
SCOPE=all
for arg in "$@"; do
    case "$arg" in
        --check) MODE=check ;;
        --force) MODE=force ;;
        --api-only) SCOPE=api ;;
        *) echo "Unknown option: $arg (use --check, --force or --api-only)"; exit 2 ;;
    esac
done
ROOT=~/showmefire
REPO=$ROOT/showmefire-api
STATE_FILE=$ROOT/.deploy-config.sha256   # checksum of .env + docker-compose.yml after the last deploy
PENDING_FILE=$ROOT/.deploy-pending       # set when code was pulled but containers were not restarted
GUARD_WAIT_MIN=${GUARD_WAIT_MIN:-90}
GUARD_LEAD_MIN=${GUARD_LEAD_MIN:-10}
# ---------------------------------------------------------------- guard
# Frequent jobs that simply run again minutes later; everything else in the worker is a daily job
# that would be lost for the day if killed.
IGNORED_JOBS="maps fire_outlook"
# Heavy scheduler jobs that run every few minutes and simply resume or run again, so a restart loses nothing.
IGNORED_HEAVY="hrrr_hourly_forecast capture_latest_mrms end_of_day_archive rtma_spread_rate_pipeline forecast_v1_admin_requests"
# The scheduler container writes this file every minute (running jobs, next runs).
HEAVY_STATE=$REPO/data/current_conditions/status/scheduler_heavy.json
# forecast-v1 phases that are safe to interrupt (see guard_collect).
SAFE_FORECAST_PHASES="queued preparing acquiring_sources"
# Daily job start times (UTC HH:MM). empty_outlook_maps is scheduled in America/Chicago.
DAILY_UTC="04:30 06:40 12:05 13:00 15:00"
DAILY_CHICAGO="02:00"
GUARD_REASONS=()
guard_collect() {
    GUARD_REASONS=()
    local now_epoch job id ignored skip
    now_epoch=$(date +%s)
    # 1. Cron jobs currently running in the worker container (read from its process list).
    if docker inspect -f '{{.State.Running}}' show-me-fire-worker 2>/dev/null | grep -q true; then
        while read -r id; do
            [ -z "$id" ] && continue
            skip=false
            for ignored in $IGNORED_JOBS; do [ "$id" = "$ignored" ] && skip=true; done
            $skip || GUARD_REASONS+=("worker job '$id' is running")
        done < <(docker exec show-me-fire-worker sh -c 'for p in /proc/[0-9]*; do tr "\0" " " < $p/cmdline 2>/dev/null; echo; done' 2>/dev/null \
                 | awk '$1=="/bin/bash" && $2 ~ /run_job\.sh$/ {print $3}' | sort -u)
    fi
    # 2. forecast-v1 run (in the scheduler container once it exists, otherwise in the backend): the job file must say
    #    running/queued AND have been updated recently (a run killed by an earlier restart stops updating and is
    #    treated as stale). These are files on the host, so this works whichever container runs the job.
    local job_file=$REPO/data/forecast-v1/admin-job.json status updated age
    if [ -f "$job_file" ]; then
        status=$(jq -r '.status // "idle"' "$job_file" 2>/dev/null || echo idle)
        updated=$(jq -r '.updated_at // .started_at // empty' "$job_file" 2>/dev/null || true)
        if [ "$status" = "running" ] || [ "$status" = "queued" ]; then
            phase=$(jq -r '.phase // "?"' "$job_file" 2>/dev/null || echo "?")
            # Downloading is the part that takes "forever" (30-100+ min), but it is safe to interrupt:
            # raw downloads are kept in download-cache and the next scheduled run (every 30 min) retries.
            # Only the processing/publishing phases (~65 min: station_extraction ~55 min + ~10 min) are
            # worth waiting for, because their work is lost if killed.
            case " $SAFE_FORECAST_PHASES " in
                *" $phase "*) ;;  # interruptible
                *)
                    if [ -n "$updated" ] && age=$(( now_epoch - $(date -d "$updated" +%s 2>/dev/null || echo 0) )) && [ "$age" -lt 2700 ]; then
                        GUARD_REASONS+=("forecast-v1 run is $status (phase '$phase', updated $((age/60)) min ago)")
                    fi
                    ;;
            esac
        fi
    fi
    # Also a recent run still staging in the database.
    local cutoff staging
    cutoff=$(date -u -d '3 hours ago' +%Y-%m-%dT%H:%M:%SZ)
    staging=$(sqlite3 -readonly "$REPO/data/showmefire.db" "SELECT count(*) FROM forecast_runs WHERE status='staging' AND created_at_utc > '$cutoff';" 2>/dev/null || echo 0)
    [ "${staging:-0}" -gt 0 ] && GUARD_REASONS+=("$staging forecast-v1 run(s) still staging in the database")
    # 2b. Jobs running right now in the scheduler container (ignore a heartbeat file that has gone stale).
    if [ -f "$HEAVY_STATE" ]; then
        local beat_age hjob hskip hi
        beat_age=$(( now_epoch - $(stat -c %Y "$HEAVY_STATE") ))
        if [ "$beat_age" -lt 300 ]; then
            while read -r hjob; do
                [ -z "$hjob" ] && continue
                hskip=false
                for hi in $IGNORED_HEAVY; do [ "$hjob" = "$hi" ] && hskip=true; done
                $hskip || GUARD_REASONS+=("scheduler job '$hjob' is running")
            done < <(jq -r '(.running_jobs // [])[]' "$HEAVY_STATE" 2>/dev/null)
        fi
    fi
    # 3. A daily job is about to start.
    local t start delta lead=$(( GUARD_LEAD_MIN * 60 ))
    for t in $DAILY_UTC; do
        for day in today tomorrow; do
            start=$(date -u -d "$day $t" +%s)
            delta=$(( start - now_epoch ))
            if [ "$delta" -ge 0 ] && [ "$delta" -le "$lead" ]; then
                GUARD_REASONS+=("daily job due at $t UTC (in $((delta/60)) min)")
            fi
        done
    done
    for t in $DAILY_CHICAGO; do
        for day in today tomorrow; do
            start=$(TZ=America/Chicago date -d "$day $t" +%s)
            delta=$(( start - now_epoch ))
            if [ "$delta" -ge 0 ] && [ "$delta" -le "$lead" ]; then
                GUARD_REASONS+=("daily job due at $t Central (in $((delta/60)) min)")
            fi
        done
    done
}
guard_print() {
    local r
    for r in "${GUARD_REASONS[@]}"; do echo "  - $r"; done
}
# ---------------------------------------------------------------- --check
if [ "$MODE" = check ]; then
    guard_collect
    if [ ${#GUARD_REASONS[@]} -eq 0 ]; then
        echo "SAFE: nothing is running that a restart would interrupt."
        exit 0
    fi
    echo "NOT SAFE to restart containers right now:"
    guard_print
    exit 2
fi
# ---------------------------------------------------------------- pull
cd "$REPO"
BEFORE=$(git rev-parse HEAD)
git pull
AFTER=$(git rev-parse HEAD)
cd "$ROOT"
# Is the scheduler container part of this deployment yet? (It is added to docker-compose.yml at cutover.)
HAS_SCHEDULER=false
if docker compose config --services 2>/dev/null | grep -qx scheduler; then HAS_SCHEDULER=true; fi
SERVICES="backend worker"
CONTAINERS="show-me-fire-api show-me-fire-worker"
if [ "$HAS_SCHEDULER" = true ]; then
    SERVICES="$SERVICES scheduler"
    CONTAINERS="$CONTAINERS show-me-fire-scheduler"
fi
# What does this deploy need?
#   .env is gitignored and docker-compose.yml lives outside the repo, so git can't tell us when they
#   change. Compare a checksum against the one saved after the last deploy. `docker compose restart`
#   does not re-read either file; only `up -d` does.
CONFIG_SUM=$(cat "$REPO/.env" "$ROOT/docker-compose.yml" | sha256sum | cut -d' ' -f1)
CONFIG_CHANGED=false
if [ ! -f "$STATE_FILE" ] || [ "$(cat "$STATE_FILE")" != "$CONFIG_SUM" ]; then
    CONFIG_CHANGED=true
fi
ACTION=none
if git -C "$REPO" diff --name-only "$BEFORE" "$AFTER" | grep -qE '^(requirements.*\.txt|Dockerfile)$'; then
    ACTION=rebuild
elif [ "$CONFIG_CHANGED" = true ]; then
    ACTION=recreate
elif [ "$BEFORE" != "$AFTER" ]; then
    ACTION=restart
fi
# A previous run pulled code but was stopped by the guard: finish that job now, at the strongest level.
if [ -f "$PENDING_FILE" ]; then
    PENDING=$(cat "$PENDING_FILE")
    echo "Earlier deploy is still pending ($PENDING)."
    if [ "$PENDING" = rebuild ] || [ "$ACTION" = rebuild ]; then ACTION=rebuild
    elif [ "$ACTION" = none ] || [ "$ACTION" = restart ]; then ACTION=$PENDING
    fi
fi
if [ "$ACTION" = none ]; then
    echo "No new commits pulled; nothing to deploy."
    exit 0
fi
# ---------------------------------------------------------------- guard
if [ "$MODE" = force ]; then
    echo "--force: skipping the safety guard."
elif [ "$SCOPE" = api ] && [ "$ACTION" = restart ]; then
    # Only safe once the heavy jobs live in their own container; otherwise this would kill them with the backend.
    if ! docker inspect -f '{{.State.Running}}' show-me-fire-scheduler 2>/dev/null | grep -q true; then
        echo "--api-only needs the scheduler container running (without it the heavy jobs run inside the backend)."
        echo "Use plain ./deploy.sh instead."
        exit 2
    fi
    echo "--api-only: restarting only the backend; the scheduler and worker are untouched."
else
    deadline=$(( $(date +%s) + GUARD_WAIT_MIN * 60 ))
    while true; do
        guard_collect
        [ ${#GUARD_REASONS[@]} -eq 0 ] && break
        echo "$(date +%H:%M:%S) Waiting for a safe moment to restart ($ACTION):"
        guard_print
        if [ "$(date +%s)" -ge "$deadline" ]; then
            echo "$ACTION" > "$PENDING_FILE"
            echo "Gave up after ${GUARD_WAIT_MIN} min. The new code is pulled but containers were NOT restarted."
            echo "Run ./deploy.sh again later (it will finish this deploy), or ./deploy.sh --force to restart now."
            exit 1
        fi
        sleep 30
    done
    echo "Safe to restart."
fi
# ---------------------------------------------------------------- apply
case "$ACTION" in
    rebuild)
        echo "Dependency/Dockerfile changes -- full rebuild."
        docker compose up -d --build
        # Drop build cache untouched for a week; recent layers stay so the next rebuild is still fast.
        docker builder prune -f --filter until=168h || echo "Build cache prune failed (non-fatal)."
        ;;
    recreate)
        echo ".env or docker-compose.yml changed -- recreating containers so the new config is applied."
        docker compose up -d
        ;;
    restart)
        if [ "$SCOPE" = api ]; then
            echo "Code-only change, API only -- restarting backend."
            docker compose restart backend
        else
            echo "Code-only changes -- restarting $SERVICES without rebuild."
            docker compose restart $SERVICES
        fi
        ;;
esac
echo "$CONFIG_SUM" > "$STATE_FILE"
rm -f "$PENDING_FILE"
# ---------------------------------------------------------------- health check
# The containers must be running and not crash-looping.
echo "Checking container health..."
sleep 5
FAILED=false
for name in $CONTAINERS; do
    for attempt in 1 2 3 4 5 6; do
        STATUS=$(docker inspect -f '{{.State.Status}}' "$name" 2>/dev/null || echo missing)
        [ "$STATUS" = "running" ] && break
        sleep 5
    done
    if [ "$STATUS" != "running" ]; then
        echo "ERROR: $name is $STATUS. Last log lines:"
        docker logs --tail 30 "$name" 2>&1 || true
        FAILED=true
    else
        sleep 5
        if [ "$(docker inspect -f '{{.State.Status}}' "$name")" != "running" ]; then
            echo "ERROR: $name stopped again after starting. Last log lines:"
            docker logs --tail 30 "$name" 2>&1 || true
            FAILED=true
        else
            echo "  $name: running"
        fi
    fi
done
if [ "$FAILED" = true ]; then
    echo "Deployment FINISHED WITH ERRORS -- see above."
    exit 1
fi
echo "Deployment complete!"
