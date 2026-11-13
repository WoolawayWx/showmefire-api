#!/bin/bash
# Deploy the latest showmefire-api code.
#
#   ./deploy.sh            pull, work out which containers need a restart, show the plan, ask, wait for a safe
#                          moment (only for the containers being restarted), then restart just those
#   ./deploy.sh --check    only report whether it is safe to restart everything right now
#   ./deploy.sh --yes      don't ask (for scripts); still waits for a safe moment
#   ./deploy.sh --api-only restart only the backend, no wait (the heavy jobs live in the scheduler container)
#   ./deploy.sh --all      restart backend, worker and scheduler, with the full guard
#   ./deploy.sh --force    skip the safety guard (emergencies only; kills running jobs)
#
# Which container needs what (the repo is bind-mounted at /app in all of them):
#   backend    runs the web API and the quick scheduler jobs; Python code is read once at start, so any code change restarts it.
#   scheduler  runs the heavy jobs (forecast-v1, hourly HRRR, archives...); restarted when code other than routers/, main.py,
#              tests or docs changed, because it may import it. Restarting it kills a running forecast-v1 run.
#   worker     runs cron; every cron job starts a fresh Python process, so it picks up new code by itself. Restarted only when
#              its own entrypoint changes, or after a rebuild / compose / .env change. Restarting it kills running cron jobs.
# A Dockerfile/requirements change (rebuild) or a docker-compose.yml/.env change (recreate) affects every container.
# Without a scheduler container (an unsplit deployment) the backend also runs the heavy jobs, so restarting it is guarded too.
# Env: GUARD_WAIT_MIN (default 90) = how long to wait before giving up; GUARD_LEAD_MIN (default 10)
# = how long before a daily job's start time we refuse to restart.
set -e
MODE=deploy
FORCE_SCOPE=""      # "", api or all
ASSUME_YES=false
for arg in "$@"; do
    case "$arg" in
        --check) MODE=check ;;
        --force) MODE=force ;;
        --api-only) FORCE_SCOPE=api ;;
        --all) FORCE_SCOPE=all ;;
        --yes|-y) ASSUME_YES=true ;;
        *) echo "Unknown option: $arg (use --check, --yes, --api-only, --all or --force)"; exit 2 ;;
    esac
done
ROOT=~/showmefire
REPO=$ROOT/showmefire-api
STATE_FILE=$ROOT/.deploy-config.sha256   # checksum of .env + docker-compose.yml after the last deploy
PENDING_FILE=$ROOT/.deploy-pending       # set when code was pulled but containers were not restarted (line 1: action, line 2: services)
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
# What the guard looks at, set from the restart plan (all on for --check and for a rebuild/recreate):
GUARD_CRON=true      # cron jobs running in the worker
GUARD_HEAVY=true     # forecast-v1 and the heavy scheduler jobs
GUARD_REASONS=()
guard_collect() {
    GUARD_REASONS=()
    local now_epoch job id ignored skip
    now_epoch=$(date +%s)
    # 1. Cron jobs currently running in the worker container (read from its process list).
    if [ "$GUARD_CRON" = true ] && docker inspect -f '{{.State.Running}}' show-me-fire-worker 2>/dev/null | grep -q true; then
        while read -r id; do
            [ -z "$id" ] && continue
            skip=false
            for ignored in $IGNORED_JOBS; do [ "$id" = "$ignored" ] && skip=true; done
            $skip || GUARD_REASONS+=("worker job '$id' is running")
        done < <(docker exec show-me-fire-worker sh -c 'for p in /proc/[0-9]*; do tr "\0" " " < $p/cmdline 2>/dev/null; echo; done' 2>/dev/null \
                 | awk '$1=="/bin/bash" && $2 ~ /run_job\.sh$/ {print $3}' | sort -u)
    fi
    if [ "$GUARD_HEAVY" = true ]; then
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
    fi
    # 3. A daily job is about to start (cron in the worker, or a daily heavy job in the scheduler).
    if [ "$GUARD_CRON" = true ] || [ "$GUARD_HEAVY" = true ]; then
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
    fi
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
CHANGED_FILES=$(git -C "$REPO" diff --name-only "$BEFORE" "$AFTER")
# Which containers exist / are up? (The scheduler is added to docker-compose.yml at cutover.)
HAS_SCHEDULER=false
if docker compose config --services 2>/dev/null | grep -qx scheduler; then HAS_SCHEDULER=true; fi
SCHEDULER_UP=false
if docker inspect -f '{{.State.Running}}' show-me-fire-scheduler 2>/dev/null | grep -q true; then SCHEDULER_UP=true; fi
CONTAINERS="show-me-fire-api show-me-fire-worker"
[ "$HAS_SCHEDULER" = true ] && CONTAINERS="$CONTAINERS show-me-fire-scheduler"
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
if echo "$CHANGED_FILES" | grep -qE '^(requirements.*\.txt|Dockerfile)$'; then
    ACTION=rebuild
elif [ "$CONFIG_CHANGED" = true ]; then
    ACTION=recreate
elif [ "$BEFORE" != "$AFTER" ]; then
    ACTION=restart
fi
# A previous run pulled code but was stopped by the guard: finish that job now, at the strongest level.
PENDING_SERVICES=""
if [ -f "$PENDING_FILE" ]; then
    PENDING=$(sed -n 1p "$PENDING_FILE")
    PENDING_SERVICES=$(sed -n 2p "$PENDING_FILE")
    echo "Earlier deploy is still pending ($PENDING${PENDING_SERVICES:+: $PENDING_SERVICES})."
    if [ "$PENDING" = rebuild ] || [ "$ACTION" = rebuild ]; then ACTION=rebuild
    elif [ "$ACTION" = none ] || [ "$ACTION" = restart ]; then ACTION=$PENDING
    fi
fi
if [ "$ACTION" = none ]; then
    echo "No new commits pulled; nothing to deploy."
    exit 0
fi
# ---------------------------------------------------------------- plan: which containers?
RESTART_BACKEND=false
RESTART_SCHEDULER=false
RESTART_WORKER=false
REASON_BACKEND="" REASON_SCHEDULER="" REASON_WORKER=""
NOTE_SKIPPED=()
plan_restart() {
    if [ "$ACTION" = rebuild ] || [ "$ACTION" = recreate ]; then
        RESTART_BACKEND=true; RESTART_WORKER=true
        [ "$HAS_SCHEDULER" = true ] && RESTART_SCHEDULER=true
        REASON_BACKEND="$ACTION"; REASON_WORKER="$ACTION"; REASON_SCHEDULER="$ACTION"
        return
    fi
    local code heavy CODE_ONLY
    # Anything that is not documentation, tests or a shell/cron script is code the backend has to reload.
    code=$(echo "$CHANGED_FILES" | grep -vE '^(docs/|tests/|.*\.md$|scripts/.*\.sh$|scripts/localhost_forward\.py$)' || true)
    # The scheduler may import anything except the web routers, main.py, tests and docs.
    CODE_ONLY="$code"
    heavy=$(echo "$CODE_ONLY" | grep -vE '^(routers/|main\.py$)' || true)
    if [ -n "$code" ]; then
        RESTART_BACKEND=true
        REASON_BACKEND="code changed ($(echo "$code" | head -3 | tr '\n' ' ')...)"
    else
        NOTE_SKIPPED+=("backend: only docs, tests or cron scripts changed")
    fi
    if [ "$HAS_SCHEDULER" = true ]; then
        if [ -n "$heavy" ]; then
            RESTART_SCHEDULER=true
            REASON_SCHEDULER="code it may import changed ($(echo "$heavy" | head -3 | tr '\n' ' ')...)"
        elif [ -n "$code" ]; then
            NOTE_SKIPPED+=("scheduler: only API code (routers/, main.py) changed")
        fi
    fi
    # Cron scripts start a fresh Python process each run, so new code reaches them without a restart.
    if echo "$CHANGED_FILES" | grep -qE '^scripts/(worker-entrypoint\.sh|localhost_forward\.py)$'; then
        RESTART_WORKER=true
        REASON_WORKER="worker entrypoint changed"
    elif [ -n "$code" ]; then
        NOTE_SKIPPED+=("worker: cron jobs pick up new code on their next run")
    fi
}
plan_restart
# Finish an earlier, guard-blocked deploy too.
if [ -f "$PENDING_FILE" ] && [ -z "$PENDING_SERVICES" ] && [ "$ACTION" = restart ]; then
    PENDING_SERVICES="backend scheduler worker"      # an older pending file did not record the plan: do everything
fi
for s in $PENDING_SERVICES; do
    case "$s" in
        backend) RESTART_BACKEND=true; REASON_BACKEND="${REASON_BACKEND:-pending from an earlier deploy}" ;;
        scheduler) [ "$HAS_SCHEDULER" = true ] && RESTART_SCHEDULER=true && REASON_SCHEDULER="${REASON_SCHEDULER:-pending from an earlier deploy}" ;;
        worker) RESTART_WORKER=true; REASON_WORKER="${REASON_WORKER:-pending from an earlier deploy}" ;;
    esac
done
set_scope_api() { RESTART_BACKEND=true; RESTART_SCHEDULER=false; RESTART_WORKER=false; }
set_scope_all() { RESTART_BACKEND=true; RESTART_WORKER=true; [ "$HAS_SCHEDULER" = true ] && RESTART_SCHEDULER=true; }
[ "$FORCE_SCOPE" = api ] && [ "$ACTION" = restart ] && set_scope_api
[ "$FORCE_SCOPE" = all ] && set_scope_all
if [ "$ACTION" = restart ] && [ "$RESTART_BACKEND" = false ] && [ "$RESTART_SCHEDULER" = false ] && [ "$RESTART_WORKER" = false ]; then
    echo "Pulled $(echo "$CHANGED_FILES" | wc -l) changed file(s); nothing needs a restart (${NOTE_SKIPPED[*]:-only docs, tests or cron scripts changed})."
    echo "$CONFIG_SUM" > "$STATE_FILE"
    rm -f "$PENDING_FILE"
    exit 0
fi
save_pending() {
    local list=""
    [ "$RESTART_BACKEND" = true ] && list="$list backend"
    [ "$RESTART_SCHEDULER" = true ] && list="$list scheduler"
    [ "$RESTART_WORKER" = true ] && list="$list worker"
    printf '%s\n%s\n' "$ACTION" "${list# }" > "$PENDING_FILE"
}
describe_plan() {
    echo
    echo "Deploy plan ($ACTION):"
    if [ "$ACTION" = rebuild ] || [ "$ACTION" = recreate ]; then
        echo "  recreate every container (the image or the compose/.env config changed)"
    else
        [ "$RESTART_BACKEND" = true ]   && echo "  restart backend   - ${REASON_BACKEND:-requested}"
        [ "$RESTART_SCHEDULER" = true ] && echo "  restart scheduler - ${REASON_SCHEDULER:-requested}  (kills a running forecast-v1 run)"
        [ "$RESTART_WORKER" = true ]    && echo "  restart worker    - ${REASON_WORKER:-requested}  (kills running cron jobs)"
        local n
        for n in "${NOTE_SKIPPED[@]}"; do echo "  leave alone       - $n"; done
        if [ "$HAS_SCHEDULER" = true ] && [ "$RESTART_SCHEDULER" = false ] && [ "$SCHEDULER_UP" = false ]; then
            echo "  note: the scheduler container is not running; the backend would run the heavy jobs itself."
        fi
    fi
    echo
}
# Ask (only on a terminal, and not when a scope was forced or --yes was given).
if [ "$MODE" != force ] && [ "$ASSUME_YES" = false ] && [ -z "$FORCE_SCOPE" ] && [ -t 0 ] && [ -t 1 ]; then
    describe_plan
    if [ "$ACTION" = restart ]; then
        read -r -p "[Enter] go   [a] API only   [e] everything (full guard)   [q] quit > " CHOICE
        case "$CHOICE" in
            a|A) set_scope_api; echo "-> API only." ;;
            e|E) set_scope_all; echo "-> everything." ;;
            q|Q) save_pending; echo "Cancelled. The new code is pulled; run ./deploy.sh again to finish."; exit 0 ;;
            *) ;;
        esac
    else
        read -r -p "[Enter] go   [q] quit > " CHOICE
        case "$CHOICE" in q|Q) save_pending; echo "Cancelled. Run ./deploy.sh again to finish."; exit 0 ;; esac
    fi
else
    describe_plan
fi
# The guard only looks at what is about to be restarted.
GUARD_CRON=false
GUARD_HEAVY=false
if [ "$ACTION" = rebuild ] || [ "$ACTION" = recreate ] || [ "$FORCE_SCOPE" = all ]; then
    GUARD_CRON=true; GUARD_HEAVY=true
else
    [ "$RESTART_WORKER" = true ] && GUARD_CRON=true
    [ "$RESTART_SCHEDULER" = true ] && GUARD_HEAVY=true
    # Without a running scheduler container the backend runs the heavy jobs, so restarting it is guarded like the scheduler.
    if [ "$RESTART_BACKEND" = true ] && [ "$SCHEDULER_UP" = false ]; then GUARD_HEAVY=true; fi
fi
# ---------------------------------------------------------------- guard
if [ "$MODE" = force ]; then
    echo "--force: skipping the safety guard."
elif [ "$GUARD_CRON" = false ] && [ "$GUARD_HEAVY" = false ]; then
    echo "Restarting only containers that run nothing a restart would lose; no wait needed."
else
    deadline=$(( $(date +%s) + GUARD_WAIT_MIN * 60 ))
    while true; do
        guard_collect
        [ ${#GUARD_REASONS[@]} -eq 0 ] && break
        echo "$(date +%H:%M:%S) Waiting for a safe moment to restart ($ACTION):"
        guard_print
        if [ "$(date +%s)" -ge "$deadline" ]; then
            save_pending
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
        SERVICES=""
        [ "$RESTART_BACKEND" = true ] && SERVICES="$SERVICES backend"
        [ "$RESTART_SCHEDULER" = true ] && SERVICES="$SERVICES scheduler"
        [ "$RESTART_WORKER" = true ] && SERVICES="$SERVICES worker"
        echo "Restarting:$SERVICES"
        docker compose restart $SERVICES
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
