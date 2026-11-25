#!/bin/bash
# Deploy the upstream revision on the ARM64 host. See docs/long_jobs.md.
set -eo pipefail

MODE=deploy
SCOPE=normal
ASSUME_YES=false
FORCE=false
for arg in "$@"; do
    case "$arg" in
        --check) MODE=check ;;
        --long) MODE=long ;;
        --force) FORCE=true ;;
        --api-only) SCOPE=api ;;
        --all) SCOPE=all ;;
        --yes|-y) ASSUME_YES=true ;;
        *) echo "Unknown option: $arg" >&2; exit 2 ;;
    esac
done

ROOT=${SMF_DEPLOY_ROOT:-"$HOME/showmefire"}
ROOT=$(cd "$ROOT" && pwd -P)
REPO=$ROOT/showmefire-api
STATE=$ROOT/.deploy-state
STATE_FILE=$ROOT/.deploy-config.sha256
PENDING_FILE=$ROOT/.deploy-pending
LONG_PENDING_FILE=$ROOT/.long-pending
LONG_LOCK_PATH=/app/.locks/long-deploy.lock
GUARD_WAIT_MIN=${GUARD_WAIT_MIN:-90}
GUARD_LEAD_MIN=${GUARD_LEAD_MIN:-10}
DEPLOY_HEALTH_WAIT_SECONDS=${DEPLOY_HEALTH_WAIT_SECONDS:-120}

cd "$ROOT"
exec 9>"$ROOT/.deploy.lock"
flock -n 9 || { echo "Another deployment is in progress." >&2; exit 1; }

CONFIG=$(docker compose config --format json)
HAS_SCHEDULER=false
HAS_LONG=false
SPLIT=false
if jq -e '.services.scheduler' >/dev/null <<< "$CONFIG"; then HAS_SCHEDULER=true; fi
if jq -e '.services.forecast and .services["worker-long"]' >/dev/null <<< "$CONFIG"; then HAS_LONG=true; fi
if jq -e '
    .services.backend.environment.SMF_SCHEDULER_GROUPS == "api" and
    .services.scheduler.command == ["python", "-m", "core.scheduler_worker", "--groups", "heavy"] and
    .services.worker.environment.SMF_CRON_SET == "short" and
    .services.forecast.command == ["python", "-m", "core.scheduler_worker", "--groups", "forecast"] and
    .services["worker-long"].environment.SMF_CRON_SET == "long"
' >/dev/null <<< "$CONFIG"; then SPLIT=true; fi

container_name() {
    case "$1" in
        backend) echo show-me-fire-api ;;
        *) echo "show-me-fire-$1" ;;
    esac
}

inspect_container() {
    docker inspect -f "$2" "$(container_name "$1")" 2>/dev/null
}

is_running() {
    [ "$(inspect_container "$1" '{{.State.Running}}')" = true ]
}

uses_long_lock() {
    [ "$(inspect_container "$1" '{{index .Config.Labels "smf.long_job_lock"}}')" = 1 ] &&
        inspect_container "$1" '{{range .Config.Env}}{{println .}}{{end}}' |
            grep -Fx "SMF_LONG_JOB_LOCK=$LONG_LOCK_PATH" >/dev/null
}

forecast_heartbeat() {
    local revision=${1:-} host started written now
    host=$(inspect_container forecast '{{.Config.Hostname}}') || return 1
    started=$(inspect_container forecast '{{.State.StartedAt}}') || return 1
    started=$(date -d "$started" +%s) || return 1
    written=$(jq -er --arg revision "$revision" --arg host "$host" '
        select(.host == $host and .groups == ["forecast"] and ($revision == "" or .git_sha == $revision))
        | .written_at
    ' "$LONG_STATE" 2>/dev/null) || return 1
    written=$(date -d "$written" +%s) || return 1
    now=$(date +%s)
    [ "$written" -ge "$started" ] && [ "$written" -le "$now" ] && [ "$((now - written))" -lt 120 ]
}

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
# Daily jobs that run in the long containers (12z/09z forecasts, ensemble). When those are separate they no longer matter
# for restarting the backend, scheduler or worker, only for moving the long containers (guard_collect_long).
LONG_DAILY_UTC="01:30 06:40 10:30 12:05 13:00 15:00 16:30 19:30 22:30"
[ "$HAS_LONG" = true ] && DAILY_UTC="04:30 06:40"
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
    if [ "$GUARD_HEAVY" = true ] && { [ "$HAS_LONG" = false ] || ! is_running forecast; }; then
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
    fi
    if [ "$GUARD_HEAVY" = true ]; then
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
# ---------------------------------------------------------------- long containers
# The long containers run a pinned image, so the question is only "may I stop them now?": no forecast-v1 run in progress
# (same rules as the old guard), no long cron job running, no long daily job about to start.
LONG_STATE=$REPO/data/current_conditions/status/scheduler_forecast.json
LONG_REASONS=()
guard_collect_long() {
    LONG_REASONS=()
    local now_epoch id job_file status updated age cutoff staging t day start delta lead processes
    now_epoch=$(date +%s)
    if docker inspect -f '{{.State.Running}}' show-me-fire-worker-long 2>/dev/null | grep -q true; then
        if processes=$(docker top show-me-fire-worker-long -eo args); then
            while read -r id; do
                [ -n "$id" ] && LONG_REASONS+=("long cron job '$id' is running")
            done < <(awk '$1 ~ /(^|\/)bash$/ && $2 ~ /run_job\.sh$/ {print $3}' <<< "$processes")
        else
            LONG_REASONS+=("could not inspect long cron jobs")
        fi
    fi
    job_file=$REPO/data/forecast-v1/admin-job.json
    if [ -f "$job_file" ]; then
        status=$(jq -r '.status // "idle"' "$job_file" 2>/dev/null || echo idle)
        updated=$(jq -r '.updated_at // .started_at // empty' "$job_file" 2>/dev/null || true)
        if [ "$status" = running ] && [ -n "$updated" ] \
            && age=$(( now_epoch - $(date -d "$updated" +%s 2>/dev/null || echo 0) )) && [ "$age" -lt 2700 ]; then
            LONG_REASONS+=("forecast-v1 admin run is active")
        fi
    fi
    cutoff=$(date -u -d '3 hours ago' +%Y-%m-%dT%H:%M:%SZ)
    if [ -f "$REPO/data/showmefire.db" ]; then
        if staging=$(sqlite3 -readonly "$REPO/data/showmefire.db" "SELECT count(*) FROM forecast_runs WHERE status='staging' AND created_at_utc > '$cutoff';"); then
            [ "${staging:-0}" -gt 0 ] && LONG_REASONS+=("$staging forecast-v1 run(s) still staging in the database")
        else
            LONG_REASONS+=("could not check forecast staging")
        fi
    fi
    if is_running forecast && ! uses_long_lock forecast && ! forecast_heartbeat; then
        LONG_REASONS+=("legacy forecast worker has no fresh heartbeat")
    fi
    if [ -f "$LONG_STATE" ] && [ $(( now_epoch - $(stat -c %Y "$LONG_STATE") )) -lt 300 ]; then
        while read -r id; do
            [ -z "$id" ] && continue
            LONG_REASONS+=("forecast worker job '$id' is running")
        done < <(jq -r '(.running_jobs // [])[]' "$LONG_STATE" 2>/dev/null)
    fi
    # Before the cutover the OLD scheduler container still runs forecast-v1 (its heartbeat lists it): starting the forecast
    # container now would start a second run, so treat that run as busy too.
    local HEAVY_BEAT=$REPO/data/current_conditions/status/scheduler_heavy.json
    if [ -f "$HEAVY_BEAT" ] && [ $(( now_epoch - $(stat -c %Y "$HEAVY_BEAT") )) -lt 300 ] \
       && jq -e '(.running_jobs // []) | index("run_forecast_v1_shadow")' "$HEAVY_BEAT" >/dev/null 2>&1; then
        LONG_REASONS+=("forecast-v1 is running in the scheduler container (old layout)")
    fi
    lead=$(( GUARD_LEAD_MIN * 60 ))
    for t in $LONG_DAILY_UTC; do
        for day in today tomorrow; do
            start=$(date -u -d "$day $t" +%s)
            delta=$(( start - now_epoch ))
            if [ "$delta" -ge 0 ] && [ "$delta" -le "$lead" ]; then
                LONG_REASONS+=("long daily job due at $t UTC (in $((delta/60)) min)")
            fi
        done
    done
    # The forecast scheduler runs its daily retention job in Central time.
    for day in today tomorrow; do
        start=$(TZ=America/Chicago date -d "$day 03:20" +%s)
        delta=$(( start - now_epoch ))
        if [ "$delta" -ge 0 ] && [ "$delta" -le "$lead" ]; then
            LONG_REASONS+=("forecast retention due at 03:20 Central (in $((delta/60)) min)")
        fi
    done
}
if [ "$MODE" = check ]; then
    guard_collect
    if [ "$HAS_LONG" = true ]; then
        guard_collect_long
        GUARD_REASONS+=("${LONG_REASONS[@]}")
    fi
    if [ "${#GUARD_REASONS[@]}" -gt 0 ]; then
        guard_print
        exit 2
    fi
    echo "Safe to restart."
    exit 0
fi

if [ "$MODE" = long ] && [ "$HAS_LONG" = false ]; then
    echo "No long worker services are configured."
    exit 0
fi

BEFORE=$(git -C "$REPO" rev-parse HEAD)
TARGET=$BEFORE
if [ "$MODE" = deploy ]; then
    git -C "$REPO" fetch --quiet
    TARGET=$(git -C "$REPO" rev-parse '@{upstream}')
    git -C "$REPO" merge-base --is-ancestor "$BEFORE" "$TARGET" ||
        { echo "Cannot fast-forward the deployed checkout to $TARGET." >&2; exit 1; }
    if ! git -C "$REPO" diff --quiet HEAD --; then
        echo "The deployed checkout has tracked changes. Commit or move them before deploying." >&2
        exit 1
    fi
fi

CONFIG_SUM=$(cat "$REPO/.env" "$ROOT/docker-compose.yml" | sha256sum | cut -d' ' -f1)
LEGACY_CONFIG=$(cat "$STATE_FILE" 2>/dev/null || true)
LEGACY_ACTION=
LEGACY_SERVICES=
if [ -f "$PENDING_FILE" ]; then
    LEGACY_ACTION=$(sed -n '1p' "$PENDING_FILE")
    LEGACY_SERVICES=$(sed -n '2p' "$PENDING_FILE")
fi

MAIN_SERVICES=(backend)
if jq -e '.services.worker' >/dev/null <<< "$CONFIG"; then MAIN_SERVICES+=(worker); fi
if [ "$HAS_SCHEDULER" = true ]; then MAIN_SERVICES+=(scheduler); fi
EXTRA_SERVICES=()
while IFS= read -r service; do
    EXTRA_SERVICES+=("$service")
done < <(jq -r '.services | keys[] | select(. != "backend" and . != "worker" and . != "scheduler" and . != "forecast" and . != "worker-long")' <<< "$CONFIG")

# Each service keeps a revision and config checksum even when another service
# has already applied the same commit. The long workers advance as a pair.
read_state() {
    APPLIED=$BEFORE
    APPLIED_CONFIG=$LEGACY_CONFIG
    APPLIED_IMAGE=
    if [ -f "$STATE/$1" ]; then
        read -r APPLIED APPLIED_CONFIG APPLIED_IMAGE < "$STATE/$1"
    elif [ "$1" = long ]; then
        APPLIED=$(inspect_container forecast '{{index .Config.Labels "smf.git_sha"}}' || true)
        APPLIED_IMAGE=$(inspect_container forecast '{{.Image}}' || true)
        [ -n "$APPLIED" ] || APPLIED=unknown
    fi
    [ -n "$APPLIED_CONFIG" ] || APPLIED_CONFIG=unknown
}

write_state() {
    local temp
    temp=$(mktemp "$STATE/.state.XXXXXX") || return 1
    printf '%s %s %s\n' "$2" "$3" "${4:-}" > "$temp" && mv -f "$temp" "$STATE/$1"
}

classify() {
    local service=$1 changed path
    ACTION=none
    if ! changed=$(git -C "$REPO" diff --name-only --no-renames "$APPLIED" "$TARGET" -- 2>/dev/null); then
        ACTION=rebuild
        return
    fi
    while IFS= read -r path; do
        case "$path" in
            '') ;;
            Dockerfile|.dockerignore|requirements*.txt|patches/rrfs.py)
                ACTION=rebuild
                return ;;
            docs/*|tests/*|*.md) ;;
            maps/advisory_category_maps.py)
                # The API alert poll imports it, and the short map cron reads it on its next run.
                if [ "$SPLIT" = true ]; then
                    if [ "$service" = backend ]; then ACTION=restart; fi
                    continue
                fi
                case "$service" in backend|scheduler|long) ACTION=restart ;; esac ;;
            routers/*|main.py)
                if [ "$service" = backend ]; then ACTION=restart; fi ;;
            scripts/worker-entrypoint.sh|scripts/localhost_forward.py)
                case "$service" in worker|long) ACTION=restart ;; esac ;;
            scripts/*.sh)
                if [ "$service" = long ]; then ACTION=restart; fi ;;
            *)
                case "$service" in backend|scheduler|long) ACTION=restart ;; esac ;;
        esac
    done <<< "$changed"
}

plan_service() {
    local service=$1 pending ignored
    read_state "$service"
    classify "$service"
    if [ ! -f "$STATE/$service" ] && [ "$service" != long ] &&
        { [ -z "$LEGACY_SERVICES" ] || [[ " $LEGACY_SERVICES " == *" $service "* ]]; }; then
        case "$LEGACY_ACTION" in
            rebuild) ACTION=rebuild ;;
            recreate) [ "$ACTION" = rebuild ] || ACTION=recreate ;;
            restart) [ "$ACTION" != none ] || ACTION=restart ;;
        esac
    fi
    if [ "$ACTION" != rebuild ] && [ "$APPLIED_CONFIG" != "$CONFIG_SUM" ]; then ACTION=recreate; fi
    if [ -f "$STATE/pending-$service" ]; then
        read -r pending ignored < "$STATE/pending-$service"
        case "$pending" in
            recreate) [ "$ACTION" = rebuild ] || ACTION=recreate ;;
            restart) [ "$ACTION" != none ] || ACTION=restart ;;
        esac
    fi
    if [ "$service" = long ]; then
        if [ "$ACTION" = none ] &&
            { [ -f "$LONG_PENDING_FILE" ] || ! is_running forecast || ! is_running worker-long ||
              [ "$(inspect_container worker-long '{{.Image}}')" != "$APPLIED_IMAGE" ] ||
              [ "$(inspect_container forecast '{{.Image}}')" != "$APPLIED_IMAGE" ]; }; then
            ACTION=restart
        fi
    elif [ "$ACTION" = none ] && { [ "$SCOPE" = all ] || [ "$SCOPE" = api ]; }; then
        ACTION=restart
    fi
}

make_plan() {
    local service
    RESTART=()
    RECREATE=()
    BUILD_MAIN=false
    BUILD_LONG=false
    EXTRAS=false
    NEED_IMAGE=false
    echo "Deploy plan for ${TARGET:0:12}:"
    if [ "$MODE" = deploy ]; then
        for service in "${MAIN_SERVICES[@]}"; do
            plan_service "$service"
            if [ "$SCOPE" = api ] && [ "$service" != backend ]; then
                if [ "$ACTION" != none ]; then echo "  $service: deferred (API only)"; fi
                continue
            fi
            if [ "$ACTION" = recreate ] && ! docker image inspect showmefire-api:latest >/dev/null 2>&1; then
                ACTION=rebuild
            fi
            echo "  $service: $ACTION"
            case "$ACTION" in
                restart) RESTART+=("$service") ;;
                recreate|rebuild)
                    RECREATE+=("$service")
                    if [ "$ACTION" = rebuild ]; then BUILD_MAIN=true; fi ;;
            esac
        done
        read_state extras
        if [ "$SCOPE" != api ] && [ "$APPLIED_CONFIG" != "$CONFIG_SUM" ]; then
            EXTRAS=true
            if [ "${#EXTRA_SERVICES[@]}" -gt 0 ]; then echo "  ${EXTRA_SERVICES[*]}: reconcile Compose config"; fi
        fi
    fi
    if [ "$HAS_LONG" = true ]; then
        plan_service long
        if [ "$ACTION" = none ]; then
            echo "  forecast, worker-long: no change"
        elif [ "$SCOPE" = api ]; then
            echo "  forecast, worker-long: deferred (API only)"
        else
            BUILD_LONG=true
            echo "  forecast, worker-long: apply when idle"
        fi
    fi
    if $BUILD_MAIN || $BUILD_LONG; then
        NEED_IMAGE=true
        IMAGE="showmefire-api:$TARGET"
        if image_matches_target; then
            echo "  image: reuse $IMAGE"
        else
            echo "  image: build $IMAGE locally"
        fi
    else
        echo "  image: no build"
    fi
}

image_matches_target() {
    [ "$(docker image inspect -f '{{index .Config.Labels "smf.git_sha"}}' "$IMAGE" 2>/dev/null)" = "$TARGET" ]
}

make_plan
if [ "$MODE" = deploy ] && ! $FORCE && ! $ASSUME_YES && [ "$SCOPE" = normal ] && [ -t 0 ] && [ -t 1 ]; then
    read -r -p "[Enter] go   [a] API only   [e] all standard services   [q] quit > " CHOICE
    case "$CHOICE" in
        a|A) SCOPE=api; make_plan ;;
        e|E) SCOPE=all; make_plan ;;
        '') ;;
        *) echo "Cancelled. The live checkout is unchanged."; exit 0 ;;
    esac
fi

if $BUILD_LONG && ! jq -e --arg lock "$LONG_LOCK_PATH" --arg source "$REPO/.locks" '
    [.services.forecast, .services["worker-long"]]
    | all(.environment.SMF_LONG_JOB_LOCK == $lock and
          any(.volumes[]?; .type == "bind" and .source == $source and .target == "/app/.locks"))
' >/dev/null <<< "$CONFIG"; then
    echo "Long workers: failed preflight. Apply the shared job lock settings from docs/server/docker-compose.yml." >&2
    exit 1
fi

mkdir -p "$STATE"
# Record initial baselines before changing HEAD; a failed or partial deploy
# retries from these revisions instead of treating the new checkout as applied.
for service in "${MAIN_SERVICES[@]}" extras; do
    [ -f "$STATE/$service" ] && continue
    read_state "$service"
    if [ "$service" != extras ] && [ -n "$LEGACY_ACTION" ] &&
        { [ -z "$LEGACY_SERVICES" ] || [[ " $LEGACY_SERVICES " == *" $service "* ]]; }; then
        # No trustworthy applied revision in the old pending format.
        APPLIED=unknown
    fi
    write_state "$service" "$APPLIED" "$APPLIED_CONFIG"
done
if [ "$HAS_LONG" = true ] && [ ! -f "$STATE/long" ]; then
    read_state long
    write_state long "$APPLIED" "$APPLIED_CONFIG" "$APPLIED_IMAGE"
fi

prepare_image() (
    local snapshot start
    if image_matches_target; then
        echo "Reusing image $IMAGE."
        exit 0
    fi
    snapshot=$(mktemp -d) || exit 1
    trap 'rm -rf "$snapshot"' EXIT
    git -C "$REPO" archive "$TARGET" | tar -xf - -C "$snapshot" || exit 1
    start=$SECONDS
    docker build --progress=plain --build-arg "GIT_SHA=$TARGET" --tag "$IMAGE" "$snapshot" || exit 1
    image_matches_target || exit 1
    echo "Image built in $((SECONDS - start))s."
)

if $BUILD_MAIN && ! prepare_image; then
    echo "Standard services: failed to prepare $IMAGE." >&2
    exit 1
fi

if [ "$MODE" = deploy ]; then
    GUARD_CRON=false
    GUARD_HEAVY=false
    for service in "${RESTART[@]}" "${RECREATE[@]}"; do
        case "$service" in
            worker) GUARD_CRON=true ;;
            scheduler) GUARD_HEAVY=true ;;
            backend) if ! is_running scheduler; then GUARD_HEAVY=true; fi ;;
        esac
    done
    if ! $FORCE && { $GUARD_CRON || $GUARD_HEAVY; }; then
        deadline=$((SECONDS + GUARD_WAIT_MIN * 60))
        while true; do
            guard_collect
            [ "${#GUARD_REASONS[@]}" -eq 0 ] && break
            echo "Waiting for a safe restart:"
            guard_print
            if [ "$SECONDS" -ge "$deadline" ]; then exit 1; fi
            sleep 30
        done
    fi

    for service in "${RESTART[@]}"; do write_state "pending-$service" restart "$TARGET"; done
    for service in "${RECREATE[@]}"; do write_state "pending-$service" recreate "$TARGET"; done
    git -C "$REPO" merge --ff-only --quiet "$TARGET"
    if $BUILD_MAIN; then docker tag "$IMAGE" showmefire-api:latest; fi
    if [ "${#RECREATE[@]}" -gt 0 ]; then
        docker compose up -d --no-build --no-deps "${RECREATE[@]}"
    fi
    if [ "${#RESTART[@]}" -gt 0 ]; then docker compose restart --no-deps "${RESTART[@]}"; fi
    if $EXTRAS && [ "${#EXTRA_SERVICES[@]}" -gt 0 ]; then
        docker compose up -d --no-deps "${EXTRA_SERVICES[@]}"
    fi

    for service in "${RESTART[@]}" "${RECREATE[@]}"; do
        deadline=$((SECONDS + DEPLOY_HEALTH_WAIT_SECONDS))
        until is_running "$service" &&
            { [ "$service" != backend ] || curl -fsS -m 5 http://localhost:5001/health >/dev/null; }; do
            if [ "$SECONDS" -ge "$deadline" ]; then
                echo "$service: failed verification." >&2
                docker logs --tail 30 "$(container_name "$service")" >&2 || true
                exit 1
            fi
            sleep 2
        done
        write_state "$service" "$TARGET" "$CONFIG_SUM"
        rm -f "$STATE/pending-$service"
        echo "$service: applied ${TARGET:0:12}"
    done

    if $EXTRAS; then
        for service in "${EXTRA_SERVICES[@]}"; do
            if [ -z "$(docker compose ps --status running --quiet "$service")" ]; then
                echo "$service: failed verification." >&2
                exit 1
            fi
        done
    fi

    for service in "${MAIN_SERVICES[@]}"; do
        if [ "$SCOPE" = api ] && [ "$service" != backend ]; then continue; fi
        read_state "$service"
        if [ "$APPLIED" != "$TARGET" ] || [ "$APPLIED_CONFIG" != "$CONFIG_SUM" ]; then
            write_state "$service" "$TARGET" "$CONFIG_SUM"
        fi
    done
    if $EXTRAS; then write_state extras "$TARGET" "$CONFIG_SUM"; fi
    if [ "$SCOPE" != api ]; then
        if [ "$LEGACY_CONFIG" != "$CONFIG_SUM" ]; then printf '%s\n' "$CONFIG_SUM" > "$STATE_FILE"; fi
        rm -f "$PENDING_FILE"
    fi
fi

promote_long() (
    local expected deadline reason
    mkdir -p "$REPO/.locks" || return 1
    exec 8>>"$REPO/.locks/long-deploy.lock"
    if ! flock -n -x 8; then
        echo "Long workers: deferred (a guarded job is active)."
        return 2
    fi
    guard_collect_long
    if [ "${#LONG_REASONS[@]}" -gt 0 ]; then
        echo "Long workers: deferred."
        for reason in "${LONG_REASONS[@]}"; do echo "  - $reason"; done
        return 2
    fi

    read_state long
    expected=$(docker image inspect -f '{{.Id}}' "$IMAGE") || return 1
    if [ -n "$APPLIED_IMAGE" ] && [ "$APPLIED_IMAGE" != "$expected" ]; then
        docker tag "$APPLIED_IMAGE" showmefire-api:long-previous || return 1
    fi
    docker tag "$IMAGE" showmefire-api:long || return 1
    docker compose up -d --no-build --no-deps forecast worker-long || return 1

    deadline=$((SECONDS + DEPLOY_HEALTH_WAIT_SECONDS))
    until is_running forecast && is_running worker-long &&
        [ "$(inspect_container forecast '{{.Image}}')" = "$expected" ] &&
        [ "$(inspect_container worker-long '{{.Image}}')" = "$expected" ] &&
        uses_long_lock forecast && uses_long_lock worker-long &&
        [ "$(docker exec show-me-fire-worker-long cat /proc/1/comm 2>/dev/null)" = cron ] &&
        forecast_heartbeat "$TARGET"; do
        if [ "$SECONDS" -ge "$deadline" ]; then return 1; fi
        sleep 2
    done
    write_state long "$TARGET" "$CONFIG_SUM" "$expected" || return 1
    rm -f "$LONG_PENDING_FILE" || return 1
    echo "Long workers: applied ${TARGET:0:12}"
)

if $BUILD_LONG; then
    printf '%s\n' "$TARGET" > "$LONG_PENDING_FILE"
    if ! prepare_image; then
        echo "Long workers: failed to prepare $IMAGE. Retry with ./deploy.sh --long." >&2
        exit 1
    fi
    rollout_start=$SECONDS
    code=0
    promote_long || code=$?
    case "$code" in
        0) echo "Long rollout verified in $((SECONDS - rollout_start))s." ;;
        2) echo "Retry the deferred rollout with ./deploy.sh --long."; exit 0 ;;
        *)
            echo "Long workers: failed verification or startup. Previous verified image is retained when available." >&2
            docker logs --tail 30 show-me-fire-forecast >&2 || true
            docker logs --tail 30 show-me-fire-worker-long >&2 || true
            exit 1 ;;
    esac
fi
echo "Deployment complete."
