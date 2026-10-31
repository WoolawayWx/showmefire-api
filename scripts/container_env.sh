#!/bin/bash
# Source this from a cron-run script. cron starts jobs with a near-empty environment, so settings given to the container
# (docker `environment:` / `env_file:`) are invisible to bash even though `docker exec ... printenv` shows them. Python code
# that reads its own .env is unaffected; shell scripts that check a variable before starting Python (the fire outlook
# switch) are not. This copies the container's variables from PID 1 (the process docker started with that environment)
# into this shell and the children it starts. Anything already set is left alone, so a value passed on the command line
# or by a caller still wins. No-op when PID 1's environment cannot be read (local runs, non-Linux).

_smf_env_file="${SMF_CONTAINER_ENV_FILE:-/proc/1/environ}"
if [ -r "$_smf_env_file" ]; then
    while IFS= read -r -d '' _smf_kv; do
        _smf_name="${_smf_kv%%=*}"
        case "$_smf_name" in
            ''|*[!A-Za-z0-9_]*|[0-9]*) continue ;;
        esac
        if [ -z "${!_smf_name+x}" ]; then
            export "$_smf_kv"
        fi
    done < "$_smf_env_file"
fi
unset _smf_env_file _smf_kv _smf_name
