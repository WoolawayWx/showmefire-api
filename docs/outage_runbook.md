# API outage runbook

Background: 2026-11-16 the API container stayed `Up` at ~100% CPU for ~35 min (issue #46).

## Detect
- `curl -m 10 http://localhost:5001/health` (host) — times out when hung.
- `docker ps` shows `(unhealthy)` once the healthcheck has failed 3 times (~90 s); the `autoheal` container then restarts it.
- `docker stats --no-stream show-me-fire-api` — ~100% CPU with no new log lines = spinning.

## Capture evidence BEFORE restarting
```
docker kill -s USR1 show-me-fire-api     # faulthandler dumps every thread's stack to the container log
docker logs --tail 300 show-me-fire-api  # read the dump
```
Note: autoheal may restart the container first; if so, the dump is only available if you were quicker. To investigate
a recurrence without autoheal interfering: `docker update --label-rm autoheal` is not possible live, so stop autoheal
(`docker stop show-me-fire-autoheal`) first, capture, then restart it.

## Recover
`docker restart show-me-fire-api` (or let autoheal do it).

## What was changed after #46
- `services/hrrr_hourly/store.py::point_series`: one HDF5 read at a time on a private 4-thread pool, 8 s deadline, 3 attempts,
  returns `{"error": "unavailable"}` instead of raising. The hourly route answers 503 with `Retry-After`.
- `/health` is `async def`, `/health/ready` runs in the threadpool.
- Compose: healthcheck on `backend` + `autoheal` service; `deploy.sh` also curls `/health`.

Root cause of the busy loop was never captured; if it recurs, attach the USR1 dump to the issue.
