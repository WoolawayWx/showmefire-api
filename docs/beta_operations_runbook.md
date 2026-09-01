# Beta operations runbook

The beta lane is designed to collect evidence without changing the operational forecast path. It has four separate boundaries:

1. A beta forecast runs in a subprocess with its own output, cache, archive, and status paths under `data/testbed/forecast`.
2. `FORECAST_WRITE_DATABASE=false` prevents operational database updates.
3. `uploadForecast=false` prevents publishing beta files through the production upload path.
4. `MODEL_SHADOW_ENABLED=false` prevents a Testbed rerun from contaminating production shadow evidence.
5. Beta verification reads the matching operational observation archive but writes reports only under `data/testbed/verification`.

The admin **Beta Models** page is the operating console. It combines the registry, promotion blockers, shadow health, drift state, Testbed freshness, isolation checks, and beta-versus-stable outcome metrics.

## Daily workflow

1. The scheduler triggers a fresh isolated beta forecast at 9:00 AM Central (`run_scheduled_beta_forecast`), ahead of the 10 AM–9 PM Central observation window it will be verified against. You can also open `/testbed` and run one manually from the admin control at any time — the scheduled trigger backs off with a log line if one is already running.
2. Confirm the job reaches `completed` and inspect the generated maps and station comparison.
3. The scheduler attempts verification at 11:40 PM Central, immediately before nightly archiving. You can also open `/admin/models` and select **Verify latest beta forecast** while the matching observation archive is still local.
4. Review category MAE, continuous-score MAE, exact match, within-one-category rate, bias, elevated-event support, and the beta-minus-stable deltas.
5. Investigate any failed isolation check, auto-disabled shadow, drift flag, stale output, or promotion blocker before considering promotion.

Without step 1 happening daily, verification has nothing fresh to score: it silently skips (logged as "Beta verification skipped") whenever the most recent beta forecast predates the observation archive still available on disk.

A negative MAE delta means beta was closer to observations. A positive exact-match or within-one delta means beta improved that rate. Reports with fewer than `BETA_VERIFICATION_MINIMUM_SUPPORT` matched station-hours remain in `collecting_evidence`; the default minimum is 50.

## Evidence files

- `data/testbed/verification/YYYYMMDD.json` — date-scoped result, replaced atomically only when that date is deliberately re-verified.
- `data/testbed/verification/latest.json` — latest verification result for the scorecard.
- `data/testbed/verification/history.json` — rolling history, default 90 days.

`BETA_VERIFICATION_HISTORY_LIMIT` controls history retention. Verification never writes `reports/validation_history.json` or any production map, archive, or database table.

## Promotion policy

The scorecard is evidence, not an automatic promotion mechanism. Promotion remains explicit and continues to use the model registry's existing structural, metadata, checksum, smoke, shadow, and promotion gates. A beta should not be promoted while:

- any isolation check fails;
- the relevant shadow is unhealthy or auto-disabled;
- required drift evidence is flagged;
- the registry reports promotion blockers;
- verification support is too small or elevated-event support is absent;
- beta accuracy is materially worse than stable without an understood reason.

This keeps evaluation reversible and production changes deliberate.

## Observed Rothermel spread rate

The spread-rate Testbed layer is separate from beta forecast verification. It
polls for RTMA every 15 minutes and requires a promoted
`fire_behavior_static` asset contract plus 120–168 cached RTMA hours.

Build and publish the static bundle on the desktop by following
`ShowMeFire-Models/docs/observed_spread_rate_runbook.md`. On this server:

```bash
python pipelines/import_model.py \
  --model fire_behavior_static \
  --tag fire_behavior_static-vVERSION-beta.N \
  --repo OWNER/ShowMeFire-Models
python pipelines/promote_model.py --model fire_behavior_static
python -c "from services.rtma_capture import warmup_rtma_cache; print(warmup_rtma_cache(days=7))"
```

Check `/api/testbed/spread-rate/status` before requesting artifacts. A missing
stable bundle is an operator setup error; `warming` is expected until the
causal RTMA requirement is met.
