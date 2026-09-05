"""Authenticated archive inventory and resumable sync controls."""
from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from core.security import verify_token
from services.archive_store import ArchiveStore

router = APIRouter(prefix='/api/admin/archive', tags=['archive-admin'])


def require_admin(token):
    if not verify_token(token):
        raise HTTPException(status_code=401, detail='Unauthorized')


@router.get('/status')
def status(days: int = Query(30, ge=1, le=366), token: str | None = None):
    require_admin(token)
    return ArchiveStore().status(days)


@router.post('/sync', status_code=202)
def sync(background_tasks: BackgroundTasks, token: str | None = None):
    require_admin(token)
    store = ArchiveStore()
    if not store.configured:
        raise HTTPException(status_code=503, detail='R2 archive credentials are not configured')
    try:
        with store.lock():
            pass
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    background_tasks.add_task(store.sync)
    return {'status': 'queued'}


@router.get('/processing')
def processing(token: str | None = None):
    require_admin(token)
    import json
    from services.forecast_jobs import get_beta_forecast_status
    from services.forecast_09z_jobs import get_09z_forecast_status
    store = ArchiveStore()
    backfill = store.root / 'archive/raw_data/synoptic_backfill_manifest.json'
    observations = {'status': 'not_started', 'days': {}}
    if backfill.exists():
        try:
            observations = json.loads(backfill.read_text())
        except (OSError, ValueError):
            observations = {'status': 'unreadable', 'days': {}}
    counts = {}
    for day in observations.get('days', {}).values():
        state = day.get('status', 'unknown')
        counts[state] = counts.get(state, 0) + 1
    primary_path = store.root / 'status.json'
    try:
        primary = json.loads(primary_path.read_text()) if primary_path.exists() else {'status': 'unknown'}
    except (OSError, ValueError):
        primary = {'status': 'unreadable'}
    return {'primary': primary.get('ForecastFireDanger', {'status': 'unknown'}),
            'realtime_observations': primary.get('RealtimeFireDanger', {'status': 'unknown'}), 'beta': get_beta_forecast_status(),
            'forecast_09z': get_09z_forecast_status(),
            'observations': {'status': observations.get('status', 'tracked'), 'days_by_status': counts,
                'failures': [{'date': date, **value} for date, value in observations.get('days', {}).items() if value.get('status') == 'failed'][-30:]}}
