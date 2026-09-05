"""Persist optional AI feedback separately from immutable scoring evidence."""
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from core.config import REPORTS_DIR
from ai.cloudflare import CloudflareAIClient
from ai.verification_summary import generate_verification_summary
from routers.verification import _read_summary, _load_history


def fingerprint(report):
    evidence = {k: v for k, v in report.items() if k != 'ai_summary'}
    return hashlib.sha256(json.dumps(evidence, sort_keys=True, default=str).encode()).hexdigest()


def feedback_status(date):
    report = _read_summary(date)
    path = Path(REPORTS_DIR) / report['date'] / 'ai_feedback.json'
    try:
        saved = json.loads(path.read_text())
        if saved.get('report_fingerprint') == fingerprint(report) and saved.get('ai_summary'):
            return {**saved, 'status': 'ready', 'configured': CloudflareAIClient().configured}
    except (OSError, ValueError):
        pass
    text = report.get('ai_summary')
    configured = CloudflareAIClient().configured
    return {'date': report['date'], 'ai_summary': text,
            'status': 'ready' if text else 'not_generated' if configured else 'not_configured',
            'configured': configured}


def generate_feedback(date):
    report = _read_summary(date)
    if not CloudflareAIClient().configured:
        raise RuntimeError('Cloudflare Workers AI credentials are not configured')
    history = sorted((e for e in _load_history() if e['date'] < report['date']), key=lambda e: e['date'])[-30:]
    text = generate_verification_summary(report, report.get('comparison_rows', []), history, strict=True)
    result = {'date': report['date'], 'ai_summary': text,
              'report_fingerprint': fingerprint(report), 'generated_at': datetime.now(timezone.utc).isoformat()}
    directory = Path(REPORTS_DIR) / report['date']
    directory.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=directory, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as output:
            json.dump(result, output)
        # A concurrently rerun report must never receive feedback from old evidence.
        if fingerprint(_read_summary(date)) != result['report_fingerprint']:
            raise RuntimeError('Report changed during generation. Please retry.')
        os.replace(temp, directory / 'ai_feedback.json')
    finally:
        Path(temp).unlink(missing_ok=True)
    return {**result, 'status': 'ready', 'configured': True}
