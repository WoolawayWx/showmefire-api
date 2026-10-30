"""Seven-day Show Me Fire outlook: calculation core (see _plans/PLAN.md, _plans/CLAUDE_PLAN.md).

Pure numpy modules, no scheduler, no network:
    moisture   time-aware dead-fuel moisture advance + hourly interpolation / precipitation allocation
    windows    28 six-hour windows, concurrent category tests, streaming scenario agreement
"""
