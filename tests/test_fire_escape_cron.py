"""The fire escape cron step must not fail silently: exit codes of scripts/run_fire_escape.py, and forecasts.sh's env setup."""
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def script(monkeypatch):
    spec = importlib.util.spec_from_file_location("run_fire_escape_script", ROOT / "scripts" / "run_fire_escape.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", ["run_fire_escape.py"])
    return module


def test_off_is_a_quiet_success(script, monkeypatch, capsys):
    monkeypatch.delenv("SMF_FIRE_ESCAPE_ENABLED", raising=False)
    assert script.main() == 0
    assert json.loads(capsys.readouterr().out)["status"] == "skipped"


def test_enabled_but_skipped_exits_nonzero(script, monkeypatch):
    monkeypatch.setenv("SMF_FIRE_ESCAPE_ENABLED", "1")
    monkeypatch.setattr(script.job, "run", lambda *a, **k: {"status": "skipped", "reason": "no cached HRRR file for 2026-11-17"})
    assert script.main() == script.EXIT_COULD_NOT_RUN


def test_enabled_and_ok_exits_zero(script, monkeypatch):
    monkeypatch.setenv("SMF_FIRE_ESCAPE_ENABLED", "1")
    monkeypatch.setattr(script.job, "run", lambda *a, **k: {"status": "ok"})
    assert script.main() == 0


def test_forecasts_sh_loads_container_env_before_the_fire_escape_step():
    text = (ROOT / "scripts" / "forecasts.sh").read_text()
    source = text.index('. "$SCRIPT_DIR/container_env.sh"')
    step = text.index('run_step "Fire escape shadow maps"')
    assert source < step
    assert re.search(r'run_step "Fire escape shadow maps" .*degraded', text)
