"""Loads and runs the detection-confidence model package (api/detection-confidence-model/) from inside the API.

That folder's scripts import each other by bare name (`from config import ...`, `from features import ...`) so they
also run standalone, and two training packages under api/ use the same names, so they cannot simply be imported.
Loading one by file path alone does not work either: its siblings are not on the import path (the nightly job hit
`ModuleNotFoundError: No module named 'config'`). This puts the package folder first on sys.path for the duration of
the load, hides any same-named module another package already imported, and restores everything afterwards. The
loaded module keeps its own references, so it keeps working once the path is put back.
"""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

PACKAGE_DIR = Path(__file__).resolve().parent.parent / "detection-confidence-model"
_PACKAGE_MODULES = ("config", "features", "train", "evaluate", "retrain")


def load_module(name: str) -> ModuleType:
    """The package's `name` module (config, features, train, evaluate or retrain), loaded in isolation."""
    if name not in _PACKAGE_MODULES:
        raise ValueError(f"unknown detection-confidence module: {name}")
    saved = {module: sys.modules.pop(module, None) for module in _PACKAGE_MODULES}
    sys.path.insert(0, str(PACKAGE_DIR))
    try:
        spec = importlib.util.spec_from_file_location(f"detection_confidence_{name}", PACKAGE_DIR / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(PACKAGE_DIR))
        for module in _PACKAGE_MODULES:
            sys.modules.pop(module, None)
            if saved[module] is not None:
                sys.modules[module] = saved[module]


def retrain_and_gate() -> dict:
    """Train a candidate, evaluate it, and register it only if it passes the gate (see retrain.py)."""
    return load_module("retrain").retrain_and_gate()
