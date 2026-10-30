"""
Six-hour outlook windows and scenario agreement.

Windows are [0,6), [6,12) ... [162,168) hours from initialization. Hourly index h is the
interval [h, h+1) and belongs to window h // 6. Temperature, humidity, wind and fuel are
tested together at each hourly step (the canonical rule needs all three at once), then
the per-window maximum is taken - never a max of each variable separately.

Agreement is the equal-weight fraction of ALL members reaching a threshold at least once
in the window. A cell/window where any member is missing is NaN (NoData), never a
shrunken denominator. Critical-or-higher can never exceed Elevated-or-higher because a
Critical category is also >= Elevated. Members are folded in one at a time
(`AgreementAccumulator`) so only one member's hourly fields are in memory.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

import numpy as np

from services.rule_uncertainty import CATEGORY_CRITICAL, CATEGORY_ELEVATED, MISSING_CATEGORY

WINDOW_HOURS = 6
N_WINDOWS = 28
HORIZON_HOURS = WINDOW_HOURS * N_WINDOWS  # 168


def window_bounds() -> list[tuple[int, int]]:
    return [(w * WINDOW_HOURS, (w + 1) * WINDOW_HOURS) for w in range(N_WINDOWS)]


def hourly_categories(fm, rh, wind_kts, categorize: Callable, dampen: Optional[Callable] = None,
                      snow_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """(hours, y, x) -> (hours, y, x) int8. Same order as the daily product: canonical rule,
    seasonal dampening, then snow forces Low. MISSING_CATEGORY is preserved."""
    cat = categorize(fm, rh, wind_kts).astype("int16")
    if dampen is not None:
        cat = dampen(cat, fm, rh, wind_kts).astype("int16")
    if snow_mask is not None:
        cat = np.where(np.broadcast_to(snow_mask, cat.shape) & (cat != MISSING_CATEGORY), 0, cat)
    return cat.astype("int8")


def window_summary(hourly_cat: np.ndarray) -> Dict[str, np.ndarray]:
    """(168, y, x) hourly categories -> per-window max category (28, y, x) int8 (MISSING if any hour
    in the window is missing) and hours at/above Elevated and Critical (uint8; 255 where missing)."""
    if hourly_cat.shape[0] != HORIZON_HOURS:
        raise ValueError(f"expected {HORIZON_HOURS} hourly steps, got {hourly_cat.shape[0]}")
    grid = hourly_cat.shape[1:]
    by_window = hourly_cat.reshape((N_WINDOWS, WINDOW_HOURS) + grid)
    missing = (by_window == MISSING_CATEGORY).any(axis=1)
    peak = by_window.max(axis=1).astype("int8")
    peak = np.where(missing, MISSING_CATEGORY, peak).astype("int8")
    out = {"max_category": peak}
    for name, thr in (("hours_elevated", CATEGORY_ELEVATED), ("hours_critical", CATEGORY_CRITICAL)):
        hours = (by_window >= thr).sum(axis=1).astype("uint8")
        out[name] = np.where(missing, 255, hours).astype("uint8")
    return out


class AgreementAccumulator:
    """Streaming equal-weight agreement over a fixed member count."""

    def __init__(self, n_members: int, grid_shape: tuple[int, int]):
        self.n_members = n_members
        self.added = 0
        shape = (N_WINDOWS,) + tuple(grid_shape)
        self._elevated = np.zeros(shape, dtype="uint8")
        self._critical = np.zeros(shape, dtype="uint8")
        self._missing = np.zeros(shape, dtype=bool)

    def add(self, max_category: np.ndarray) -> None:
        if max_category.shape != self._elevated.shape:
            raise ValueError(f"member shape {max_category.shape} != {self._elevated.shape}")
        if self.added >= self.n_members:
            raise ValueError("more members added than declared")
        miss = max_category == MISSING_CATEGORY
        self._missing |= miss
        self._elevated += ((max_category >= CATEGORY_ELEVATED) & ~miss).astype("uint8")
        self._critical += ((max_category >= CATEGORY_CRITICAL) & ~miss).astype("uint8")
        self.added += 1

    def finalize(self) -> Dict[str, np.ndarray]:
        if self.added != self.n_members:
            raise ValueError(f"{self.added} of {self.n_members} members added; refusing to publish partial agreement")
        denom = float(self.n_members)  # fixed denominator, never the count of valid members
        elevated = np.where(self._missing, np.nan, self._elevated / denom).astype("float32")
        critical = np.where(self._missing, np.nan, self._critical / denom).astype("float32")
        return {"elevated": elevated, "critical": critical, "missing": self._missing.copy(),
                "n_members": self.n_members}
