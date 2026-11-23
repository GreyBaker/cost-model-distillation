"""Small, ML-free helpers for cumulative runtime-budget experiments."""
from __future__ import annotations

from functools import lru_cache
import math
import os
import time


@lru_cache(maxsize=1)
def _windows_clock():
    import ctypes
    clock = ctypes.WinDLL("kernel32", use_last_error=True).QueryUnbiasedInterruptTime
    clock.argtypes = [ctypes.POINTER(ctypes.c_ulonglong)]
    clock.restype = ctypes.c_int
    return clock


def awake_seconds():
    """Monotonic elapsed time excluding OS suspend (including Windows standby)."""
    if os.name == "nt":
        import ctypes
        value = ctypes.c_ulonglong()
        if not _windows_clock()(ctypes.byref(value)):
            raise ctypes.WinError(ctypes.get_last_error())
        return value.value / 10_000_000
    return time.monotonic()


def budget_reached(consumed, budget):
    return budget is not None and consumed >= budget


def budget_lr_factor(consumed, budget):
    """Table A.1 schedule fractions: 5% warmup, cooldown from 75% to 100%."""
    if not math.isfinite(consumed) or consumed < 0 or not math.isfinite(budget) or budget <= 0:
        raise ValueError("Invalid compute-budget schedule")
    fraction = consumed / budget
    return max(0.0, min(1.0, fraction / 0.05, (1.0 - fraction) / 0.25))


def branch_plan(config, rng):
    """One immutable branch per microbatch; balanced mixtures have exact counts."""
    size = config["gradient_accumulation_steps"]
    fraction = config["lmbda"] if config["algorithm"] == "gkd" else 0.0
    if config["mixture_mode"] == "balanced":
        online = round(size * fraction)
        if not math.isclose(online, size * fraction, abs_tol=1e-9):
            raise ValueError("Balanced mixture needs an integral number of online microbatches")
        plan = [True] * online + [False] * (size - online)
        if 0 < online < size:
            rng.shuffle(plan)
        return plan
    return [fraction == 1 or (fraction > 0 and rng.random() < fraction) for _ in range(size)]
