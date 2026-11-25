"""Narrow resume compatibility: batch partition changes, not recipe changes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

from .core import atomic_json

PARTITION = {"microbatch_size", "gradient_accumulation_steps"}
RETIRED_MONITOR = {"rouge_every_steps", "rouge_examples", "review_step"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def effective_batch(value):
    sizes = [value.get(key) for key in PARTITION]
    if any(type(size) is not int or size < 1 for size in sizes):
        return None
    return sizes[0] * sizes[1]


def config_compatible(saved, current):
    # Removing diagnostics does not change optimizer updates or their lineage.
    # Keep historical configs/checkpoint metadata intact and record migration.
    ignored = PARTITION | (RETIRED_MONITOR if saved.get("role") == current.get("role") == "teacher" else set())
    return (effective_batch(saved) is not None
            and effective_batch(saved) == effective_batch(current)
            and {k: v for k, v in saved.items() if k not in ignored}
            == {k: v for k, v in current.items() if k not in ignored})


def signatures_compatible(saved, current):
    # Repository changes are controlled by the operator. Record source hashes,
    # but never treat them as an authorization gate for valid saved training state.
    return ({k: v for k, v in saved.items() if k != "code"}
            == {k: v for k, v in current.items() if k != "code"})


def record_migration(directory, saved, current, checkpoint=None):
    if saved == current:
        return
    event = {"from": saved, "to": current, "checkpoint": str(checkpoint) if checkpoint else None}
    path = Path(directory) / (digest(event) + ".json")
    if not path.exists():
        atomic_json(path, {**event, "unix_time": time.time(),
                          "note": "Batch partition/code/retired-monitor migration; effective examples and semantic recipe preserved. "
                                  "Floating point, dropout and generated trajectories need not be bitwise identical."})
