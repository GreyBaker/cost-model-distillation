"""Saved-pilot planning only; does not stop, schedule, or launch training."""
from __future__ import annotations

import json
import math
import statistics

from . import checkpoints
from .core import read_json


def project(value, start, update_seconds, save_seconds, validation_seconds, initialization_seconds,
            monitor_seconds=0.0):
    if (type(start) is not int or not 0 <= start <= value["max_steps"]
            or not all(math.isfinite(x) for x in (update_seconds, save_seconds, validation_seconds,
                                                 initialization_seconds, monitor_seconds))
            or update_seconds <= 0 or min(save_seconds, validation_seconds, initialization_seconds, monitor_seconds) < 0):
        raise ValueError("Invalid projection inputs")
    target = value["max_steps"]
    if start == target:
        return {"hours": 0.0, "updates": 0, "checkpoints": 0, "validations": 0, "greedy_monitors": 0}
    saves = target // value["save_every_steps"] - start // value["save_every_steps"]
    if target % value["save_every_steps"]:
        saves += 1
    saves += int(start == 0)  # Step-zero restart anchor.
    validations = target // value["eval_every_steps"] - start // value["eval_every_steps"] + int(start == 0)
    every = value.get("rouge_every_steps")
    monitors = target // every - start // every + int(start == 0) if every else 0
    seconds = ((target - start) * update_seconds + saves * save_seconds
               + validations * validation_seconds + initialization_seconds + monitors * monitor_seconds)
    return {"hours": seconds / 3600, "updates": target - start, "checkpoints": saves,
            "validations": validations, "greedy_monitors": monitors}


def report(work, value, hourly_rate):
    if isinstance(hourly_rate, bool) or not math.isfinite(hourly_rate) or hourly_rate < 0:
        raise ValueError("Use an actual finite nonnegative USD/hour rate")
    run_dir = work / "runs" / value["run_name"]
    updates, saves, validations, starts, monitors = [], [], [], [], []
    for path in sorted((run_dir / "sessions").glob("*.jsonl")):
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        warmed = [row for row in records if row["event"] == "update"]
        updates.extend(warmed[1:] or warmed)
        saves.extend(row for row in records if row["event"] == "checkpoint")
        validations.extend(row for row in records if row["event"] == "validation")
        starts.extend(row for row in records if row["event"] == "initialized")
        monitors.extend(row for row in records if row["event"] == "greedy_monitor")
    if not updates or not saves:
        raise ValueError("Need saved update/checkpoint pilots first")
    def mean(rows, key, n):
        recent = sorted(rows, key=lambda row: row["wall_time_unix"])[-n:]
        return statistics.mean(row[key] for row in recent) if recent else 0.0
    timings = (mean(updates, "update_seconds", 100), mean(saves, "seconds", 10),
               mean(validations, "seconds", 5), mean(starts, "seconds", 5),
               mean(monitors, "seconds", 5))
    choices = checkpoints.completed(run_dir)
    step = read_json(choices[-1] / "COMPLETE.json")["step"] if choices else 0
    full = project(value, 0, *timings)
    remaining = project(value, step, *timings)
    return {"role": value["role"], "last_complete_step": step, "mean_update_seconds": timings[0],
            "mean_greedy_monitor_seconds": timings[4],
            "hypothetical_full_training": full, "remaining_training": remaining,
            "hypothetical_full_compute_usd": full["hours"] * hourly_rate,
            "remaining_compute_usd": remaining["hours"] * hourly_rate,
             "warning": "Projection, not a time/billing cap. Includes diagnostic greedy monitoring when configured. "
                        "Excludes fit-pilot overhead, downloads/tokenization, final export/full ROUGE, extra "
                        "restarts/lost work, storage/tax. Measure each role on the intended GPU; short pilots are noisy."}
