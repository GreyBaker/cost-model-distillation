from __future__ import annotations

import json
import math
import statistics

from .checkpoints import completed
from .core import read_json


def project(config, steps, update_seconds, save_seconds, validation_seconds, initialization_seconds,
            start_step=0, extra_restarts=0):
    """Throughput projection, not a promise/credit entitlement or cost hard cap."""
    if (not all(math.isfinite(x) for x in (update_seconds, save_seconds, validation_seconds, initialization_seconds))
            or type(steps) is not int or steps < 0 or type(start_step) is not int or start_step < 0
            or type(extra_restarts) is not int or extra_restarts < 0 or update_seconds <= 0
            or min(save_seconds, validation_seconds, initialization_seconds) < 0):
        raise ValueError("Invalid projection inputs")
    if steps == 0:
        return {"hours": 0.0, "process_starts": 0, "checkpoints": 0, "validations": 0}
    target = start_step + steps
    interval = config["save_every_steps"]
    saves = target // interval - start_step // interval
    if target % interval:
        saves += 1  # Final/pilot save when not already on a periodic save boundary.
    saves += extra_restarts  # Conservative allowance for extra graceful pause saves.
    validations = target // config["eval_every_steps"] - start_step // config["eval_every_steps"]
    if start_step == 0:
        validations += 1  # Initial diagnostic; resumed processes don't repeat it.
    starts = 1 + extra_restarts
    total = (steps * update_seconds + saves * save_seconds + validations * validation_seconds
             + starts * initialization_seconds)
    return {"hours": total / 3600, "process_starts": starts, "checkpoints": saves, "validations": validations}


def report(run_dir, config, hourly_rate, credits_usd=0.0, extra_restarts=0):
    if hourly_rate < 0 or credits_usd < 0 or not math.isfinite(hourly_rate + credits_usd):
        raise ValueError("Costs/credits must be finite nonnegative USD values")
    if read_json(run_dir / "run.json")["config"] != config:
        raise ValueError("Estimate config differs from the measured run")
    updates, saves, validations, initializations = [], [], [], []
    for path in sorted((run_dir / "sessions").glob("*.jsonl")):
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # Abrupt termination may leave a truncated trailing log line.
        for index, record in enumerate(records):
            record.setdefault("wall_time_unix", path.stat().st_mtime + index * 1e-6)
        session_updates = [(record["wall_time_unix"], record["update_seconds"]) for record in records if record.get("event") == "update"]
        updates.extend(session_updates[2:] or session_updates)
        saves.extend((record["wall_time_unix"], record["seconds"]) for record in records if record.get("event") == "checkpoint")
        validations.extend((record["wall_time_unix"], record["seconds"]) for record in records if record.get("event") == "validation")
        initializations.extend((record["wall_time_unix"], record["seconds"]) for record in records if record.get("event") == "initialized")
    if not updates or not saves:
        raise ValueError("Run a saved pilot first; need measured updates AND a full checkpoint write")
    updates = [seconds for _, seconds in sorted(updates)[-200:]]
    save_mean = statistics.mean(seconds for _, seconds in sorted(saves)[-10:])
    val_mean = statistics.mean(seconds for _, seconds in sorted(validations)[-5:]) if validations else 0.0
    init_mean = statistics.mean(seconds for _, seconds in sorted(initializations)[-5:]) if initializations else 0.0
    choices = completed(run_dir)
    saved_step = read_json(choices[-1] / "COMPLETE.json")["step"] if choices else 0
    full = project(config, config["max_steps"], statistics.mean(updates), save_mean, val_mean, init_mean,
                   extra_restarts=extra_restarts)
    remaining = project(config, max(0, config["max_steps"] - saved_step), statistics.mean(updates),
                        save_mean, val_mean, init_mean, start_step=saved_step, extra_restarts=extra_restarts)
    return {"update_samples": len(updates), "mean_update_seconds": statistics.mean(updates),
            "median_update_seconds": statistics.median(updates), "mean_checkpoint_seconds": save_mean,
            "mean_validation_seconds": val_mean, "mean_initialization_seconds": init_mean,
            "last_complete_step": saved_step, "estimated_full_run": full, "estimated_remaining": remaining,
            "extra_restart_allowance": extra_restarts,
            "projection_mode": "continuous run; measured update/step-based checkpoint/validation overhead; no automatic wall-time pauses",
            "hourly_rate_usd": hourly_rate, "full_compute_usd_before_credits": full["hours"] * hourly_rate,
            "remaining_compute_usd_before_credits": remaining["hours"] * hourly_rate,
            "remaining_compute_usd_after_entered_credits": max(0, remaining["hours"] * hourly_rate - credits_usd),
            "warning": "Projection only; full-run figure is a hypothetical fresh run, not spend to date. Excludes storage, tax, setup/download, final export/ROUGE and lost work. Small pilots are noisy; use 20+ representative warmed updates on the intended GPU. Credits must be entered as confirmed USD value, not assumed credit units."}
