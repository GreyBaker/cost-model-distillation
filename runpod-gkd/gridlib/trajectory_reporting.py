"""Measured full-trajectory forecast; never charge prefix training repeatedly."""
from __future__ import annotations

import math
from pathlib import Path

from . import schedule
from .shared import read_json, sha256


def estimate(root, selected, settings):
    from .evaluation import verify
    from .policy import VALIDATION_N
    from .recovery import training_microbatch
    pointer = root / "baseline/latest_evaluation.json"
    if not pointer.is_file():
        raise ValueError("Finish the pilot's full baseline evaluation before estimating the grid")
    output = Path(read_json(pointer)["path"])
    baseline = verify(output, read_json(output / "COMPLETE.json")["identity"])
    if baseline["n"] != VALIDATION_N:
        raise ValueError("Full baseline evaluation missing")
    mb = selected["microbatch"]
    off = selected["off_policy"]["mean_update_seconds"] / (32 // mb)
    on = selected["on_policy"]["mean_update_seconds"] / (32 // mb)
    save_time = max(selected[key]["checkpoint_seconds"] for key in ("off_policy", "on_policy"))
    startup = max(0.0, max(
        (selected[key]["wall_seconds"] - selected[key]["checkpoint_seconds"]
         - sum(x["training_seconds"] for x in selected[key]["updates"])
         - selected[key]["resumed_update"]["training_seconds"]) / 2
        for key in ("off_policy", "on_policy")))
    rows = []
    for item in schedule.trajectories():
        seconds, previous_groups = 0.0, 0
        for _, groups, _ in schedule.boundaries(item["on_policy_percent"]):
            full, short = divmod(groups - previous_groups, item["groups_per_update"])
            for count, multiplicity in ((item["groups_per_update"], full), (short, int(short > 0))):
                online = count * item["group_online"]
                offline = count * (item["group_size"] - item["group_online"])
                seconds += multiplicity * (math.ceil(offline / mb) * off + math.ceil(online / mb) * on)
            previous_groups = groups
        n = len(schedule.EVALUATION_POINTS)
        saves = 1 + len(schedule.boundaries(item["on_policy_percent"]))
        # A separate independent full-state copy for each retained milestone.
        checkpoint_seconds = (saves + n) * save_time
        rows.append({
            "cell": item["id"], "trajectory": item["id"], "points": item["points"], "updates": item["updates"],
            "recovery_saves_including_initial": saves, "retained_checkpoints": n,
            "evaluations": n, "estimated_training_seconds": seconds,
            "estimated_checkpoint_seconds": checkpoint_seconds,
            "estimated_initialization_seconds": n * startup,
            "estimated_evaluation_seconds": n * baseline["evaluation_seconds"],
            "estimated_total_seconds": seconds + checkpoint_seconds + n * (startup + baseline["wall_seconds"]),
        })
    active = training_microbatch(root, selected)
    return {
        "basis": "Measured fit pilot and common full validation; each continuous mixture trained ONCE",
        "training_design": schedule.DESIGN, "trajectories": rows,
        "pilot_sha256": sha256(root / "pilot.json"),
        "pilot_microbatch": mb, "active_training_microbatch": active,
        "evaluation_pilot_batch": (read_json(root / "evaluation_pilot.json")["batch_size"]
                                   if (root / "evaluation_pilot.json").exists() else None),
        "evaluation_pilot_sha256": (sha256(root / "evaluation_pilot.json")
                                   if (root / "evaluation_pilot.json").exists() else None),
        "forecast_stale_after_oom": active != mb,
        "training_hours": sum(row["estimated_training_seconds"] for row in rows) / 3600,
        "evaluation_hours": sum(row["estimated_evaluation_seconds"] for row in rows) / 3600,
        "total_grid_hours": sum(row["estimated_total_seconds"] for row in rows) / 3600,
        "baseline_already_spent_hours": baseline["wall_seconds"] / 3600,
        "warning": "Full campaign projection, not remaining time or billing cap. Prefix costs are not "
                   "summed repeatedly. Checkpoint-copy time uses save-time proxy. Forced-length pilot "
                   "and mixed overhead vary; failures, replay, storage and pilot costs excluded. "
                   "After OOM, timings remain the original pilot forecast.",
    }
