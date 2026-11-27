from __future__ import annotations

import math
from pathlib import Path

from .shared import SFT_CODE, read_json, sha256

ROOT = Path(__file__).resolve().parents[1]
VALIDATION_N = 11332
TRAIN_N = 204045
PROTOCOL = {
    "training_generation": "categorical", "training_temperature": 1.0,
    "top_k": 0, "top_p": 1.0, "evaluation_generation": "greedy",
    "loss": "forward_kl_teacher_to_student", "loss_temperature": 1.0,
    "loss_normalization": "token_mean_per_sequence_then_mean_examples",
    "off_policy_prefixes": "human_xsum_references", "dropout": 0.0,
    "prefix": "summarize: ", "max_source_tokens": 1024, "max_target_tokens": 64,
    "point_costs": {"off_policy_example": 1, "on_policy_example": 2},
    "mixture": "exact_fraction_each_update", "budget_policy": "never_exceed",
    "lr_schedule": "5_percent_warmup_75_percent_cooldown_update_point_midpoint",
}


def settings(path=None):
    value = read_json(path or ROOT / "settings.json")
    expected = {"seed", "data_seed", "effective_batch_size", "learning_rate",
                "pilot_microbatches", "pilot_updates",
                "allocator_fraction", "pilot_reserved_fraction_limit", "evaluation_batch_cap"}
    optional = {"evaluation_batch_safety_fraction"}
    if not expected <= set(value) or set(value) - expected - optional:
        raise ValueError("Unknown/missing settings fields")
    for key in ("seed", "data_seed", "effective_batch_size",
                "pilot_updates", "evaluation_batch_cap"):
        if type(value[key]) is not int or value[key] < (0 if key.endswith("seed") else 1):
            raise ValueError(f"Invalid integer setting: {key}")
    if value["effective_batch_size"] != 32 or value["pilot_updates"] < 3:
        raise ValueError("Effective batch is 32; pilot needs warmup plus two timed updates")
    for key in ("learning_rate", "allocator_fraction", "pilot_reserved_fraction_limit"):
        x = value[key]
        if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0:
            raise ValueError(f"Invalid finite positive setting: {key}")
    if not value["pilot_reserved_fraction_limit"] < value["allocator_fraction"] <= 1:
        raise ValueError("Require reserved-memory headroom below allocator fraction <= 1")
    safety = value.get("evaluation_batch_safety_fraction", .9)
    if isinstance(safety, bool) or not isinstance(safety, (int, float)) or not math.isfinite(safety) or not 0 < safety < 1:
        raise ValueError("Evaluation batch safety fraction must be finite and strictly between 0 and 1")
    sizes = value["pilot_microbatches"]
    if (not isinstance(sizes, list) or not sizes or sizes != sorted(set(sizes))
            or any(type(x) is not int or x < 1 or 32 % x for x in sizes)):
        raise ValueError("Pilot microbatches must be sorted unique divisors of 32")
    return value


def update_spec(item, completed_updates):
    from .schedule import require_trajectory, update
    require_trajectory(item)
    if type(completed_updates) is not int or not 0 <= completed_updates < item["updates"]:
        raise ValueError("No next update")
    return update(item, completed_updates)


def lr_factor(completed_points, next_points, total_points):
    if not 0 <= completed_points < total_points or not 0 < next_points <= total_points - completed_points:
        raise ValueError("Invalid point-schedule position")
    position = (completed_points + next_points / 2) / total_points
    return min(1.0, position / .05, (1 - position) / .25)


def chunks(rows, branch_flags, microbatch, rng):
    """Stratified per-update mixture; branch-homogeneous microbatches may be short."""
    if len(rows) != len(branch_flags) or microbatch < 1:
        raise ValueError("Invalid mixture partition")
    grouped = []
    for online in (False, True):
        subset = [row for row, flag in zip(rows, branch_flags) if flag == online]
        grouped.extend((online, subset[i:i + microbatch]) for i in range(0, len(subset), microbatch))
    rng.shuffle(grouped)
    return grouped


def code_signature():
    paths = [ROOT / "distill_grid.py", *sorted((ROOT / "gridlib").glob("*.py")),
             *sorted((SFT_CODE / "sftlib").glob("*.py"))]
    external = ROOT.parent / "runpod-external-teacher"
    if (external / "external_teacher.py").is_file():
        paths += [external / "external_teacher.py", *sorted((external / "external_workflow").glob("*.py"))]
    return {p.relative_to(ROOT.parent).as_posix(): sha256(p) for p in paths}


def plan():
    from .schedule import plan as continuous_plan
    return continuous_plan()
