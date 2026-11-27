from __future__ import annotations

from fractions import Fraction
import math
from pathlib import Path

from .shared import SFT_CODE, read_json, sha256

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = (500, 1000, 5000, 10000, 50000)
PERCENTAGES = (0, 25, 50, 75, 100)
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
                "save_every_updates", "pilot_microbatches", "pilot_updates",
                "allocator_fraction", "pilot_reserved_fraction_limit", "evaluation_batch_cap"}
    optional = {"evaluation_batch_safety_fraction"}
    if not expected <= set(value) or set(value) - expected - optional:
        raise ValueError("Unknown/missing settings fields")
    for key in ("seed", "data_seed", "effective_batch_size", "save_every_updates",
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


def cell(budget, percent, batch_size=32):
    if budget not in BUDGETS or percent not in PERCENTAGES or batch_size != 32:
        raise ValueError("Unknown grid cell/effective batch")
    fraction = Fraction(percent, 100)
    online, group_size = fraction.numerator, fraction.denominator
    group_cost = group_size + online
    groups = budget // group_cost
    groups_per_update = batch_size // group_size
    return {
        "id": f"p{budget:05d}_on{percent:03d}", "budget_points": budget,
        "on_policy_percent": percent, "group_online": online,
        "group_size": group_size, "group_cost": group_cost, "groups": groups,
        "groups_per_update": groups_per_update,
        "on_policy_examples": groups * online,
        "off_policy_examples": groups * (group_size - online),
        "examples": groups * group_size, "points": groups * group_cost,
        "unspent_points": budget % group_cost,
        "updates": math.ceil(groups / groups_per_update),
    }


def cells():
    return [cell(budget, percent) for budget in BUDGETS for percent in PERCENTAGES]


def update_spec(item, completed_updates):
    if type(completed_updates) is not int or not 0 <= completed_updates < item["updates"]:
        raise ValueError("No next update")
    if "training_design" in item:
        from .schedule import update
        return update(item, completed_updates)
    groups = min(item["groups_per_update"],
                 item["groups"] - completed_updates * item["groups_per_update"])
    online = groups * item["group_online"]
    examples = groups * item["group_size"]
    return {"groups": groups, "on": online, "off": examples - online,
            "examples": examples, "points": examples + online}


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


def legacy_plan():
    grid = cells()
    return {"operation": "read-only plan; no ML, network, GPU or training",
            "protocol": PROTOCOL, "cells": grid,
            "storage": "One short-lived on-Pod disk, shared with SFT; target 150 GB available "
                       "for the entire workflow, including environment/caches/temp; no separate disk assumed",
            "totals": {key: sum(row[key] for row in grid)
                       for key in ("budget_points", "points", "examples", "on_policy_examples",
                                   "off_policy_examples", "updates", "unspent_points")},
            "validation_generations_including_baseline": VALIDATION_N * (len(grid) + 1),
            "evaluation_batch_policy": "Once-only student-inference pilot: stress-fit search, tested safety ceiling, "
                                       "then warmed-throughput sweep; smallest batch within 2% of fastest tested. "
                                       "Shared by baseline/all cells; historical pilots remain reusable unchanged",
            "runtime_scenarios": [
                {"assumed_points_per_second": rate,
                 "training_hours": sum(row["points"] for row in grid) / rate / 3600}
                for rate in (10, 50, 100)],
            "estimate_warning": "A100 planning scenarios, NOT measured A100 throughput. "
                                "Add 1.64h evaluation IF 50 summaries/s, plus pilot/IO/loading. "
                                 "Use measured pilot estimate before authorizing execution."}


def plan():
    from .schedule import plan as continuous_plan
    return continuous_plan()
