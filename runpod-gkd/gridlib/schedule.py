"""Continuous, independently initialized mixtures with immutable point milestones.

Legacy independent-budget grids remain readable/resumable without this design.
"""
from __future__ import annotations

from bisect import bisect_left
from fractions import Fraction
from functools import lru_cache
import math

FINAL_POINTS = 2_560_000
EVALUATION_POINTS = (10_000, 20_000, 40_000, 80_000, 160_000,
                     320_000, 640_000, 1_280_000, FINAL_POINTS)
RECOVERY_POINTS = 5_000
PERCENTAGES = (0, 25, 50, 75, 100)
DESIGN = {
    "kind": "continuous-point-milestones-v1",
    "final_points": FINAL_POINTS,
    "evaluation_points": list(EVALUATION_POINTS),
    "recovery_every_points": RECOVERY_POINTS,
    "recovery_retention": 2,
    "boundary_policy": "complete_exact_mixture_groups_at_or_below_nominal_points",
    "update_policy": "batch_32_except_short_update_at_recovery_boundary",
    "lr_horizon": "full_trajectory_spendable_points",
}


def enabled(bound):
    design = bound.get("training_design")
    if design is not None and design != DESIGN:
        raise ValueError("Unknown/changed continuous training design")
    return design is not None


@lru_cache(maxsize=5)
def boundaries(percent):
    if percent not in PERCENTAGES:
        raise ValueError("Unknown on-policy percentage")
    fraction = Fraction(percent, 100)
    size, online = fraction.denominator, fraction.numerator
    cost, per_update = size + online, 32 // size
    rows, previous_groups, steps = [], 0, 0
    for nominal in range(RECOVERY_POINTS, FINAL_POINTS + 1, RECOVERY_POINTS):
        groups = nominal // cost
        steps += math.ceil((groups - previous_groups) / per_update)
        rows.append((nominal, groups, steps))
        previous_groups = groups
    return tuple(rows)


def trajectory(percent):
    fraction = Fraction(percent, 100)
    online, size = fraction.numerator, fraction.denominator
    _, groups, updates = boundaries(percent)[-1]
    cost = size + online
    return {
        "id": f"on{percent:03d}", "training_design": DESIGN["kind"],
        "budget_points": FINAL_POINTS, "on_policy_percent": percent,
        "group_online": online, "group_size": size, "group_cost": cost,
        "groups": groups, "groups_per_update": 32 // size,
        "on_policy_examples": groups * online,
        "off_policy_examples": groups * (size - online),
        "examples": groups * size, "points": groups * cost,
        "unspent_points": FINAL_POINTS - groups * cost, "updates": updates,
    }


def trajectories():
    return [trajectory(percent) for percent in PERCENTAGES]


def groups_after(item, step):
    if type(step) is not int or not 0 <= step <= item["updates"]:
        raise ValueError("Invalid continuous update count")
    if step == 0:
        return 0
    rows = boundaries(item["on_policy_percent"])
    index = bisect_left(boundary_steps(item["on_policy_percent"]), step)
    previous_groups, previous_step = (rows[index - 1][1:] if index else (0, 0))
    return min(rows[index][1],
               previous_groups + (step - previous_step) * item["groups_per_update"])


@lru_cache(maxsize=5)
def boundary_steps(percent):
    return tuple(row[2] for row in boundaries(percent))


def accounting(item, step):
    groups = groups_after(item, step)
    return {"step": step, "cursor": groups * item["group_size"],
            "points": groups * item["group_cost"],
            "on_policy_examples": groups * item["group_online"],
            "off_policy_examples": groups * (item["group_size"] - item["group_online"])}


def update(item, step):
    before, after = groups_after(item, step), groups_after(item, step + 1)
    groups = after - before
    on, examples = groups * item["group_online"], groups * item["group_size"]
    return {"groups": groups, "on": on, "off": examples - on,
            "examples": examples, "points": examples + on}


def milestone(percent, nominal):
    if nominal not in EVALUATION_POINTS:
        raise ValueError("Only declared milestones may be retained/evaluated")
    item = trajectory(percent)
    _, groups, steps = boundaries(percent)[nominal // RECOVERY_POINTS - 1]
    return {**item, "id": f"p{nominal:07d}_on{percent:03d}",
            "trajectory_id": item["id"], "budget_points": nominal,
            "groups": groups, "updates": steps, "points": groups * item["group_cost"],
            "examples": groups * item["group_size"],
            "on_policy_examples": groups * item["group_online"],
            "off_policy_examples": groups * (item["group_size"] - item["group_online"]),
            "unspent_points": nominal - groups * item["group_cost"]}


def milestones():
    return [milestone(percent, nominal) for percent in PERCENTAGES for nominal in EVALUATION_POINTS]


def training_directory(root, item):
    return root / ("trajectories" if "training_design" in item else "cells") / item["id"]


def evaluation_directory(root, item):
    if item is None:
        return root / "baseline"
    return root / ("milestones" if "trajectory_id" in item else "cells") / item["id"]


def plan():
    from .policy import PROTOCOL, VALIDATION_N
    runs = trajectories()
    return {
        "operation": "read-only plan; no ML, network, GPU or training",
        "protocol": PROTOCOL, "training_design": DESIGN,
        "trajectories": runs, "milestones": milestones(),
        "totals": {key: sum(row[key] for row in runs)
                   for key in ("budget_points", "points", "examples", "on_policy_examples",
                               "off_policy_examples", "updates", "unspent_points")},
        "training_trajectories": len(runs), "retained_evaluated_checkpoints": len(milestones()),
        "validation_generations_including_baseline": VALIDATION_N * (len(milestones()) + 1),
        "storage": "One disposable Pod disk; 45 permanent full-state milestone checkpoints plus "
                   "two rolling restart checkpoints per mixture, transient saves, models and data.",
        "warning": "Milestones are correlated prefixes, not independent budget runs. "
                   "Training costs are counted once per mixture, not once per evaluation.",
    }
