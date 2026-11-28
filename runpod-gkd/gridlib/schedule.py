"""The only supported grid: continuous mixtures with immutable point milestones."""
from __future__ import annotations

from bisect import bisect_left
from fractions import Fraction
from functools import lru_cache
import math

FINAL_POINTS = 160_000
DEFAULT_END_POINTS = FINAL_POINTS
# This is a new training recipe, not a truncation of the old 2.56M LR schedule.
EVALUATION_POINTS = (10_000, 20_000, 40_000, 80_000, FINAL_POINTS)
RECOVERY_POINTS = 5_000
PERCENTAGES = (0, 25, 50, 75, 100)
DESIGN = {
    "kind": "continuous-point-milestones-v2",
    "final_points": FINAL_POINTS,
    "evaluation_points": list(EVALUATION_POINTS),
    "recovery_every_points": RECOVERY_POINTS,
    "recovery_retention": 2,
    "boundary_policy": "complete_exact_mixture_groups_at_or_below_nominal_points",
    "update_policy": "batch_32_except_short_update_at_recovery_boundary",
    "lr_horizon": "nominal_trajectory_points",
    "lr_warmup_points": 8_000,
    "lr_cooldown_begin_points": 120_000,
    "lr_cooldown_end_points": FINAL_POINTS,
}


def require_design(bound):
    if bound.get("training_design") != DESIGN:
        raise ValueError("Unsupported grid/LR recipe. The 160K horizon (8K warmup, 120K–160K cooldown) "
                         "requires continuous-point-milestones-v2; long-horizon checkpoints cannot be relabeled. "
                         "Preserve old results and use a NEW grid directory; approved pairs remain reusable.")


def validate_root(root, *, allow_new=False):
    """Reject obsolete/unowned work before creating locks, caches or status files."""
    from .shared import read_json
    path = root / "binding.json"
    if path.is_file():
        bound = read_json(path)
        require_design(bound)
        if (root / "cells").exists():
            raise ValueError("Old independent-cell grid directories are unsupported; use a NEW grid directory.")
        return bound
    if not allow_new:
        raise ValueError("No continuous grid binding; run the experiment with the single run command "
                         "in a NEW grid directory.")
    if root.exists() and any(entry.name not in (".controller.lock", ".worker.lock")
                             for entry in root.iterdir()):
        raise ValueError("Nonempty grid directory without a continuous binding; use a NEW grid directory.")
    return None


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


def require_trajectory(item):
    if item != trajectory(item.get("on_policy_percent", -1)):
        raise ValueError("Only a declared full continuous trajectory may be trained.")


def require_milestone(item):
    if item != milestone(item.get("on_policy_percent", -1), item.get("budget_points")):
        raise ValueError("Only a declared continuous milestone may be evaluated.")


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


def evaluation_points(end_points=DEFAULT_END_POINTS):
    """Operational scope, deliberately NOT part of immutable training identity."""
    if type(end_points) is not int or end_points not in EVALUATION_POINTS:
        raise ValueError("Endpoint must be one of the declared evaluation milestones")
    return tuple(point for point in EVALUATION_POINTS if point <= end_points)


def active_milestones(end_points=DEFAULT_END_POINTS):
    return [milestone(percent, nominal) for percent in PERCENTAGES
            for nominal in evaluation_points(end_points)]


def campaign(end_points=DEFAULT_END_POINTS):
    return {"end_points": end_points, "evaluation_points": list(evaluation_points(end_points)),
            "lr_horizon_nominal_points": FINAL_POINTS,
            "lr_warmup_points": DESIGN["lr_warmup_points"],
            "lr_cooldown_begin_points": DESIGN["lr_cooldown_begin_points"],
            "lr_cooldown_end_points": FINAL_POINTS,
            "extension_policy": "full-state endpoint retained; beyond 160K requires an explicit new LR stage",
            "resume_policy": "same full-state trajectory; skip mixtures already beyond endpoint"}


def training_directory(root, item):
    require_trajectory(item)
    return root / "trajectories" / item["id"]


def evaluation_directory(root, item):
    if item is None:
        return root / "baseline"
    require_milestone(item)
    return root / "milestones" / item["id"]


def plan(end_points=DEFAULT_END_POINTS):
    from .policy import PROTOCOL, VALIDATION_N
    scope = campaign(end_points)
    runs = [milestone(percent, end_points) for percent in PERCENTAGES]
    selected = active_milestones(end_points)
    return {
        "operation": "read-only plan; no ML, network, GPU or training",
        "protocol": PROTOCOL, "training_design": DESIGN, "campaign": scope,
        "trajectories": runs, "milestones": selected,
        "full_horizon_trajectories": trajectories(),
        "totals": {key: sum(row[key] for row in runs)
                   for key in ("budget_points", "points", "examples", "on_policy_examples",
                               "off_policy_examples", "updates", "unspent_points")},
        "training_trajectories": len(runs), "retained_evaluated_checkpoints": len(selected),
        "validation_generations_including_baseline": VALIDATION_N * (len(selected) + 1),
        "storage": f"One disposable Pod disk; {len(selected)} permanent full-state milestone checkpoints plus "
                   "two rolling restart checkpoints per mixture, transient saves, models and data. "
                   "Existing later artifacts are preserved and need additional space.",
        "warning": "Milestones are correlated prefixes, not independent budget runs. "
                   "Training costs are counted once per mixture, not once per evaluation.",
    }
