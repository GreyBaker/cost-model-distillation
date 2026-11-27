"""Atomic permanent copies of selected full-state checkpoints, never quality gates."""
from __future__ import annotations

import os
import shutil
import uuid

from . import schedule
from .shared import (atomic_json, checkpoints, check_space, read_json, sha256,
                     sync_directory, sync_tree)


def verify(root, bound, item, selected):
    from .engine import check_state
    from .controller import training_config
    run_dir = root / "trajectories" / item["trajectory_id"]
    run = read_json(run_dir / "run.json")
    signatures = {"binding_sha256": sha256(root / "binding.json"),
                  "pilot_sha256": sha256(root / "pilot.json")}
    trajectory = schedule.trajectory(item["on_policy_percent"])
    if (run["config"] != training_config(bound, trajectory, selected)
            or run["signatures"] != signatures
            or run["initial_student"] != bound["inputs"]["paths"]["student_checkpoint"]):
        raise ValueError("Milestone trajectory provenance/configuration changed")
    owner = schedule.evaluation_directory(root, item)
    record = read_json(owner / "snapshot.json")
    checkpoint = owner / "checkpoint"
    if (record["milestone"] != item or record["run_id"] != run["run_id"]
            or record["signatures"] != signatures
            or record["checkpoint_complete_sha256"] != sha256(checkpoint / "COMPLETE.json")):
        raise ValueError("Milestone snapshot identity changed")
    state = record["accounting"]
    check_state(state, trajectory)
    marker = checkpoints.verify(checkpoint, run["run_id"])
    if state["step"] != item["updates"] or state["points"] != item["points"] or marker["step"] != state["step"]:
        raise ValueError("Milestone checkpoint/point accounting differs")
    metadata = read_json(checkpoint / "metadata.json")
    if metadata["config"] != run["config"] or metadata["signatures"] != signatures:
        raise ValueError("Retained checkpoint recipe/provenance differs")
    return checkpoint, record


def retain(root, bound, trajectory, selected, checkpoint, state):
    """Called before advancing beyond a recovery boundary; safe to retry on resume."""
    matches = [item for item in schedule.milestones()
               if item["trajectory_id"] == trajectory["id"] and item["updates"] == state["step"]]
    if not matches:
        return
    item = matches[0]
    owner = schedule.evaluation_directory(root, item)
    if owner.exists():
        verify(root, bound, item, selected)
        return
    run = read_json(schedule.training_directory(root, trajectory) / "run.json")
    marker = checkpoints.verify(checkpoint, run["run_id"])
    parent = owner.parent
    parent.mkdir(parents=True, exist_ok=True)
    check_space(parent, sum(info["bytes"] for info in marker["files"].values()) + 1024**3)
    scratch = parent / ("." + owner.name + "-" + uuid.uuid4().hex + ".incomplete")
    scratch.mkdir()
    # Actual independent bytes, not links into the pruned recovery directory.
    shutil.copytree(checkpoint, scratch / "checkpoint")
    atomic_json(scratch / "snapshot.json", {
        "milestone": item, "run_id": run["run_id"], "accounting": dict(state),
        "signatures": run["signatures"],
        "checkpoint_complete_sha256": sha256(checkpoint / "COMPLETE.json"),
    })
    checkpoints.verify(scratch / "checkpoint", run["run_id"])
    sync_tree(scratch)
    os.replace(scratch, owner)
    sync_directory(parent)
    verify(root, bound, item, selected)
