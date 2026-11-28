"""Atomic permanent copies of selected full-state checkpoints, never quality gates."""
from __future__ import annotations

import os
import shutil
import uuid

from . import schedule
from .shared import (atomic_json, checkpoints, check_space, read_json, sha256,
                     sync_directory, sync_tree)


def verify_run(root, bound, trajectory, selected):
    from .controller import training_config
    schedule.require_design(bound)
    schedule.require_trajectory(trajectory)
    run = read_json(schedule.training_directory(root, trajectory) / "run.json")
    signatures = {"binding_sha256": sha256(root / "binding.json"),
                  "pilot_sha256": sha256(root / "pilot.json")}
    if (run["config"] != training_config(bound, trajectory, selected)
            or run["signatures"] != signatures
            or run["initial_student"] != bound["inputs"]["paths"]["student_checkpoint"]):
        raise ValueError("Trajectory provenance/configuration changed")
    return run


def latest_restart(root, bound, trajectory, selected, *, full=True):
    """Read-only latest full-state restart; permanent copies survive rolling loss."""
    run_dir = schedule.training_directory(root, trajectory)
    if not (run_dir / "run.json").exists():
        if run_dir.exists():
            raise ValueError(f"Unowned trajectory directory: {run_dir}")
        return None, None
    run = verify_run(root, bound, trajectory, selected)
    candidates = [(read_json(path / "COMPLETE.json")["step"], path, None)
                  for path in checkpoints.completed(run_dir)]
    for item in schedule.milestones():
        owner = schedule.evaluation_directory(root, item)
        if item["trajectory_id"] == trajectory["id"] and (owner / "snapshot.json").exists():
            candidates.append((item["updates"], owner / "checkpoint", item))
    if not candidates:
        if not full:
            return None, None  # A first save can fail; reporting must still show pending work.
        raise ValueError("Initialized trajectory has no complete full-state restart")
    step, checkpoint, item = max(candidates, key=lambda row: row[0])
    if type(step) is not int or not 0 <= step <= trajectory["updates"]:
        raise ValueError("Invalid saved trajectory update count")
    if full and item is not None:
        checkpoint, _ = verify(root, bound, item, selected)
        marker = read_json(checkpoint / "COMPLETE.json")  # Already fully verified above.
    else:
        marker = (checkpoints.verify(checkpoint, run["run_id"]) if full
                  else read_json(checkpoint / "COMPLETE.json"))
    metadata = read_json(checkpoint / "metadata.json")
    if (marker["step"] != step or marker["run_id"] != run["run_id"]
            or metadata["step"] != step or metadata["run_id"] != run["run_id"]
            or metadata["config"] != run["config"] or metadata["signatures"] != run["signatures"]
            or "training_state.pt" not in marker["files"]):
        raise ValueError("Restart checkpoint recipe/provenance differs")
    return checkpoint, marker


def verify(root, bound, item, selected):
    schedule.require_design(bound)
    schedule.require_milestone(item)
    from .engine import check_state
    trajectory = schedule.trajectory(item["on_policy_percent"])
    run = verify_run(root, bound, trajectory, selected)
    signatures = {"binding_sha256": sha256(root / "binding.json"),
                  "pilot_sha256": sha256(root / "pilot.json")}
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
    schedule.require_design(bound)
    schedule.require_trajectory(trajectory)
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
