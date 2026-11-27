from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path

from .shared import atomic_json, read_json


def select_pilot(results, candidates):
    eligible = []
    for size in candidates:
        branches = [row for row in results if row["microbatch"] == size]
        if len(branches) != 2 or {x["online"] for x in branches} != {False, True}:
            raise ValueError("Incomplete/duplicate pilot branch coverage")
        if all(row["status"] == "pass" for row in branches):
            if any(not math.isfinite(x["mean_update_seconds"]) or x["mean_update_seconds"] <= 0
                   for x in branches):
                raise ValueError("Invalid pilot timing")
            runtimes = [row["runtime"] for row in branches]
            if runtimes[0] != runtimes[1]:
                raise ValueError("Pilot hardware/software differs across branches")
            eligible.append((math.sqrt(math.prod(x["mean_update_seconds"] for x in branches)), size, branches))
    if not eligible:
        raise ValueError("No common passing microbatch with reserved-memory headroom. Grid will not start.")
    _, size, branches = min(eligible, key=lambda row: (row[0], row[1]))
    return {"microbatch": size, "runtime": branches[0]["runtime"],
            "off_policy": next(x for x in branches if not x["online"]),
            "on_policy": next(x for x in branches if x["online"])}


def estimate(root, selected, settings):
    from .trajectory_reporting import estimate as trajectory_estimate
    return trajectory_estimate(root, selected, settings)


def records(path):
    if not path.exists():
        return []
    result = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                result.append(json.loads(line))
            except json.JSONDecodeError:
                # A killed process can leave only its final append incomplete.
                if stream.read().strip():
                    raise
    return result


def summary(root):
    from . import schedule
    schedule.validate_root(root)
    stages = records(root / "stages.jsonl")
    baseline_metrics = {}
    baseline_pointer = root / "baseline/latest_evaluation.json"
    if baseline_pointer.exists():
        baseline_metrics = read_json(Path(read_json(baseline_pointer)["path"]) / "metrics.json")
    baseline = {"status": baseline_metrics.get("status", "pending"), "training_seconds": 0,
                "training_time_scope": "No additional GKD; prior SFT preparation time is external.",
                "evaluation_seconds": baseline_metrics.get("evaluation_seconds"),
                "scores": baseline_metrics.get("scores", {})}
    rows = []
    for item in schedule.milestones():
        run_dir = schedule.evaluation_directory(root, item)
        training_owner = root / "trajectories" / item["trajectory_id"]
        training_path = run_dir / "snapshot.json"
        training = read_json(training_path) if training_path.exists() else {}
        if training:
            training = {**training["accounting"], "status": "complete",
                        "microbatch": read_json(run_dir / "checkpoint/metadata.json")
                        .get("execution_config", {}).get("microbatch")}
        evaluation = {}
        pointer = run_dir / "latest_evaluation.json"
        if pointer.exists():
            evaluation = read_json(Path(read_json(pointer)["path"]) / "metrics.json")
        events = [event for path in sorted((training_owner / "sessions").glob("*.jsonl")) for event in records(path)
                  if event.get("step", 0) <= item["updates"]]
        batch_sessions = [event["microbatch"] for event in sorted(events, key=lambda e: e.get("unix_time", 0))
                          if event["event"] == "initialized" and "microbatch" in event]
        own_stages = [x for x in stages if x["cell"] == item["id"] or
                      (x["cell"] == item["trajectory_id"]
                       and x.get("until_points") == item["budget_points"])]
        train_wall = sum(x["seconds"] for x in own_stages if x["operation"] == "train")
        eval_wall = sum(x["seconds"] for x in own_stages if x["operation"] == "evaluate")
        status = training.get("status", "pending")
        if status == "complete":
            status = "complete" if evaluation.get("status") == "complete" else "trained"
        r2 = evaluation.get("scores", {}).get("rouge2")
        baseline_r2 = baseline["scores"].get("rouge2")
        row = {"cell": item["id"], "on_policy_percent": item["on_policy_percent"],
               "budget_points": item["budget_points"], "planned_points": item["points"],
               "unspent_points": item["unspent_points"], "status": status,
               "points": training.get("points"), "on_policy_examples": training.get("on_policy_examples"),
               "off_policy_examples": training.get("off_policy_examples"), "updates": training.get("step"),
               "source_tokens": training.get("source_tokens"), "target_tokens": training.get("target_tokens"),
               "microbatch": training.get("microbatch"), "session_microbatches": batch_sessions,
               "training_seconds": training.get("training_seconds"),
               "recorded_update_seconds_including_replay": sum(e.get("training_seconds", 0)
                                                              for e in events if e["event"] == "update"),
               "training_wall_seconds": train_wall,
               "evaluation_seconds": evaluation.get("evaluation_seconds"),
               "evaluation_batch_size": read_json(Path(read_json(pointer)["path"]) / "COMPLETE.json")
                                        .get("identity", {}).get("batch_size") if pointer.exists() else None,
               "evaluation_batching_recovery": read_json(run_dir / "evaluation_batching.json")
                                               if (run_dir / "evaluation_batching.json").exists() else None,
               "evaluation_wall_seconds": eval_wall,
               "total_stage_wall_seconds": train_wall + eval_wall,
               "rouge1": evaluation.get("scores", {}).get("rouge1"),
               "rouge2": r2,
               "rouge2_minus_sft": None if r2 is None or baseline_r2 is None else r2 - baseline_r2,
               "rougeL": evaluation.get("scores", {}).get("rougeL")}
        rows.append(row)
    finished_logs = {x["log"] for x in stages}
    unmeasured = [x for x in records(root / "stage_starts.jsonl") if x["log"] not in finished_logs]
    result = {"rows": rows, "baseline": baseline,
               "evaluation_pilot": (read_json(root / "evaluation_pilot.json")
                                    if (root / "evaluation_pilot.json").exists() else None),
               "evaluation_pilot_provenance": [read_json(path) for path in
                                               sorted((root / "evaluation_pilot_provenance").glob("*.json"))],
               "evaluation_pilot_wall_seconds": sum(x["seconds"] for x in stages
                                                     if x["operation"] == "evaluation_pilot"),
              "complete_cells": sum(x["status"] == "complete" for x in rows),
              "unmeasured_interrupted_stages": unmeasured,
              "shared_stage_wall_seconds": sum(x["seconds"] for x in stages if x["cell"] == "shared"),
              "oom_training_recovery": read_json(root / "training_batching.json")
                                      if (root / "training_batching.json").exists() else None,
               "time_accounting": "training_seconds = committed synchronized update time including generation/scoring/backward. "
                                  "Stage wall includes initialization/checkpoint/evaluation and recorded failed attempts. "
                                  "Shared pilot/baseline/controller overhead is separate; evaluation_pilot_wall_seconds "
                                  "is a component of shared_stage_wall_seconds, not an additional cost. "
                                  "Killed controllers may leave unmeasured stages."}
    result["training_design"] = schedule.DESIGN
    result["complete_milestones"] = result["complete_cells"]
    result["complete_trajectories"] = sum(
        all(row["status"] == "complete" for row in rows if row["on_policy_percent"] == percent)
        for percent in schedule.PERCENTAGES)
    result["committed_training_seconds"] = sum(
        read_json(path)["training_seconds"] for item in schedule.trajectories()
        if (path := schedule.training_directory(root, item) / "training_summary.json").exists())
    result["recorded_training_stage_wall_seconds"] = sum(
        event["seconds"] for event in stages if event["operation"] == "train")
    result["time_accounting"] += (
        " Milestone training_seconds and example/update/token counts are CUMULATIVE prefixes "
        "of one trajectory per mixture; DO NOT sum them across milestones. "
        "committed_training_seconds counts each trajectory once. "
        "Row training_wall_seconds covers only that milestone's incremental training stage.")
    atomic_json(root / "summary.json", result)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    (root / "summary.csv").write_text(buffer.getvalue(), encoding="utf-8")
    def show(value, divisor=1):
        return "—" if value is None else f"{value / divisor:.3f}"
    lines = ["# Point-budget GKD results", "",
             f"Common SFT baseline R2: **{show(baseline['scores'].get('rouge2'))}**; "
             f"validation minutes: {show(baseline['evaluation_seconds'], 60)}. "
             "Prior SFT training is not charged to GKD.", "",
             "| Cell | Status | R2 | Training min | Eval min | Stage wall min |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    lines.extend(f"| {row['cell']} | {row['status']} | {show(row['rouge2'])} | "
                 f"{show(row['training_seconds'], 60)} | {show(row['evaluation_seconds'], 60)} | "
                 f"{show(row['total_stage_wall_seconds'], 60)} |" for row in rows)
    lines.extend(["", result["time_accounting"], ""])
    (root / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
    return result
