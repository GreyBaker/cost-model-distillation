"""Offline integrity/count/coverage audit. Reaggregates saved scores, not ROUGE."""
from __future__ import annotations

import json
import math
from pathlib import Path

from . import evaluation
from .controller import check_completed_training, saved_pilot
from .policy import VALIDATION_N
from .shared import read_json
from . import schedule


def prediction_audit(output, expected=None, expected_n=VALIDATION_N):
    metrics = read_json(output / "metrics.json")
    ids, sums, counters = {}, {k: 0.0 for k in metrics["scores"]}, {k: 0 for k in metrics["counts"]}
    with (output / "predictions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            if not line.endswith("\n"):
                raise ValueError("Incomplete final prediction line")
            row = json.loads(line)
            key = str(row["id"])
            if key in ids or row["split"] != "validation":
                raise ValueError("Duplicate ID or non-validation prediction")
            ids[key] = row["reference"]
            if expected is not None and expected.get(key) != row["reference"]:
                raise ValueError("ID/reference differs from the common baseline")
            for name in sums:
                score = row["scores"][name]
                if not math.isfinite(score) or not 0 <= score <= 100:
                    raise ValueError("Invalid saved ROUGE score")
                sums[name] += score
            for name in counters:
                counters[name] += int(row[name])
            tokens = row["generated_ids"]
            eos = 1 in tokens
            if (not tokens or len(tokens) > 64 or (eos and tokens.index(1) != len(tokens) - 1)
                    or row["terminated_with_eos"] != eos
                    or row["hit_generation_cap"] != (len(tokens) == 64 and not eos)
                    or row["empty_prediction"] != (not row["prediction"].strip())):
                raise ValueError("Generation EOS/cap/empty flags differ")
    if len(ids) != expected_n or (expected is not None and ids != expected) or counters != metrics["counts"]:
        raise ValueError("Wrong coverage/reference/flag totals")
    for name, total in sums.items():
        if not math.isclose(total / expected_n, metrics["scores"][name], abs_tol=1e-10):
            raise ValueError("Saved-score aggregate differs")
    return ids


def audit(root, bound, end_points=schedule.DEFAULT_END_POINTS):
    schedule.require_design(bound)
    selected = saved_pilot(root)
    baseline = Path(read_json(root / "baseline/latest_evaluation.json")["path"])
    evaluation.verify(baseline, evaluation.identity(root, bound, None, selected))
    expected = prediction_audit(baseline)
    from .milestones import latest_restart
    progress, skipped_trajectories = {}, []
    for trajectory in schedule.trajectories():
        _, marker = latest_restart(root, bound, trajectory, selected)
        progress[trajectory["id"]] = marker["step"] if marker else 0
        if progress[trajectory["id"]] > schedule.milestone(trajectory["on_policy_percent"], end_points)["updates"]:
            skipped_trajectories.append(trajectory["id"])
    verified, skipped = [], []
    for item in schedule.active_milestones(end_points):
        if item["trajectory_id"] in skipped_trajectories:
            skipped.append({"cell": item["id"], "reason": "trajectory already beyond active endpoint"})
            continue
        if not check_completed_training(root, bound, item, selected):
            if progress[item["trajectory_id"]] > item["updates"]:
                skipped.append({"cell": item["id"], "reason": "no retained snapshot before current checkpoint"})
                continue
            raise ValueError(f"Incomplete cell: {item['id']}")
        owner = schedule.evaluation_directory(root, item)
        output = Path(read_json(owner / "latest_evaluation.json")["path"])
        evaluation.verify(output, evaluation.identity(root, bound, item, selected))
        prediction_audit(output, expected)
        verified.append(item["id"])
    return {"status": "pass_with_skips" if skipped else "pass", "cells": verified, "validation_n": VALIDATION_N,
            "campaign": schedule.campaign(end_points), "skipped": skipped,
            "skipped_trajectories": skipped_trajectories,
            "training_design": schedule.DESIGN,
            "checks": ["immutable common initialization", "final checkpoint/evaluation hashes",
                       "point and exact-mixture counts", "exact validation ID/reference agreement",
                       "no duplicates", "reaggregated saved score means/counters", "EOS/cap/empty flags"],
            "limitations": "Does not recompute ROUGE from text or establish real-ML restart equivalence/seed robustness. "
                           "Skipped items have no evaluation coverage/score audit; only their latest restart "
                           "bytes and recipe are verified. Campaign completion is not full-horizon completion."}
