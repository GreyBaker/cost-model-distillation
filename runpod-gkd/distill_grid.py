"""Opt-in, post-SFT XSum GKD grid. Default is a read-only plan, not execution."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from gridlib import policy, schedule
from gridlib.shared import RunLock, atomic_json, read_json
from sftlib.console import output as display


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path(os.environ.get("GKD_WORK", "work")))
    parser.add_argument("--sft-work", type=Path, default=Path(os.environ.get("SFT_WORK", "../runpod-sft/work")))
    parser.add_argument("--settings", type=Path, default=policy.ROOT / "settings.json")
    parser.add_argument("--json", action="store_true", help="JSON for plan/summary/estimate/audit reports")
    parser.add_argument("--end-points", type=int, choices=schedule.EVALUATION_POINTS,
                        default=schedule.DEFAULT_END_POINTS,
                        help="Active endpoint up to 160000; the intended LR horizon remains 160K")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("plan", help="Read-only grid/count/runtime scenarios; no ML imports")
    commands.add_parser("estimate", help="Read-only measured full-grid estimate from saved pilot/baseline")
    run = commands.add_parser("run", help="ON POD: automatically bind, calibrate, validate, train/resume and audit")
    run.add_argument("--accept-estimate", action="store_true", help=argparse.SUPPRESS)
    commands.add_parser("summary", help="Write JSON/CSV/Markdown summary without ML imports")
    commands.add_parser("audit", help="Offline final-grid hashes/counts/coverage audit; no ML imports")
    for name in ("_train", "_evaluate", "_pilot", "_evaluation_pilot", "_runtime"):
        worker = commands.add_parser(name, help=argparse.SUPPRESS)
        worker.add_argument("--cell")
        worker.add_argument("--microbatch", type=int)
        worker.add_argument("--online", type=int, choices=[0, 1])
        worker.add_argument("--output", type=Path)
        worker.add_argument("--session-updates", type=int)
        worker.add_argument("--until-points", type=int)
        worker.add_argument("--attempt-result", type=Path)
    args = parser.parse_args()
    operation = args.command or "plan"
    settings = policy.settings(args.settings)
    if operation == "plan":
        display({**policy.plan(args.end_points), "settings": settings}, "Point-budget GKD plan", args.json)
        return
    root, sft_work = args.work_dir.resolve(), args.sft_work.resolve()
    schedule.validate_root(root, allow_new=operation == "run")
    if operation == "summary":
        from gridlib.reporting import summary
        if not root.is_dir():
            raise ValueError("No existing grid work directory")
        with RunLock(root / ".controller.lock"):
            display(summary(root, args.end_points), "Grid results", args.json)
        return
    from gridlib.inputs import binding, separate_work
    separate_work(root, sft_work)
    if operation == "estimate":
        from gridlib.controller import saved_pilot
        from gridlib.reporting import estimate
        # Read only: no locks/directories/files are created and no CUDA is touched.
        bound = binding(root, sft_work, settings, full=False)
        display(estimate(root, saved_pilot(root), settings, args.end_points), "Grid runtime estimate", args.json)
        return
    lock_name = ".worker.lock" if operation.startswith("_") else ".controller.lock"
    with RunLock(root / lock_name):
        bound = binding(root, sft_work, settings, create=operation == "run", full=operation == "run")
        if operation.startswith("_"):
            from sftlib.core import configure_local_storage
            configure_local_storage(root)
        if operation == "audit":
            from gridlib.audit import audit
            result = audit(root, bound, args.end_points)
            atomic_json(root / "audit.json", result)
            display(result, "Grid integrity audit", args.json)
        elif operation == "run":
            from gridlib.controller import run
            from sftlib.core import configure_local_storage
            configure_local_storage(root)
            try:
                run(root, bound, args.settings.resolve(), args.end_points)
            except BaseException as error:
                atomic_json(root / "status.json", {"status": "failed", "error": str(error)})
                raise
        elif operation == "_runtime":
            from gridlib.backend import runtime
            _, info = runtime(settings)
            atomic_json(root / "runtime.json", info)
        elif operation == "_pilot":
            from gridlib.engine import pilot_worker
            if args.microbatch not in settings["pilot_microbatches"] or args.online is None or args.output is None:
                raise ValueError("Explicit pilot worker arguments required")
            pilot_worker(root, bound, args.microbatch, bool(args.online), args.output.resolve())
        elif operation == "_evaluation_pilot":
            from gridlib.controller import saved_pilot
            from gridlib.evalpilot import worker
            if args.microbatch is None or args.output is None:
                raise ValueError("Explicit evaluation-pilot batch and output required")
            output = args.output.resolve()
            if not output.is_relative_to(root / "evaluation_pilots"):
                raise ValueError("Evaluation-pilot results must stay in this grid's evaluation_pilots directory")
            worker(root, bound, saved_pilot(root), args.microbatch, output)
        elif operation in ("_train", "_evaluate"):
            from gridlib.controller import saved_pilot
            from gridlib.recovery import (OOM_EXIT_CODE, evaluation_microbatch, training_microbatch)
            selected = saved_pilot(root)
            candidates = schedule.trajectories() if operation == "_train" else schedule.milestones()
            item = next((x for x in candidates if x["id"] == args.cell), None)
            if item is None and not (operation == "_evaluate" and args.cell == "baseline"):
                raise ValueError("Explicit grid cell required")
            if args.attempt_result is not None and not args.attempt_result.resolve().is_relative_to(root / "attempts"):
                raise ValueError("Attempt reports must stay inside this grid's attempts directory")
            actual_batch = (training_microbatch(root, selected) if operation == "_train"
                            else evaluation_microbatch(root, bound, item, selected))
            if args.microbatch is not None and args.microbatch != actual_batch:
                raise ValueError("Worker microbatch differs from the recorded active batching policy")
            import torch
            try:
                if operation == "_train":
                    from gridlib.engine import train
                    if args.session_updates is not None and args.session_updates < 1:
                        raise ValueError("Session pilot update cap must be positive")
                    train(root, bound, item, selected, args.session_updates, microbatch=actual_batch,
                          until_points=args.until_points if args.until_points is not None else args.end_points)
                else:
                    from gridlib.evaluation import evaluate
                    if args.output is None:
                        raise ValueError("Explicit evaluation output required")
                    output = args.output.resolve()
                    owner = schedule.evaluation_directory(root, item)
                    if not output.is_relative_to(owner):
                        raise ValueError("Evaluation must stay within its own cell/baseline directory")
                    evaluate(root, bound, item, selected, output)
                    atomic_json(owner / "latest_evaluation.json", {"path": str(output)})
            except torch.cuda.OutOfMemoryError as error:
                if args.attempt_result is None:
                    raise
                atomic_json(args.attempt_result.resolve(),
                            {"status": "cuda_oom", "operation": operation[1:], "cell": args.cell,
                             "microbatch": actual_batch, "error": str(error)})
                raise SystemExit(OOM_EXIT_CODE)


if __name__ == "__main__":
    main()
