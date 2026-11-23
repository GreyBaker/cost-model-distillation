"""Opt-in, post-SFT XSum GKD grid. Default is a read-only plan, not execution."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from gridlib import policy
from gridlib.shared import RunLock, atomic_json, read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path(os.environ.get("GKD_WORK", "work")))
    parser.add_argument("--sft-work", type=Path, default=Path(os.environ.get("SFT_WORK", "../runpod-sft/work")))
    parser.add_argument("--settings", type=Path, default=policy.ROOT / "settings.json")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("plan", help="Read-only grid/count/runtime scenarios; no ML imports")
    commands.add_parser("bind", help="After SFT only: verify/freeze completed pair (hashes; no GPU/model load)")
    commands.add_parser("pilot", help="ON POD: once-only joint-fit pilot, save/reload, then full common baseline")
    commands.add_parser("estimate", help="Read-only measured full-grid estimate from saved pilot/baseline")
    run = commands.add_parser("run", help="Print estimate; execute only with --accept-estimate")
    run.add_argument("--accept-estimate", action="store_true")
    commands.add_parser("summary", help="Write JSON/CSV/Markdown summary without ML imports")
    commands.add_parser("audit", help="Offline final-grid hashes/counts/coverage audit; no ML imports")
    for name in ("_train", "_evaluate", "_pilot", "_runtime"):
        worker = commands.add_parser(name, help=argparse.SUPPRESS)
        worker.add_argument("--cell")
        worker.add_argument("--microbatch", type=int)
        worker.add_argument("--online", type=int, choices=[0, 1])
        worker.add_argument("--output", type=Path)
        worker.add_argument("--session-updates", type=int)
    args = parser.parse_args()
    operation = args.command or "plan"
    settings = policy.settings(args.settings)
    if operation == "plan":
        print(json.dumps({**policy.plan(), "settings": settings}, indent=2))
        return
    root, sft_work = args.work_dir.resolve(), args.sft_work.resolve()
    if operation == "summary":
        from gridlib.reporting import summary
        if not root.is_dir():
            raise ValueError("No existing grid work directory")
        with RunLock(root / ".controller.lock"):
            print(json.dumps(summary(root), indent=2))
        return
    from gridlib.inputs import binding, separate_work
    separate_work(root, sft_work)
    if operation == "estimate" or (operation == "run" and not args.accept_estimate):
        from gridlib.controller import saved_pilot
        from gridlib.reporting import estimate
        # Read only: no locks/directories/files are created and no CUDA is touched.
        bound = binding(root, sft_work, settings, full=False)
        print(json.dumps(estimate(root, saved_pilot(root), settings), indent=2))
        return
    lock_name = ".worker.lock" if operation.startswith("_") else ".controller.lock"
    if operation in ("pilot", "run", "_runtime", "_pilot", "_train", "_evaluate"):
        from sftlib.core import configure_local_storage
        configure_local_storage(root)
    with RunLock(root / lock_name):
        bound = binding(root, sft_work, settings, create=operation == "bind", full=operation == "bind")
        if operation == "bind":
            print("Verified completed 100K/8K SFT pair; no model loaded or GPU used.")
        elif operation == "audit":
            from gridlib.audit import audit
            result = audit(root, bound)
            atomic_json(root / "audit.json", result)
            print(json.dumps(result, indent=2))
        elif operation == "pilot":
            from gridlib.controller import pilot
            try:
                pilot(root, bound, args.settings.resolve())
            except BaseException as error:
                atomic_json(root / "status.json", {"status": "pilot_failed", "error": str(error)})
                raise
        elif operation == "run":
            from gridlib.controller import run
            # Verify immutable backing bytes again before a long grid/resume.
            binding(root, sft_work, settings, full=True)
            try:
                run(root, bound, args.settings.resolve())
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
        elif operation in ("_train", "_evaluate"):
            from gridlib.controller import saved_pilot
            selected = saved_pilot(root)
            item = next((x for x in policy.cells() if x["id"] == args.cell), None)
            if item is None and not (operation == "_evaluate" and args.cell == "baseline"):
                raise ValueError("Explicit grid cell required")
            if operation == "_train":
                from gridlib.engine import train
                if args.session_updates is not None and args.session_updates < 1:
                    raise ValueError("Session pilot update cap must be positive")
                train(root, bound, item, selected, args.session_updates)
            else:
                from gridlib.evaluation import evaluate
                if args.output is None:
                    raise ValueError("Explicit evaluation output required")
                output = args.output.resolve()
                owner = root / "cells" / item["id"] if item else root / "baseline"
                if not output.is_relative_to(owner):
                    raise ValueError("Evaluation must stay within its own cell/baseline directory")
                evaluate(root, bound, item, selected, output)
                atomic_json(owner / "latest_evaluation.json", {"path": str(output)})


if __name__ == "__main__":
    main()
