"""Run paired SFT, then the distillation grid. Default: print a plan only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from gridlib.policy import ROOT
from gridlib.shared import SFT_CODE, RunLock, atomic_json, read_json
from gridlib.inputs import inspect_pair, separate_work
from gridlib.schedule import validate_root
from sftlib.core import configure_local_storage
from sftlib.console import message, output


class Stop:
    """Forward stop requests to the isolated child session; launch nothing else."""
    def __init__(self):
        self.requested, self.child, self.previous = False, None, {}

    def handle(self, number, frame):
        self.requested = True
        if self.child is not None and self.child.poll() is None:
            try:
                if os.name == "posix":
                    os.killpg(self.child.pid, number)
                else:
                    self.child.send_signal(number)
            except ProcessLookupError:
                pass
        print("Pipeline stop requested; waiting for the current stage, with no next stage.", flush=True)

    def __enter__(self):
        for number in (signal.SIGINT, signal.SIGTERM):
            self.previous[number] = signal.signal(number, self.handle)
        return self

    def __exit__(self, *args):
        for number, previous in self.previous.items():
            signal.signal(number, previous)


def invoke(argv, stop):
    if stop.requested:
        return False
    stop.child = subprocess.Popen(argv, start_new_session=os.name == "posix")
    if stop.requested:  # Handle a signal arriving during Popen.
        stop.handle(signal.SIGTERM, None)
    code = stop.child.wait()
    stop.child = None
    if stop.requested:
        return False
    if code:
        raise subprocess.CalledProcessError(code, argv)
    return True


def commands(sft_work, grid_work, reuse_student=False, reuse_sft=False):
    if reuse_student and reuse_sft:
        raise ValueError("Choose student-only reuse or complete-pair reuse")
    sft = [sys.executable, "-u", str(SFT_CODE / "prepare_models.py"), "--work-dir", str(sft_work)]
    grid = [sys.executable, "-u", str(ROOT / "distill_grid.py"),
            "--work-dir", str(grid_work), "--sft-work", str(sft_work)]
    if reuse_sft:
        preparation = []
    elif reuse_student:
        preparation = [("verify_student", [*sft, "verify-student"]),
                       ("prepare_sft", [*sft, "prepare", "--role", "teacher"])]
    else:
        preparation = [("prepare_sft", [*sft, "prepare"])]
    sft_flags = (["--reuse-sft"] if reuse_sft else
                 ["--accept-reconstructed-recipe", *(["--reuse-student"] if reuse_student else [])])
    return [*preparation,
        ("reuse_sft" if reuse_sft else "sft", [*sft, "run", *sft_flags]),
        ("grid", [*grid, "run"]),
    ]


def run(disk, accept_grid_estimate=False, reuse_student=False, reuse_sft=False):
    disk = Path(disk).expanduser().resolve(strict=True)
    if not disk.is_dir():
        raise ValueError("POD_DISK must be an existing on-Pod directory")
    sft_work = Path(os.environ.get("SFT_WORK", disk / "sft-work")).expanduser().resolve()
    grid_work = Path(os.environ.get("GKD_WORK", disk / "gkd-grid-work")).expanduser().resolve()
    state_dir = disk / "pipeline"
    for path in (sft_work, grid_work, state_dir):
        if any(parent.name.lower() == "results_and_temp" for parent in (path, *path.parents)):
            raise ValueError("results_and_temp is immutable; restore inputs into separate on-Pod work directories")
    for path in (SFT_CODE.resolve(), ROOT.resolve(), sft_work, grid_work):
        if not path.is_relative_to(disk):
            raise ValueError("Source packages and work directories must all be on POD_DISK")
    separate_work(grid_work, sft_work)
    validate_root(grid_work, allow_new=True)
    if (state_dir.is_relative_to(sft_work) or sft_work.is_relative_to(state_dir)
            or state_dir.is_relative_to(grid_work) or grid_work.is_relative_to(state_dir)):
        raise ValueError("Use separate sft-work, gkd-grid-work and pipeline directories")
    os.environ["POD_DISK"] = str(disk)
    configure_local_storage(disk)
    with RunLock(state_dir / ".lock"), Stop() as stop:
        def status(phase, **extra):
            atomic_json(state_dir / "status.json", {"phase": phase, **extra})
            details = " | ".join(f"{key.replace('_', ' ')}: {value}"
                                 for key, value in extra.items() if key != "command")
            message(f"Pipeline · {phase.replace('_', ' ')}" + (f" | {details}" if details else ""))

        try:
            stages = commands(sft_work, grid_work, reuse_student, reuse_sft)
            for index, (name, argv) in enumerate(stages, 1):
                if stop.requested:
                    status("paused")
                    return
                status(name, command=argv)
                message(f"Stage {index}/{len(stages)}")
                started = time.monotonic()
                try:
                    if not invoke(argv, stop):
                        status("paused", stage=name)
                        return
                finally:
                    with (state_dir / "stages.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({"stage": name, "seconds": time.monotonic() - started,
                                                 "command": argv}) + "\n")
                # Graceful pauses can exit zero: never mistake them for completion.
                if name in ("sft", "reuse_sft"):
                    if read_json(sft_work / "status.json").get("phase") != "complete":
                        status("paused", stage=name)
                        return
                    inspect_pair(sft_work, full=False)  # Grid run verifies all backing bytes next.
                elif name == "grid":
                    if read_json(grid_work / "status.json").get("status") != "complete":
                        status("paused", stage=name)
                        return
            status("complete", results=str(grid_work / "summary.json"),
                   warning="Download results now; the disk is disposable. Paid compute is not stopped.")
        except BaseException as error:
            status("failed", error=str(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pod-disk", type=Path, default=os.environ.get("POD_DISK"))
    parser.add_argument("--json", action="store_true", help="Emit the read-only plan as JSON")
    subs = parser.add_subparsers(dest="command")
    plan = subs.add_parser("plan", help="Read-only stage plan (default)")
    plan_reuse = plan.add_mutually_exclusive_group()
    plan_reuse.add_argument("--reuse-student", action="store_true", help="Plan reuse of an existing complete student role")
    plan_reuse.add_argument("--reuse-sft", action="store_true", help="Plan reuse of both completed SFT roles")
    execute = subs.add_parser("run", help="Historical SFT workflow, then a single automatic grid run")
    execute.add_argument("--accept-reconstructed-recipe", action="store_true")
    execute.add_argument("--accept-grid-estimate", action="store_true", help=argparse.SUPPRESS)
    execute_reuse = execute.add_mutually_exclusive_group()
    execute_reuse.add_argument("--reuse-student", action="store_true",
                         help="Verify SFT_WORK/student and skip student preparation/training/export/evaluation")
    execute_reuse.add_argument("--reuse-sft", action="store_true",
                               help="Reuse both completed SFT roles, rebuild pair paths, launch no SFT workers")
    args = parser.parse_args()
    if args.command != "run":
        output({"operation": "plan only; no setup/downloads/ML/GPU work",
                "stages": [name for name, _ in commands("<SFT_WORK>", "<GKD_WORK>",
                          getattr(args, "reuse_student", False), getattr(args, "reuse_sft", False))],
                "reuse_student": getattr(args, "reuse_student", False),
                "reuse_sft": getattr(args, "reuse_sft", False),
                "prerequisites": "One disposable POD_DISK, both source folders and the SFT venv already set up.",
                "approval": "SFT training requires recipe acceptance; --reuse-sft only verifies restored roles. "
                             "The grid runs automatically, including calibration and audit."},
               "SFT to GKD pipeline", args.json)
        return
    if args.pod_disk is None:
        parser.error("Set POD_DISK or pass --pod-disk before run")
    if not args.reuse_sft and not args.accept_reconstructed_recipe:
        parser.error("SFT training requires --accept-reconstructed-recipe; use --reuse-sft to reuse both complete roles")
    run(args.pod_disk, args.accept_grid_estimate, args.reuse_student, args.reuse_sft)


if __name__ == "__main__":
    main()
