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
from sftlib.core import configure_local_storage


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


def commands(sft_work, grid_work):
    sft = [sys.executable, "-u", str(SFT_CODE / "prepare_models.py"), "--work-dir", str(sft_work)]
    grid = [sys.executable, "-u", str(ROOT / "distill_grid.py"),
            "--work-dir", str(grid_work), "--sft-work", str(sft_work)]
    return [
        ("prepare_sft", [*sft, "prepare"]),
        ("sft", [*sft, "run", "--accept-reconstructed-recipe"]),
        ("bind", [*grid, "bind"]),
        ("pilot", [*grid, "pilot"]),
        ("estimate", [*grid, "estimate"]),
        ("grid", [*grid, "run", "--accept-estimate"]),
        ("audit", [*grid, "audit"]),
    ]


def run(disk, accept_grid_estimate=False):
    disk = Path(disk).expanduser().resolve(strict=True)
    if not disk.is_dir():
        raise ValueError("POD_DISK must be an existing on-Pod directory")
    sft_work = Path(os.environ.get("SFT_WORK", disk / "sft-work")).expanduser().resolve()
    grid_work = Path(os.environ.get("GKD_WORK", disk / "gkd-grid-work")).expanduser().resolve()
    state_dir = disk / "pipeline"
    for path in (SFT_CODE.resolve(), ROOT.resolve(), sft_work, grid_work):
        if not path.is_relative_to(disk):
            raise ValueError("Source packages and work directories must all be on POD_DISK")
    separate_work(grid_work, sft_work)
    if (state_dir.is_relative_to(sft_work) or sft_work.is_relative_to(state_dir)
            or state_dir.is_relative_to(grid_work) or grid_work.is_relative_to(state_dir)):
        raise ValueError("Use separate sft-work, gkd-grid-work and pipeline directories")
    os.environ["POD_DISK"] = str(disk)
    configure_local_storage(disk)
    with RunLock(state_dir / ".lock"), Stop() as stop:
        def status(phase, **extra):
            atomic_json(state_dir / "status.json", {"phase": phase, **extra})
            print(json.dumps({"phase": phase, **extra}), flush=True)

        try:
            for name, argv in commands(sft_work, grid_work):
                if stop.requested:
                    status("paused")
                    return
                if name == "grid" and not accept_grid_estimate:
                    status("awaiting_grid_approval", estimate=str(grid_work / "estimate.json"))
                    print("Review the estimate above, then rerun with --accept-grid-estimate. "
                          "Existing SFT/pilot results will be reused.", flush=True)
                    return
                status(name, command=argv)
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
                if name == "sft":
                    if read_json(sft_work / "status.json").get("phase") != "complete":
                        status("paused", stage=name)
                        return
                    inspect_pair(sft_work, full=False)  # bind performs full byte verification next.
                elif name == "pilot":
                    if read_json(grid_work / "status.json").get("status") != "pilot_ready":
                        status("paused", stage=name)
                        return
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
    subs = parser.add_subparsers(dest="command")
    subs.add_parser("plan", help="Read-only stage plan (default)")
    execute = subs.add_parser("run", help="ON POD: prepare/train SFT, then pilot/estimate/grid/audit")
    execute.add_argument("--accept-reconstructed-recipe", action="store_true", required=True)
    execute.add_argument("--accept-grid-estimate", action="store_true",
                         help="Authorize unattended grid launch after its measured estimate is printed; "
                              "otherwise pause for review before grid training")
    args = parser.parse_args()
    if args.command != "run":
        print(json.dumps({"operation": "plan only; no setup/downloads/ML/GPU work",
                          "stages": [name for name, _ in commands("<SFT_WORK>", "<GKD_WORK>")],
                          "prerequisites": "One disposable POD_DISK, both source folders and the SFT venv already set up.",
                          "approval": "run requires recipe acceptance; without --accept-grid-estimate it pauses before the grid."},
                         indent=2))
        return
    if args.pod_disk is None:
        parser.error("Set POD_DISK or pass --pod-disk before run")
    run(args.pod_disk, args.accept_grid_estimate)


if __name__ == "__main__":
    main()
