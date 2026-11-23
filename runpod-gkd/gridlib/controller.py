"""Serial Pod orchestration; no model imports in plans/estimates/reports."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from . import evaluation, reporting
from .policy import ROOT, cells
from .shared import atomic_json, checkpoints, read_json, sha256
from .engine import append, check_state
from .recovery import (CudaOOM, OOM_EXIT_CODE, evaluation_microbatch, evaluation_path,
                       lower, train_with_recovery)


def command(root, sft_work, settings_path, operation, *extra):
    return [sys.executable, "-u", str(ROOT / "distill_grid.py"), "--work-dir", str(root),
            "--sft-work", str(sft_work), "--settings", str(settings_path), operation, *map(str, extra)]


class Runner:
    def __init__(self, root, sft_work, settings_path, stop):
        self.root, self.sft_work, self.settings_path, self.stop = root, sft_work, settings_path, stop

    def call(self, operation, cell_id, *extra):
        if self.stop.requested:
            return False
        logs = self.root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        log = logs / f"{cell_id}-{operation}-{time.time_ns()}.log"
        started = time.monotonic()
        code = None
        attempt = self.root / "attempts" / (uuid.uuid4().hex + ".json")
        append(self.root / "stage_starts.jsonl",
               {"cell": cell_id, "operation": operation, "log": str(log), "unix_time": time.time()})
        print(json.dumps({"cell": cell_id, "operation": operation, "log": str(log)}), flush=True)
        try:
            with log.open("x", encoding="utf-8") as stream:
                worker_extra = [*extra]
                if operation in ("train", "evaluate"):
                    worker_extra += ["--attempt-result", str(attempt)]
                self.stop.child = subprocess.Popen(
                    command(self.root, self.sft_work, self.settings_path, "_" + operation, *worker_extra),
                    cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=os.name != "nt",
                    env={**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
                if self.stop.requested:
                    import signal
                    self.stop.child.send_signal(signal.SIGTERM)
                code = self.stop.child.wait()
                self.stop.child = None
            if code and not self.stop.requested:
                if code == OOM_EXIT_CODE and attempt.is_file() and operation in ("train", "evaluate"):
                    report = read_json(attempt)
                    expected_cell = str(extra[extra.index("--cell") + 1])
                    expected_batch = int(extra[extra.index("--microbatch") + 1])
                    if (report.get("status") == "cuda_oom" and report.get("operation") == operation
                            and report.get("cell") == expected_cell and report.get("microbatch") == expected_batch):
                        raise CudaOOM({**report, "attempt_result": str(attempt), "log": str(log)})
                raise RuntimeError(f"{cell_id} {operation} failed: inspect {log}")
            return not self.stop.requested
        finally:
            append(self.root / "stages.jsonl",
                   {"cell": cell_id, "operation": operation, "seconds": time.monotonic() - started,
                    "return_code": code, "log": str(log)})
            if cell_id != "shared":
                reporting.summary(self.root)


def saved_pilot(root):
    pilot = read_json(root / "pilot.json")
    if pilot["binding_sha256"] != sha256(root / "binding.json"):
        raise ValueError("Saved pilot belongs to different inputs/settings/code")
    for entry in pilot["result_files"]:
        if sha256(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("Saved pilot result changed")
    return pilot["selected"]


def ensure_evaluation(root, bound, item, selected, runner):
    while True:
        try:
            return evaluation_attempt(root, bound, item, selected, runner)
        except CudaOOM as error:
            if item is not None:
                from .recovery import checkpoint_for_retry
                checkpoint_for_retry(root, item)
            initial = min(selected["microbatch"], bound["settings"]["evaluation_batch_cap"])
            lower(root, evaluation_path(root, item), initial,
                  evaluation_microbatch(root, bound, item, selected),
                  cell=item["id"] if item else "baseline", failure=error.report)


def evaluation_attempt(root, bound, item, selected, runner):
    owner = root / "cells" / item["id"] if item else root / "baseline"
    owner.mkdir(parents=True, exist_ok=True)
    pointer = owner / "latest_evaluation.json"
    expected = evaluation.identity(root, bound, item, selected)
    if pointer.exists():
        output = Path(read_json(pointer)["path"])
        evaluation.verify(output, expected)
        return True
    output = owner / "evaluation"
    if output.exists():
        if (output / "COMPLETE.json").exists():
            evaluation.verify(output, expected)
            atomic_json(pointer, {"path": str(output)})
            return True
        output = owner / ("evaluation-retry-" + uuid.uuid4().hex)
    return runner.call("evaluate", item["id"] if item else "shared",
                       "--cell", item["id"] if item else "baseline", "--output", output,
                       "--microbatch", evaluation_microbatch(root, bound, item, selected))


def check_completed_training(root, bound, item, selected):
    run_dir = root / "cells" / item["id"]
    path = run_dir / "training_summary.json"
    if not path.exists():
        return False
    result = read_json(path)
    if result.get("status") != "complete":
        return False
    run = read_json(run_dir / "run.json")
    if (result["cell"] != item or run["config"] !=
            {"cell": item, "microbatch": selected["microbatch"], "settings": bound["settings"],
             "protocol": bound["protocol"]}
            or run["signatures"] != {"binding_sha256": sha256(root / "binding.json"),
                                      "pilot_sha256": sha256(root / "pilot.json")}
            or run["initial_student"] != bound["inputs"]["paths"]["student_checkpoint"]):
        raise ValueError("Completed cell provenance/configuration changed")
    check_state(result, item)
    if result["step"] != item["updates"]:
        raise ValueError("Completed cell has incorrect update count")
    checkpoint = checkpoints.resolve(run_dir)
    marker = checkpoints.verify(checkpoint, run["run_id"])
    if str(checkpoint) != result["checkpoint"] or marker["step"] != result["step"]:
        raise ValueError("Completed cell checkpoint/summary disagree")
    return True


def pilot(root, bound, settings_path):
    from sftlib.sequence import ChildStop
    with ChildStop() as stop:
        runner = Runner(root, bound["inputs"]["sft_work"], settings_path, stop)
        if not runner.call("runtime", "shared"):
            return
        runtime = read_json(root / "runtime.json")
        request = {"binding_sha256": sha256(root / "binding.json"), "runtime": runtime}
        request_path = root / "pilot_request.json"
        if request_path.exists() and read_json(request_path) != request:
            raise ValueError("Pilot hardware/software/inputs changed; use a new grid directory")
        atomic_json(request_path, request)
        if (root / "pilot.json").exists():
            selected = saved_pilot(root)
            if selected["runtime"] != runtime:
                raise ValueError("Saved pilot is not applicable to the current GPU/software")
            print(f"Reusing once-only pilot: microbatch {selected['microbatch']}", flush=True)
        else:
            results, files = [], []
            for size in bound["settings"]["pilot_microbatches"]:
                for online in (False, True):
                    output = root / "pilots" / f"mb{size:02d}_{'on' if online else 'off'}.json"
                    if not output.exists():
                        if not runner.call("pilot", "shared", "--microbatch", size, "--online", int(online),
                                           "--output", output):
                            return
                    result = read_json(output)
                    if (result["microbatch"] != size or result["online"] != online
                            or result["binding_sha256"] != request["binding_sha256"]
                            or (result.get("runtime") is not None and result["runtime"] != runtime)
                            or result["status"] not in ("pass", "oom", "headroom")):
                        raise ValueError("Invalid or foreign pilot result")
                    results.append(result)
                    files.append({"path": str(output), "sha256": sha256(output)})
            selected = reporting.select_pilot(results, bound["settings"]["pilot_microbatches"])
            atomic_json(root / "pilot.json", {"binding_sha256": request["binding_sha256"],
                        "selected": selected, "result_files": files})
        if not ensure_evaluation(root, bound, None, selected, runner):
            return
        forecast = reporting.estimate(root, selected, bound["settings"])
        atomic_json(root / "estimate.json", forecast)
        atomic_json(root / "status.json", {"status": "pilot_ready",
                                         "microbatch": selected["microbatch"]})
        print(json.dumps(forecast, indent=2), flush=True)


def run(root, bound, settings_path):
    from sftlib.sequence import ChildStop
    selected = saved_pilot(root)
    # Full measured forecast is printed before ANY grid cell is started.
    forecast = reporting.estimate(root, selected, bound["settings"])
    atomic_json(root / "estimate.json", forecast)
    print(json.dumps(forecast, indent=2), flush=True)
    with ChildStop() as stop:
        runner = Runner(root, bound["inputs"]["sft_work"], settings_path, stop)
        if not runner.call("runtime", "shared"):
            return
        if read_json(root / "runtime.json") != selected["runtime"]:
            raise ValueError("GPU/software differs from the once-only pilot; use a new grid directory")
        if not ensure_evaluation(root, bound, None, selected, runner):
            return
        for item in cells():
            if stop.requested:
                break
            if not check_completed_training(root, bound, item, selected):
                if not train_with_recovery(root, item, selected, runner):
                    break
                if not check_completed_training(root, bound, item, selected):
                    break
            if not ensure_evaluation(root, bound, item, selected, runner):
                break
            reporting.summary(root)
        result = reporting.summary(root)
        atomic_json(root / "status.json", {"status": "complete" if result["complete_cells"] == 25 else "paused",
                                         "complete_cells": result["complete_cells"]})
