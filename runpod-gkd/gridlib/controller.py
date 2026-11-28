"""Serial Pod orchestration; no model imports in plans/estimates/reports."""
from __future__ import annotations

from pathlib import Path
import sys
import time
import uuid

from . import evaluation, reporting
from .policy import ROOT
from .shared import atomic_json, read_json, sha256
from .engine import append
from .recovery import (CudaOOM, OOM_EXIT_CODE, evaluation_microbatch, evaluation_path,
                        evaluation_initial, lower, train_with_recovery)
from . import evalpilot
from . import schedule
from sftlib.console import message, output as display, run_logged


def command(root, sft_work, settings_path, operation, *extra, end_points=schedule.DEFAULT_END_POINTS):
    return [sys.executable, "-u", str(ROOT / "distill_grid.py"), "--work-dir", str(root),
            "--sft-work", str(sft_work), "--settings", str(settings_path),
            "--end-points", str(end_points), operation, *map(str, extra)]


class Runner:
    def __init__(self, root, sft_work, settings_path, stop, end_points=schedule.DEFAULT_END_POINTS):
        self.root, self.sft_work, self.settings_path, self.stop = root, sft_work, settings_path, stop
        self.end_points = end_points

    def call(self, operation, cell_id, *extra):
        if self.stop.requested:
            return False
        logs = self.root / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        log = logs / f"{cell_id}-{operation}-{time.time_ns()}.log"
        started = time.monotonic()
        code = None
        attempt = self.root / "attempts" / (uuid.uuid4().hex + ".json")
        stage_context = ({"until_points": int(extra[extra.index("--until-points") + 1])}
                         if "--until-points" in extra else {})
        append(self.root / "stage_starts.jsonl",
               {"cell": cell_id, "operation": operation, "log": str(log), "unix_time": time.time(),
                **stage_context})
        message(f"{cell_id} · {operation} | log: {log}")
        try:
            worker_extra = [*extra]
            if operation in ("train", "evaluate"):
                worker_extra += ["--attempt-result", str(attempt)]
            code = run_logged(
                command(self.root, self.sft_work, self.settings_path, "_" + operation, *worker_extra,
                        end_points=self.end_points),
                log, self.stop, cwd=ROOT,
                env={"PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
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
                    "return_code": code, "log": str(log), **stage_context})
            if cell_id != "shared":
                reporting.summary(self.root, self.end_points)


def saved_pilot(root):
    schedule.validate_root(root)
    pilot = read_json(root / "pilot.json")
    if pilot["binding_sha256"] != sha256(root / "binding.json"):
        raise ValueError("Saved pilot belongs to different inputs/settings/code")
    for entry in pilot["result_files"]:
        if sha256(Path(entry["path"])) != entry["sha256"]:
            raise ValueError("Saved pilot result changed")
    return pilot["selected"]


def ensure_evaluation(root, bound, item, selected, runner):
    schedule.require_design(bound)
    while True:
        try:
            return evaluation_attempt(root, bound, item, selected, runner)
        except CudaOOM as error:
            if item is not None:
                evaluation.source(root, bound, item)
            initial = evaluation_initial(root, bound, selected)
            lower(root, evaluation_path(root, item), initial,
                  evaluation_microbatch(root, bound, item, selected),
                  cell=item["id"] if item else "baseline", failure=error.report)


def evaluation_attempt(root, bound, item, selected, runner):
    schedule.require_design(bound)
    owner = schedule.evaluation_directory(root, item)
    owner.mkdir(parents=True, exist_ok=True)
    pointer = owner / "latest_evaluation.json"
    expected = evaluation.identity(root, bound, item, selected)
    if pointer.exists():
        output = Path(read_json(pointer)["path"])
        evaluation.verify(output, expected)
        return True
    output = owner / "evaluation"
    # A worker may publish COMPLETE then die before latest_evaluation.json.
    # Verify ALL completed candidates before repairing: foreign/corrupt retries
    # fail closed even if another valid completion exists. Partial dirs are kept.
    completed = [path for path in [output, *sorted(owner.glob("evaluation-retry-*"))]
                 if (path / "COMPLETE.json").exists()]
    for candidate in completed:
        evaluation.verify(candidate, expected)
    if completed:
        atomic_json(pointer, {"path": str(completed[0])})
        return True
    if output.exists():
        output = owner / ("evaluation-retry-" + uuid.uuid4().hex)
    return runner.call("evaluate", item["id"] if item else "shared",
                       "--cell", item["id"] if item else "baseline", "--output", output,
                       "--microbatch", evaluation_microbatch(root, bound, item, selected))


def training_config(bound, item, selected):
    return {"cell": item, "microbatch": selected["microbatch"], "settings": bound["settings"],
            "protocol": bound["protocol"]}


def check_completed_training(root, bound, item, selected):
    from .milestones import verify
    schedule.require_design(bound)
    if not (schedule.evaluation_directory(root, item) / "snapshot.json").exists():
        return False
    verify(root, bound, item, selected)
    return True


def pilot(root, bound, settings_path, end_points=schedule.DEFAULT_END_POINTS):
    schedule.require_design(bound)
    from sftlib.sequence import ChildStop
    with ChildStop() as stop:
        runner = Runner(root, bound["inputs"]["sft_work"], settings_path, stop, end_points)
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
        if not evalpilot.ensure(root, bound, selected, runner):
            return
        if not ensure_evaluation(root, bound, None, selected, runner):
            return
        forecast = reporting.estimate(root, selected, bound["settings"], end_points)
        atomic_json(root / "estimate.json", forecast)
        atomic_json(root / "status.json", {"status": "pilot_ready",
                                          "microbatch": selected["microbatch"],
                                          "evaluation_batch_size": evaluation_initial(root, bound, selected)})
        display(forecast, "Grid runtime estimate")


def run(root, bound, settings_path, end_points=schedule.DEFAULT_END_POINTS):
    schedule.require_design(bound)
    from sftlib.sequence import ChildStop
    selected = saved_pilot(root)
    # Full measured forecast is printed before ANY grid cell is started.
    forecast = reporting.estimate(root, selected, bound["settings"], end_points)
    atomic_json(root / "estimate.json", forecast)
    display(forecast, "Grid runtime estimate")
    with ChildStop() as stop:
        runner = Runner(root, bound["inputs"]["sft_work"], settings_path, stop, end_points)
        if not runner.call("runtime", "shared"):
            return
        if read_json(root / "runtime.json") != selected["runtime"]:
            raise ValueError("GPU/software differs from the once-only pilot; use a new grid directory")
        if not evalpilot.ensure(root, bound, selected, runner):
            return
        if not ensure_evaluation(root, bound, None, selected, runner):
            return
        return run_trajectories(root, bound, selected, runner, stop, end_points)


def run_trajectories(root, bound, selected, runner, stop, end_points=schedule.DEFAULT_END_POINTS):
    """Resume each mixture's single full-state trajectory between evaluated prefixes."""
    schedule.require_design(bound)
    scope = schedule.campaign(end_points)
    from .milestones import latest_restart
    for trajectory in schedule.trajectories():
        if stop.requested:
            break
        _, marker = latest_restart(root, bound, trajectory, selected)
        step = marker["step"] if marker else 0
        target = schedule.milestone(trajectory["on_policy_percent"], end_points)
        if step > target["updates"]:
            message(f"{trajectory['id']} already beyond {end_points:,} nominal points; moving to next mixture.")
            continue
        for nominal in schedule.evaluation_points(end_points):
            if stop.requested:
                break
            item = schedule.milestone(trajectory["on_policy_percent"], nominal)
            message(f"Trajectory {trajectory['id']} · milestone {nominal:,} nominal points "
                    f"({item['points']:,} actual; update {item['updates']:,})")
            if not check_completed_training(root, bound, item, selected):
                if step > item["updates"]:
                    message(f"{item['id']} has no retained snapshot and is behind the current checkpoint; skipping.")
                    continue
                if not train_with_recovery(root, trajectory, selected, runner, until_points=nominal):
                    break
                if not check_completed_training(root, bound, item, selected):
                    break
                step = item["updates"]
            if not ensure_evaluation(root, bound, item, selected, runner):
                break
            reporting.summary(root, end_points)
        else:
            continue
        break
    result = reporting.summary(root, end_points)
    atomic_json(root / "status.json", {
        "status": "complete" if result["campaign_complete"] else "paused",
        "campaign": scope, "skipped_trajectories": result["skipped_trajectories"],
        "training_design": schedule.DESIGN, "complete_cells": result["complete_cells"],
        "complete_milestones": result["complete_cells"],
        "complete_trajectories": result["complete_trajectories"],
    })
