"""Serial subprocess controller; holds no model or CUDA allocations itself."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from . import assets, checkpoints, data, inference
from .core import ROOT, SPEC, RunLock, atomic_json, code_signature, config, read_json, sha256

ORDER = ("student", "teacher")


def command(root, config_dir, role, operation, *extra):
    return [sys.executable, "-u", str(ROOT / "prepare_models.py"), "--work-dir", str(root),
            "--config-dir", str(config_dir), operation, "--role", role, *map(str, extra)]


def training_complete(work, value):
    run_dir = work / "runs" / value["run_name"]
    run = read_json(run_dir / "run.json")
    checkpoint = checkpoints.resolve(run_dir)
    marker = checkpoints.verify(checkpoint, run["run_id"])
    return run["config"] == value and marker["step"] == value["max_steps"]


class ChildStop:
    def __init__(self):
        self.requested, self.child, self.previous = False, None, {}

    def handle(self, number, frame):
        if self.requested:
            if self.child and self.child.poll() is None:
                self.child.kill()
            raise KeyboardInterrupt("Second signal: abort controller; use latest complete role checkpoint")
        self.requested = True
        if self.child and self.child.poll() is None:
            self.child.send_signal(number)
        print("Sequence stop requested; no next role will be started.", flush=True)

    def __enter__(self):
        for number in (signal.SIGINT, signal.SIGTERM):
            self.previous[number] = signal.signal(number, self.handle)
        return self

    def __exit__(self, *args):
        for number, previous in self.previous.items():
            signal.signal(number, previous)


def run(root, config_dir):
    values = {role: config(role, config_dir) for role in ORDER}
    if any(values["student"][name] != values["teacher"][name]
           for name in ("prefix", "max_source_tokens", "max_target_tokens")):
        raise ValueError("Both roles must share prompt/source/target protocol for later GKD")
    definition = {"order": list(ORDER), "configs": values, "assets_spec_sha256": sha256(SPEC),
                  "code": code_signature(), "automatic_time_limit": None}
    root.mkdir(parents=True, exist_ok=True)
    with RunLock(root / ".sequence.lock"):
        path = root / "sequence.json"
        if path.exists() and read_json(path) != definition:
            raise ValueError("Sequence/config/code changed; use a new work directory, not a silent migration")
        if not path.exists():
            atomic_json(path, definition)
        # Fail before a lengthy student run if teacher inputs were not prepared.
        for role in ORDER:
            assets.verify(root / role, values[role])
            data.verify(root / role, values[role])
        logs = root / "logs"
        logs.mkdir(exist_ok=True)
        results = {}

        def status(phase, **extra):
            atomic_json(root / "status.json", {"phase": phase, "wall_time_unix": time.time(), **extra})
            print(json.dumps({"phase": phase, **extra}), flush=True)

        with ChildStop() as stop:
            def child(role, operation, *extra):
                if stop.requested:
                    return False
                log = logs / f"{role}-{operation}-{time.time_ns()}.log"
                status(f"{role}_{operation}", log=str(log))
                with log.open("x", encoding="utf-8") as stream:
                    stop.child = subprocess.Popen(command(root, config_dir, role, operation, *extra),
                                                  cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                                  start_new_session=os.name != "nt",
                                                  env={**os.environ, "PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
                    if stop.requested:
                        stop.child.send_signal(signal.SIGTERM)
                    code = stop.child.wait()
                    stop.child = None
                if code and not stop.requested:
                    raise RuntimeError(f"{role} {operation} failed; inspect {log}")
                return not stop.requested

            try:
                for role in ORDER:
                    if stop.requested:
                        status("paused")
                        return
                    value = values[role]
                    work = root / role
                    run_dir = work / "runs" / value["run_name"]
                    extra = ["--resume", "latest"] if (run_dir / "run.json").exists() else ["--accept-reconstructed-recipe"]
                    if not child(role, "train", *extra):
                        status("paused", role=role)
                        return
                    if not training_complete(work, value):
                        status("paused", role=role, reason="role did not finish its update budget")
                        return  # In particular, never advance after a saved pilot/graceful pause.
                    if not child(role, "export"):
                        status("paused", role=role)
                        return
                    output = work / "evaluation"
                    recorded = work / "latest_evaluation.json"
                    if recorded.exists() and read_json(recorded).get("n") == data.COUNTS["validation"]:
                        output = Path(read_json(recorded)["path"])
                    elif output.exists():
                        metrics_path = output / "metrics.json"
                        if (not metrics_path.exists() or read_json(metrics_path).get("n") != data.COUNTS["validation"]
                                or read_json(metrics_path).get("status") != "complete"):
                            output = work / ("evaluation-retry-" + uuid.uuid4().hex)
                    if not child(role, "evaluate", "--output-dir", output, "--limit", "0"):
                        status("paused", role=role)
                        return
                    export_dir = inference.export_path(work, value)
                    inference.verify_export(export_dir)
                    metrics = inference.verify_evaluation(work, value, output, export_dir, data.COUNTS["validation"])
                    checkpoint = checkpoints.resolve(run_dir)
                    results[role] = {"fp32_checkpoint": str(checkpoint), "bf16_export": str(export_dir),
                                     "export_complete_sha256": sha256(export_dir / "COMPLETE.json"),
                                     "evaluation": str(output), "metrics": metrics}
                    atomic_json(root / "results.json", results)
                atomic_json(root / "prepared_pair.json", {"status": "complete", "roles": results,
                            "student_for_fresh_gkd_training": results["student"]["fp32_checkpoint"],
                            "teacher_for_frozen_gkd_scoring": results["teacher"]["bf16_export"],
                            "warning": "Update counts align with GKD Appendix A.2; full SFT hyperparameters are reconstructed. Keep immutable for independent downstream cells."})
                status("complete", pair=str(root / "prepared_pair.json"))
            except BaseException as error:
                status("paused" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
                raise
