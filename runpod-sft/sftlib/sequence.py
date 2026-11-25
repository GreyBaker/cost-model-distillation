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
from .resume import config_compatible, record_migration
from .console import message, run_logged

ORDER = ("student", "teacher")


def command(root, config_dir, role, operation, *extra):
    return [sys.executable, "-u", str(ROOT / "prepare_models.py"), "--work-dir", str(root),
            "--config-dir", str(config_dir), operation, "--role", role, *map(str, extra)]


def training_complete(work, value):
    run_dir = work / "runs" / value["run_name"]
    run = read_json(run_dir / "run.json")
    checkpoint = checkpoints.resolve(run_dir)
    marker = checkpoints.verify(checkpoint, run["run_id"])
    return config_compatible(run["config"], value) and marker["step"] == value["max_steps"]


def verified_role(root, config_dir, role):
    """Read-only verification of a restored SFT role; never load/train a model."""
    value = config(role, config_dir)
    work = (root / role).resolve()
    run_dir = work / "runs" / value["run_name"]
    if not (run_dir / "run.json").is_file():
        raise ValueError(f"Restore the complete pre-SFTed {role} role under SFT_WORK/{role} first")
    run = read_json(run_dir / "run.json")
    if not config_compatible(run["config"], value):
        raise ValueError(f"Restored {role} recipe differs; use its compatible {role} config")
    assets.verify(work, value)
    data.verify(work, value)
    if (run["signatures"]["assets"] != sha256(work / "assets/manifest.json")
            or run["signatures"]["prepared"] != sha256(work / "prepared/manifest.json")):
        raise ValueError(f"Restored {role} assets/prepared data differ from its training provenance")
    checkpoint = checkpoints.resolve(run_dir)
    marker = checkpoints.verify(checkpoint, run["run_id"])
    if marker["step"] != value["max_steps"]:
        raise ValueError(f"Reuse requires the complete {value['max_steps']:,}-update {role}, not partial SFT")
    export_dir = inference.export_path(work, value)
    exported = inference.verify_export(export_dir)
    provenance = read_json(export_dir / "provenance.json")
    if (exported.get("role") != role or exported.get("step") != value["max_steps"]
            or provenance.get("partial") is not False or provenance["source_run"] != run
            or provenance["source_checkpoint_complete_sha256"] != sha256(checkpoint / "COMPLETE.json")):
        raise ValueError(f"Restored {role} export does not match its final FP32 checkpoint")
    # A copied role can retain an old absolute latest_evaluation pointer. Discover
    # complete local evaluations without modifying that pointer or any manifest.
    candidates = [work / "evaluation"]
    pointer = work / "latest_evaluation.json"
    if pointer.is_file():
        pointed = Path(read_json(pointer)["path"]).resolve()
        if pointed.is_relative_to(work):
            candidates.append(pointed)
    candidates.extend(sorted(work.glob("evaluation-retry-*")))
    seen = set()
    for output in candidates:
        output = output.resolve()
        if output in seen or not output.is_relative_to(work):
            continue
        seen.add(output)
        metrics_path = output / "metrics.json"
        if not metrics_path.is_file():
            continue
        metrics = read_json(metrics_path)
        if metrics.get("status") != "complete" or metrics.get("n") != data.COUNTS["validation"]:
            continue
        metrics = inference.verify_evaluation(work, value, output, export_dir, data.COUNTS["validation"])
        return {"fp32_checkpoint": str(checkpoint), "bf16_export": str(export_dir),
                "export_complete_sha256": sha256(export_dir / "COMPLETE.json"),
                "evaluation": str(output), "metrics": metrics}
    raise ValueError(f"Restore a complete 11,332-example {role} evaluation before reuse")


def verified_student(root, config_dir):
    return verified_role(root, config_dir, "student")


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


def run(root, config_dir, reuse_student=False, reuse_sft=False):
    if reuse_student and reuse_sft:
        raise ValueError("Choose student-only reuse or complete-pair reuse, not both")
    values = {role: config(role, config_dir) for role in ORDER}
    if any(values["student"][name] != values["teacher"][name]
           for name in ("prefix", "max_source_tokens", "max_target_tokens")):
        raise ValueError("Both roles must share prompt/source/target protocol for later GKD")
    # Fail before creating outputs, downloading/training teacher, or launching GPU
    # workers. Source hashes and historical GPU/software are not reuse blockers.
    reuse_roles = ORDER if reuse_sft else ("student",) if reuse_student else ()
    reused = {role: verified_role(root, config_dir, role) for role in reuse_roles}
    definition = {"order": list(ORDER), "configs": values, "assets_spec_sha256": sha256(SPEC),
                  "code": code_signature(), "automatic_time_limit": None}
    root.mkdir(parents=True, exist_ok=True)
    with RunLock(root / ".sequence.lock"):
        path = root / "sequence.json"
        if path.exists():
            saved = read_json(path)
            if (set(saved) != set(definition)
                    or any(saved[k] != definition[k] for k in definition if k not in ("configs", "code"))
                    or set(saved["configs"]) != set(values)
                    or any(not config_compatible(saved["configs"][role], values[role]) for role in ORDER)):
                raise ValueError("Sequence effective batch/recipe changed; use a new work directory")
            record_migration(root / "migrations", saved, definition)
        if not path.exists():
            atomic_json(path, definition)
        # Fail before a lengthy student run if teacher inputs were not prepared.
        for role in ORDER:
            if role in reused:
                continue
            assets.verify(root / role, values[role])
            data.verify(root / role, values[role])
        logs = root / "logs"
        logs.mkdir(exist_ok=True)
        results = dict(reused)
        if reused:
            atomic_json(root / "reused_sft.json", {
                "status": "verified", "roles": reused, "training_launched_for_reused_roles": False,
                "source_checkpoint_complete_sha256": {
                    role: sha256(Path(result["fp32_checkpoint"]) / "COMPLETE.json")
                    for role, result in reused.items()},
                "note": "Restored role folders verified read-only; no train/export/evaluation worker for reused roles."})

        def status(phase, **extra):
            atomic_json(root / "status.json", {"phase": phase, "wall_time_unix": time.time(), **extra})
            details = " | ".join(f"{key.replace('_', ' ')}: {value}" for key, value in extra.items())
            message(f"SFT · {phase.replace('_', ' ')}" + (f" | {details}" if details else ""))

        with ChildStop() as stop:
            def child(role, operation, *extra):
                if stop.requested:
                    return False
                log = logs / f"{role}-{operation}-{time.time_ns()}.log"
                status(f"{role}_{operation}", log=str(log))
                code = run_logged(command(root, config_dir, role, operation, *extra), log, stop,
                                  cwd=ROOT, env={"PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
                if code and not stop.requested:
                    raise RuntimeError(f"{role} {operation} failed; inspect {log}")
                return not stop.requested

            try:
                for role in ORDER:
                    if stop.requested:
                        status("paused")
                        return
                    if role in reused:
                        status(f"{role}_reused", checkpoint=reused[role]["fp32_checkpoint"])
                        continue
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
