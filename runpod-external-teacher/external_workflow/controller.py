"""Serial subprocess execution; parent never loads a model or initializes CUDA."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import sys
import time
import uuid

from .common import (OOM_EXIT, REPO, ROOT, RunLock, append, atomic_json, check_space, code,
                     digest, operational_paths, pair_root, read_json, runtime, sha256)
from sftlib.console import message, run_logged
from sftlib.core import CONFIGS, configure_local_storage
from sftlib.sequence import ChildStop, verified_student
from .reporting import display_quality


class ExternalOOM(RuntimeError):
    pass


def worker_command(work, stage, python=None, batch=None, attempt=None):
    argv = [str(python or sys.executable), "-u", "-B", str(ROOT / "external_teacher.py"),
            "--work-dir", str(work), "_worker", "--stage", stage]
    if batch is not None:
        argv += ["--batch-size", str(batch)]
    if attempt is not None:
        argv += ["--attempt-result", str(attempt)]
    return argv


class Runner:
    def __init__(self, work, stop):
        self.work, self.stop = work, stop

    def call(self, argv, name, *, oom_stage=None, batch=None, attempt=None):
        if self.stop.requested:
            return False
        log = self.work / "logs" / f"{name}-{uuid.uuid4().hex}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        message(f"External workflow: {name} | {log}")
        started, result = time.monotonic(), None
        try:
            result = run_logged(argv, log, self.stop, cwd=ROOT,
                                env={"PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
            if self.stop.requested:
                return False
            if result:
                if result == OOM_EXIT and oom_stage and attempt is not None and attempt.is_file():
                    report = read_json(attempt)
                    if (report.get("status") == "cuda_oom" and report.get("stage") == oom_stage
                            and report.get("batch_size") == batch
                            and report.get("definition_sha256") == digest(read_json(self.work / "run.json")["definition"])):
                        raise ExternalOOM(str(report))
                raise RuntimeError(f"{name} failed; inspect {log}. Non-OOM errors are not retried.")
            return True
        finally:
            append(self.work / "stages.jsonl", {"stage": name, "seconds": time.monotonic() - started,
                                              "return_code": result, "log": str(log), "command": argv})

    def worker(self, stage, python=None, batch=None):
        attempt = self.work / "attempts" / f"{uuid.uuid4().hex}.json"
        return self.call(worker_command(self.work, stage, python, batch, attempt), stage,
                         oom_stage=stage if stage.endswith("-validation") else None,
                         batch=batch, attempt=attempt)


def definition(args):
    from gridlib.policy import settings
    settings(args.settings)  # Reject a bad grid recipe before lengthy imports/evaluations.
    restored_identity = None
    if args.student == "restored":
        if args.restored_root is None or args.reference_python is None or not args.reference_python.is_file():
            raise ValueError("Restored mode needs a complete copied SFT root and the isolated reference Python")
        restored = verified_student(args.restored_root.resolve(), CONFIGS)
        restored_identity = sha256(Path(restored["fp32_checkpoint"]) / "COMPLETE.json")
    return {"schema": "external-run-v1", "student": args.student,
            "restored_root": str(args.restored_root.resolve()) if args.student == "restored" else None,
            "reference_python": str(args.reference_python.expanduser().absolute()) if args.student == "restored" else None,
            "restored_checkpoint_complete_sha256": restored_identity,
            "models_lock_sha256": sha256(ROOT / "models.lock.json"),
            "grid_settings_sha256": sha256(args.settings.resolve()),
            "grid_work": str(args.grid_work.resolve())}


def storage_projection(value):
    """Conservative remaining-space estimate from pinned sizes, not a fit claim."""
    from gridlib import schedule
    milestone_count = len(schedule.active_milestones())
    locked = read_json(ROOT / "models.lock.json")
    specs = [locked["models"]["teacher"], locked["evaluation_data"]]
    if value["student"] == "public":
        specs.append(locked["models"]["public_student"])
        student_bytes = next(item["bytes"] for item in specs[-1]["files"] if item["path"] == "model.safetensors")
    else:
        from sftlib.core import config
        from sftlib.checkpoints import resolve
        run = Path(value["restored_root"]) / "student/runs" / config("student")["run_name"]
        student_bytes = sum(path.stat().st_size for path in resolve(run).glob("*.safetensors"))
        if student_bytes <= 0:
            raise ValueError("Restored student has no safe FP32 initialization weights")
    teacher_bytes = next(item["bytes"] for item in specs[0]["files"] if item["path"] == "model.safetensors")
    parts = {
        "locked_sources": sum(item["bytes"] for spec in specs for item in spec["files"]),
        "canonical_models": student_bytes + (teacher_bytes + 1) // 2,
        # Conservative optimizer upper allowance: another full model per state,
        # though factored Adafactor typically needs substantially less.
        "grid_5_trajectories_two_rolling_restarts_each": 5 * 2 * 2 * student_bytes,
        f"grid_{milestone_count}_permanent_full_state_milestones": milestone_count * 2 * student_bytes,
        "grid_pilot_and_transient_checkpoints": 10 * student_bytes,
        "prepared_tables_and_data_caches_allowance": 10 * 1024 ** 3,
        "reports_and_miscellaneous_reserve": 5 * 1024 ** 3,
    }
    return {"components_bytes": parts, "projected_remaining_bytes": sum(parts.values()),
            "basis": "Pinned raw sizes; BF16 teacher at half FP32; conservative checkpoint/data allowances. "
                     "Already installed environments and copied restored inputs are excluded from remaining space. "
                     "Not a guarantee: stage-specific free-space guards still apply."}


def run(args):
    disk, work, grid = operational_paths(args.pod_disk, args.work_dir, args.grid_work,
        args.restored_root if args.student == "restored" else None,
        args.reference_python if args.student == "restored" else None)
    from gridlib.schedule import validate_root
    validate_root(grid, allow_new=True)
    os.environ["POD_DISK"] = str(disk)
    from plan import validate
    validate(read_json(ROOT / "scheme.json"), read_json(ROOT / "models.lock.json"))
    configure_local_storage(work)
    runtime(work)  # No GPU call; fail before downloads under the wrong environment.
    value = definition(args)
    with RunLock(work / ".controller.lock"), ChildStop() as stop:
        path = work / "run.json"
        if path.exists():
            original = read_json(path)
            # Old thresholds are archived metadata, not a resume/admission policy.
            # Grid destination is operational, not part of immutable pair evidence.
            # Preserve the original definition/hashes even when selecting a new grid.
            effective = lambda definition: {key: item for key, item in definition.items()
                                            if key not in ("thresholds", "grid_work")}
            if effective(original["definition"]) != effective(value):
                raise ValueError("Mode, source or settings changed; preserve this external work root")
            value = original["definition"]  # Preserve hashes of all completed legacy artifacts.
            if original.get("source_fingerprints") != code():
                message("WARNING: source changed; valid artifacts/resumes are reused without a code-hash gate.")
        else:
            if any(p.name != ".controller.lock" for p in work.iterdir()):
                raise ValueError("Inspect nonempty external work root without run.json; no files overwritten")
            storage = storage_projection(value)
            storage["required_initial_free_bytes"] = max(
                int(args.minimum_free_gib * 1024 ** 3), storage["projected_remaining_bytes"])
            check_space(work, storage["required_initial_free_bytes"])
            atomic_json(path, {"definition": value, "run_id": uuid.uuid4().hex, "source_fingerprints": code()})
            atomic_json(work / "storage_plan.json", storage)
        pair_root(work).mkdir(exist_ok=True)
        for target, source in ((work / "models.lock.json", ROOT / "models.lock.json"),
                               (pair_root(work) / "models.lock.json", ROOT / "models.lock.json"),
                               (work / "grid_settings.json", args.settings.resolve())):
            if target.exists():
                if sha256(target) != sha256(source):
                    raise ValueError(f"Frozen configuration changed: {target}")
            else:
                shutil.copyfile(source, target)
        append(work / "source_provenance.jsonl", {"source_fingerprints": code()})
        append(work / "experiment_invocations.jsonl", {
            "grid_work": str(grid), "pair_definition_sha256": digest(value),
            "authorization": "explicit single-command run; no manual approval pauses"})
        runner = Runner(work, stop)

        def status(phase, **extra):
            atomic_json(work / "status.json", {"phase": phase, **extra})

        try:
            stages = ["native-runtime"]
            if args.student == "restored":
                stages += ["reference-runtime"]
            stages += ["acquire", "corpus"]
            if args.student == "restored":
                stages += ["legacy-tokenizer"]
            stages += ["tokenizers", "prepared"]
            if args.student == "restored":
                stages += ["legacy-probe"]
            stages += ["student", "teacher"]
            for stage in stages:
                status(stage)
                python = value["reference_python"] if stage.startswith("legacy-") or stage == "reference-runtime" else None
                if not runner.worker(stage, python):
                    status("paused", stage=stage)
                    return
            for role, initial in (("student", 16), ("teacher", 4)):
                state_path = work / f"{role}_validation_batch.json"
                if state_path.exists():
                    record = read_json(state_path)
                    if record.get("definition_sha256") != digest(value):
                        raise ValueError("Foreign validation recovery batch")
                    batch = record["batch_size"]
                    if type(batch) is not int or not 1 <= batch <= initial:
                        raise ValueError("Invalid saved validation batch")
                else:
                    batch = initial
                while not stop.requested:
                    status(role + "_validation", batch_size=batch)
                    try:
                        if not runner.worker(role + "-validation", batch=batch):
                            status("paused")
                            return
                        break
                    except ExternalOOM as error:
                        if batch == 1:
                            raise RuntimeError("External full validation OOM at batch1; stopping") from error
                        batch = max(1, batch // 2)
                        atomic_json(state_path, {"definition_sha256": digest(value), "batch_size": batch,
                                                 "last_cuda_oom": str(error)})
                if stop.requested:
                    status("paused")
                    return
            if not runner.worker("quality"):
                status("paused")
                return
            report = read_json(pair_root(work) / "reports/quality/report.json")
            display_quality(work, report)
            from .pair import approve
            approve(work)
            prefix = [sys.executable, "-u", "-B", str(REPO / "runpod-gkd/distill_grid.py"),
                      "--sft-work", str(pair_root(work)), "--work-dir", str(grid),
                      "--settings", str(work / "grid_settings.json")]
            status("grid_running", grid=str(grid))
            if not runner.call([*prefix, "run"], "grid-run"):
                status("paused")
                return
            if read_json(grid / "status.json").get("status") != "complete":
                status("paused")
                return
            status("complete", grid=str(grid))
            message("External-teacher grid complete. Download outputs before terminating the disposable Pod.")
        except BaseException as error:
            status("paused" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
            raise


def development_smoke(args):
    """Explicit opt-in development diagnostic; never update standard run status."""
    work = args.work_dir.expanduser().resolve()
    saved = read_json(work / "run.json")["definition"]
    disk, work, _ = operational_paths(args.pod_disk, work, saved["grid_work"])
    os.environ["POD_DISK"] = str(disk)
    configure_local_storage(work)
    runtime(work)
    status_path = work / "development_smoke_status.json"
    with RunLock(work / ".controller.lock"), ChildStop() as stop:
        runner = Runner(work, stop)
        try:
            if not runner.worker("smoke"):
                atomic_json(status_path, {"phase": "paused", "development_only": True})
                return
            report = pair_root(work) / "reports/smoke/report.json"
            if read_json(report).get("passed") is not True:
                raise ValueError("Development smoke/restart checks did not pass")
            atomic_json(status_path, {"phase": "complete", "development_only": True, "report": str(report)})
            message(f"Development smoke/restart checks complete: {report}. "
                    "This report is not required or included in new pair approvals.")
        except BaseException as error:
            atomic_json(status_path, {"phase": "paused" if isinstance(error, KeyboardInterrupt) else "failed",
                                      "development_only": True, "error": str(error)})
            raise
