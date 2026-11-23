"""Queue one 100K-update common SFT preparation, then repeat matched-budget GKD."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import time

from gkdproto.common import ROOT, atomic_json, environment, read_json, sha256
from run_gkd_study import PREPARED, Study, keep_awake, load_metrics, prepare_data, recipe

SFT_STEPS = 100000
SFT_BATCHES = (2, 4, 8, 16, 32)
CELLS = {"off_policy", "on_policy", "mixed"}


def preparation_recipe(microbatch, pilot=False):
    config = recipe("common_sft", ROOT / "assets/student", microbatch, algorithm="sft")
    config.update(max_steps=3 if pilot else SFT_STEPS,
                  schedule_horizon=3 if pilot else SFT_STEPS,
                  warmup_steps=0 if pilot else 2000,
                  save_every=3 if pilot else 500,
                  compute_budget_seconds=None, budget_schedule=False, max_wall_minutes=None)
    return config


def select_sft_batch(results):
    passing = [item for item in results if item["status"] == "complete"]
    if not passing:
        raise RuntimeError("No student-only SFT batch passed; no 100K run will start")
    return min(passing, key=lambda item: item["warmed_update_seconds"])["microbatch"]


def validate_completed_study(directory):
    if read_json(directory / "status.json")["phase"] != "complete":
        raise ValueError("Preceding study did not complete successfully; follow-up will not start")
    results = read_json(directory / "results.json")
    if set(results) != CELLS:
        raise ValueError("Preceding study is missing a distillation approach")
    for name, result in results.items():
        status = result["training"]
        if status["status"] != "complete" or status["consumed_seconds"] < status["compute_budget_seconds"]:
            raise ValueError(f"Preceding {name} did not finish its training budget")
        record = result["evaluation"]
        metrics = record["metrics"]
        if metrics["status"] != "complete" or metrics["n"] != 11332 or metrics["split"] != "validation":
            raise ValueError(f"Preceding {name} has incomplete evaluation")
        if sha256(Path(record["path"]) / "predictions.jsonl") != metrics["predictions_sha256"]:
            raise ValueError(f"Preceding {name} prediction hash changed")


def _wait_windows(pid, lock_started):
    """Block on the exact predecessor process handle; no periodic polling."""
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
    kernel.GetProcessTimes.restype = wintypes.BOOL
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x00100000 | 0x1000, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        created, exited, system, user = (wintypes.FILETIME() for _ in range(4))
        if not kernel.GetProcessTimes(handle, *[ctypes.byref(item) for item in (created, exited, system, user)]):
            raise ctypes.WinError(ctypes.get_last_error())
        born = ((created.dwHighDateTime << 32) | created.dwLowDateTime) / 10_000_000 - 11644473600
        if born > lock_started + 2:
            raise ValueError("Predecessor PID was reused by a different process")
        if kernel.WaitForSingleObject(handle, 0xFFFFFFFF) != 0:
            raise ctypes.WinError(ctypes.get_last_error())
    finally:
        kernel.CloseHandle(handle)


def await_study(directory, wait):
    status = directory / "status.json"
    phase = read_json(status).get("phase") if status.exists() else None
    if phase == "complete":
        validate_completed_study(directory)
        return
    if phase in ("failed", "interrupted"):
        raise ValueError("Preceding study failed/interrupted; finish it before starting this follow-up")
    if not wait:
        raise ValueError("Preceding study is unfinished; add --wait to queue without using the GPU")
    lock = read_json(directory / "RUNNING.lock")
    try:
        if os.name == "nt":
            _wait_windows(lock["pid"], lock["started_unix"])
        elif hasattr(os, "pidfd_open"):
            import select
            descriptor = os.pidfd_open(lock["pid"])
            try:
                select.select([descriptor], [], [])
            finally:
                os.close(descriptor)
        else:
            raise RuntimeError("Blocking predecessor wait is unsupported here; rerun after it completes")
    except OSError:
        # The predecessor can finish in the small interval before opening its handle.
        if not status.exists() or read_json(status).get("phase") != "complete":
            raise
    validate_completed_study(directory)


def code_signature():
    paths = [*sorted((ROOT / "gkdproto").glob("*.py")), ROOT / "run_gkd_study.py", Path(__file__).resolve()]
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def verify_student_assets():
    manifest_path = ROOT / "assets/manifest.json"
    manifest = read_json(manifest_path)
    student = manifest["assets"]["student"]
    if (manifest["status"] != "complete" or not student["complete"]
            or student["repo_id"] != "google/t5-small-lm-adapt"
            or student["revision"] != "4ff52753a4ddd906de166c573db99280719b22f9"):
        raise ValueError("Expected the pinned complete T5-77M LM-adapt base")
    for item in student["files"]:
        name = item["path"]
        if Path(name).name != name:
            raise ValueError("Invalid student asset path")
        path = ROOT / "assets/student" / name
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Student asset changed: {name}")
    return {"manifest_sha256": sha256(manifest_path), "student": student}


def run_preparation(study):
    saved = study.directory / "sft_pilot_results.json"
    if saved.exists():
        results = read_json(saved)["results"]
    else:
        results = []
        for size in SFT_BATCHES:
            name = f"pilot_sft_b{size}"
            config = preparation_recipe(size, pilot=True)
            config["run_name"] = name
            code, log = study.train(name, config, allow_failure=True)
            if code:
                message = log.read_text(encoding="utf-8") if log else read_json(study.directory / name / "status.json").get("error", "")
                if "out of memory" not in message.lower():
                    raise RuntimeError(f"SFT pilot failed for a non-OOM reason: {log}")
                results.append({"microbatch": size, "status": "oom"})
            else:
                metrics = load_metrics(study.directory / name / "metrics.jsonl")
                results.append({"microbatch": size, "status": "complete",
                                "warmed_update_seconds": statistics.mean(row["update_seconds"] for row in metrics[1:]),
                                "peak_allocated_bytes": metrics[-1]["peak_allocated_bytes"],
                                "peak_reserved_bytes": metrics[-1]["peak_reserved_bytes"]})
            atomic_json(study.directory / "sft_pilot_progress.json", {"results": results})
        selected = select_sft_batch(results)
        atomic_json(saved, {"results": results, "selected_microbatch": selected,
                            "note": "Student-only pilots discarded; preparation starts afresh from the pinned base."})
    selected = select_sft_batch(results)
    rate = next(row["warmed_update_seconds"] for row in results if row["microbatch"] == selected and row["status"] == "complete")
    study.status("sft_pilots_complete", selected_microbatch=selected,
                 preliminary_preparation_hours=SFT_STEPS * rate / 3600)
    study.train("common_sft", preparation_recipe(selected))
    checkpoint = study.checkpoint("common_sft")
    status = read_json(study.directory / "common_sft/status.json")
    config = read_json(checkpoint / "metadata.json")["config"]
    if status["step"] != SFT_STEPS or config["compute_budget_seconds"] is not None or config["max_wall_minutes"] is not None:
        raise ValueError("Common SFT did not complete the declared time-unbounded 100K update preparation")
    frozen = {"path": str(checkpoint), "step": SFT_STEPS,
              "files": read_json(checkpoint / "COMPLETE.json")["files"],
              "note": "Paper-aligned student SFT update count; other SFT hyperparameters are reconstructed."}
    destination = study.directory / "frozen_initialization.json"
    if destination.exists() and read_json(destination) != frozen:
        raise ValueError("Frozen common SFT initialization changed")
    atomic_json(destination, frozen)
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--after-study", type=Path, default=ROOT / "runs/gkd_jordiclive_b60_mb1_seed17")
    parser.add_argument("--preparation-dir", type=Path, default=ROOT / "runs/gkd_jordiclive_sft100k_seed17")
    parser.add_argument("--distillation-dir", type=Path, default=ROOT / "runs/gkd_jordiclive_sft100k_b60_seed17")
    parser.add_argument("--budget-minutes", type=float, default=60)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--wait", action="store_true", help="Block on predecessor process exit without starting a concurrent GPU job")
    args = parser.parse_args()
    if not math.isfinite(args.budget_minutes) or args.budget_minutes <= 0:
        parser.error("Distillation budget must be finite and positive")
    directory = args.preparation_dir.resolve()
    previous = args.after_study.resolve()
    repeated = args.distillation_dir.resolve()
    if len({directory, previous, repeated}) != 3:
        parser.error("Previous study, preparation and repeated study must be separate directories")
    spec = {"after_study": str(previous), "preparation_dir": str(directory), "distillation_dir": str(repeated),
            "student_sft_updates": SFT_STEPS, "sft_time_limit": None, "sft_effective_batch": 32,
            "sft_pilot_microbatches": list(SFT_BATCHES), "sft_save_every_updates": 500,
            "distillation_budget_minutes": args.budget_minutes, "distillation_microbatch": 1,
            "seed": 17, "environment": environment(), "code_sha256": code_signature(),
            "evidence_boundary": "100K student SFT steps are specified in GKD A.2; the full teacher/student SFT recipe is not recovered."}
    print(json.dumps(spec, indent=2), flush=True)
    if not args.run:
        print("Plan only. --run --wait queues preparation after successful predecessor completion.")
        return
    directory.mkdir(parents=True, exist_ok=True)
    plan = directory / "followup.json"
    if plan.exists():
        if read_json(plan) != spec:
            raise ValueError("Follow-up definition/code/software changed; do not silently migrate the queued run")
    elif any(directory.iterdir()):
        raise FileExistsError("Refusing an unrecognized nonempty preparation directory")
    else:
        atomic_json(plan, spec)
    lock = directory / "RUNNING.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"pid": os.getpid(), "started_unix": time.time()}))
    study = Study(directory, spec)
    try:
        with keep_awake():
            study.status("queued", after_study=str(previous))
            await_study(previous, args.wait)
            if code_signature() != spec["code_sha256"] or environment() != spec["environment"]:
                raise ValueError("Code/software changed while queued; inspect before running")
            prepare_data()
            provenance = verify_student_assets()
            provenance_path = directory / "student_provenance.json"
            if provenance_path.exists() and read_json(provenance_path) != provenance:
                raise ValueError("Pinned student provenance changed")
            atomic_json(provenance_path, provenance)
            checkpoint = run_preparation(study)
            study.command("repeat_distillation", ["run_gkd_study.py", "--run", "--study-dir", repeated,
                          "--student-init", checkpoint, "--microbatch", "1", "--budget-minutes", str(args.budget_minutes)])
            validate_completed_study(repeated)
            study.status("complete", common_initialization=str(checkpoint), repeated_study=str(repeated))
    except BaseException as error:
        study.status("interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
