"""Fastest-update SFT partition search and checkpoint-only CUDA-OOM recovery."""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import sys
import time
import uuid

from . import assets, checkpoints, data
from .core import ROOT, atomic_json, check_teacher_work, code_signature, read_json, sha256
from .resume import PARTITION, config_compatible, effective_batch
from .console import message, run_logged

OOM_EXIT_CODE = 86
PILOT_UPDATES = 3
RESERVED_FRACTION_LIMIT = .9
SELECTION_POLICY = "min_mean_warmed_update_seconds_v1"


def candidates(value):
    total = effective_batch(value)
    if total is None:
        raise ValueError("Invalid SFT effective batch")
    return [size for size in range(total, 0, -1) if total % size == 0]


def partition(value, microbatch):
    if type(microbatch) is not int or microbatch not in candidates(value):
        raise ValueError("SFT microbatch must divide the unchanged effective batch")
    return {**value, "microbatch_size": microbatch,
            "gradient_accumulation_steps": effective_batch(value) // microbatch}


def context(work, value, runtime):
    return {"recipe": {k: v for k, v in value.items() if k not in PARTITION},
            "selection_policy": SELECTION_POLICY,
            "effective_batch": effective_batch(value), "runtime": runtime,
            "assets_sha256": sha256(work / "assets/manifest.json"),
            "prepared_sha256": sha256(work / "prepared/manifest.json"),
            "candidates": candidates(value), "pilot_updates": PILOT_UPDATES,
            "warmup_updates": 1, "timed_updates": PILOT_UPDATES - 1,
            "reserved_fraction_limit": RESERVED_FRACTION_LIMIT}


def latest(work, value):
    run_dir = work / "runs" / value["run_name"]
    if not (run_dir / "run.json").exists():
        return None
    run = read_json(run_dir / "run.json")
    if not config_compatible(run["config"], value):
        raise ValueError("Existing SFT run has a different effective batch/recipe")
    checkpoint = checkpoints.resolve(run_dir)
    checkpoints.verify(checkpoint, run["run_id"])
    return checkpoint


def source_anchor(work, value):
    checkpoint = latest(work, value)
    if checkpoint:
        return {"kind": "checkpoint", "path": str(checkpoint),
                "complete_sha256": sha256(checkpoint / "COMPLETE.json"),
                "step": read_json(checkpoint / "COMPLETE.json")["step"]}
    return {"kind": "pinned_base", "path": str(work / "assets/model"), "step": 0,
            "assets_sha256": sha256(work / "assets/manifest.json")}


def verify_trial(result, expected, size):
    if (result.get("context") != expected or type(result.get("microbatch")) is not int
            or result["microbatch"] != size
            or result.get("status") not in ("pass", "oom", "headroom")):
        raise ValueError("Foreign/invalid SFT fit-pilot result")
    if result["status"] != "oom":
        peak = result.get("peak_reserved_bytes")
        if type(peak) is not int or peak < 1:
            raise ValueError("Missing SFT pilot memory measurement")
        limit = expected["runtime"]["total_memory"] * RESERVED_FRACTION_LIMIT
        if (result["status"] == "pass") != (peak <= limit):
            raise ValueError("SFT pilot headroom status differs from measured peak")
        if result.get("save_reload_verified") is not True:
            raise ValueError("SFT fit pilot did not verify a full checkpoint reload")
        records = result.get("updates", [])
        if len(records) != PILOT_UPDATES or not isinstance(result.get("resumed_update"), dict):
            raise ValueError("SFT pilot lacks warmup/timed/reloaded update measurements")
        for row in [*records, result["resumed_update"]]:
            if (not math.isfinite(row["update_seconds"]) or row["update_seconds"] <= 0
                    or not math.isfinite(row["loss"])
                    or row["peak_reserved_bytes"] > peak):
                raise ValueError("Invalid SFT pilot update timing/loss/memory")
    return result["status"] == "pass"


def warmed_seconds(result):
    """Use only synchronized training updates after warmup, never IO/validation."""
    rows = result["updates"][1:]
    return sum(row["update_seconds"] for row in rows) / len(rows)


def ranking(trials):
    return sorted((trial for trial in trials if trial["status"] == "pass"),
                  key=lambda trial: (warmed_seconds(trial), -trial["microbatch"]))


def saved_pilot(work, value, runtime):
    result = read_json(work / "throughput_pilot.json")
    expected = context(work, value, runtime)
    if result["context"] != expected:
        raise ValueError("SFT pilot hardware/software/inputs/recipe changed; use a new work directory")
    sizes = candidates(value)
    selected = result["microbatch"]
    if type(selected) is not int or selected not in sizes:
        raise ValueError("Invalid saved SFT microbatch")
    files = result["result_files"]
    if len(files) != len(sizes):
        raise ValueError("SFT throughput pilot lacks measurements for every partition")
    trials = []
    for size, entry in zip(sizes, files):
        path = Path(entry["path"])
        if sha256(path) != entry["sha256"]:
            raise ValueError("SFT pilot candidate hash changed")
        trial = read_json(path)
        verify_trial(trial, expected, size)
        if trial.get("anchor") != result["anchor"]:
            raise ValueError("SFT pilot candidates mixed different initialization checkpoints")
        trials.append(trial)
    ordered = ranking(trials)
    if not ordered or ordered[0]["microbatch"] != selected:
        raise ValueError("SFT pilot did not select the fastest passing partition")
    if (result.get("mean_warmed_update_seconds") != warmed_seconds(ordered[0])
            or result.get("accumulation") != effective_batch(value) // selected):
        raise ValueError("SFT throughput selection timing/accumulation disagrees with candidate evidence")
    return result


def ensure_pilot(work, value, runtime, launch):
    if (work / "throughput_pilot.json").exists():
        result = saved_pilot(work, value, runtime)
        atomic_json(work / "pilot_reuse" / f"{uuid.uuid4().hex}.json", {
            "code": code_signature(), "unix_time": time.time(),
            "pilot_code": result.get("code", {}),
            "note": "Source hashes are provenance only; the once-only throughput pilot is reused."})
        if result.get("code") != code_signature():
            message("WARNING: SFT throughput pilot predates current source; reused without a code-hash gate.")
        return result
    expected = context(work, value, runtime)
    request_path = work / "throughput_pilot_request.json"
    if request_path.exists():
        request = read_json(request_path)
        if request["context"] != expected:
            raise ValueError("Incomplete SFT pilot request changed; use a new work directory")
    else:
        prior = {name: sha256(work / name)
                 for name in ("microbatch_pilot.json", "training_batching.json")
                 if (work / name).is_file()}
        request = {"context": expected, "anchor": source_anchor(work, value), "code": code_signature(),
                   "prior_largest_fit_artifacts": prior}
        atomic_json(request_path, request)
    files, trials = [], []
    for size in candidates(value):
        output = work / "throughput_pilots" / f"mb-{size:05d}.json"
        if not output.exists() and not launch("_sft_pilot", "--microbatch", size, "--output", output):
            return None
        trial = read_json(output)
        verify_trial(trial, expected, size)
        if trial.get("anchor") != request["anchor"]:
            raise ValueError("SFT pilot source checkpoint changed between candidates")
        if trial.get("code") != code_signature():
            message("WARNING: reusing an SFT pilot candidate measured under different source code.")
        files.append({"path": str(output), "sha256": sha256(output)})
        trials.append(trial)
        if trial["status"] == "pass":
            message(f"{value['role'].title()} SFT {size} × {effective_batch(value) // size}: "
                    f"{warmed_seconds(trial):.3f}s/warmed update.")
    ordered = ranking(trials)
    if not ordered:
        raise RuntimeError(f"{value['role']} SFT has no passing partition, including microbatch 1, with memory headroom")
    best = ordered[0]
    size = best["microbatch"]
    result = {"context": expected, "microbatch": size,
              "accumulation": effective_batch(value) // size, "anchor": request["anchor"],
              "result_files": files, "code": code_signature(),
              "mean_warmed_update_seconds": warmed_seconds(best),
              "selection": "minimum mean warmed update_seconds across all passing partitions; exact ties prefer larger microbatch"}
    atomic_json(work / "throughput_pilot.json", result)
    saved_pilot(work, value, runtime)
    message(f"{value['role'].title()} SFT pilot selected {size} × {result['accumulation']} "
            f"at {warmed_seconds(best):.3f}s/warmed update.")
    return result


def active_microbatch(work, value, runtime):
    initial = saved_pilot(work, value, runtime)["microbatch"]
    path = work / "throughput_batching.json"
    if not path.exists():
        return initial
    result = read_json(path)
    if (result.get("pilot_sha256") != sha256(work / "throughput_pilot.json")
            or type(result.get("microbatch")) is not int
            or result["microbatch"] not in candidates(value)
            or result["microbatch"] > initial):
        raise ValueError("Foreign/invalid SFT recovery batching state")
    previous = initial
    sizes = candidates(value)
    history = result.get("history")
    if not isinstance(history, list) or not history:
        raise ValueError("Missing SFT recovery history")
    for event in history:
        smaller = [size for size in sizes if size < previous]
        if (not smaller or event.get("from_microbatch") != previous
                or event.get("to_microbatch") != smaller[0]
                or event.get("failure", {}).get("status") != "cuda_oom"
                or type(event.get("step")) is not int or not 0 <= event["step"] <= value["max_steps"]):
            raise ValueError("Invalid SFT CUDA-OOM reduction chain")
        previous = smaller[0]
    if previous != result["microbatch"]:
        raise ValueError("SFT recovery selection differs from its recorded OOM chain")
    return result["microbatch"]


class CudaOOM(RuntimeError):
    def __init__(self, report):
        self.report = report
        super().__init__(f"CUDA OOM for {report['role']} at microbatch {report['microbatch']}")


def launch_worker(root, config_dir, value, stop, operation, *extra):
    work = root / value["role"]
    attempt = uuid.uuid4().hex
    log = work / "logs" / f"{operation}-{attempt}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    report = work / "attempts" / f"{attempt}.json"
    argv = [sys.executable, "-u", str(ROOT / "prepare_models.py"),
            "--work-dir", str(root), "--config-dir", str(config_dir),
            operation, "--role", value["role"], *map(str, extra)]
    if operation == "_train":
        argv += ["--attempt-result", str(report)]
    started, code = time.monotonic(), None
    try:
        code = run_logged(argv, log, stop, cwd=ROOT,
                          env={"PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
        if stop.requested:
            return False
        if code:
            if operation == "_train" and code == OOM_EXIT_CODE and report.is_file():
                failure = read_json(report)
                batch = int(extra[extra.index("--microbatch") + 1])
                if (failure.get("status") == "cuda_oom" and failure.get("role") == value["role"]
                        and failure.get("operation") == "train" and failure.get("microbatch") == batch):
                    raise CudaOOM({**failure, "attempt_result": str(report), "log": str(log)})
            raise RuntimeError(f"SFT {operation} failed, not automatically retried; inspect {log}")
        return True
    finally:
        with (work / "stages.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"operation": operation, "seconds": time.monotonic() - started,
                                     "return_code": code, "log": str(log)}) + "\n")


def run(root, config_dir, value, *, resume=None, session_updates=None, pilot_only=False):
    """Model-free supervisor; each probe/retry runs in a new isolated process."""
    from .sequence import ChildStop
    work = root / value["role"]
    check_teacher_work(work, value)
    assets.verify(work, value)
    data.verify(work, value)
    run_exists = (work / "runs" / value["run_name"] / "run.json").exists()
    if not pilot_only and run_exists and not resume:
        raise FileExistsError("SFT run exists; use --resume latest (or the paired run command)")
    if not pilot_only and resume and not run_exists:
        raise ValueError("No SFT run to resume")
    checkpoint = latest(work, value) if run_exists else None
    start = read_json(checkpoint / "COMPLETE.json")["step"] if checkpoint else 0
    if start == value["max_steps"]:
        message(f"{value['role'].title()} SFT already complete; no pilot or training launched.")
        return
    if session_updates is not None and (type(session_updates) is not int or session_updates < 1):
        raise ValueError("Session update cap must be positive")
    target = min(value["max_steps"], start + session_updates) if session_updates else value["max_steps"]
    with ChildStop() as stop:
        launch = lambda operation, *extra: launch_worker(root, config_dir, value, stop, operation, *extra)
        runtime_path = work / "pilot_runtime.json"
        if not launch("_sft_runtime", "--output", runtime_path):
            return
        runtime = read_json(runtime_path)
        if checkpoint is not None:
            original = read_json(checkpoint.parent.parent / "run.json")
            if original["runtime"] != runtime["environment"]:
                raise ValueError("SFT resume GPU/software differs; no fit probes or training launched")
            if (original["signatures"]["assets"] != sha256(work / "assets/manifest.json")
                    or original["signatures"]["prepared"] != sha256(work / "prepared/manifest.json")):
                raise ValueError("SFT resume inputs differ from checkpoint provenance")
        if ensure_pilot(work, value, runtime, launch) is None or pilot_only:
            return
        while not stop.requested:
            checkpoint = latest(work, value)
            step = read_json(checkpoint / "COMPLETE.json")["step"] if checkpoint else 0
            if step >= target:
                return
            size = active_microbatch(work, value, runtime)
            extra = ["--microbatch", size]
            if target > step:
                extra += ["--session-updates", target - step]
            if checkpoint:
                extra += ["--resume", "latest"]
            try:
                launch("_train", *extra)
                return  # A graceful zero-exit pause must not be restarted automatically.
            except CudaOOM as error:
                run_dir = work / "runs" / value["run_name"]
                if (not (run_dir / "run.json").exists() or not checkpoints.completed(run_dir)):
                    raise RuntimeError("SFT OOM before a complete restart; no partial-gradient or base restart") from error
                checkpoint = latest(work, value)
                if checkpoint is None:
                    raise RuntimeError("SFT OOM before a complete restart; no partial-gradient or base restart")
                smaller = [candidate for candidate in candidates(value) if candidate < size]
                if not smaller:
                    raise RuntimeError("SFT CUDA OOM at microbatch 1; no further reduction")
                path = work / "throughput_batching.json"
                history = read_json(path)["history"] if path.exists() else []
                event = {"from_microbatch": size, "to_microbatch": smaller[0],
                         "checkpoint": str(checkpoint),
                         "step": read_json(checkpoint / "COMPLETE.json")["step"],
                         "unix_time": time.time(), "failure": error.report}
                atomic_json(path, {"pilot_sha256": sha256(work / "throughput_pilot.json"),
                                   "microbatch": smaller[0], "history": [*history, event]})
                message(f"SFT OOM: restart {value['role']} from update {event['step']:,} "
                        f"at {smaller[0]} × {effective_batch(value) // smaller[0]}.")


def runtime_worker(value, output):
    import torch
    from .training import runtime
    device, info = runtime(value)
    atomic_json(output, {"environment": info, "total_memory": torch.cuda.get_device_properties(device).total_memory})


def pilot_worker(work, value, microbatch, output):
    """Discarded full-shape updates + complete optimizer/RNG save/reload/update."""
    import torch
    from . import modeling, training
    started = time.monotonic()
    request = read_json(work / "throughput_pilot_request.json")
    value = partition(value, microbatch)
    if output.exists():
        raise FileExistsError("Never overwrite a completed SFT pilot candidate")
    pilot_run = None
    try:
        assets.verify(work, value)
        data.verify(work, value)
        device, info = training.runtime(value)
        runtime = {"environment": info, "total_memory": torch.cuda.get_device_properties(device).total_memory}
        if context(work, value, runtime) != request["context"]:
            raise ValueError("SFT pilot runtime/recipe/inputs changed")
        anchor = request["anchor"]
        source = Path(anchor["path"])
        if anchor["kind"] == "checkpoint":
            run = read_json(source.parent.parent / "run.json")
            if (source.parent.parent != work / "runs" / value["run_name"]
                    or not config_compatible(run["config"], value)
                    or sha256(source / "COMPLETE.json") != anchor["complete_sha256"]):
                raise ValueError("SFT pilot source is not this role's captured complete checkpoint")
            checkpoints.verify(source, run["run_id"])
        elif anchor["kind"] != "pinned_base" or source != work / "assets/model":
            raise ValueError("Invalid SFT pilot initialization")
        training.seed(value["seed"])
        tokenizer, model = modeling.load(source, device, True, value)
        opt = training.optimizer(model, value)
        if anchor["kind"] == "checkpoint":
            state = torch.load(source / "training_state.pt", map_location="cpu", weights_only=True)
            if (state["step"] != anchor["step"] or state["cursor"] != state["step"] * effective_batch(value)
                    or not config_compatible(state.get("execution_config", run["config"]), value)):
                raise ValueError("SFT pilot checkpoint state/cursor/recipe disagrees with its captured source")
            opt.load_state_dict(state["optimizer"])
            training.restore_rng(state["rng"])
            del state
        datasets = data.load(work)
        train = datasets["train"]
        if len(train) != data.COUNTS["train"] or len(datasets["validation"]) != data.COUNTS["validation"]:
            raise ValueError("SFT pilot requires the full prepared pool")
        # Synthetic worst-shape stress, not a scan for a naturally long row
        # (which might still have a short target).
        row = train[0]
        def stretch(tokens, length):
            content = tokens[:-1] or [tokenizer.eos_token_id]
            return (content * math.ceil((length - 1) / len(content)))[:length - 1] + [tokenizer.eos_token_id]
        stress = {**row, "input_ids": stretch(row["input_ids"], value["max_source_tokens"]),
                  "labels": stretch(row["labels"], value["max_target_tokens"])}
        rows = [stress] * effective_batch(value)
        for group in opt.param_groups:
            group["lr"] = value["learning_rate"]
        initialization_seconds = time.monotonic() - started
        torch.cuda.reset_peak_memory_stats(device)
        records = []
        from .console import progress
        with progress(PILOT_UPDATES, desc=f"{value['role']} SFT pilot mb={microbatch}", unit="update") as bar:
            for _ in range(PILOT_UPDATES):
                records.append(training.update(model, tokenizer, opt, rows, value, device))
                bar.update()
        pilot_run = output.parent / f"{output.stem}-restart-{uuid.uuid4().hex}"
        pilot_run.mkdir(parents=True)
        atomic_json(pilot_run / "run.json", {"run_id": uuid.uuid4().hex, "config": value,
                    "signatures": {"context": request["context"]}, "kind": "discarded-sft-fit-pilot"})
        before = time.monotonic()
        saved = checkpoints.save(pilot_run, model, tokenizer, {
            "step": PILOT_UPDATES, "cursor": PILOT_UPDATES * effective_batch(value),
            "optimizer": opt.state_dict(), "rng": training.rng_state(),
            "execution_config": value, "synthetic_pilot": True}, "discarded-sft-pilot")
        checkpoints.verify(saved, read_json(pilot_run / "run.json")["run_id"])
        checkpoint_seconds = time.monotonic() - before
        peak = torch.cuda.max_memory_reserved(device)
        del model, opt
        torch.cuda.empty_cache()
        tokenizer, model = modeling.load(saved, device, True, value)
        opt = training.optimizer(model, value)
        state = torch.load(saved / "training_state.pt", map_location="cpu", weights_only=True)
        opt.load_state_dict(state["optimizer"])
        training.restore_rng(state["rng"])
        del state
        resumed = training.update(model, tokenizer, opt, rows, value, device)
        before = time.monotonic()
        validation = training.validation_loss(model, tokenizer, datasets["validation"], value, device)
        validation_seconds = time.monotonic() - before
        peak = max(peak, torch.cuda.max_memory_reserved(device))
        result = {"status": "pass" if peak <= runtime["total_memory"] * RESERVED_FRACTION_LIMIT else "headroom",
                  "peak_reserved_bytes": peak, "updates": records, "resumed_update": resumed,
                  "validation": validation, "save_reload_verified": True,
                  "checkpoint_seconds": checkpoint_seconds, "validation_seconds": validation_seconds,
                  "initialization_seconds": initialization_seconds,
                  "pilot_checkpoint": str(saved),
                  "pilot_checkpoint_complete": read_json(saved / "COMPLETE.json"),
                  "pilot_checkpoint_discarded": True,
                  "mean_warmed_update_seconds": sum(r["update_seconds"] for r in records[1:]) / (PILOT_UPDATES - 1)}
    except torch.cuda.OutOfMemoryError as error:
        result = {"status": "oom", "error": str(error)}
    finally:
        # Delete only this invocation's newly created, uniquely named scratch
        # restart, never a real run or another invocation's incomplete output.
        if pilot_run is not None:
            if pilot_run.is_symlink() or pilot_run.parent.resolve() != output.parent.resolve():
                raise ValueError("Unsafe SFT pilot scratch cleanup target")
            shutil.rmtree(pilot_run)
    atomic_json(output, {**result, "microbatch": microbatch, "context": request["context"],
                         "anchor": request["anchor"], "code": code_signature(),
                         "wall_seconds": time.monotonic() - started})
