"""Independent-cell training, strict full-state restart, joint-fit pilots."""
from __future__ import annotations

import heapq
import json
from pathlib import Path
import random
import time
import uuid

from .policy import PROTOCOL, TRAIN_N, chunks, lr_factor, update_spec
from .shared import Cursor, atomic_json, checkpoints, read_json, sha256
from sftlib.console import duration, message, progress


def append(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"unix_time": time.time(), **record}, allow_nan=False) + "\n")
        stream.flush()


def new_state():
    return {"step": 0, "cursor": 0, "points": 0, "on_policy_examples": 0,
            "off_policy_examples": 0, "training_seconds": 0.0,
            "source_tokens": 0, "target_tokens": 0}


def check_state(state, item):
    steps = state["step"]
    if type(steps) is not int or not 0 <= steps <= item["updates"]:
        raise ValueError("Invalid saved update count")
    groups = min(item["groups"], steps * item["groups_per_update"])
    expected = {"cursor": groups * item["group_size"], "points": groups * item["group_cost"],
                "on_policy_examples": groups * item["group_online"],
                "off_policy_examples": groups * (item["group_size"] - item["group_online"])}
    if any(state[k] != v for k, v in expected.items()):
        raise ValueError("Checkpoint point/mixture/data-cursor accounting differs")


def train(root, bound, item, selected, session_updates=None, microbatch=None):
    started = time.monotonic()
    import torch
    from sftlib.training import StopRequest, seed, rng_state, restore_rng
    from . import backend
    settings = bound["settings"]
    microbatch = selected["microbatch"] if microbatch is None else microbatch
    if type(microbatch) is not int or not 1 <= microbatch <= selected["microbatch"] or 32 % microbatch:
        raise ValueError("Microbatch must be a positive divisor of 32 no larger than the original pilot selection")
    run_dir = root / "cells" / item["id"]
    run_file = run_dir / "run.json"
    value = {"cell": item, "microbatch": selected["microbatch"], "settings": settings,
             "protocol": PROTOCOL}
    signatures = {"binding_sha256": sha256(root / "binding.json"),
                  "pilot_sha256": sha256(root / "pilot.json")}
    checkpoint, marker = None, None
    if run_file.exists():
        run = read_json(run_file)
        if run["config"] != value or run["signatures"] != signatures:
            raise ValueError("Cell configuration/pilot/input identity changed")
        # Never silently restart an initialized cell that has no complete restart.
        checkpoint = checkpoints.resolve(run_dir)
        marker = checkpoints.verify(checkpoint, run["run_id"])
    else:
        if run_dir.exists():
            raise FileExistsError(f"Inspect nonempty/unowned cell directory: {run_dir}")
    message(f"Loading {item['id']} · microbatch {microbatch}"
            + (" · resuming latest checkpoint..." if checkpoint else " · fresh common SFT initialization..."))
    device, runtime, tokenizer, student, teacher, datasets = backend.load_pair(
        bound, settings, checkpoint)
    if runtime != selected["runtime"]:
        raise ValueError("Hardware/software changed since pilot; use a new grid directory")
    if run_file.exists():
        if runtime != run["runtime"]:
            raise ValueError("Resume runtime changed")
    else:
        run_dir.mkdir(parents=True)
        run = {"run_id": uuid.uuid4().hex, "config": value, "signatures": signatures,
               "runtime": runtime, "initial_student": bound["inputs"]["paths"]["student_checkpoint"]}
        atomic_json(run_file, run)
    seed(settings["seed"])
    opt = backend.optimizer(student, settings["learning_rate"])
    state = new_state()
    cursor = Cursor(len(datasets["train"]), settings["data_seed"])
    branch_rng = random.Random(settings["seed"] + 2)
    if checkpoint:
        loaded = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
        execution = loaded.get("execution_config", value)
        saved_batch = execution.get("microbatch")
        if ({k: v for k, v in execution.items() if k != "microbatch"}
                != {k: v for k, v in value.items() if k != "microbatch"}
                or type(saved_batch) is not int or not 1 <= saved_batch <= selected["microbatch"]
                or 32 % saved_batch):
            raise ValueError("Checkpoint execution recipe/effective-batch partition is invalid")
        opt.load_state_dict(loaded["optimizer"])
        state = loaded["accounting"]
        check_state(state, item)
        if marker["step"] != state["step"] or loaded["step"] != state["step"]:
            raise ValueError("Restart marker/state step mismatch")
        cursor.position = state["cursor"]
        branch_rng.setstate(loaded["branch_rng"])
        restore_rng(loaded["rng"])
        del loaded
    session = run_dir / "sessions" / (uuid.uuid4().hex + ".jsonl")
    append(session, {"event": "initialized", "seconds": time.monotonic() - started,
                     "step": state["step"], "from_checkpoint": str(checkpoint) if checkpoint else None,
                     "microbatch": microbatch, "effective_batch_size": 32})
    last_saved = state["step"] if checkpoint else -1
    start_step = state["step"]
    bar = progress(item["points"], initial=state["points"], desc=f"GKD {item['id']}", unit="pt")

    def save(reason):
        nonlocal last_saved
        if last_saved == state["step"]:
            return
        before = time.monotonic()
        checkpoints.save(run_dir, student, tokenizer,
                         {"step": state["step"], "accounting": dict(state),
                          "optimizer": opt.state_dict(), "rng": rng_state(),
                           "branch_rng": branch_rng.getstate(),
                           "execution_config": {**value, "microbatch": microbatch}}, reason)
        last_saved = state["step"]
        append(session, {"event": "checkpoint", "step": last_saved,
                         "seconds": time.monotonic() - before})
        message(f"{item['id']} checkpoint saved: update {last_saved:,}, "
                f"{state['points']:,}/{item['points']:,} points.")
        atomic_json(run_dir / "training_summary.json",
                    {"status": "paused", "cell": item, **state,
                     "checkpoint": str(checkpoints.resolve(run_dir)),
                     "initial_student": run["initial_student"], "signatures": signatures,
                     "microbatch": microbatch})

    try:
        torch.cuda.reset_peak_memory_stats(device)
        with StopRequest() as stop:
            # A failed first update can resume from an owned zero-update restart.
            save("initial state")
            while state["step"] < item["updates"] and not stop.requested:
                spec = update_spec(item, state["step"])
                # Include host row selection/mixture planning in synchronized update time.
                torch.cuda.synchronize(device)
                before = time.monotonic()
                rows = [datasets["train"][i] for i in cursor.take(spec["examples"])]
                flags = [True] * spec["on"] + [False] * spec["off"]
                branch_rng.shuffle(flags)
                batches = chunks(rows, flags, microbatch, branch_rng)
                rate = settings["learning_rate"] * lr_factor(state["points"], spec["points"], item["points"])
                for group in opt.param_groups:
                    group["lr"] = rate
                result = backend.update(student, teacher, tokenizer, opt, batches, len(rows), device)
                torch.cuda.synchronize(device)
                result["training_seconds"] = time.monotonic() - before
                state["step"] += 1
                state["cursor"] = cursor.position
                state["points"] += spec["points"]
                state["on_policy_examples"] += spec["on"]
                state["off_policy_examples"] += spec["off"]
                for key in ("training_seconds", "source_tokens", "target_tokens"):
                    state[key] += result[key]
                check_state(state, item)
                append(session, {"event": "update", "step": state["step"], "points": state["points"],
                                 "on": spec["on"], "off": spec["off"], "learning_rate": rate,
                                 "microbatch": microbatch, "microbatches": len(batches), **result})
                bar.set_postfix(loss=f"{result['loss']:.4f}", lr=f"{rate:.2g}",
                                update=f"{state['step']}/{item['updates']}", mb=microbatch,
                                VRAM=f"{result.get('peak_allocated_bytes', 0) / 1024**3:.1f}GiB",
                                train=duration(state["training_seconds"]), refresh=False)
                bar.update(spec["points"])
                stopping = stop.requested or (session_updates is not None
                                              and state["step"] - start_step >= session_updates)
                if stopping or state["step"] == item["updates"] or state["step"] % settings["save_every_updates"] == 0:
                    save("pause/final" if stopping or state["step"] == item["updates"] else "periodic")
                if stopping:
                    break
            save("session end")
        status = "complete" if state["step"] == item["updates"] else "paused"
        atomic_json(run_dir / "training_summary.json",
                    {"status": status, "cell": item, **state,
                     "checkpoint": str(checkpoints.resolve(run_dir)),
                     "initial_student": run["initial_student"], "signatures": signatures,
                     "microbatch": microbatch})
    except BaseException as error:
        atomic_json(run_dir / "failure.json", {"error": str(error), "last_saved_step": last_saved,
                    "recovery": "Resume only this cell's latest COMPLETE checkpoint; preserve incomplete files."})
        raise
    finally:
        bar.close()
        append(session, {"event": "session_end", "seconds": time.monotonic() - started})


def pilot_worker(root, bound, microbatch, online, output):
    """Fresh process per candidate/branch, forced max-length online memory stress."""
    started = time.monotonic()
    import torch
    from sftlib.training import seed, rng_state, restore_rng
    from . import backend
    settings = bound["settings"]
    try:
        device, runtime, tokenizer, student, teacher, datasets = backend.load_pair(bound, settings)
        seed(settings["seed"])
        dataset = datasets["train"]
        if len(dataset) != TRAIN_N:
            raise ValueError("Pilot must hold the full prepared training pool")
        # A bounded top-96 heap, not a duplicate copy of the full data.
        indices = heapq.nlargest(96, range(len(dataset)),
                                key=lambda i: (len(dataset[i]["input_ids"]), len(dataset[i]["labels"])))
        rows = [dataset[i] for i in indices[:32]]
        batches = [(online, rows[i:i + microbatch]) for i in range(0, 32, microbatch)]
        opt = backend.optimizer(student, settings["learning_rate"])
        torch.cuda.reset_peak_memory_stats(device)
        records = []
        with progress(settings["pilot_updates"], desc=f"Pilot mb={microbatch} {'online' if online else 'offline'}",
                      unit="update") as bar:
            for _ in range(settings["pilot_updates"]):
                record = backend.update(student, teacher, tokenizer, opt, batches, 32, device, force_cap=True)
                records.append(record)
                bar.set_postfix(loss=f"{record['loss']:.4f}",
                                VRAM=f"{record['peak_reserved_bytes'] / 1024**3:.1f}GiB", refresh=False)
                bar.update()
        # Checkpoint save and real reload/update are part of fit, not just forward/backward.
        pilot_run = output.parent / (output.stem + "-restart")
        pilot_run.mkdir(parents=True)
        run = {"run_id": uuid.uuid4().hex, "config": {"microbatch": microbatch, "online": online},
               "signatures": {"binding": sha256(root / "binding.json")}}
        atomic_json(pilot_run / "run.json", run)
        message(f"Pilot mb={microbatch} {'online' if online else 'offline'}: checking checkpoint save/reload...")
        before = time.monotonic()
        checkpoint = checkpoints.save(pilot_run, student, tokenizer,
                                     {"step": settings["pilot_updates"], "optimizer": opt.state_dict(),
                                      "rng": rng_state()}, "joint-fit-pilot")
        save_seconds = time.monotonic() - before
        checkpoints.verify(checkpoint, run["run_id"])
        # Release both models before reloading the saved student plus frozen teacher.
        del opt, student, teacher, datasets, dataset, batches
        torch.cuda.empty_cache()
        device, runtime_after, tokenizer, student, teacher, datasets = backend.load_pair(bound, settings, checkpoint)
        if runtime_after != runtime:
            raise ValueError("Pilot runtime changed during reload")
        opt = backend.optimizer(student, settings["learning_rate"])
        loaded = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
        opt.load_state_dict(loaded["optimizer"])
        restore_rng(loaded["rng"])
        del loaded
        batches = [(online, rows[i:i + microbatch]) for i in range(0, 32, microbatch)]
        resumed = backend.update(student, teacher, tokenizer, opt, batches, 32, device, force_cap=True)
        peak = max(row["peak_reserved_bytes"] for row in records + [resumed])
        status = "pass" if peak <= runtime["total_memory"] * settings["pilot_reserved_fraction_limit"] else "headroom"
        result = {"status": status, "microbatch": microbatch, "online": online, "runtime": runtime,
                  "updates": records, "resumed_update": resumed,
                  "mean_update_seconds": sum(x["training_seconds"] for x in records[1:]) / (len(records) - 1),
                  "checkpoint_seconds": save_seconds, "peak_reserved_bytes": peak,
                  "wall_seconds": time.monotonic() - started,
                  "stress": "1024/64 fixed scoring; online minimum 63 / maximum 64 new tokens"}
    except torch.cuda.OutOfMemoryError as error:
        result = {"status": "oom", "microbatch": microbatch, "online": online,
                  "error": str(error), "wall_seconds": time.monotonic() - started}
    result["binding_sha256"] = sha256(root / "binding.json")
    atomic_json(output, result)
    message(f"Pilot mb={microbatch} {'online' if online else 'offline'}: {result['status'].upper()}"
            + (f", {result['mean_update_seconds']:.2f}s/update" if "mean_update_seconds" in result else ""))
