from __future__ import annotations

import json
import random
import signal
import time
import uuid

from . import assets, checkpoints, data, modeling
from .core import Cursor, atomic_json, checkpoint_due, code_signature, environment, lr_for_update, read_json, sha256
from .resume import config_compatible, record_migration, signatures_compatible
from .console import duration, message, progress


def seed(value):
    import numpy as np
    import torch
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def rng_state():
    import numpy as np
    import torch
    state = np.random.get_state()
    return {"python": random.getstate(), "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all(),
            "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]]}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    value = state["numpy"]
    np.random.set_state((value[0], np.asarray(value[1], dtype=np.uint32), value[2], value[3], value[4]))


class StopRequest:
    def __init__(self):
        self.requested, self.previous = False, {}

    def handler(self, number, frame):
        if self.requested:
            raise KeyboardInterrupt("Second interrupt: abort; resume latest complete checkpoint")
        self.requested = True
        print("Stop requested: finish current optimizer update, save and pause.", flush=True)

    def __enter__(self):
        for number in (signal.SIGINT, signal.SIGTERM):
            self.previous[number] = signal.signal(number, self.handler)
        return self

    def __exit__(self, *args):
        for number, previous in self.previous.items():
            signal.signal(number, previous)


def validation_loss(model, tokenizer, dataset, value, device):
    import torch
    order = list(range(len(dataset)))
    random.Random(value["seed"] + 1).shuffle(order)
    indices = order[:value["eval_examples"]]
    total, tokens = 0.0, 0
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for start in range(0, len(indices), value["microbatch_size"]):
                rows = [dataset[index] for index in indices[start:start + value["microbatch_size"]]]
                inputs, labels = data.batch(tokenizer, rows, device)
                loss = modeling.loss_sum(model, inputs, labels)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite validation loss")
                total += float(loss.item())
                tokens += data.target_count(rows)
    finally:
        model.train(was_training)
    return {"validation_loss": total / tokens, "validation_n": len(indices), "validation_target_tokens": tokens}


def runtime(value):
    import torch
    device = modeling.cuda_device(value)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    info = environment()
    info.update(gpu=torch.cuda.get_device_name(device), cuda=torch.version.cuda,
                visible_gpus=torch.cuda.device_count(), capability=list(torch.cuda.get_device_capability(device)))
    return device, info


def optimizer(model, value):
    from transformers import Adafactor
    return Adafactor(model.parameters(), lr=value["learning_rate"], eps=(1e-30, 1e-3),
                     clip_threshold=1.0, decay_rate=-0.8, beta1=None, weight_decay=0.0,
                     scale_parameter=False, relative_step=False, warmup_init=False)


def update(model, tokenizer, opt, rows, value, device):
    """Same full-effective-batch SFT update for real training and discarded fit probes."""
    import torch
    count = value["microbatch_size"] * value["gradient_accumulation_steps"]
    if len(rows) != count:
        raise ValueError("SFT update must preserve its complete effective batch")
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    before = time.monotonic()
    denominator = data.target_count(rows)
    total_loss, source_tokens, target_tokens = 0.0, 0, 0
    for offset in range(0, count, value["microbatch_size"]):
        batch = rows[offset:offset + value["microbatch_size"]]
        inputs, labels = data.batch(tokenizer, batch, device, limits=value)
        source_tokens += int(inputs["attention_mask"].sum().item())
        target_tokens += data.target_count(batch)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = modeling.loss_sum(model, inputs, labels) / denominator
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite SFT loss; no precision fallback")
        total_loss += float(loss.detach().item())
        loss.backward()
        del loss, inputs, labels
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
    opt.step()
    opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    return {"loss": total_loss, "gradient_norm": float(norm),
            "source_tokens": source_tokens, "target_tokens": target_tokens,
            "update_seconds": time.monotonic() - before,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}


def train(work, value, resume=None, session_updates=None):
    import torch
    started = time.monotonic()
    if session_updates is not None and (type(session_updates) is not int or session_updates < 1):
        raise ValueError("Pilot cap must be a positive additional update count")
    run_dir = work / "runs" / value["run_name"]
    assets.verify(work, value)
    data.verify(work, value)
    device, runtime_info = runtime(value)
    signatures = {"assets": sha256(work / "assets/manifest.json"),
                  "prepared": sha256(work / "prepared/manifest.json"), "code": code_signature()}
    checkpoint = None
    if resume:
        run = read_json(run_dir / "run.json")
        if (not config_compatible(run["config"], value)
                or not signatures_compatible(run["signatures"], signatures) or run["runtime"] != runtime_info):
            raise ValueError("Resume refused: effective batch, recipe, assets/data, software or GPU changed")
        checkpoint = checkpoints.resolve(run_dir, resume)
        marker = checkpoints.verify(checkpoint, run["run_id"])
        record_migration(run_dir / "migrations",
                         {"config": run["config"], "signatures": run["signatures"]},
                         {"config": value, "signatures": signatures}, checkpoint)
        if marker["step"] >= value["max_steps"]:
            print("This role already completed its full update budget; no training run.")
            return
    else:
        if run_dir.exists():
            raise FileExistsError(f"Run exists; use --resume latest: {run_dir}")
        run_dir.mkdir(parents=True)
        run = {"run_id": uuid.uuid4().hex, "config": value, "signatures": signatures, "runtime": runtime_info}
        atomic_json(run_dir / "run.json", run)
    session_id = uuid.uuid4().hex
    log_dir = run_dir / "sessions"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{session_id}.jsonl"
    atomic_json(log_dir / f"{session_id}.json", {"checkpoint": str(checkpoint) if checkpoint else None,
                "session_updates": session_updates, "runtime": runtime_info,
                "execution_config": value, "execution_signatures": signatures})
    atomic_json(run_dir / "status.json", {"status": "initializing", "session_id": session_id})
    seed(value["seed"])
    message(f"Loading {value['role']} SFT model" + (" and restart state..." if checkpoint else "..."))
    tokenizer, model = modeling.load(checkpoint or work / "assets/model", device, True, value)
    opt = optimizer(model, value)
    datasets = data.load(work)
    cursor = Cursor(len(datasets["train"]), value["data_seed"])
    step, tokens_seen = 0, {"source": 0, "target": 0}
    if checkpoint:
        state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
        if not config_compatible(state.get("execution_config", run["config"]), value):
            raise ValueError("Saved execution configuration is not batch-partition compatible")
        if not signatures_compatible(state.get("execution_signatures", run["signatures"]), signatures):
            raise ValueError("Saved execution assets/data are not compatible")
        opt.load_state_dict(state["optimizer"])
        step = state["step"]
        cursor.position = state["cursor"]
        tokens_seen = state["tokens_seen"]
        if step != marker["step"] or cursor.position != step * value["microbatch_size"] * value["gradient_accumulation_steps"]:
            raise ValueError("Checkpoint step/example cursor mismatch")
        restore_rng(state["rng"])
        del state
    start_step, last_saved = step, step if checkpoint else -1
    torch.cuda.reset_peak_memory_stats(device)
    bar = progress(value["max_steps"], initial=step, desc=f"{value['role'].title()} SFT", unit="update")

    def log(record):
        record["wall_time_unix"] = time.time()
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
        if record["event"] == "update":
            bar.set_postfix(loss=f"{record['loss']:.4f}", lr=f"{record['learning_rate']:.2g}",
                            VRAM=f"{record['peak_allocated_bytes'] / 1024**3:.1f}GiB",
                            mb=value["microbatch_size"], refresh=False)
            bar.update(1)
        elif record["event"] == "checkpoint":
            message(f"{value['role'].title()} checkpoint saved at update {record['step']:,} "
                    f"({duration(record['seconds'])}).")
        elif record["event"] == "validation":
            message(f"{value['role'].title()} validation at update {record['step']:,}: "
                    f"CE {record['validation_loss']:.4f} ({duration(record['seconds'])}).")
        elif record["event"] == "initialized":
            message(f"{value['role'].title()} SFT ready at update {step:,}; "
                    f"microbatch {value['microbatch_size']} × accumulation "
                    f"{value['gradient_accumulation_steps']} (loaded in {duration(record['seconds'])}).")
        elif record["event"] == "session_end":
            message(f"{value['role'].title()} SFT session ended at {step:,}/{value['max_steps']:,} updates.")

    def status(state):
        atomic_json(run_dir / "status.json", {"status": state, "role": value["role"], "step": step,
                    "examples_seen": cursor.position, "tokens_seen": tokens_seen,
                    "latest_saved_step": last_saved, "session_id": session_id})

    def save(reason):
        nonlocal last_saved
        if last_saved != step:
            before = time.monotonic()
            checkpoints.save(run_dir, model, tokenizer, {"step": step, "cursor": cursor.position,
                              "tokens_seen": dict(tokens_seen), "optimizer": opt.state_dict(), "rng": rng_state(),
                             "execution_config": value, "execution_signatures": signatures}, reason)
            last_saved = step
            log({"event": "checkpoint", "step": step, "seconds": time.monotonic() - before, "reason": reason})

    try:
        with StopRequest() as stop:
            log({"event": "initialized", "step": step, "seconds": time.monotonic() - started})
            status("running")
            if not checkpoint and not stop.requested:
                # A full restart exists before the first update/validation can OOM.
                save("initial restart anchor")
                before = time.monotonic()
                result = validation_loss(model, tokenizer, datasets["validation"], value, device)
                log({"event": "validation", "step": 0, "seconds": time.monotonic() - before, **result})
            while step < value["max_steps"] and not stop.requested:
                learning_rate = lr_for_update(value, step)
                for group in opt.param_groups:
                    group["lr"] = learning_rate
                count = value["microbatch_size"] * value["gradient_accumulation_steps"]
                before = time.monotonic()
                rows = [datasets["train"][index] for index in cursor.take(count)]
                input_row_seconds = time.monotonic() - before
                result = update(model, tokenizer, opt, rows, value, device)
                result["update_seconds"] += input_row_seconds
                tokens_seen["source"] += result["source_tokens"]
                tokens_seen["target"] += result["target_tokens"]
                step += 1
                log({"event": "update", "step": step, "learning_rate": learning_rate,
                     "examples_seen": cursor.position, "tokens_seen": dict(tokens_seen),
                     "microbatch_size": value["microbatch_size"],
                     "gradient_accumulation_steps": value["gradient_accumulation_steps"], **result})
                stopping = stop.requested or (session_updates is not None and step - start_step >= session_updates)
                if checkpoint_due(value, step, stopping):
                    save("pause/final" if stopping or step == value["max_steps"] else "periodic")
                    status("running")
                if not stopping and step % value["eval_every_steps"] == 0:
                    before = time.monotonic()
                    result = validation_loss(model, tokenizer, datasets["validation"], value, device)
                    log({"event": "validation", "step": step, "seconds": time.monotonic() - before, **result})
                if stopping:
                    break
            save("session end")
            log({"event": "session_end", "step": step, "seconds": time.monotonic() - started})
            status("complete" if step == value["max_steps"] else "paused")
    except BaseException as error:
        atomic_json(run_dir / "status.json", {"status": "failed", "role": value["role"], "latest_saved_step": last_saved,
                    "last_completed_step": step, "session_id": session_id, "error": str(error),
                    "recovery": "Resume latest COMPLETE checkpoint only; inspect incomplete saves"})
        raise
    finally:
        bar.close()
