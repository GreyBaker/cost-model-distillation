from __future__ import annotations

import json
import random
import signal
import time
import uuid

from . import assets, checkpoints, data, modeling
from .core import (Cursor, atomic_json, checkpoint_due, code_signature, environment, lr_for_update,
                   read_json, sha256)


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
    return {"python": random.getstate(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all(),
            "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]]}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])
    values = state["numpy"]
    np.random.set_state((values[0], np.asarray(values[1], dtype=np.uint32), values[2], values[3], values[4]))


class StopRequest:
    def __init__(self):
        self.requested = False
        self.previous = {}

    def handler(self, number, frame):
        if self.requested:
            raise KeyboardInterrupt("Second interrupt: abort now; resume latest complete checkpoint")
        self.requested = True
        print("Stop requested: finish the current optimizer update, then save and exit.", flush=True)

    def __enter__(self):
        for number in (signal.SIGINT, signal.SIGTERM):
            self.previous[number] = signal.signal(number, self.handler)
        return self

    def __exit__(self, *exc):
        for number, previous in self.previous.items():
            signal.signal(number, previous)


def validation_loss(model, tokenizer, dataset, config, device):
    import torch
    order = list(range(len(dataset)))
    random.Random(config["seed"] + 1).shuffle(order)
    indices = order[:config["eval_examples"]]
    was_training = model.training
    model.eval()
    total = 0.0
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for index in indices:
                inputs, labels = data.batch(tokenizer, [dataset[index]], device)
                value = modeling.loss(model, inputs, labels)
                if not torch.isfinite(value):
                    raise FloatingPointError("Non-finite validation loss")
                total += float(value.item())
    finally:
        model.train(was_training)
    return {"validation_loss": total / len(indices), "validation_n": len(indices)}


def train(work, config, resume=None, session_updates=None):
    import torch
    from transformers import Adafactor

    started = time.monotonic()
    if session_updates is not None and session_updates < 1:
        raise ValueError("Pilot update cap must be positive")
    run_dir = work / "runs" / config["run_name"]
    previous_run = read_json(run_dir / "run.json") if resume else None
    assets.verify(work, previous_run["run_id"] if previous_run else None)
    data.verify(work, config)
    device = modeling.cuda_device(config)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    runtime = environment()
    runtime.update(gpu=torch.cuda.get_device_name(device), cuda=torch.version.cuda,
                   visible_gpus=torch.cuda.device_count(), capability=list(torch.cuda.get_device_capability(device)))
    signatures = {"assets": sha256(work / "assets" / "manifest.json"),
                  "prepared": sha256(work / "prepared" / "manifest.json"), "code": code_signature()}
    checkpoint = None
    if resume:
        run = previous_run
        if run["config"] != config or run["signatures"] != signatures or run["runtime"] != runtime:
            raise ValueError("Resume refused: config, code, assets/data, software or GPU changed")
        checkpoint = checkpoints.resolve_checkpoint(run_dir, resume)
        marker = checkpoints.verify(checkpoint, run["run_id"])
        if marker["step"] >= config["max_steps"]:
            print("Training already complete. Use export/evaluate; no new updates run.")
            return
    else:
        if run_dir.exists():
            raise FileExistsError(f"Run exists; use --resume latest: {run_dir}")
        run_dir.mkdir(parents=True)
        run = {"run_id": uuid.uuid4().hex, "config": config, "signatures": signatures, "runtime": runtime}
        atomic_json(run_dir / "run.json", run)
    session_id = uuid.uuid4().hex
    log_dir = run_dir / "sessions"
    log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"{session_id}.jsonl"
    atomic_json(log_dir / f"{session_id}.json", {"checkpoint": str(checkpoint) if checkpoint else None,
                "session_updates": session_updates, "runtime": runtime})
    atomic_json(run_dir / "status.json", {"status": "initializing", "session_id": session_id})
    seed(config["seed"])
    tokenizer, model = modeling.load(checkpoint or work / "assets" / "model", device, True, config)
    optimizer = Adafactor(model.parameters(), lr=config["learning_rate"], eps=(1e-30, 1e-3),
                         clip_threshold=1.0, decay_rate=-0.8, beta1=None, weight_decay=0.0,
                         scale_parameter=False, relative_step=False, warmup_init=False)
    datasets = data.load(work)
    cursor = Cursor(len(datasets["train"]), config["data_seed"])
    step = 0
    tokens_seen = {"source": 0, "target": 0}
    if checkpoint:
        state = torch.load(checkpoint / "training_state.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
        if step != marker["step"]:
            raise ValueError("Checkpoint marker/state step mismatch")
        cursor.position = state["cursor"]
        tokens_seen = state["tokens_seen"]
        expected = step * config["microbatch_size"] * config["gradient_accumulation_steps"]
        if cursor.position != expected:
            raise ValueError("Checkpoint step/example cursor mismatch")
        restore_rng(state["rng"])
        del state
    start_step = step
    last_saved = step if checkpoint else -1
    torch.cuda.reset_peak_memory_stats(device)

    def log(record):
        record["wall_time_unix"] = time.time()
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
        print(json.dumps(record), flush=True)

    def save(reason):
        nonlocal last_saved
        if last_saved != step:
            before = time.monotonic()
            checkpoints.save(run_dir, model, tokenizer,
                             {"step": step, "cursor": cursor.position, "tokens_seen": dict(tokens_seen),
                              "optimizer": optimizer.state_dict(), "rng": rng_state()}, reason)
            last_saved = step
            log({"event": "checkpoint", "step": step, "seconds": time.monotonic() - before, "reason": reason})

    try:
        with StopRequest() as stop:
            log({"event": "initialized", "step": step, "seconds": time.monotonic() - started})
            if not checkpoint and not stop.requested:
                before = time.monotonic()
                result = validation_loss(model, tokenizer, datasets["validation"], config, device)
                log({"event": "validation", "step": 0, "seconds": time.monotonic() - before, **result})
            while step < config["max_steps"]:
                if stop.requested:
                    break
                optimizer.zero_grad(set_to_none=True)
                learning_rate = lr_for_update(config, step)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate
                torch.cuda.synchronize(device)
                before = time.monotonic()
                total_loss = 0.0
                for _ in range(config["gradient_accumulation_steps"]):
                    rows = [datasets["train"][index] for index in cursor.take(config["microbatch_size"])]
                    inputs, labels = data.batch(tokenizer, rows, device)
                    tokens_seen["source"] += int(inputs["attention_mask"].sum().item())
                    tokens_seen["target"] += int(labels.ne(-100).sum().item())
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        value = modeling.loss(model, inputs, labels)
                    if not torch.isfinite(value):
                        raise FloatingPointError("Non-finite SFT loss; no precision fallback")
                    total_loss += value.detach().item() / config["gradient_accumulation_steps"]
                    (value / config["gradient_accumulation_steps"]).backward()
                    del value, inputs, labels
                # Infinity computes/checks the norm without additional clipping;
                # Adafactor's own RMS update clipping is enabled instead.
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                torch.cuda.synchronize(device)
                log({"event": "update", "step": step, "loss": total_loss, "learning_rate": learning_rate,
                     "gradient_norm": float(norm), "examples_seen": cursor.position,
                     "microbatches_completed": step * config["gradient_accumulation_steps"],
                     "tokens_seen": dict(tokens_seen),
                     "update_seconds": time.monotonic() - before,
                     "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                     "peak_reserved_bytes": torch.cuda.max_memory_reserved(device)})
                stopping = stop.requested or (session_updates is not None and step - start_step >= session_updates)
                if checkpoint_due(config, step, stopping):
                    save("pause/final" if stopping or step == config["max_steps"] else "periodic")
                if not stopping and step % config["eval_every_steps"] == 0:
                    before = time.monotonic()
                    result = validation_loss(model, tokenizer, datasets["validation"], config, device)
                    log({"event": "validation", "step": step, "seconds": time.monotonic() - before, **result})
                if stopping:
                    break
            save("session end")
            log({"event": "session_end", "step": step, "seconds": time.monotonic() - started})
            atomic_json(run_dir / "status.json", {"status": "complete" if step == config["max_steps"] else "paused",
                        "step": step, "examples_seen": cursor.position, "latest_saved_step": last_saved,
                        "tokens_seen": tokens_seen,
                        "session_id": session_id, "session_seconds": time.monotonic() - started})
    except BaseException as error:
        # No emergency serialization of partially accumulated/updated parameters.
        atomic_json(run_dir / "status.json", {"status": "failed", "latest_saved_step": last_saved,
                    "last_completed_step": step, "session_id": session_id, "error": str(error),
                    "recovery": "Resume the latest COMPLETE checkpoint; inspect any incomplete saves"})
        raise
