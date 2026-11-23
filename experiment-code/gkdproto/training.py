from __future__ import annotations

import json
import math
import os
from pathlib import Path
import random
import shutil
import time
import uuid

from .common import (ROOT, atomic_json, environment, load_rows, read_json, require_assets,
                     resolve, restore_rng, rng_state, seed_everything, sha256, weight_signature)
from .modeling import (autocast, check_pair, decoder_inputs, device_and_precision, divergence_loss,
                       encode_sources, generate_targets, labels_from_ids, load_model,
                        load_tokenizer, reference_labels)
from .budget import awake_seconds, branch_plan, budget_lr_factor, budget_reached

DEFAULTS = {
    "algorithm": "gkd", "student_path": "assets/student", "teacher_path": "assets/teacher",
    "assets_dir": "assets", "prepared_dir": "prepared/smoke", "teacher_cache_dir": None,
    "device": "cuda:0", "teacher_device": "cuda:0", "precision": "bfloat16",
    "allow_cpu": False, "prefix": "summarize: ", "max_source_tokens": 256,
    "max_target_tokens": 32, "microbatch_size": 1, "gradient_accumulation_steps": 4,
    "max_steps": 8, "learning_rate": 0.0001, "weight_decay": 0.0, "warmup_steps": 1,
    "schedule_horizon": 8, "max_grad_norm": 1.0, "beta": 0.5, "lmbda": 0.0,
    "sampling_temperature": 1.0, "off_policy_source": "reference", "seed": 17,
    "data_seed": None, "gradient_checkpointing": True, "save_every": 4,
    "max_wall_minutes": None,
    "optimizer": "adamw", "compute_budget_seconds": None, "budget_schedule": False,
    "mixture_mode": "bernoulli", "pad_to_max_length": False,
    "cuda_memory_fraction": None, "keep_checkpoints": None, "record_code_signature": False,
}


def configuration(path: Path) -> dict:
    provided = read_json(path)
    unknown = set(provided) - (set(DEFAULTS) | {"run_name"})
    if unknown:
        raise ValueError(f"Unknown config fields (possible typo): {sorted(unknown)}")
    config = {**DEFAULTS, **provided}
    if config["data_seed"] is None:
        config["data_seed"] = config["seed"] + 104729
    if not config.get("run_name") or Path(config["run_name"]).name != config["run_name"]:
        raise ValueError("run_name must be a nonempty single directory name")
    for key in ("assets_dir", "prepared_dir", "student_path", "teacher_path", "teacher_cache_dir"):
        if config[key] is not None:
            config[key] = str(resolve(config[key]))
    for key in ("microbatch_size", "gradient_accumulation_steps", "max_steps", "schedule_horizon",
                "max_source_tokens", "max_target_tokens", "save_every"):
        if not isinstance(config[key], int) or isinstance(config[key], bool) or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if config["algorithm"] not in ("gkd", "sft"):
        raise ValueError("algorithm must be gkd or sft")
    if config["off_policy_source"] not in ("reference", "cached_teacher"):
        raise ValueError("off_policy_source must be reference or cached_teacher")
    if not 0 <= config["lmbda"] <= 1 or not 0 <= config["beta"] <= 1:
        raise ValueError("lmbda and beta must be in [0,1]")
    if config["algorithm"] == "sft" and config["lmbda"] != 0:
        raise ValueError("SFT is an off-policy CE control; lmbda must be 0")
    if not 0 <= config["warmup_steps"] <= config["schedule_horizon"] or config["max_steps"] > config["schedule_horizon"]:
        raise ValueError("Invalid warmup/horizon; shorter cells retain the declared full-horizon schedule")
    for key in ("learning_rate", "max_grad_norm", "sampling_temperature"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if config["max_wall_minutes"] is not None and (not math.isfinite(config["max_wall_minutes"]) or config["max_wall_minutes"] <= 0):
        raise ValueError("max_wall_minutes must be positive")
    if config["optimizer"] not in ("adamw", "adafactor"):
        raise ValueError("Unknown optimizer")
    budget = config["compute_budget_seconds"]
    if budget is not None and (isinstance(budget, bool) or not math.isfinite(budget) or budget <= 0):
        raise ValueError("compute_budget_seconds must be finite and positive")
    for key in ("budget_schedule", "pad_to_max_length", "record_code_signature"):
        if type(config[key]) is not bool:
            raise ValueError(f"{key} must be boolean")
    if config["budget_schedule"] and budget is None:
        raise ValueError("Budget-relative schedule requires a compute budget")
    if config["mixture_mode"] not in ("bernoulli", "balanced"):
        raise ValueError("Unknown mixture mode")
    branch_plan(config, random.Random(0))
    fraction = config["cuda_memory_fraction"]
    if fraction is not None and (isinstance(fraction, bool) or not math.isfinite(fraction) or not 0 < fraction <= 1):
        raise ValueError("Invalid CUDA allocator fraction")
    keep = config["keep_checkpoints"]
    if keep is not None and (type(keep) is not int or keep < 2):
        raise ValueError("Keep at least two complete checkpoints")
    return config


class Cursor:
    def __init__(self, rows: list[dict], seed: int, position: int = 0):
        self.rows, self.seed, self.position = rows, seed, position
        self.epoch = None
        self.order = []

    def take(self, count: int) -> list[dict]:
        batch = []
        for _ in range(count):
            epoch, offset = divmod(self.position, len(self.rows))
            if epoch != self.epoch:
                self.order = list(range(len(self.rows)))
                random.Random(self.seed + epoch).shuffle(self.order)
                self.epoch = epoch
            batch.append(self.rows[self.order[offset]])
            self.position += 1
        return batch


def _cache(config: dict, tokenizer, rows: list[dict]) -> dict | None:
    # A fully on-policy run never requires an off-policy teacher cache.
    if config["lmbda"] == 1 or config["off_policy_source"] != "cached_teacher":
        return None
    if config["teacher_cache_dir"] is None:
        raise ValueError("cached_teacher source requires teacher_cache_dir")
    directory = Path(config["teacher_cache_dir"])
    manifest = read_json(directory / "manifest.json")
    required = {"prefix": config["prefix"], "max_source_tokens": config["max_source_tokens"],
                "max_target_tokens": config["max_target_tokens"], "precision": config["precision"],
                "train_sha256": sha256(Path(config["prepared_dir"]) / "train.jsonl"),
                "teacher_signature": weight_signature(Path(config["teacher_path"]))}
    if manifest.get("status") != "complete" or any(manifest.get(k) != v for k, v in required.items()):
        raise ValueError("Teacher cache incomplete or incompatible with this run")
    path = directory / "targets.jsonl"
    if sha256(path) != manifest["targets_sha256"]:
        raise ValueError("Teacher cache hash mismatch")
    targets = {row["id"]: row["target_ids"] for row in load_rows(path)}
    if set(targets) != {row["id"] for row in rows}:
        raise ValueError("Teacher cache IDs do not match the prepared training pool")
    for ids in targets.values():
        if not ids or len(ids) > config["max_target_tokens"] or any(not isinstance(i, int) or i < 0 or i >= len(tokenizer) for i in ids):
            raise ValueError("Invalid cached target IDs")
    return targets


def _save_checkpoint(run_dir, step, student, tokenizer, optimizer, scheduler, cursor, branch_rng, counts, config, signatures, runtime, consumed_seconds=0.0):
    import torch

    parent = run_dir / "checkpoints"
    parent.mkdir(exist_ok=True)
    final = parent / f"step-{step:06d}"
    if final.exists():
        raise FileExistsError(f"Checkpoint already exists: {final}")
    temporary = parent / f".step-{step:06d}-{uuid.uuid4().hex}.incomplete"
    temporary.mkdir()
    student.save_pretrained(temporary, safe_serialization=True)
    tokenizer.save_pretrained(temporary)
    torch.save({"step": step, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "rng": rng_state(), "cursor": cursor.position, "branch_rng": branch_rng.getstate(),
                 "counts": counts, "consumed_seconds": consumed_seconds}, temporary / "training_state.pt")
    atomic_json(temporary / "metadata.json", {"config": config, "signatures": signatures, "runtime": runtime})
    files = {p.name: sha256(p) for p in temporary.iterdir() if p.is_file()}
    atomic_json(temporary / "COMPLETE.json", {"step": step, "state_sha256": files["training_state.pt"], "files": files})
    os.replace(temporary, final)
    atomic_json(run_dir / "latest_checkpoint.json", {"path": str(final), "step": step})
    if config["keep_checkpoints"] is not None:
        owned = []
        for candidate in sorted(parent.glob("step-*")):
            if not candidate.is_dir() or not (candidate / "COMPLETE.json").is_file():
                raise ValueError(f"Refusing to prune unrecognized checkpoint: {candidate}")
            if read_json(candidate / "metadata.json") != {"config": config, "signatures": signatures, "runtime": runtime}:
                raise ValueError(f"Refusing to prune foreign checkpoint: {candidate}")
            owned.append(candidate)
        for candidate in owned[:-config["keep_checkpoints"]]:
            shutil.rmtree(candidate)
    return final


def _synchronize(devices):
    import torch
    for device in set(devices):
        if device.type == "cuda":
            torch.cuda.synchronize(device)


def train(args) -> None:
    import torch
    import torch.nn.functional as F

    config = configuration(args.config.resolve())
    assets_manifest = require_assets(Path(config["assets_dir"]))
    prepared = Path(config["prepared_dir"])
    data_manifest = read_json(prepared / "manifest.json")
    train_path = prepared / "train.jsonl"
    if sha256(train_path) != data_manifest["subsets"]["train"]["sha256"]:
        raise ValueError("Prepared training data changed")
    rows = load_rows(train_path)
    if any(row["split"] != "train" for row in rows):
        raise ValueError("Training input contains non-training examples")
    student_path, teacher_path = Path(config["student_path"]), Path(config["teacher_path"])
    signatures = {"train_sha256": sha256(train_path), "student_init": weight_signature(student_path),
                  "teacher": weight_signature(teacher_path) if config["algorithm"] == "gkd" else None,
                   "assets_manifest_sha256": sha256(Path(config["assets_dir"]) / "manifest.json")}
    if config["record_code_signature"]:
        signatures["code_sha256"] = {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    if config["lmbda"] < 1 and config["off_policy_source"] == "cached_teacher":
        if config["teacher_cache_dir"] is None:
            raise ValueError("cached_teacher source requires teacher_cache_dir")
        cache_dir = Path(config["teacher_cache_dir"])
        signatures["teacher_cache_sha256"] = sha256(cache_dir / "targets.jsonl")
        signatures["teacher_cache_manifest_sha256"] = sha256(cache_dir / "manifest.json")
    device = device_and_precision(config["device"], config["precision"])
    teacher_device = device_and_precision(config["teacher_device"], config["precision"]) if config["algorithm"] == "gkd" else device
    if config["cuda_memory_fraction"] is not None:
        for d in set([device, teacher_device]):
            if d.type == "cuda":
                torch.cuda.set_per_process_memory_fraction(config["cuda_memory_fraction"], d)
    if not config["allow_cpu"] and (device.type == "cpu" or teacher_device.type == "cpu"):
        raise ValueError("CPU model execution requires allow_cpu=true; no implicit offloading")
    runtime = environment()
    runtime["devices"] = {str(d): torch.cuda.get_device_name(d) if d.type == "cuda" else "cpu" for d in set([device, teacher_device])}
    runtime["cuda_runtime"] = torch.version.cuda
    run_dir = args.run_dir.resolve() if args.run_dir else ROOT / "runs" / config["run_name"]
    resume = args.resume.resolve() if args.resume else None
    state = None
    if resume:
        if not run_dir.exists() or not (resume / "COMPLETE.json").exists():
            raise ValueError("Resume requires the existing run directory and a complete checkpoint")
        saved = read_json(resume / "metadata.json")
        if saved != {"config": config, "signatures": signatures, "runtime": runtime}:
            raise ValueError("Resume configuration, initial assets/data, environment or device signature changed")
        if read_json(run_dir / "resolved_config.json") != config:
            raise ValueError("Run directory belongs to a different cell")
        complete = read_json(resume / "COMPLETE.json")
        if sha256(resume / "training_state.pt") != complete["state_sha256"]:
            raise ValueError("Resume state hash mismatch")
        for name, digest in complete.get("files", {}).items():
            if Path(name).name != name or sha256(resume / name) != digest:
                raise ValueError("Resume checkpoint file hash mismatch")
        state = torch.load(resume / "training_state.pt", map_location="cpu", weights_only=True)
    else:
        if run_dir.exists():
            raise FileExistsError(f"Run directory exists; use its own --resume checkpoint or a new directory: {run_dir}")
        run_dir.mkdir(parents=True)
        atomic_json(run_dir / "resolved_config.json", config)
        atomic_json(run_dir / "inputs.json", {"signatures": signatures, "runtime": runtime,
                                               "assets": assets_manifest, "prepared": data_manifest,
                                               "warning": "If using the downloaded base checkpoints, they are not XSum-supervised initializations."})
    atomic_json(run_dir / "status.json", {"status": "initializing", "resume": str(resume) if resume else None})
    seed_everything(config["seed"])
    tokenizer = load_tokenizer(student_path)
    student = load_model(resume or student_path, device, config["precision"], trainable=True, dropout=0.0)
    if config["gradient_checkpointing"]:
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.config.use_cache = False
    teacher = None
    if config["algorithm"] == "gkd":
        teacher_tokenizer = load_tokenizer(teacher_path)
        teacher = load_model(teacher_path, teacher_device, config["precision"], trainable=False)
        check_pair(student, teacher, tokenizer, teacher_tokenizer)
    cache = _cache(config, tokenizer, rows)
    if config["optimizer"] == "adafactor":
        from transformers import Adafactor
        optimizer = Adafactor(student.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"],
                              relative_step=False, scale_parameter=False, warmup_init=False,
                              beta1=None, clip_threshold=1.0)
    else:
        optimizer = torch.optim.AdamW(student.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    # Fixed warmup + constant schedule: stopping early never rescales the schedule.
    def lr_factor(index):
        return min(1.0, (index + 1) / max(1, config["warmup_steps"]))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_factor)
    branch_rng = random.Random(config["seed"] + 1000003)
    cursor = Cursor(rows, config["data_seed"])
    counts = {"on_policy_examples": 0, "off_policy_examples": 0, "on_policy_tokens": 0,
              "off_policy_tokens": 0, "on_policy_microbatches": 0, "off_policy_microbatches": 0}
    step = 0
    consumed_seconds = 0.0
    if state:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        cursor.position = state["cursor"]
        branch_rng.setstate(state["branch_rng"])
        counts = state["counts"]
        step = state["step"]
        consumed_seconds = state.get("consumed_seconds", 0.0)
        restore_rng(state["rng"])
    metric_path = run_dir / "metrics.jsonl"
    if resume and metric_path.exists():
        history = [json.loads(line) for line in metric_path.read_text(encoding="utf-8").splitlines() if line]
        if any(item["step"] > step for item in history):
            shutil.copy2(metric_path, run_dir / f"metrics-before-resume-{uuid.uuid4().hex}.jsonl")
            metric_path.write_text("".join(json.dumps(item) + "\n" for item in history if item["step"] <= step), encoding="utf-8")
    for d in set([device, teacher_device]):
        if d.type == "cuda":
            torch.cuda.reset_peak_memory_stats(d)
    session_started = time.perf_counter()
    last_saved_step = step if resume else -1
    atomic_json(run_dir / "status.json", {"status": "running", "step": step})
    try:
        while step < config["max_steps"] and not budget_reached(consumed_seconds, config["compute_budget_seconds"]):
            _synchronize([device, teacher_device])
            started = awake_seconds()
            optimizer.zero_grad(set_to_none=True)
            if config["budget_schedule"]:
                factor = budget_lr_factor(consumed_seconds, config["compute_budget_seconds"])
                for group in optimizer.param_groups:
                    group["lr"] = config["learning_rate"] * factor
            update_loss = 0.0
            learning_rate_used = optimizer.param_groups[0]["lr"]
            for on_policy in branch_plan(config, branch_rng):
                batch = cursor.take(config["microbatch_size"])
                sources = encode_sources(tokenizer, batch, config["prefix"], config["max_source_tokens"], device,
                                         pad_to_max_length=config["pad_to_max_length"])
                if on_policy:
                    with autocast(device, config["precision"]):
                        ids = generate_targets(student, tokenizer, sources, config["max_target_tokens"], True, config["sampling_temperature"])
                    labels = labels_from_ids(ids, tokenizer.eos_token_id, tokenizer.pad_token_id)
                    del ids
                elif cache is not None:
                    targets = [cache[row["id"]] for row in batch]
                    width = max(map(len, targets))
                    ids = torch.full((len(targets), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
                    position_mask = torch.zeros_like(ids, dtype=torch.bool)
                    for i, target in enumerate(targets):
                        ids[i, :len(target)] = torch.tensor(target, device=device)
                        position_mask[i, :len(target)] = True
                    labels = labels_from_ids(ids, tokenizer.eos_token_id, tokenizer.pad_token_id, position_mask)
                    del ids, position_mask
                else:
                    labels = reference_labels(tokenizer, batch, config["max_target_tokens"], device)
                if config["pad_to_max_length"]:
                    labels = F.pad(labels, (0, config["max_target_tokens"] - labels.shape[1]), value=-100)
                branch = "on_policy" if on_policy else "off_policy"
                counts[f"{branch}_examples"] += len(batch)
                counts[f"{branch}_tokens"] += int(labels.ne(-100).sum().item())
                counts[f"{branch}_microbatches"] += 1
                teacher_logits = None
                if teacher is not None:
                    teacher_sources = {key: value.to(teacher_device) for key, value in sources.items()}
                    teacher_labels = labels.to(teacher_device)
                    with torch.no_grad(), autocast(teacher_device, config["precision"]):
                        teacher_outputs = teacher(**teacher_sources, **decoder_inputs(teacher, teacher_labels), use_cache=False)
                    teacher_logits = teacher_outputs.logits.to(device)
                    del teacher_outputs, teacher_sources, teacher_labels
                with autocast(device, config["precision"]):
                    student_outputs = student(**sources, **decoder_inputs(student, labels), use_cache=False)
                if teacher_logits is not None:
                    loss = divergence_loss(student_outputs.logits, teacher_logits, labels, config["beta"])
                else:
                    losses = F.cross_entropy(student_outputs.logits.float().transpose(1, 2), labels,
                                             reduction="none", ignore_index=-100)
                    mask = labels.ne(-100)
                    loss = (losses.sum(dim=1) / mask.sum(dim=1)).mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite loss; no automatic precision/model changes applied")
                update_loss += float(loss.detach().item()) / config["gradient_accumulation_steps"]
                (loss / config["gradient_accumulation_steps"]).backward()
                del loss, student_outputs, teacher_logits, labels, sources
            if config["optimizer"] == "adafactor":
                gradient_norm = torch.linalg.vector_norm(torch.stack([
                    torch.linalg.vector_norm(p.grad.detach().float()) for p in student.parameters() if p.grad is not None]))
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError("Nonfinite student gradients")
            else:
                gradient_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), config["max_grad_norm"], error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            step += 1
            if teacher is not None and any(p.grad is not None for p in teacher.parameters()):
                raise RuntimeError("Teacher unexpectedly has gradients")
            _synchronize([device, teacher_device])
            update_seconds = awake_seconds() - started
            consumed_seconds += update_seconds
            metrics = {"step": step, "loss": update_loss, "learning_rate_used": learning_rate_used,
                       "gradient_norm": float(gradient_norm), "update_seconds": update_seconds,
                       "consumed_seconds": consumed_seconds, "compute_budget_seconds": config["compute_budget_seconds"],
                       "wall_time_unix": time.time(),
                       "cursor": cursor.position, **counts,
                       "peak_allocated_bytes": {str(d): torch.cuda.max_memory_allocated(d) for d in set([device, teacher_device]) if d.type == "cuda"},
                       "peak_reserved_bytes": {str(d): torch.cuda.max_memory_reserved(d) for d in set([device, teacher_device]) if d.type == "cuda"}}
            with metric_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")
            print(json.dumps(metrics), flush=True)
            budget_stop = (budget_reached(consumed_seconds, config["compute_budget_seconds"])
                           or (config["max_wall_minutes"] is not None and (time.perf_counter() - session_started) / 60 >= config["max_wall_minutes"]))
            if step % config["save_every"] == 0 or step == config["max_steps"] or budget_stop:
                _save_checkpoint(run_dir, step, student, tokenizer, optimizer, scheduler, cursor, branch_rng, counts, config, signatures, runtime, consumed_seconds)
                last_saved_step = step
                atomic_json(run_dir / "status.json", {"status": "running", "step": step,
                                                       "consumed_seconds": consumed_seconds, "counts": counts,
                                                       "latest_checkpoint_step": last_saved_step})
            if budget_stop:
                break
        complete = step == config["max_steps"] or budget_reached(consumed_seconds, config["compute_budget_seconds"])
        atomic_json(run_dir / "status.json", {"status": "complete" if complete else "paused",
                                                 "stop_reason": "compute_budget" if budget_reached(consumed_seconds, config["compute_budget_seconds"]) else "update_or_session_limit",
                                                 "step": step, "counts": counts, "consumed_seconds": consumed_seconds,
                                                 "compute_budget_seconds": config["compute_budget_seconds"],
                                                 "latest_checkpoint_step": last_saved_step})
    except BaseException as error:
        # Do not serialize partly accumulated gradients as a resumable complete update.
        atomic_json(run_dir / "status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                                "last_completed_update": step, "latest_checkpoint_step": last_saved_step,
                                                "error": str(error), "resume_policy": "Use the last COMPLETE checkpoint only"})
        raise
