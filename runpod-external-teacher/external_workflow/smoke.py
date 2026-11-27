"""Disposable real GKD updates, native save/reload and smaller-batch restart."""
from __future__ import annotations

import gc
from pathlib import Path
import random
import shutil
import time

from .common import (atomic_json, cached, identity, pair_root, publish, read_json,
                     runtime, sha256, temporary, verify)
from .evaluation import model_path


def bound_inputs(work):
    return {"sft_work": str(pair_root(work)),
            "paths": {"student_checkpoint": str(model_path(work, "student")),
                      "teacher_export": str(model_path(work, "teacher"))}}


def close_tree(a, b):
    import torch
    if isinstance(a, torch.Tensor):
        if a.dtype.is_floating_point:
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)
        elif not torch.equal(a, b):
            raise ValueError("Restart RNG/integer tensor changed")
    elif isinstance(a, dict):
        if set(a) != set(b):
            raise ValueError("Restart state keys changed")
        for key in a:
            close_tree(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        if len(a) != len(b):
            raise ValueError("Restart sequence changed")
        for x, y in zip(a, b):
            close_tree(x, y)
    elif a != b:
        raise ValueError("Restart scalar/accounting changed")


def weights(directory):
    from safetensors.torch import load_file
    result = {}
    for path in sorted(Path(directory).glob("*.safetensors")):
        result.update(load_file(path))
    return result


def run(work):
    import torch
    from gridlib import backend, engine, policy, schedule
    from sftlib import checkpoints
    from sftlib.training import seed
    final = pair_root(work) / "reports/smoke"
    expected = identity(work, "smoke", models={
        role: sha256(model_path(work, role) / "COMPLETE.json") for role in ("student", "teacher")})
    if cached(final, expected):
        return
    runtime(work, gpu=True)
    for role in ("student", "teacher"):
        verify(model_path(work, role), identity(work, "model", role=role))
    settings = policy.settings()
    bound = {"inputs": bound_inputs(work), "settings": settings, "protocol": policy.PROTOCOL,
             "training_design": schedule.DESIGN}
    device, info, tokenizer, student, teacher, data = backend.load_pair(bound, settings)
    if teacher.training or any(p.requires_grad for p in teacher.parameters()):
        raise ValueError("Grid teacher is not frozen/eval")
    versions = [p._version for p in teacher.parameters()]
    seed(settings["seed"])
    rows = [data["train"][i] for i in range(32)]
    batches = policy.chunks(rows, [False] * 16 + [True] * 16, 2, random.Random(17))
    opt = backend.optimizer(student, settings["learning_rate"])
    fit = backend.update(student, teacher, tokenizer, opt, batches, 32, device)
    if versions != [p._version for p in teacher.parameters()] or any(p.grad is not None for p in teacher.parameters()):
        raise ValueError("Frozen teacher was modified by a GKD update")
    del opt, student, teacher, data, batches
    gc.collect()
    torch.cuda.empty_cache()
    scratch = temporary(final)
    a, b = scratch / "restart_a", scratch / "restart_b"
    a.mkdir()
    atomic_json(a / "binding.json", {"kind": "disposable-external-smoke", **bound})
    selected = {"microbatch": 2, "runtime": info}
    atomic_json(a / "pilot.json", {"selected": selected, "kind": "disposable-smoke-not-grid-pilot"})
    item = schedule.trajectory(50)
    before = time.monotonic()
    engine.train(a, bound, item, selected, session_updates=1, microbatch=2)
    gc.collect()
    torch.cuda.empty_cache()
    shutil.copytree(a, b)  # Only our newly created scratch run; no user data overwritten.
    for root in (a, b):
        engine.train(root, bound, item, selected, session_updates=1, microbatch=2)
        gc.collect()
        torch.cuda.empty_cache()
    run_a, run_b = schedule.training_directory(a, item), schedule.training_directory(b, item)
    ca, cb = checkpoints.resolve(run_a), checkpoints.resolve(run_b)
    sa = torch.load(ca / "training_state.pt", map_location="cpu", weights_only=True)
    sb = torch.load(cb / "training_state.pt", map_location="cpu", weights_only=True)
    # Wall-clock duration is not an optimizer/RNG/data-cursor state component.
    for state in (sa, sb):
        state["accounting"].pop("training_seconds")
    close_tree(sa, sb)
    close_tree(weights(ca), weights(cb))
    engine.train(a, bound, item, selected, session_updates=1, microbatch=1)
    state = torch.load(checkpoints.resolve(run_a) / "training_state.pt", map_location="cpu", weights_only=True)
    engine.check_state(state["accounting"], item)
    if (state["accounting"]["step"], state["accounting"]["cursor"], state["accounting"]["points"]) != (3, 96, 144):
        raise ValueError("Smaller-partition restart changed effective batch/point accounting")
    evidence = {"branch_a_step2_complete_sha256": sha256(ca / "COMPLETE.json"),
                "branch_b_step2_complete_sha256": sha256(cb / "COMPLETE.json"),
                "smaller_partition_step3_complete_sha256": sha256(checkpoints.resolve(run_a) / "COMPLETE.json")}
    # These exact directories were created above under our fresh scratch root.
    # Keep hashes/reports, not trained smoke models that could be mistaken for initialization.
    shutil.rmtree(a)
    shutil.rmtree(b)
    atomic_json(scratch / "report.json", {"passed": True, "runtime": info, "fit_update": fit,
                "checkpoint_evidence": evidence,
                "same_partition_restart": "matching optimizer/RNG/cursor and close FP32 weights",
                "smaller_partition_restart": {"microbatch": 1, "step": 3, "examples": 96, "points": 144},
                "seconds": time.monotonic() - before, "local_sft_updates": 0,
                "warning": "All smoke students are discarded; no real cell starts from these checkpoints."})
    publish(scratch, final, expected)
