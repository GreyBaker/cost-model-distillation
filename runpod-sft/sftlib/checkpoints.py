from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import uuid

from .core import atomic_json, check_space, read_json, sync_directory, sync_tree, tree_manifest, verify_tree

PATTERN = re.compile(r"step-\d{8}")


def completed(run_dir):
    parent = run_dir / "checkpoints"
    if not parent.exists():
        return []
    run_id = read_json(run_dir / "run.json")["run_id"]
    found = []
    for path in parent.iterdir():
        if PATTERN.fullmatch(path.name) and path.is_dir() and not path.is_symlink() and (path / "COMPLETE.json").is_file():
            marker = read_json(path / "COMPLETE.json")
            if marker.get("run_id") != run_id or path.name != f"step-{marker['step']:08d}":
                raise ValueError(f"Foreign/mislabeled checkpoint: {path}")
            found.append((marker["step"], path))
    return [path for _, path in sorted(found)]


def resolve(run_dir, selection="latest"):
    choices = completed(run_dir)
    if selection == "latest":
        if not choices:
            raise ValueError("No complete checkpoint to resume/export")
        return choices[-1]
    path = Path(selection).resolve()
    if path not in [item.resolve() for item in choices]:
        raise ValueError("Select only this run's complete checkpoints")
    return path


def verify(path, run_id):
    marker = read_json(path / "COMPLETE.json")
    if marker.get("run_id") != run_id:
        raise ValueError("Checkpoint belongs to another run")
    verify_tree(path, marker["files"])
    return marker


def prune(run_dir):
    choices = completed(run_dir)
    for path in choices[:-2]:
        if path.parent.resolve() != (run_dir / "checkpoints").resolve() or path.is_symlink():
            raise ValueError("Unsafe checkpoint deletion target")
        shutil.rmtree(path)


def save(run_dir, model, tokenizer, state, reason):
    import torch
    run = read_json(run_dir / "run.json")
    step = state["step"]
    parent = run_dir / "checkpoints"
    parent.mkdir(exist_ok=True)
    final = parent / f"step-{step:08d}"
    if final.exists():
        raise FileExistsError(f"Never overwrite checkpoint: {final}")
    model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    check_space(parent, int(model_bytes * 1.1) + 1024**3)
    temporary = parent / (f".step-{step:08d}-" + uuid.uuid4().hex + ".incomplete")
    temporary.mkdir()
    atomic_json(temporary / "OWNER.json", {"run_id": run["run_id"], "step": step})
    model.save_pretrained(temporary, safe_serialization=True, max_shard_size="2GB")
    tokenizer.save_pretrained(temporary)
    torch.save(state, temporary / "training_state.pt")
    atomic_json(temporary / "metadata.json", {"run_id": run["run_id"], "step": step, "reason": reason,
                                             "config": run["config"], "signatures": run["signatures"]})
    sync_tree(temporary)
    files = tree_manifest(temporary)
    atomic_json(temporary / "COMPLETE.json", {"run_id": run["run_id"], "step": step, "files": files})
    sync_directory(temporary)
    os.replace(temporary, final)
    sync_directory(parent)
    atomic_json(run_dir / "latest_checkpoint.json", {"path": final.name, "step": step})
    prune(run_dir)
    return final
