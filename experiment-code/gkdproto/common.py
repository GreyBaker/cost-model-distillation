from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import random
from importlib import metadata

ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    ids = [str(row["id"]) for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise ValueError(f"Empty data or duplicate IDs: {path}")
    return rows


def write_rows(path: Path, rows) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def environment() -> dict:
    packages = {}
    for package in ("torch", "transformers", "accelerate", "datasets", "numpy", "rouge-score"):
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def require_assets(base: Path) -> dict:
    manifest = read_json(base / "manifest.json")
    if manifest.get("status") != "complete" or not all(x.get("complete") for x in manifest["assets"].values()):
        raise RuntimeError("Assets are incomplete. Finish download_assets.py first.")
    return manifest


def weight_signature(path: Path) -> dict:
    """Lightweight drift guard; downloader provides cryptographic asset hashes.

    Does not rehash the multi-GB teacher on every cell. Mutable custom checkpoints
    must be frozen by the researcher before running the grid.
    """
    files = sorted(p for p in path.iterdir() if p.is_file() and (
        p.suffix in (".bin", ".safetensors") or p.name.endswith(".index.json") or p.name == "config.json"
    ))
    if not files:
        raise ValueError(f"No local model files: {path}")
    return {p.name: {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in files}


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> dict:
    import numpy as np
    import torch
    state = np.random.get_state()
    return {
        "python": random.getstate(), "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
    }


def restore_rng(state: dict) -> None:
    import numpy as np
    import torch
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise RuntimeError("Resume requires the same visible CUDA device count")
        torch.cuda.set_rng_state_all(state["cuda"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
